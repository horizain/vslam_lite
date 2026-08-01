#include "vslam/viewer.h"

#include <pangolin/pangolin.h>
#include <opencv2/imgproc.hpp>

namespace vslam {

Viewer::Viewer() {}

void Viewer::start() {
    if (running_) return;
    running_ = true;
    render_thread_ = std::thread(&Viewer::renderLoop, this);
}

void Viewer::stop() {
    running_ = false;
    quit_ = true;
    if (render_thread_.joinable()) render_thread_.join();
}

void Viewer::updateFrame(const cv::Mat& image,
                         const std::vector<cv::KeyPoint>& keypoints,
                         const std::vector<Vec3>& trajectory) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // 视频流优先彩色：单通道（灰度）输入则转 BGR 显示
    if (image.channels() == 1) {
        cv::cvtColor(image, display_img_, cv::COLOR_GRAY2BGR);
    } else {
        display_img_ = image.clone();
    }

    // 绿色特征点
    for (const auto& kp : keypoints)
        cv::circle(display_img_, kp.pt, 3, cv::Scalar(0, 255, 0), -1);

    // 状态文字叠画在图像底部（黑底白字）
    if (!status_text_.empty()) {
        int h = display_img_.rows;
        // 半透明黑底条
        cv::Mat roi = display_img_(cv::Rect(0, h - 30, display_img_.cols, 30));
        cv::Mat overlay;
        roi.copyTo(overlay);
        overlay = overlay * 0.5;
        overlay.copyTo(roi);
        // 白色文字
        cv::putText(display_img_, status_text_,
                    cv::Point(8, h - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255, 255, 255), 1);
    }

    trajectory_ = trajectory;
}

void Viewer::setStatus(const std::string& text) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    status_text_ = text;
}

void Viewer::renderLoop() {
    pangolin::CreateWindowAndBind("VSLAM Viewer", 1280, 560);
    glEnable(GL_DEPTH_TEST);

    // 左右分栏：左=2D 位置图与轨迹，右=视频流+特征点
    auto& main_panel = pangolin::Display("main")
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .SetLayout(pangolin::LayoutHorizontal);

    pangolin::View& traj_view = pangolin::Display("trajectory").SetAspect(1.0);
    pangolin::View& img_view = pangolin::Display("image").SetAspect(640.0 / 480.0);
    main_panel.AddDisplay(traj_view);  // 左：轨迹
    main_panel.AddDisplay(img_view);   // 右：视频流

    pangolin::GlTexture image_texture(640, 480, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    while (!pangolin::ShouldQuit() && !quit_.load()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ===== 左：2D 轨迹图 =====
        traj_view.Activate();
        {
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();

            double range = 50.0;
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                if (!trajectory_.empty()) {
                    double mx = 0, mz = 0;
                    for (auto& p : trajectory_) { mx = std::max(mx, std::abs(p.x())); mz = std::max(mz, std::abs(p.z())); }
                    range = std::max({mx, mz, 5.0}) * 1.5;
                }
            }
            glOrtho(-range, range, -range, range, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glColor3f(0.4f, 0.4f, 0.4f);
            glBegin(GL_LINES);
            glVertex2f(-range, 0); glVertex2f(range, 0);
            glVertex2f(0, -range); glVertex2f(0, range);
            glEnd();

            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                if (trajectory_.size() >= 2) {
                    glColor3f(0.0f, 0.9f, 1.0f);
                    glLineWidth(2.0f);
                    glBegin(GL_LINE_STRIP);
                    for (auto& p : trajectory_) glVertex2f((float)p.x(), (float)p.z());
                    glEnd();

                    auto& last = trajectory_.back();
                    glPointSize(8.0f);
                    glColor3f(1.0f, 0.3f, 0.3f);
                    glBegin(GL_POINTS);
                    glVertex2f((float)last.x(), (float)last.z());
                    glEnd();
                }
            }
        }

        // ===== 右：视频流 + 绿色特征点 =====
        img_view.Activate();
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (!display_img_.empty()) {
                int w = display_img_.cols, h = display_img_.rows;
                img_view.SetAspect((double)w / (double)h);  // 宽高比自适应图像尺寸
                if (image_texture.width != w || image_texture.height != h)
                    image_texture.Reinitialise(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
                cv::Mat rgb;
                cv::cvtColor(display_img_, rgb, cv::COLOR_BGR2RGB);
                image_texture.Upload(rgb.data, GL_RGB, GL_UNSIGNED_BYTE);
                image_texture.RenderToViewportFlipY();
            }
        }

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    pangolin::DestroyWindow("VSLAM Viewer");
}

} // namespace vslam
