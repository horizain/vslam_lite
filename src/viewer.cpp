#include "vslam/viewer.h"

#include <pangolin/pangolin.h>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <algorithm>
#include <cmath>

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
                         const std::vector<Vec3>& trajectory,
                         const SE3& pose_cw,
                         const cv::Mat& image_right) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto to_bgr = [](const cv::Mat& img, cv::Mat& out) {
        if (img.channels() == 1) cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
        else if (img.channels() == 4) cv::cvtColor(img, out, cv::COLOR_BGRA2BGR);
        else out = img.clone();
    };

    // 双目上下拼接，避免 KITTI 这类超宽图横向拼接后宽高比超过 6:1，
    // 在 Viewer 中被严重压扁；特征点仍画在左目上。
    if (!image_right.empty()) {
        cv::Mat left_bgr, right_bgr;
        to_bgr(image, left_bgr);
        to_bgr(image_right, right_bgr);
        if (right_bgr.cols != left_bgr.cols) {
            cv::resize(right_bgr, right_bgr,
                       cv::Size(left_bgr.cols,
                                cvRound(right_bgr.rows * (double)left_bgr.cols / right_bgr.cols)));
        }
        for (const auto& kp : keypoints)
            cv::circle(left_bgr, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
        cv::putText(left_bgr, "LEFT", cv::Point(12, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
        cv::putText(right_bgr, "RIGHT", cv::Point(12, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
        cv::vconcat(left_bgr, right_bgr, display_img_);
    } else {
        // 单目：视频流优先彩色（单通道灰度则转 BGR）
        to_bgr(image, display_img_);
        // 绿色特征点
        for (const auto& kp : keypoints)
            cv::circle(display_img_, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
    }

    // 状态栏放在图像外部，不遮挡画面。按分隔符自动换行，避免为了塞进
    // 窄画面而把字体缩得看不清。
    if (!status_text_.empty()) {
        const double font_scale = std::clamp(display_img_.cols / 1400.0, 0.65, 0.9);
        const int thickness = 2;
        const int margin = 12;
        const int max_width = display_img_.cols - margin * 2;

        std::vector<std::string> parts;
        size_t begin = 0;
        while (begin < status_text_.size()) {
            size_t end = status_text_.find(" | ", begin);
            parts.push_back(status_text_.substr(
                begin, end == std::string::npos ? end : end - begin));
            if (end == std::string::npos) break;
            begin = end + 3;
        }

        std::vector<std::string> lines;
        std::string line;
        for (const auto& part : parts) {
            std::string candidate = line.empty() ? part : line + " | " + part;
            if (!line.empty() && cv::getTextSize(candidate, cv::FONT_HERSHEY_SIMPLEX,
                                                  font_scale, thickness, nullptr).width > max_width) {
                lines.push_back(line);
                line = part;
            } else {
                line = std::move(candidate);
            }
        }
        if (!line.empty()) lines.push_back(line);

        int baseline = 0;
        const int text_height = cv::getTextSize("Ag", cv::FONT_HERSHEY_SIMPLEX,
                                                font_scale, thickness, &baseline).height;
        const int line_height = text_height + baseline + 8;
        const int status_height = margin + line_height * (int)lines.size();
        cv::copyMakeBorder(display_img_, display_img_, 0, status_height, 0, 0,
                           cv::BORDER_CONSTANT, cv::Scalar(24, 24, 24));
        const int image_height = display_img_.rows - status_height;
        for (size_t i = 0; i < lines.size(); i++) {
            cv::putText(display_img_, lines[i],
                        cv::Point(margin, image_height + margin + text_height + (int)i * line_height),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale,
                        cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
    }

    trajectory_ = trajectory;
    camera_pose_wc_ = pose_cw.inverse();
}

void Viewer::setStatus(const std::string& text) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    status_text_ = text;
}

void Viewer::renderLoop() {
    pangolin::CreateWindowAndBind("VSLAM Viewer", 1600, 900);
    glEnable(GL_DEPTH_TEST);

    // 视频区固定占 72%，轨迹区占 28%；不再让 LayoutHorizontal 等分后
    // 再按宽高比压缩视频，保证单目和双目画面都有足够显示面积。
    pangolin::View& traj_view = pangolin::Display("trajectory")
        .SetBounds(0.0, 1.0, 0.0, 0.28, 1.0);
    pangolin::View& img_view = pangolin::Display("image")
        .SetBounds(0.0, 1.0, 0.28, 1.0, 640.0 / 480.0);

    pangolin::GlTexture image_texture(640, 480, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    while (!pangolin::ShouldQuit() && !quit_.load()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ===== 左：2D 轨迹图 =====
        traj_view.Activate();
        {
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();

            double range = 50.0;
            // 只显示最近 kMaxTrajPts 个轨迹点（长跑后轨迹点几十万，全量绘制会卡渲染）
            constexpr size_t kMaxTrajPts = 3000;
            size_t traj_start = 0;
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                if (trajectory_.size() > kMaxTrajPts)
                    traj_start = trajectory_.size() - kMaxTrajPts;
                if (!trajectory_.empty()) {
                    // C++23 ranges：views::drop+transform 映射坐标 → ranges::max 取最大绝对值
                    auto xs = trajectory_ | std::views::drop(traj_start)
                                          | std::views::transform([](const Vec3& p) { return std::abs(p.x()); });
                    auto zs = trajectory_ | std::views::drop(traj_start)
                                          | std::views::transform([](const Vec3& p) { return std::abs(p.z()); });
                    range = std::max({std::ranges::max(xs), std::ranges::max(zs), 5.0}) * 1.5;
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
                    for (size_t i = traj_start; i < trajectory_.size(); i++)
                        glVertex2f((float)trajectory_[i].x(), (float)trajectory_[i].z());
                    glEnd();

                    auto& last = trajectory_.back();
                    glPointSize(8.0f);
                    glColor3f(1.0f, 0.3f, 0.3f);
                    glBegin(GL_POINTS);
                    glVertex2f((float)last.x(), (float)last.z());
                    glEnd();

                    // 红色箭头表示相机光轴在 x-z 平面的朝向。原地旋转时位置应固定，
                    // 只有箭头转动，便于直接区分“姿态变化”和“平移轨迹”。
                    Vec3 forward = camera_pose_wc_.q * Vec3(0.0, 0.0, 1.0);
                    double forward_xz = std::hypot(forward.x(), forward.z());
                    if (forward_xz > 1e-6) {
                        double arrow_length = std::max(range * 0.08, 0.5);
                        glLineWidth(3.0f);
                        glBegin(GL_LINES);
                        glVertex2f((float)last.x(), (float)last.z());
                        glVertex2f((float)(last.x() + arrow_length * forward.x() / forward_xz),
                                   (float)(last.z() + arrow_length * forward.z() / forward_xz));
                        glEnd();
                    }
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
                if (!rgb.isContinuous()) rgb = rgb.clone();
                // OpenGL 默认 GL_UNPACK_ALIGNMENT=4；RGB 每像素 3 字节，当图像行宽
                // 不是 4 的倍数时会跨行错读，表现为视频逐行倾斜。
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
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
