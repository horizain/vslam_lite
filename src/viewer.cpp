#include "vslam/viewer.h"

#include <pangolin/pangolin.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

namespace vslam {

namespace {

std::vector<std::string> splitStatus(const std::string& status) {
    std::vector<std::string> parts;
    size_t begin = 0;
    while (begin < status.size()) {
        const size_t end = status.find(" | ", begin);
        parts.push_back(status.substr(begin, end == std::string::npos
                                               ? std::string::npos : end - begin));
        if (end == std::string::npos) break;
        begin = end + 3;
    }
    return parts;
}

cv::Mat makeStatusImage(const std::string& status) {
    constexpr int width = 640;
    constexpr int height = 240;
    constexpr int margin = 24;
    constexpr double font_scale = 0.72;
    constexpr int thickness = 2;

    cv::Mat bgr(height, width, CV_8UC3, cv::Scalar(24, 28, 32));
    cv::rectangle(bgr, cv::Point(0, 0), cv::Point(width - 1, height - 1),
                  cv::Scalar(58, 68, 76), 2);
    cv::putText(bgr, "VSLAM / LIVE STATUS", cv::Point(margin, 34),
                cv::FONT_HERSHEY_SIMPLEX, 0.72, cv::Scalar(180, 220, 255), 2,
                cv::LINE_AA);
    cv::line(bgr, cv::Point(margin, 48), cv::Point(width - margin, 48),
             cv::Scalar(70, 82, 92), 1, cv::LINE_AA);

    std::vector<std::string> lines;
    std::string current;
    for (const auto& part : splitStatus(status)) {
        const std::string candidate = current.empty() ? part : current + " | " + part;
        if (!current.empty() &&
            cv::getTextSize(candidate, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                            thickness, nullptr).width > width - margin * 2) {
            lines.push_back(current);
            current = part;
        } else {
            current = candidate;
        }
    }
    if (!current.empty()) lines.push_back(current);
    if (lines.empty()) lines.emplace_back("Waiting for frames...");

    cv::Scalar state_color(224, 224, 224);
    if (status.find("LOST") != std::string::npos) {
        state_color = cv::Scalar(100, 100, 255);
    } else if (status.find("TRACKING") != std::string::npos) {
        state_color = cv::Scalar(100, 235, 130);
    } else if (status.find("RECOVER") != std::string::npos ||
               status.find("RELOC") != std::string::npos) {
        state_color = cv::Scalar(80, 215, 255);
    }

    constexpr int line_height = 38;
    for (size_t i = 0; i < lines.size() && i < 4; ++i) {
        cv::putText(bgr, lines[i], cv::Point(margin, 82 + (int)i * line_height),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, state_color, thickness,
                    cv::LINE_AA);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

double gridStep(double range) {
    const double target = std::max(range / 4.0, 1.0);
    const double power = std::pow(10.0, std::floor(std::log10(target)));
    const double normalized = target / power;
    if (normalized < 2.0) return power;
    if (normalized < 5.0) return 2.0 * power;
    return 5.0 * power;
}

/// 在 3D 地图视图中绘制平行于 XZ 平面的参考网格（KITTI 中 y 为竖直轴）。
void drawGrid3D(double center_x, double center_y, double center_z,
                double range, double step) {
    glColor3f(0.14f, 0.18f, 0.21f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (double x = center_x - range; x <= center_x + range + 1e-6; x += step) {
        glVertex3f((float)x, (float)center_y, (float)(center_z - range));
        glVertex3f((float)x, (float)center_y, (float)(center_z + range));
    }
    for (double z = center_z - range; z <= center_z + range + 1e-6; z += step) {
        glVertex3f((float)(center_x - range), (float)center_y, (float)z);
        glVertex3f((float)(center_x + range), (float)center_y, (float)z);
    }
    glEnd();
    // 中心轴线（X 与 Z 方向）
    glColor3f(0.32f, 0.38f, 0.42f);
    glLineWidth(1.5f);
    glBegin(GL_LINES);
    glVertex3f((float)(center_x - range), (float)center_y, (float)center_z);
    glVertex3f((float)(center_x + range), (float)center_y, (float)center_z);
    glVertex3f((float)center_x, (float)center_y, (float)(center_z - range));
    glVertex3f((float)center_x, (float)center_y, (float)(center_z + range));
    glEnd();
}

/// 用金字塔线框绘制当前相机视锥。T_wc 为相机→世界位姿（TUM 约定）。
void drawCameraFrustum(const SE3& T_wc, double size) {
    const Vec3 origin = T_wc.t;
    const Vec3 corner[4] = {
        T_wc * Vec3(-size, -size, size),
        T_wc * Vec3(size, -size, size),
        T_wc * Vec3(size, size, size),
        T_wc * Vec3(-size, size, size),
    };
    glColor3f(1.0f, 0.62f, 0.18f);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for (int i = 0; i < 4; i++) {
        glVertex3f((float)origin.x(), (float)origin.y(), (float)origin.z());
        glVertex3f((float)corner[i].x(), (float)corner[i].y(),
                   (float)corner[i].z());
    }
    for (int i = 0; i < 4; i++) {
        const int j = (i + 1) % 4;
        glVertex3f((float)corner[i].x(), (float)corner[i].y(),
                   (float)corner[i].z());
        glVertex3f((float)corner[j].x(), (float)corner[j].y(),
                   (float)corner[j].z());
    }
    glEnd();
}

/// 3D 地图视图的共享输入状态：记录最近一次用户交互时间，
/// 供跟随模式判断是否应暂时让出视角（避免每帧重设视角覆盖拖拽）。
struct MapViewInputState {
    std::chrono::steady_clock::time_point last_input =
        std::chrono::steady_clock::now();
};

/// 包装 Handler3D：用户拖动/滚轮/按键时更新 last_input。
class MapViewHandler3D : public pangolin::Handler3D {
public:
    MapViewHandler3D(pangolin::OpenGlRenderState& cam_state,
                     MapViewInputState* input_state)
        : pangolin::Handler3D(cam_state), input_(input_state) {}

    void Keyboard(pangolin::View& view, unsigned char key, int x, int y,
                  bool pressed) override {
        if (pressed) input_->last_input = std::chrono::steady_clock::now();
        pangolin::Handler3D::Keyboard(view, key, x, y, pressed);
    }
    void Mouse(pangolin::View& view, pangolin::MouseButton button, int x, int y,
               bool pressed, int button_state) override {
        if (pressed) input_->last_input = std::chrono::steady_clock::now();
        pangolin::Handler3D::Mouse(view, button, x, y, pressed, button_state);
    }
    void MouseMotion(pangolin::View& view, int x, int y,
                     int button_state) override {
        if (button_state & (pangolin::MouseButtonLeft | pangolin::MouseButtonMiddle |
                            pangolin::MouseButtonRight))
            input_->last_input = std::chrono::steady_clock::now();
        pangolin::Handler3D::MouseMotion(view, x, y, button_state);
    }
    void Special(pangolin::View& view, pangolin::InputSpecial inType,
                 float x, float y, float p1, float p2, float p3, float p4,
                 int button_state) override {
        input_->last_input = std::chrono::steady_clock::now();
        pangolin::Handler3D::Special(view, inType, x, y, p1, p2, p3, p4,
                                     button_state);
    }

private:
    MapViewInputState* input_;
};

} // namespace

Viewer::Viewer() = default;

void Viewer::start() {
    if (running_) return;
    quit_ = false;
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
                         const SE3& pose_cs,
                         const cv::Mat& image_right) {
    if (image.empty()) return;

    auto toBgr = [](const cv::Mat& img) {
        cv::Mat bgr;
        if (img.channels() == 1) {
            cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        } else if (img.channels() == 4) {
            cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
        } else {
            bgr = img.clone();
        }
        return bgr;
    };

    cv::Mat rendered_bgr = toBgr(image);
    if (!image_right.empty()) {
        cv::Mat right_bgr = toBgr(image_right);
        if (right_bgr.cols != rendered_bgr.cols) {
            cv::resize(right_bgr, right_bgr,
                       cv::Size(rendered_bgr.cols,
                                cvRound(right_bgr.rows *
                                        (double)rendered_bgr.cols / right_bgr.cols)));
        }
        if (show_features_.load(std::memory_order_relaxed)) {
            for (const auto& kp : keypoints)
                cv::circle(rendered_bgr, kp.pt, 3, cv::Scalar(0, 235, 120), -1,
                           cv::LINE_AA);
        }
        cv::putText(rendered_bgr, "LEFT", cv::Point(16, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.78, cv::Scalar(0, 220, 255), 2,
                    cv::LINE_AA);
        cv::putText(right_bgr, "RIGHT", cv::Point(16, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.78, cv::Scalar(0, 220, 255), 2,
                    cv::LINE_AA);
        cv::vconcat(rendered_bgr, right_bgr, rendered_bgr);
    } else if (show_features_.load(std::memory_order_relaxed)) {
        for (const auto& kp : keypoints)
            cv::circle(rendered_bgr, kp.pt, 3, cv::Scalar(0, 235, 120), -1,
                       cv::LINE_AA);
    }

    cv::Mat rendered_rgb;
    cv::cvtColor(rendered_bgr, rendered_rgb, cv::COLOR_BGR2RGB);

    std::lock_guard<std::mutex> lock(data_mutex_);
    display_img_ = std::move(rendered_rgb);

    // Viewer 只需要有限长度的尾部轨迹。上游使用 getTrajectory(max_points)
    // 时这里不会再复制随运行时间增长的完整历史。
    constexpr size_t kStoredTrajectoryPoints = Viewer::kMaxTrajectoryPoints;
    const size_t start = trajectory.size() > kStoredTrajectoryPoints
        ? trajectory.size() - kStoredTrajectoryPoints : 0;
    trajectory_.assign(trajectory.begin() + static_cast<std::ptrdiff_t>(start),
                       trajectory.end());
    camera_pose_wc_ = pose_cs.inverse();
    ++image_revision_;
    ++trajectory_revision_;
}

void Viewer::setStatus(const std::string& text) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    status_img_ = makeStatusImage(text);
    ++status_revision_;
}

void Viewer::updateMapPoints(const std::vector<Vec3>& world_points) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    map_points_ = world_points;
    ++map_points_revision_;
}

void Viewer::renderLoop() {
    pangolin::CreateWindowAndBind("VSLAM Dashboard", 1440, 900);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.035f, 0.045f, 0.055f, 1.0f);

    constexpr int panel_width = 220;
    constexpr int map_width = 460;
    constexpr int status_height = 180;

    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panel_width));
    pangolin::View& image_view = pangolin::Display("image")
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width),
                   pangolin::Attach::Pix(-map_width), 1.0);
    // 右上：默认 2D X-Z 轨迹视图；打开 Show 3D map 后切换为可交互 3D 视图
    //（鼠标左键旋转 / 右键拖动 / 滚轮缩放，'r' 复位视角）。
    pangolin::OpenGlRenderState map_cam(
        pangolin::ProjectionMatrix(640, 640, 480, 480, 320, 320, 0.1, 2000.0),
        pangolin::ModelViewLookAt(4.0, -6.0, -6.0, 0.0, 0.0, 0.0,
                                  pangolin::AxisY));
    MapViewInputState map_input;
    pangolin::View& trajectory_view = pangolin::Display("trajectory")
        .SetBounds(pangolin::Attach::Pix(status_height), 1.0,
                   pangolin::Attach::Pix(-map_width), 1.0, 1.0)
        .SetHandler(new MapViewHandler3D(map_cam, &map_input));
    pangolin::View& status_view = pangolin::Display("status")
        .SetBounds(0.0, pangolin::Attach::Pix(status_height),
                   pangolin::Attach::Pix(-map_width), 1.0, 640.0 / 240.0);

    pangolin::Var<bool> show_features("ui.Show features", true, true);
    pangolin::Var<bool> show_3d_map("ui.Show 3D map", false, true);
    pangolin::Var<bool> show_map_points("ui.Show map points", true, true);
    pangolin::Var<bool> show_camera("ui.Show camera", true, true);
    pangolin::Var<bool> show_grid("ui.Show grid", true, true);
    pangolin::Var<bool> follow_camera("ui.Follow camera", true, true);
    pangolin::Var<bool> show_trail("ui.Show trail", true, true);
    pangolin::Var<bool> pause_render("ui.Pause render", false, true);
    pangolin::Var<int> trail_points("ui.Trail points",
                                    (int)Viewer::kMaxTrajectoryPoints,
                                    200, (int)Viewer::kMaxTrajectoryPoints);
    pangolin::Var<double> render_fps("ui.Render FPS", 0.0,
                                    pangolin::META_FLAG_READONLY);
    pangolin::Var<bool> save_screenshot("ui.Save screenshot", false, false);

    pangolin::GlTexture image_texture(1, 1, GL_RGB, false, 0, GL_RGB,
                                      GL_UNSIGNED_BYTE);
    pangolin::GlTexture status_texture(640, 240, GL_RGB, false, 0, GL_RGB,
                                       GL_UNSIGNED_BYTE);

    cv::Mat image_upload;
    cv::Mat status_upload;
    std::vector<Vec3> trajectory_snapshot;
    std::vector<Vec3> map_points_snapshot;
    SE3 camera_pose_snapshot;
    uint64_t uploaded_image_revision = 0;
    uint64_t uploaded_status_revision = 0;
    uint64_t copied_trajectory_revision = 0;
    uint64_t copied_map_points_revision = 0;
    int image_width = 0;
    int image_height = 0;
    bool has_image = false;
    bool image_dirty = false;
    auto fps_time = std::chrono::steady_clock::now();
    int rendered_frames = 0;

    while (!pangolin::ShouldQuit() && !quit_.load()) {
        show_features_.store(static_cast<bool>(show_features),
                             std::memory_order_relaxed);

        if (pangolin::Pushed(save_screenshot))
            pangolin::SaveWindowOnRender("vslam_dashboard");

        const bool paused = static_cast<bool>(pause_render);
        if (!paused) {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (uploaded_image_revision != image_revision_ && !display_img_.empty()) {
                // Mat 的引用计数保证生产者替换下一帧时，本地上传快照仍有效。
                image_upload = display_img_;
                uploaded_image_revision = image_revision_;
                image_width = image_upload.cols;
                image_height = image_upload.rows;
                has_image = true;
                image_dirty = true;
            }
            if (copied_trajectory_revision != trajectory_revision_) {
                trajectory_snapshot = trajectory_;
                camera_pose_snapshot = camera_pose_wc_;
                copied_trajectory_revision = trajectory_revision_;
            }
            if (copied_map_points_revision != map_points_revision_) {
                map_points_snapshot = map_points_;
                copied_map_points_revision = map_points_revision_;
            }
        }
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (uploaded_status_revision != status_revision_ && !status_img_.empty()) {
                status_upload = status_img_;
                uploaded_status_revision = status_revision_;
            }
        }

        if (has_image && image_dirty && image_upload.data) {
            image_view.SetAspect((double)image_width / (double)image_height);
            if (image_texture.width != image_width || image_texture.height != image_height)
                image_texture.Reinitialise(image_width, image_height, GL_RGB, false, 0,
                                           GL_RGB, GL_UNSIGNED_BYTE);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            image_texture.Upload(image_upload.data, GL_RGB, GL_UNSIGNED_BYTE);
            image_dirty = false;
        }
        if (!status_upload.empty()) {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            status_texture.Upload(status_upload.data, GL_RGB, GL_UNSIGNED_BYTE);
            status_upload.release();
        }

        glClear(GL_COLOR_BUFFER_BIT);

        image_view.Activate();
        if (has_image) image_texture.RenderToViewportFlipY();

        // 2D/3D 地图视图共用同一块显示区域；默认保持轻量的 X-Z 俯视图。
        size_t trail_start = 0;
        if (!trajectory_snapshot.empty()) {
            trail_start = trajectory_snapshot.size() > static_cast<size_t>(trail_points)
                ? trajectory_snapshot.size() - static_cast<size_t>(trail_points) : 0;
            if (!static_cast<bool>(show_trail)) trail_start = trajectory_snapshot.size() - 1;
        }

        if (!static_cast<bool>(show_3d_map)) {
            // 默认 2D 模式：世界系 X-Z 俯视轨迹，保留原有箭头/网格交互语义。
            trajectory_view.Activate();
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            double center_x = 0.0;
            double center_z = 0.0;
            if (static_cast<bool>(follow_camera) && !trajectory_snapshot.empty()) {
                center_x = trajectory_snapshot.back().x();
                center_z = trajectory_snapshot.back().z();
            }

            double range = 5.0;
            for (size_t i = trail_start; i < trajectory_snapshot.size(); ++i) {
                range = std::max({range,
                                  std::abs(trajectory_snapshot[i].x() - center_x),
                                  std::abs(trajectory_snapshot[i].z() - center_z)});
            }
            range *= 1.25;
            glOrtho(center_x - range, center_x + range,
                    center_z - range, center_z + range, -1.0, 1.0);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            if (static_cast<bool>(show_grid)) {
                const double step = gridStep(range);
                glColor3f(0.14f, 0.18f, 0.21f);
                glLineWidth(1.0f);
                glBegin(GL_LINES);
                for (double x = center_x - range; x <= center_x + range; x += step) {
                    glVertex2f((float)x, (float)(center_z - range));
                    glVertex2f((float)x, (float)(center_z + range));
                }
                for (double z = center_z - range; z <= center_z + range; z += step) {
                    glVertex2f((float)(center_x - range), (float)z);
                    glVertex2f((float)(center_x + range), (float)z);
                }
                glEnd();
            }

            glColor3f(0.32f, 0.38f, 0.42f);
            glLineWidth(1.5f);
            glBegin(GL_LINES);
            glVertex2f((float)(center_x - range), (float)center_z);
            glVertex2f((float)(center_x + range), (float)center_z);
            glVertex2f((float)center_x, (float)(center_z - range));
            glVertex2f((float)center_x, (float)(center_z + range));
            glEnd();

            if (trajectory_snapshot.size() >= 2 &&
                trail_start < trajectory_snapshot.size() - 1 &&
                static_cast<bool>(show_trail)) {
                glColor3f(0.0f, 0.85f, 0.95f);
                glLineWidth(2.5f);
                glBegin(GL_LINE_STRIP);
                for (size_t i = trail_start; i < trajectory_snapshot.size(); ++i)
                    glVertex2f((float)trajectory_snapshot[i].x(),
                               (float)trajectory_snapshot[i].z());
                glEnd();
            }

            if (static_cast<bool>(show_camera) && !trajectory_snapshot.empty()) {
                const Vec3& last = trajectory_snapshot.back();
                glPointSize(9.0f);
                glColor3f(1.0f, 0.32f, 0.25f);
                glBegin(GL_POINTS);
                glVertex2f((float)last.x(), (float)last.z());
                glEnd();

                const Vec3 forward = camera_pose_snapshot.q * Vec3(0.0, 0.0, 1.0);
                const double forward_xz = std::hypot(forward.x(), forward.z());
                if (forward_xz > 1e-6) {
                    const double arrow_length = std::max(range * 0.10, 0.5);
                    glColor3f(1.0f, 0.72f, 0.18f);
                    glLineWidth(3.0f);
                    glBegin(GL_LINES);
                    glVertex2f((float)last.x(), (float)last.z());
                    glVertex2f((float)(last.x() + arrow_length * forward.x() / forward_xz),
                               (float)(last.z() + arrow_length * forward.z() / forward_xz));
                    glEnd();
                }
            }
        } else {
            // 3D 模式：Follow camera 开启时只在用户空闲约 0.8s 后重新对准，
            // 避免每帧重设视角覆盖用户拖拽/平移/缩放。
            double center_x = 0.0, center_y = 0.0, center_z = 0.0;
            const bool follow_idle = static_cast<bool>(follow_camera) &&
                (std::chrono::steady_clock::now() - map_input.last_input)
                    > std::chrono::milliseconds(800);
            if (follow_idle && !trajectory_snapshot.empty()) {
                const Vec3& p = trajectory_snapshot.back();
                center_x = p.x(); center_y = p.y(); center_z = p.z();
                const double eye_dist = 3.0;
                map_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    center_x + eye_dist, center_y - eye_dist * 0.7,
                    center_z - eye_dist, center_x, center_y, center_z,
                    pangolin::AxisY));
            }

            trajectory_view.Activate(map_cam);
            glClear(GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);

            // 全局视图：网格中心 = 轨迹+点云包围盒中心，范围覆盖全部数据。
            double grid_cx = center_x, grid_cz = center_z;
            double grid_range = 5.0;
            if (!static_cast<bool>(follow_camera)) {
                double min_x = 0.0, max_x = 0.0, min_z = 0.0, max_z = 0.0;
                bool have_extent = false;
                for (const auto& p : map_points_snapshot) {
                    if (!have_extent) {
                        min_x = max_x = p.x();
                        min_z = max_z = p.z();
                        have_extent = true;
                    } else {
                        min_x = std::min(min_x, p.x()); max_x = std::max(max_x, p.x());
                        min_z = std::min(min_z, p.z()); max_z = std::max(max_z, p.z());
                    }
                }
                for (const auto& p : trajectory_snapshot) {
                    if (!have_extent) {
                        min_x = max_x = p.x();
                        min_z = max_z = p.z();
                        have_extent = true;
                    } else {
                        min_x = std::min(min_x, p.x()); max_x = std::max(max_x, p.x());
                        min_z = std::min(min_z, p.z()); max_z = std::max(max_z, p.z());
                    }
                }
                if (have_extent) {
                    grid_cx = 0.5 * (min_x + max_x);
                    grid_cz = 0.5 * (min_z + max_z);
                    grid_range = std::max({5.0, 0.5 * (max_x - min_x),
                                           0.5 * (max_z - min_z)});
                }
            }
            grid_range *= 1.15;

            if (static_cast<bool>(show_grid))
                drawGrid3D(grid_cx, 0.0, grid_cz, grid_range, gridStep(grid_range));

            // 地图点云（世界系 p_w = T_ws · p_s）
            if (static_cast<bool>(show_map_points) && !map_points_snapshot.empty()) {
                glPointSize(2.0f);
                glColor3f(0.25f, 0.85f, 0.35f);
                glBegin(GL_POINTS);
                for (const auto& pt : map_points_snapshot)
                    glVertex3f((float)pt.x(), (float)pt.y(), (float)pt.z());
                glEnd();
            }

            // 轨迹（3D 折线）
            if (trajectory_snapshot.size() >= 2 &&
                trail_start < trajectory_snapshot.size() - 1 &&
                static_cast<bool>(show_trail)) {
                glColor3f(0.0f, 0.85f, 0.95f);
                glLineWidth(2.5f);
                glBegin(GL_LINE_STRIP);
                for (size_t i = trail_start; i < trajectory_snapshot.size(); ++i)
                    glVertex3f((float)trajectory_snapshot[i].x(),
                               (float)trajectory_snapshot[i].y(),
                               (float)trajectory_snapshot[i].z());
                glEnd();
            }

            if (static_cast<bool>(show_camera) && !trajectory_snapshot.empty()) {
                const Vec3& last = trajectory_snapshot.back();
                glPointSize(9.0f);
                glColor3f(1.0f, 0.32f, 0.25f);
                glBegin(GL_POINTS);
                glVertex3f((float)last.x(), (float)last.y(), (float)last.z());
                glEnd();
                drawCameraFrustum(camera_pose_snapshot, 0.5);
            }

            glDisable(GL_DEPTH_TEST);
        }

        status_view.Activate();
        status_texture.RenderToViewportFlipY();

        const auto now = std::chrono::steady_clock::now();
        ++rendered_frames;
        const double elapsed = std::chrono::duration<double>(now - fps_time).count();
        if (elapsed >= 1.0) {
            render_fps = rendered_frames / elapsed;
            rendered_frames = 0;
            fps_time = now;
        }

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }

    // 用户关闭窗口/按 Esc（pangolin::ShouldQuit()）时把 quit_ 同步给主线程，
    // 使 run_slam/run_vo 的"跑完后保持 Viewer 打开"等待循环能正常退出。
    quit_ = true;
    pangolin::DestroyWindow("VSLAM Dashboard");
}

} // namespace vslam
