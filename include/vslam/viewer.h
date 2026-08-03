#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include <opencv2/core.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>

namespace vslam {

/// Pangolin 双窗口可视化
class Viewer {
public:
    Viewer();
    void start();
    void stop();
    bool shouldQuit() const { return quit_.load(); }

    /// 更新视频帧（优先彩色，单通道自动转 BGR）+ 特征点（绿色）+ 轨迹。
    /// 双目模式下传入 image_right → 左右目上下拼接显示；单目可省略。
    void updateFrame(const cv::Mat& image,
                     const std::vector<cv::KeyPoint>& keypoints,
                     const std::vector<Vec3>& trajectory,
                     const SE3& pose_cw,
                     const cv::Mat& image_right = cv::Mat());

    /// 更新状态文本（叠加在图像上）
    void setStatus(const std::string& text);

private:
    void renderLoop();

    std::mutex data_mutex_;
    cv::Mat          display_img_;   // 3 通道 BGR：灰度图 + 绿色特征点 + 状态文字
    std::vector<Vec3> trajectory_;
    SE3               camera_pose_wc_;  // 当前相机在世界系中的位姿，用于绘制朝向
    std::string       status_text_;

    std::thread render_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> quit_{false};
};

} // namespace vslam
