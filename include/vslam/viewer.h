#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include <opencv2/core.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace vslam {

/// Pangolin 仪表盘式可视化
class Viewer {
public:
    static constexpr size_t kMaxTrajectoryPoints = 3000;

    Viewer();
    void start();
    void stop();
    bool shouldQuit() const { return quit_.load(); }

    /// 更新视频帧、特征点和轨迹。双目模式下左右目上下排列显示。
    void updateFrame(const cv::Mat& image,
                     const std::vector<cv::KeyPoint>& keypoints,
                     const std::vector<Vec3>& trajectory,
                     const SE3& pose_cs,
                     const cv::Mat& image_right = cv::Mat());

    /// 更新状态文本（显示在右侧状态卡片）
    void setStatus(const std::string& text);

private:
    void renderLoop();

    std::mutex data_mutex_;
    cv::Mat          display_img_;   // 3 通道 RGB，渲染线程直接上传
    cv::Mat          status_img_;    // 独立状态卡片，避免改变视频宽高比
    std::vector<Vec3> trajectory_;   // 只保留可视化需要的最近轨迹点
    SE3               camera_pose_wc_;  // 当前相机在世界系中的位姿
    uint64_t image_revision_ = 0;
    uint64_t status_revision_ = 0;
    uint64_t trajectory_revision_ = 0;

    std::thread render_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> quit_{false};
    std::atomic<bool> show_features_{true};
};

} // namespace vslam
