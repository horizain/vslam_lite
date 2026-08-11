#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/point_cloud.h"
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
    /// 轨迹保留上限：覆盖 KITTI 00（4541 帧）/ EuRoC V1（2912 帧）全程，
    /// 避免长序列运行时前半段轨迹从可视化中消失。内存约 20000×24B ≈ 0.5MB。
    static constexpr size_t kMaxTrajectoryPoints = 20000;
    /// 3D 地图点云可视化硬上限（与 MapBudget.max_active_points 一致的量级）
    static constexpr size_t kMaxMapPoints = 60000;

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

    /// 更新世界系地图点云（p_w = T_ws · p_s），供 3D 地图视图绘制。
    /// 上游传入的向量按 kMaxMapPoints 上限截取；空向量表示隐藏点云。
    void updateMapPoints(const std::vector<Vec3>& world_points);

    /// 更新当前帧双目 RGB 点云；只在 3D 模式和开关打开时绘制。
    void updateColoredPointCloud(const std::vector<ColoredPoint>& points);

private:
    void renderLoop();

    std::mutex data_mutex_;
    cv::Mat          display_img_;   // 3 通道 RGB，渲染线程直接上传
    cv::Mat          status_img_;    // 独立状态卡片，避免改变视频宽高比
    std::vector<Vec3> trajectory_;   // 只保留可视化需要的最近轨迹点
    std::vector<Vec3> map_points_;   // 世界系地图点云（p_w = T_ws · p_s）
    std::vector<ColoredPoint> colored_points_;  // 当前帧双目 RGB 点云
    SE3               camera_pose_wc_;  // 当前相机在世界系中的位姿
    uint64_t image_revision_ = 0;
    uint64_t status_revision_ = 0;
    uint64_t trajectory_revision_ = 0;
    uint64_t map_points_revision_ = 0;
    uint64_t colored_points_revision_ = 0;

    std::thread render_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> quit_{false};
    std::atomic<bool> show_features_{true};
};

} // namespace vslam
