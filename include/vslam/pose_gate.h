#pragma once

#include "vslam/common.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace vslam {

/// 位姿验收质量（M1.1：从 VisualOdometry::PoseQuality 迁移，字段/公式不变）
struct PoseQuality {
    double inlier_ratio = 0.0;
    double pose_rmse    = 0.0;
    double translation  = 0.0;   // 相对运动基线的平移 (m)
    double rotation     = 0.0;   // 相对运动基线的旋转 (rad)
    bool   geometric_ok = false; // 内点/比例/RMSE 达标
    bool   motion_ok    = false; // 连续性达标（无运动基线时恒 true）
};

/// 位姿候选（M1.1，值对象）
struct PoseCandidate {
    SE3    pose_cs;        // 候选位姿（与运动基线同系：世界系 T_cw，调用方已组合 T_ws）
    int    min_inliers = 0;
    double min_ratio   = 0.0;
    double max_rmse    = 0.0;
};

/// 运动预测/先验（M1.1，值对象）
struct MotionPrediction {
    std::optional<SE3> baseline_twc;   // 运动基线（世界系 T_wc）；为空 = 无运动模型
    double max_translation = 0.0;
    double max_rotation    = 0.0;
};

/// 跟踪几何质量（M1.1，值对象）
struct TrackingQuality {
    int    inliers = 0;
    size_t total   = 0;
    double rmse    = 0.0;
};

/// 位姿决策结果（M1.1）
struct PoseDecision {
    bool accepted = false;
    PoseQuality quality;
};

/// 纯几何质量与运动连续性门（M1.1，§5.2）。
///
/// 从 VisualOdometry 迁移的四个函数（pnpReprojectionRmse / acceptPoseCandidate /
/// checkMotionContinuity，以及 acceptPose 的决策部分），公式与默认值保持不变。
/// 只接收值对象，不访问 Map/Atlas/Viewer/日志全局状态，不创建线程。
class PoseGate {
public:
    PoseGate() = default;

    /// 计算 PnP 内点的重投影 RMSE（px）。inliers 为空或全部越界 → +inf。
    [[nodiscard]] static double pnpReprojectionRmse(
        const std::vector<cv::Point3f>& pts3d,
        const std::vector<cv::Point2f>& pts2d,
        const cv::Mat& rvec, const cv::Mat& tvec,
        const std::vector<int>& inliers, const cv::Mat& K);

    /// 候选位姿相对运动基线的平移/旋转（角度），并判定是否在门限内。
    /// 与 vo.cpp 原实现逐行一致：candidate 与 baseline 均为世界系 T_wc。
    [[nodiscard]] static bool checkMotionContinuity(
        const SE3& candidate_pose_cs, const SE3& baseline_twc,
        double max_translation, double max_rotation,
        double& translation, double& rotation);

    /// 几何验收（内点/比例/RMSE）+ 连续性验收（有基线时）。与 vo.cpp 原实现一致。
    [[nodiscard]] static bool acceptPoseCandidate(
        const SE3& candidate_pose_cs, int inliers, size_t total, double rmse,
        int min_inliers, double min_ratio, double max_rmse,
        const std::optional<SE3>& baseline_twc,
        double max_translation, double max_rotation, PoseQuality& quality);

    /// M1.1 统一入口（§5.2）：候选 + 运动先验 + 几何质量 → 决策。
    /// dt 为帧间隔（秒），M1.1 保留但不参与；M3 §7.4 时间归一化连续性使用。
    [[nodiscard]] PoseDecision evaluate(const PoseCandidate& candidate,
                                        const MotionPrediction& prediction,
                                        const TrackingQuality& quality,
                                        double dt) const;
};

} // namespace vslam
