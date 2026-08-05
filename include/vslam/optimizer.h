#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/map.h"

#include <optional>

namespace vslam {

/// (Phase 2) 位姿图边：关键帧 a ↔ b 的相对位姿约束
struct LoopEdge {
    unsigned long a = 0;      // 起点关键帧 id
    unsigned long b = 0;      // 终点关键帧 id
    SE3 T_rel;                // X_b = X_a * T_rel（X 为 T_wc）
    double weight = 1.0;      // 信息权重（共视多 → 置信高）
};

/// 图优化后端（基于 g2o）
/// Phase 1: 提供局部 Bundle Adjustment
/// Phase 2: 增加 Pose Graph Optimization 和 Global BA
class Optimizer {
public:
    /// 局部 Bundle Adjustment：优化当前关键帧及其共视帧 + 地图点
    /// @param camera          相机内参
    /// @param map             地图
    /// @param active_kfs      参与优化的关键帧列表（滑动窗口，第一帧固定）
    /// @param max_iterations  g2o 最大迭代次数
    /// @param fix_points      地图点是否固定（motion-only）；默认按传感器自动选择：
    ///                        单目固定（三角化尺度由 recoverPose 决定，点自由会引入
    ///                        尺度 gauge 发散），双目/RGB-D 放开（视差绝对尺度可观测）
    /// @param max_points      参与优化的地图点数量上限（按观测数降序截断；
    ///                        地图膨胀后控制 BA 规模，防止单次 LM 到分钟级）
    static void localBundleAdjustment(
        const Camera& camera,
        Map::Ptr map,
        const std::vector<Frame::Ptr>& active_kfs,
        int max_iterations = 10,
        std::optional<bool> fix_points = std::nullopt,
        size_t max_points = 4000);

    /// 全局 BA：优化所有关键帧（+ 地图点，双目下按 fix_points 自动放开）
    static void globalBundleAdjustment(const Camera& camera, Map::Ptr map,
                                       int max_iterations = 20,
                                       std::optional<bool> fix_points = std::nullopt,
                                       size_t max_points = 4000);

    /// (Phase 2) 位姿图优化：相邻边（里程计约束）+ 回环边校正全局漂移。
    /// 顶点为所有关键帧（T_wc 语义，与 g2o EdgeSE3 群运算一致）；
    /// 第一个关键帧固定锚定坐标系；地图点在调用方按参考关键帧同步。
    /// @param odometry_edges  关键帧插入时冻结的里程计边
    /// @param loop_edges      累积保留的回环边
    /// 返回 false 表示后端不可用、约束不足或未执行优化。
    static bool poseGraphOptimization(
        Map::Ptr map,
        const std::vector<LoopEdge>& odometry_edges,
        const std::vector<LoopEdge>& loop_edges = {});
};

} // namespace vslam
