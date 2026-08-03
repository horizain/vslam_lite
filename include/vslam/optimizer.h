#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/map.h"

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
    static void localBundleAdjustment(
        const Camera& camera,
        Map::Ptr map,
        const std::vector<Frame::Ptr>& active_kfs,
        int max_iterations = 10);

    /// 全局 motion-only BA：固定地图点，优化所有关键帧位姿
    static void globalBundleAdjustment(const Camera& camera, Map::Ptr map,
                                       int max_iterations = 20);

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
