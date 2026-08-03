#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/map.h"

namespace vslam {

/// (Phase 2) 位姿图回环边：关键帧 a ↔ b 的相对位姿约束
struct LoopEdge {
    unsigned long a = 0;      // 回环帧关键帧 id
    unsigned long b = 0;      // 当前帧关键帧 id
    SE3 T_rel;                // 相对位姿（a 相机系 → b 相机系）
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

    /// 全局 Bundle Adjustment：优化所有关键帧和地图点
    static void globalBundleAdjustment(const Camera& camera, Map::Ptr map);

    /// (Phase 2) 位姿图优化：相邻边（里程计约束）+ 回环边校正全局漂移。
    /// 顶点为所有关键帧（T_wc 语义，与 g2o EdgeSE3 群运算一致）；
    /// 第一个关键帧固定锚定坐标系；地图点坐标不动（全局 BA 紧接着做精细修正）。
    /// @param loop_edges  回环边列表：{kf_id_a, kf_id_b, 相对 SE3, 权重}
    static void poseGraphOptimization(
        Map::Ptr map,
        const std::vector<LoopEdge>& loop_edges = {});
};

} // namespace vslam
