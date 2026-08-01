#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/map.h"

namespace vslam {

/// 图优化后端（基于 g2o）
/// Phase 1: 提供局部 Bundle Adjustment
/// Phase 2: 增加 Pose Graph Optimization 和 Global BA
class Optimizer {
public:
    /// 局部 Bundle Adjustment：优化当前关键帧及其共视帧 + 地图点
    /// @param camera      相机内参
    /// @param map         地图
    /// @param active_kfs  参与优化的关键帧列表（滑动窗口，第一帧固定）
    static void localBundleAdjustment(
        const Camera& camera,
        Map::Ptr map,
        const std::vector<Frame::Ptr>& active_kfs);

    /// 全局 Bundle Adjustment：优化所有关键帧和地图点
    static void globalBundleAdjustment(const Camera& camera, Map::Ptr map);

    /// (Phase 2) 位姿图优化：回环检测后校正漂移
    static void poseGraphOptimization(Map::Ptr map);
};

} // namespace vslam
