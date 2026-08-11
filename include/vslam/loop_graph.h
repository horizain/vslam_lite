#pragma once

#include "vslam/atlas.h"
#include "vslam/optimizer.h"

#include <cstddef>
#include <vector>

namespace vslam {

/// 连续 odom 相机位姿和全局相机位姿的显式分解：
/// T_wc = T_wo * T_oc。回环只改变 T_wo，前端积分只改变 T_oc。
struct GlobalCameraPoseSplit {
    SE3 T_oc;  ///< camera c -> continuous odom o
    SE3 T_wo;  ///< continuous odom o -> global world w
};

[[nodiscard]] GlobalCameraPoseSplit splitGlobalCameraPose(
    const SE3& T_wc, const SE3& T_oc);

/// 冻结前缀端点从 old_endpoint_pose_cs 校正到 new_endpoint_pose_cs 后，
/// 将锁外计算期间新增的尾段 KF 重基到新端点，同时保持
/// T_tail,endpoint = tail_pose_cs * endpoint_pose_cs^-1 严格不变。
[[nodiscard]] SE3 rebaseTailPose(
    const SE3& tail_pose_cs,
    const SE3& old_endpoint_pose_cs,
    const SE3& new_endpoint_pose_cs);

/// 尾段点跟随其参考 KF 重基，保持该 KF 相机坐标不变：
/// new_reference_pose_cs * new_point_s == old_reference_pose_cs * point_s。
[[nodiscard]] Vec3 rebaseTailPoint(
    const Vec3& point_s,
    const SE3& old_reference_pose_cs,
    const SE3& new_reference_pose_cs);

struct EssentialGraphConfig {
    /// 单个子地图进入 g2o 的最大锚点数。首尾和保留的回环端点包含在内。
    size_t max_anchors = 256;
    /// 首选均匀抽样步长；若仍超过 max_anchors，再按全段均匀限幅。
    size_t preferred_stride = 8;
};

/// 子地图内部 Essential Anchor Graph。snapshot 只含锚点及锚点间里程计边/
/// 回环边；full_keyframes 用于求解后把锚点校正渐进传播到冻结前缀全部 KF。
struct EssentialAnchorGraph {
    OptimizationSnapshot snapshot;
    std::vector<KeyframeState> full_keyframes;
};

/// 从逐 KF 冻结前缀构建有界锚点图。首尾必留；回环端点优先；普通 KF 按
/// preferred_stride/最大锚点数均匀抽样。锚点间里程计测量直接由冻结位姿组合，
/// 不从优化后的 live pose 重算。
[[nodiscard]] EssentialAnchorGraph buildEssentialAnchorGraph(
    const OptimizationSnapshot& full,
    const EssentialGraphConfig& config = {});

/// 将锚点 PGO 的世界系校正按关键帧顺序在相邻锚点间做 SE3 校正插值，返回
/// 冻结前缀每个 KF 的 pose_cs 更新。端点与优化结果逐位一致，非锚 KF 不会
/// 被整段刚体搬动到单个边界。
[[nodiscard]] std::vector<PoseUpdate> propagateAnchorCorrections(
    const EssentialAnchorGraph& graph,
    const OptimizationResult& anchor_result);

/// 构建 Atlas 层 Submap Graph。只固定最早子地图消除 gauge；其余历史/活动
/// 子地图都可被 TrackingBridge/Relocalization/LoopClosure 联合重分配。
/// 固定所有历史节点会把累计 yaw 误差锁死并集中到活动尾节点。
[[nodiscard]] OptimizationSnapshot buildSubmapGraph(const Atlas& atlas);

} // namespace vslam
