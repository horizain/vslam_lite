#pragma once

#include "vslam/common.h"
#include "vslam/map.h"
#include "vslam/optimizer.h"

#include <unordered_set>

namespace vslam {

/// 提交结果状态
enum class CommitStatus {
    COMMITTED,   // 已提交（stale 检查通过 + 质量验收通过）
    STALE,       // 快照版本过期，整笔丢弃，实时状态不变
    INVALID,     // 结果无效（valid=false 或质量指标异常），丢弃
    NOT_FOUND    // 目标地图不存在
};

/// M2：唯一 BackendCommitter——只有它能发布地图几何新版本（§14.1/§14.2）。
///
/// 优化器（M1）只产出候选增量（OptimizationResult），本类负责提交前的
/// 全部验收与原子应用：
///   1. stale 检查：快照绑定 topology/geometry revision；M6 追加 rebase
///      协议（§14.2）——几何版本未变时允许提交（拓扑仅追加 KF/点/剔除
///      不影响窗口结果有效性），几何过期整笔丢弃；
///   2. 质量验收：result.valid、有限值、最大校正受限（Local BA 合理校正
///      应为亚米级；超过阈值的 BA 结果视为地图被垃圾污染，整笔拒绝）；
///   3. 对象存活检查：按 id 查找，已被 cull/地图重建移除则跳过该项；
///   4. 一次临界区应用全部 pose/point 更新 + bumpGeometry。
///
/// commit() 的调用方必须已持有 map_mutex_ 独占锁（地图/轨迹锚点在同一
/// 提交中切换版本，见 §14.1-5）。
class BackendCommitter {
public:
    /// 提交优化结果。skip_pose：不回写位姿的 KF id 集合
    /// （活动参考帧保护，M0；该集合同样免除最大校正约束不适用——它只跳过写回）。
    /// @param max_pose_correction  最大允许位姿校正（m）；0 = 不检查
    static CommitStatus commit(
        const Map::Ptr& map,
        const OptimizationResult& result,
        const std::unordered_set<unsigned long>& skip_pose = {},
        double max_pose_correction = 10.0);

    /// stale 检查纯函数（快照版本 vs 实时版本）
    static bool isStale(const OptimizationResult& result, const Map::Ptr& map);

    /// 质量验收纯函数（valid + 有限值 + 最大校正）
    static bool passesQuality(const OptimizationResult& result,
                              double max_pose_correction = 10.0);
};

} // namespace vslam
