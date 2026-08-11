#include "vslam/backend_committer.h"

#include <cmath>
#include <limits>

namespace vslam {

bool BackendCommitter::isStale(const OptimizationResult& result,
                               const Map::Ptr& map) {
    if (!map) return true;
    // M6：几何版本是唯一硬性 stale 判据（拓扑追加不影响窗口结果有效性）；
    // 严格拓扑检查保留给回环（全图）提交路径（handleLoopCorrection 阶段 3）
    return result.base_geometry_revision != map->geometryRevision();
}

bool BackendCommitter::passesQuality(const OptimizationResult& result,
                                     double max_pose_correction) {
    if (!result.valid) return false;
    if (!std::isfinite(result.metrics.max_correction)) return false;
    if (max_pose_correction > 0.0
        && result.metrics.max_correction > max_pose_correction) {
        LOG_WARN("Committer rejected oversized correction: "
                 << result.metrics.max_correction << "m (limit "
                 << max_pose_correction << "m)");
        return false;
    }
    return true;
}

CommitStatus BackendCommitter::commit(
    const Map::Ptr& map,
    const OptimizationResult& result,
    const std::unordered_set<unsigned long>& skip_pose,
    double max_pose_correction,
    const Map::Ptr& expected_map) {
    // 调用方约定已持 map_mutex_ 独占锁（§14.1-5 原子提交）。
    if (!map) return CommitStatus::NOT_FOUND;
    // revision 只在单个 Map 实例内有意义。不同子地图会从相同版本/对象 id
    // 重新起步，因此提交者必须先验证任务绑定的 Map 身份，不能仅比较数值版本。
    if (expected_map && map != expected_map) {
        LOG_WARN("Committer dropped result for a different Map instance");
        return CommitStatus::STALE;
    }

    // 1. stale 检查（M6 追加 rebase 协议，§14.2）：
    //    - 几何版本已变（其他提交/对齐发布过新几何）→ 结果基于旧坐标，
    //      整笔丢弃（Local BA 窗口结果不可跨几何版本提交）；
    //    - 拓扑版本已变但几何未变（仅尾部追加 KF/点/剔除）→ 允许提交：
    //      窗口结果的位姿/点更新作用于快照时刻已存在的对象，新增对象
    //      不在结果中、不会被覆盖，结果本身仍有效。这解除了异步后端
    //      "前端每帧插 KF → 拓扑常变 → BA 全部过期"的死锁。
    if (result.base_geometry_revision != map->geometryRevision()) {
        LOG_WARN("Committer dropped stale geometry result (snap geo "
                 << result.base_geometry_revision << " vs live "
                 << map->geometryRevision() << ")");
        return CommitStatus::STALE;
    }
    // 2. 质量验收
    if (!passesQuality(result, max_pose_correction)) return CommitStatus::INVALID;

    // 3. 对象存活预检。拓扑追加不使窗口结果失效，但预算线程可能在 BA
    // 计算期间剔除快照对象；这时不得静默跳过一部分 pose/point 后仍发布
    // COMMITTED，否则同一优化结果会形成“位姿新、点旧”的部分事务。
    for (const auto& u : result.poses) {
        if (skip_pose.count(u.id)) continue;
        if (!map->getKeyFrame(u.id)) {
            LOG_WARN("Committer dropped result with missing keyframe " << u.id);
            return CommitStatus::STALE;
        }
    }
    for (const auto& p : result.points) {
        if (!map->getMapPoint(p.id)) {
            LOG_WARN("Committer dropped result with missing map point " << p.id);
            return CommitStatus::STALE;
        }
    }

    // 4. 一次临界区应用全部更新
    for (const auto& u : result.poses) {
        if (skip_pose.count(u.id)) continue;  // 跳过活动参考帧写回（M0）
        auto real = map->getKeyFrame(u.id);
        if (real) real->pose_cs = u.pose_cs;
    }
    for (const auto& p : result.points) {
        auto real = map->getMapPoint(p.id);
        if (real) real->pos_s = p.pos_s;
    }
    // 5. 唯一提交者发布几何新版本
    map->bumpGeometry();
    return CommitStatus::COMMITTED;
}

} // namespace vslam
