#include "vslam/resource_budget.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vslam {

ResourceBudget::ResourceBudget(const MapBudgetConfig& config) : config_(config) {}

size_t ResourceBudget::matBytes(const cv::Mat& mat) {
    return mat.empty() ? 0 : mat.total() * mat.elemSize();
}

size_t ResourceBudget::descriptorBytes(const Map::Ptr& map) {
    size_t bytes = 0;
    for (const auto& mp : map->getAllMapPoints())
        bytes += matBytes(mp->descriptor);
    for (const auto& kf : map->getAllKeyFrames())
        bytes += matBytes(kf->descriptors);
    return bytes;
}

size_t ResourceBudget::imageBytes(const Map::Ptr& map) {
    size_t bytes = 0;
    for (const auto& kf : map->getAllKeyFrames()) {
        bytes += matBytes(kf->image);
        bytes += matBytes(kf->image_gray);
        bytes += matBytes(kf->image_right);
        bytes += matBytes(kf->image_right_gray);
    }
    return bytes;
}

BudgetStatus ResourceBudget::evaluate(const Map::Ptr& map,
                                      size_t snapshot_bytes) const {
    BudgetStatus s;
    if (!map) return s;
    s.keyframes = map->keyFrameCount();
    s.points = map->mapPointCount();
    s.descriptor_bytes = descriptorBytes(map);
    s.image_bytes = imageBytes(map);
    s.snapshot_bytes = snapshot_bytes;
    s.estimated_total_bytes =
        s.descriptor_bytes + s.image_bytes + s.snapshot_bytes +
        s.keyframes * config_.overhead_bytes_per_keyframe +
        s.points * config_.overhead_bytes_per_point;

    s.over_keyframes = s.keyframes > config_.max_active_keyframes;
    s.over_points = s.points > config_.max_active_points;
    s.over_descriptor =
        s.descriptor_bytes > config_.max_descriptor_mb * MapBudgetConfig::bytes_per_mb;
    s.over_snapshot =
        s.snapshot_bytes > config_.max_snapshot_mb * MapBudgetConfig::bytes_per_mb;
    s.over_total =
        s.estimated_total_bytes > config_.max_total_estimated_mb * MapBudgetConfig::bytes_per_mb;
    s.within_budget = !(s.over_keyframes || s.over_points || s.over_descriptor ||
                        s.over_snapshot || s.over_total);
    return s;
}

size_t ResourceBudget::distinctPointCount(const Frame::Ptr& keyframe) {
    std::unordered_set<MapPointId> ids;
    for (const auto& mp : keyframe->map_points)
        if (mp) ids.insert(mp->id);
    return ids.size();
}

bool ResourceBudget::redundantPair(const Frame::Ptr& older,
                                   const Frame::Ptr& newer,
                                   const Map::Ptr& map) const {
    // 位姿差：pose_cs 的平移即相机光心在子地图系坐标，直接取范数；
    // 旋转差用相对四元数最小表示（处理 q 与 -q 等价）。
    const double dtrans = (newer->pose_cs.t - older->pose_cs.t).norm();
    if (dtrans >= config_.redundant_max_translation_m) return false;
    Eigen::Quaterniond q_rel = newer->pose_cs.q * older->pose_cs.q.inverse();
    const double angle_deg = 2.0 * std::acos(std::clamp(
        std::abs(q_rel.w()), 0.0, 1.0)) * 180.0 / M_PI;
    if (angle_deg >= config_.redundant_max_rotation_deg) return false;

    // 共视重叠率 = shared / min(count_a, count_b)
    const size_t shared = map->sharedObservationCount(older->id, newer->id);
    const size_t count_older = distinctPointCount(older);
    const size_t count_newer = distinctPointCount(newer);
    if (shared == 0 || count_older == 0 || count_newer == 0) return false;
    return static_cast<double>(shared) / std::min(count_older, count_newer) >
           config_.redundant_overlap_threshold;
}

BudgetReclaimResult ResourceBudget::reclaim(
    const Map::Ptr& map,
    const std::unordered_set<KeyframeId>& protected_keyframe_ids,
    const std::unordered_map<SubmapId, std::vector<KeyframeId>>& submap_keyframes,
    size_t snapshot_bytes,
    const std::unordered_map<SubmapId, Map::Ptr>& inactive_submaps,
    const std::function<void(const Frame::Ptr&, const Frame::Ptr&)>&
        before_keyframe_cull) const {
    BudgetReclaimResult r;
    if (!map) return r;
    if (evaluate(map, snapshot_bytes).within_budget) return r;

    // 第 1 步：删除 0 个正式 Observation 的点
    {
        std::vector<MapPointId> ids;
        for (const auto& mp : map->getAllMapPoints())
            if (mp->observationCount() == 0) ids.push_back(mp->id);
        for (const auto id : ids)
            if (map->removeMapPoint(id)) r.removed_zero_obs_points++;
        if (evaluate(map, snapshot_bytes).within_budget) return r;
    }

    // 第 2 步：删除 observationCount < weak_point_min_observations 且超过
    // weak_point_stale_kf_window 个 KF 未被跟踪命中的点。
    // "最近命中 KF" 使用 Map 旁路统计（§6.3），禁止复用 observed_count 语义。
    {
        const size_t kf_count = map->keyFrameCount();
        std::vector<MapPointId> ids;
        for (const auto& mp : map->getAllMapPoints()) {
            if (mp->observationCount() >= config_.weak_point_min_observations)
                continue;
            const size_t last_hit = map->lastHitKeyframeCount(mp->id);
            if (last_hit + config_.weak_point_stale_kf_window < kf_count)
                ids.push_back(mp->id);
        }
        for (const auto id : ids)
            if (map->removeMapPoint(id)) r.removed_weak_stale_points++;
        if (evaluate(map, snapshot_bytes).within_budget) return r;
    }

    // 第 3 步：卸载非活动 KF 的原图和灰度图，仅保留关键点、描述子、位姿
    // 与 Observation（Frame::releaseImages 语义；最近窗口 KF 保留图像）。
    {
        const auto all = map->getAllKeyFrames();
        if (!all.empty()) {
            const KeyframeId last_id = all.back()->id;
            for (const auto& kf : all) {
                if (kf->id + config_.kf_image_keep_recent <= last_id) {
                    kf->releaseImages(false);
                    r.unloaded_kf_images++;
                }
            }
        }
        if (evaluate(map, snapshot_bytes).within_budget) return r;
    }

    // 第 4 步：冗余 KF 剔除（共视重叠 > 阈值 + 相邻位姿差 < 门限；
    // 回环/子地图锚点与最近窗口 KF 必须保留）。每轮至少剔除一个，
    // 剔除后可能产生新的相邻对，循环到关键帧回到预算内或无可剔除。
    while (evaluate(map, snapshot_bytes).over_keyframes) {
        const auto all = map->getAllKeyFrames();
        if (all.size() < 2) break;
        const KeyframeId last_id = all.back()->id;
        size_t culled_this_pass = 0;
        for (size_t i = 1; i < all.size(); i++) {
            const auto& older = all[i - 1];
            const auto& newer = all[i];
            if (protected_keyframe_ids.contains(older->id)) continue;
            if (older->id + config_.kf_image_keep_recent > last_id) continue;
            if (redundantPair(older, newer, map)) {
                // VO 在同一 map 写事务内把持久轨迹从 older 原子重锚到
                // newer；回调完成后才允许删除，避免轨迹引用悬空。
                if (before_keyframe_cull) before_keyframe_cull(older, newer);
                if (map->removeKeyFrame(older->id)) {
                    r.culled_redundant_keyframes++;
                    r.culled_keyframe_ids.push_back(older->id);
                    culled_this_pass++;
                }
            }
        }
        if (culled_this_pass == 0) break;
    }
    if (evaluate(map, snapshot_bytes).within_budget) return r;

    // 第 5 步：冻结超过 max_inactive_submaps 个的非活动子地图并卸载图像
    // 缓存。保留子地图 id 最大的（最新）2 个，冻结更老的；M4 前不进行
    // 磁盘换出（§6.3：不得假装已经具备可靠的换入/换出）。冻结子地图的
    // 弱陈点一并删除（inactive_submaps 提供 Map 时）——点主体不再被访问，
    // KF 描述子保留作重定位骨架，控制 §6.5 RSS 硬门槛（<1GiB）。
    {
        std::vector<SubmapId> ids;
        for (const auto& [sid, kfs] : submap_keyframes) {
            (void)kfs;
            ids.push_back(sid);
        }
        std::ranges::sort(ids);
        if (ids.size() > config_.max_inactive_submaps) {
            r.frozen_submaps = ids.size() - config_.max_inactive_submaps;
            for (size_t i = 0; i < r.frozen_submaps; i++) {
                auto it = submap_keyframes.find(ids[i]);
                if (it == submap_keyframes.end()) continue;
                // 非活动子地图的 KF/点在子地图自己的 Map 内（Submap::map）
                Map::Ptr submap_map;
                auto mit = inactive_submaps.find(ids[i]);
                if (mit != inactive_submaps.end()) submap_map = mit->second;
                for (const auto kf_id : it->second) {
                    const auto kf = submap_map
                        ? submap_map->getKeyFrame(kf_id)
                        : map->getKeyFrame(kf_id);
                    if (kf) {
                        kf->releaseImages(false);
                        r.unloaded_submap_kf_images++;
                    }
                }
                if (submap_map) {
                    const size_t kf_count = submap_map->keyFrameCount();
                    std::vector<MapPointId> ids_to_remove;
                    for (const auto& mp : submap_map->getAllMapPoints()) {
                        if (mp->observationCount() >=
                                config_.weak_point_min_observations)
                            continue;
                        if (submap_map->lastHitKeyframeCount(mp->id) +
                                config_.weak_point_stale_kf_window <
                            kf_count)
                            ids_to_remove.push_back(mp->id);
                    }
                    for (const auto pid : ids_to_remove)
                        if (submap_map->removeMapPoint(pid))
                            r.removed_frozen_submap_points++;
                }
            }
        }
        if (evaluate(map, snapshot_bytes).within_budget) return r;
    }

    // 第 6 步：全部手段耗尽仍超预算 → 停止增加地图（调用方上报
    // Degraded + BackendOverloaded）；不得随机删除锚点。
    r.stopped_map_growth = !evaluate(map, snapshot_bytes).within_budget;
    return r;
}

} // namespace vslam
