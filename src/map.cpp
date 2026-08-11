#include "vslam/map.h"

#include <algorithm>
#include <set>
#include <unordered_set>
#include <vector>

namespace vslam {

void Map::insertMapPoint(MapPoint::Ptr mp) {
    if (!mp) return;
    if (map_points_.find(mp->id) == map_points_.end()) {
        map_points_[mp->id] = mp;
        if (mp->id >= next_mp_id_) next_mp_id_ = mp->id + 1;
        mp_count_.fetch_add(1, std::memory_order_relaxed);
        bumpTopology();
    }
}

MapPoint::Ptr Map::getMapPoint(unsigned long id) const {
    auto it = map_points_.find(id);
    return (it != map_points_.end()) ? it->second : nullptr;
}

bool Map::removeMapPoint(MapPointId id) {
    auto it = map_points_.find(id);
    if (it == map_points_.end()) return false;

    // 先清理双向关系，再从地图集合和计数中移除；调用方持有写锁时，
    // 这整个序列对其他地图读者是一个原子事务。
    const auto map_point = it->second;
    removeMapPointsUnlocked({map_point});
    map_points_.erase(it);
    mp_count_.fetch_sub(1, std::memory_order_relaxed);
    bumpTopology();
    return true;
}

size_t Map::removeMapPoints(const std::vector<MapPointId>& ids) {
    std::vector<MapPoint::Ptr> removed;
    removed.reserve(ids.size());
    for (const auto id : ids) {
        auto it = map_points_.find(id);
        if (it == map_points_.end()) continue;
        removed.push_back(it->second);
        map_points_.erase(it);
    }
    if (removed.empty()) return 0;
    mp_count_.fetch_sub(removed.size(), std::memory_order_relaxed);
    removeMapPointsUnlocked(removed);
    bumpTopology();
    return removed.size();
}

void Map::cullMapPoints(int min_observations) {
    // 按正式关键帧观测数剔除弱观测点，并同步清空双向引用。
    // 历史教训（2026-08-05 两版实验均失败）：
    // ① "只删孤儿点" → 地图无限膨胀，长序列内存失控（perf7 OOM/崩溃）；
    // ② "保留最近 30 KF 引用的单观测点" → 单观测点里垃圾点（错误深度/三角化）
    //    比例高，保留后 PnP 内点率骤降、跟踪 LOST 增 9 倍、子地图重建 37 次
    //    （perf8 ATE 174.8m）。观测数 < 2 的点对跟踪是净负资产，直接删除。
    std::vector<MapPoint::Ptr> removed;
    for (auto it = map_points_.begin(); it != map_points_.end(); ) {
        if (it->second->observationCount() < static_cast<size_t>(
                std::max(0, min_observations))) {
            removed.push_back(it->second);
            it = map_points_.erase(it);
        } else {
            ++it;
        }
    }
    if (!removed.empty())
        mp_count_.fetch_sub(removed.size(), std::memory_order_relaxed);
    // 先逐点清理正式 observation/covisibility，再对所有 KF slot 只扫描一次；
    // 这样长序列批量剔除不会对每个点重复遍历全部关键帧。
    removeMapPointsUnlocked(removed);
    if (!removed.empty())
        bumpTopology();
}

bool Map::removeKeyFrame(KeyframeId id) {
    auto it = keyframes_.find(id);
    if (it == keyframes_.end()) return false;
    const auto keyframe = it->second;
    // 撤销该 KF 上全部正式观测（反向集合 + 共视计数），并清空 slot。
    // 调用方持有写锁时，这整个序列对其他地图读者是一个原子事务。
    for (size_t i = 0; i < keyframe->map_points.size(); i++) {
        const auto map_point = keyframe->map_points[i];
        if (!map_point) continue;
        const Observation observation{id, static_cast<FeatureIndex>(i)};
        if (map_point->hasObservation(observation)) {
            removeCovisibilityUnlocked(observation, map_point);
            map_point->removeObservation(observation);
        }
        keyframe->map_points[i].reset();
    }
    keyframes_.erase(it);
    kf_count_.fetch_sub(1, std::memory_order_relaxed);
    bumpTopology();
    return true;
}

void Map::recordTrackingHit(MapPointId map_point_id) {
    // M2.2：旁路统计（§6.3）。普通帧的临时关联也计入"最近命中"，
    // 与 observed_count 语义无关；地图清空时一并重置。
    last_hit_kf_[map_point_id] = kf_count_.load(std::memory_order_relaxed);
}

size_t Map::lastHitKeyframeCount(MapPointId map_point_id) const {
    auto it = last_hit_kf_.find(map_point_id);
    return (it != last_hit_kf_.end()) ? it->second : 0;
}

bool Map::setObservation(const Frame::Ptr& keyframe,
                         FeatureIndex feature_index,
                         const MapPoint::Ptr& map_point) {
    if (!keyframe) return false;
    auto it = keyframes_.find(keyframe->id);
    if (it == keyframes_.end() || it->second.get() != keyframe.get()) return false;
    const bool changed = setObservationUnlocked(keyframe, feature_index, map_point);
    if (changed) bumpTopology();
    return changed;
}

bool Map::setObservationUnlocked(const Frame::Ptr& keyframe,
                                 FeatureIndex feature_index,
                                 const MapPoint::Ptr& map_point) {
    if (!keyframe || !map_point ||
        feature_index >= keyframe->keypoints.size() ||
        feature_index >= keyframe->map_points.size())
        return false;
    auto mp_it = map_points_.find(map_point->id);
    if (mp_it == map_points_.end() || mp_it->second.get() != map_point.get())
        return false;

    const Observation observation{keyframe->id, feature_index};
    const auto old = keyframe->map_points[feature_index];
    const auto existing_feature = map_point->featureIndex(keyframe->id);
    if (existing_feature && *existing_feature != feature_index &&
        (!old || old.get() != map_point.get())) return false;
    if (old && old.get() == map_point.get()) {
        if (existing_feature && *existing_feature != feature_index) {
            // 当前 slot 是重复 stale slot；保留另一个正式 slot，清掉该
            // KF 上同点的全部重复缓存。
            return clearDuplicateSlotsUnlocked(keyframe, map_point,
                                               existing_feature);
        }
        if (map_point->hasObservation(observation))
            return clearDuplicateSlotsUnlocked(keyframe, map_point,
                                               feature_index);
        // 先写入反向正式观测；失败时不得清理当前 slot 或修改共视。
        if (!addObservationUnlocked(observation, map_point)) return false;
        clearDuplicateSlotsUnlocked(keyframe, map_point, feature_index);
        return true;
    }

    // 目标点若已在该 KF 的另一个 feature 上正式观测，不能破坏该合法
    // 关系，也不能先动当前 slot 再返回失败。
    if (existing_feature) return false;

    // 目标反向观测必须先成功建立。这样后续旧点清理即使遇到历史 stale
    // 状态，也不会留下“slot 已换、反向观测未写入”的半事务。
    if (!addObservationUnlocked(observation, map_point)) return false;

    if (old) {
        const auto old_feature = old->featureIndex(keyframe->id);
        if (old_feature) {
            if (*old_feature == feature_index) {
                const Observation old_observation{keyframe->id, *old_feature};
                removeCovisibilityUnlocked(old_observation, old);
                old->removeObservation(old_observation);
                // 正式 slot 被重绑后，其余同点 slot 都是 stale 缓存。
                clearDuplicateSlotsUnlocked(keyframe, old, std::nullopt);
            } else {
                // 当前只是重复 stale slot；不得移除另一个合法 formal slot。
                clearDuplicateSlotsUnlocked(keyframe, old, old_feature);
            }
        } else {
            // 没有 formal 观测的旧指针全部视为 stale。
            clearDuplicateSlotsUnlocked(keyframe, old, std::nullopt);
        }
    }
    keyframe->map_points[feature_index] = map_point;
    return true;
}

bool Map::clearObservation(KeyframeId keyframe_id, FeatureIndex feature_index) {
    const bool changed = clearObservationUnlocked(keyframe_id, feature_index);
    if (changed) bumpTopology();
    return changed;
}

bool Map::clearObservationUnlocked(KeyframeId keyframe_id,
                                   FeatureIndex feature_index) {
    auto kf_it = keyframes_.find(keyframe_id);
    if (kf_it == keyframes_.end() ||
        feature_index >= kf_it->second->map_points.size()) return false;
    auto& slot = kf_it->second->map_points[feature_index];
    if (!slot) return false;
    // 复制 shared_ptr；clearDuplicateSlotsUnlocked 会清空 slot 本身，
    // 不能让随后用于扫描 stale slot 的 map_point 引用随之变 null。
    const auto map_point = slot;
    const auto formal_feature = map_point->featureIndex(keyframe_id);
    if (formal_feature && *formal_feature == feature_index) {
        const Observation observation{keyframe_id, *formal_feature};
        removeCovisibilityUnlocked(observation, map_point);
        map_point->removeObservation(observation);
        // 正式观测被清除后，所有同点重复 slot 都必须一起清掉。
        clearDuplicateSlotsUnlocked(kf_it->second, map_point, std::nullopt);
    } else if (formal_feature) {
        // 当前是 stale duplicate；保留另一个合法 formal slot。
        clearDuplicateSlotsUnlocked(kf_it->second, map_point, formal_feature);
    } else {
        // 无 formal 观测时，整个同点 stale 集合都应自愈清空。
        clearDuplicateSlotsUnlocked(kf_it->second, map_point, std::nullopt);
    }
    return true;
}

void Map::syncKeyframeObservations(const Frame::Ptr& keyframe) {
    if (!keyframe) return;
    auto kf_it = keyframes_.find(keyframe->id);
    if (kf_it == keyframes_.end() || kf_it->second.get() != keyframe.get()) return;

    bool changed = false;
    // 先删掉旧集合中已不再对应当前 slot 的观测。
    for (const auto& [id, mp] : map_points_) {
        (void)id;
        std::vector<Observation> stale;
        for (const auto& observation : mp->observations()) {
            if (observation.keyframe_id != keyframe->id) continue;
            const auto idx = static_cast<size_t>(observation.feature_index);
            if (idx >= keyframe->keypoints.size() ||
                idx >= keyframe->map_points.size() ||
                keyframe->map_points[idx].get() != mp.get())
                stale.push_back(observation);
        }
        for (const auto& observation : stale) {
            removeCovisibilityUnlocked(observation, mp);
            mp->removeObservation(observation);
            changed = true;
        }
    }
    changed = registerKeyframeObservationsUnlocked(keyframe) || changed;
    if (changed) bumpTopology();
}

bool Map::registerKeyframeObservationsUnlocked(const Frame::Ptr& keyframe) {
    if (!keyframe) return false;
    bool changed = false;
    // 把当前 slot 中的正式点加入反向集合；非法/冲突点清空。
    for (size_t i = 0; i < keyframe->map_points.size(); i++) {
        auto mp = keyframe->map_points[i];
        if (!mp) continue;
        if (i >= keyframe->keypoints.size()) {
            keyframe->map_points[i].reset();
            changed = true;
            continue;
        }
        if (!setObservationUnlocked(keyframe, static_cast<FeatureIndex>(i), mp)) {
            const Observation observation{keyframe->id,
                                          static_cast<FeatureIndex>(i)};
            const auto registered = map_points_.find(mp->id);
            if (registered == map_points_.end() ||
                registered->second.get() != mp.get() ||
                !mp->hasObservation(observation)) {
                keyframe->map_points[i].reset();
                changed = true;
            }
        } else {
            changed = true;
        }
    }
    return changed;
}

void Map::addCovisibilityUnlocked(const Observation& added,
                                  const MapPoint::Ptr& map_point) {
    for (const auto& existing : map_point->observations()) {
        if (existing.keyframe_id == added.keyframe_id) continue;
        const auto key = std::minmax(existing.keyframe_id, added.keyframe_id);
        covisibility_[key]++;
    }
}

bool Map::addObservationUnlocked(const Observation& observation,
                                 const MapPoint::Ptr& map_point) {
    if (!map_point || !map_point->addObservation(observation)) return false;
    // 只有反向正式观测确实插入后才增加共视，避免失败路径单边加权。
    addCovisibilityUnlocked(observation, map_point);
    // M2.2：正式观测即一次"跟踪命中"，记录当前关键帧计数（§6.3 旁路统计）。
    last_hit_kf_[map_point->id] = kf_count_.load(std::memory_order_relaxed);
    return true;
}

void Map::removeCovisibilityUnlocked(const Observation& removed,
                                     const MapPoint::Ptr& map_point) {
    for (const auto& existing : map_point->observations()) {
        if (existing.keyframe_id == removed.keyframe_id) continue;
        const auto key = std::minmax(existing.keyframe_id, removed.keyframe_id);
        auto it = covisibility_.find(key);
        if (it == covisibility_.end()) continue;
        if (it->second <= 1) covisibility_.erase(it);
        else --it->second;
    }
}

bool Map::clearDuplicateSlotsUnlocked(
    const Frame::Ptr& keyframe, const MapPoint::Ptr& map_point,
    std::optional<FeatureIndex> keep_feature) {
    if (!keyframe || !map_point) return false;
    bool changed = false;
    for (size_t i = 0; i < keyframe->map_points.size(); i++) {
        if (keep_feature && i == *keep_feature) continue;
        if (keyframe->map_points[i].get() != map_point.get()) continue;
        keyframe->map_points[i].reset();
        changed = true;
    }
    return changed;
}

void Map::removeMapPointsUnlocked(
    const std::vector<MapPoint::Ptr>& map_points) {
    std::unordered_set<const MapPoint*> removed;
    removed.reserve(map_points.size());
    for (const auto& map_point : map_points) {
        if (!map_point || !removed.insert(map_point.get()).second) continue;
        // formal observation 的数量通常远小于 KF 总数；共视和反向集合
        // 在这里逐条撤销，之后再统一处理可能遗留的 stale slot。
        const auto observations = map_point->observations();
        for (const auto& observation : observations) {
            removeCovisibilityUnlocked(observation, map_point);
            map_point->removeObservation(observation);
        }
        // M2.2：随点删除清理"最近命中 KF"旁路统计（§6.3）。
        last_hit_kf_.erase(map_point->id);
    }
    // 兼容迁移期间可能存在的旧 stale slot；整个批次只扫描一次所有 KF。
    for (const auto& [id, keyframe] : keyframes_) {
        (void)id;
        for (auto& slot : keyframe->map_points)
            if (slot && removed.contains(slot.get())) slot.reset();
    }
}

void Map::removeMapPointUnlocked(const MapPoint::Ptr& map_point) {
    removeMapPointsUnlocked({map_point});
}

size_t Map::sharedObservationCount(KeyframeId a, KeyframeId b) const {
    if (a == b) return 0;
    const auto key = std::minmax(a, b);
    auto it = covisibility_.find(key);
    return it == covisibility_.end() ? 0 : it->second;
}

std::vector<Map::Covisibility> Map::covisibleKeyframes(
    KeyframeId id, size_t min_shared) const {
    std::vector<Covisibility> result;
    for (const auto& [key, count] : covisibility_) {
        if (count < min_shared) continue;
        if (key.first == id) result.push_back({key.second, count});
        else if (key.second == id) result.push_back({key.first, count});
    }
    std::ranges::sort(result, [](const Covisibility& lhs,
                                 const Covisibility& rhs) {
        if (lhs.shared_points != rhs.shared_points)
            return lhs.shared_points > rhs.shared_points;
        return lhs.keyframe_id < rhs.keyframe_id;
    });
    return result;
}

bool Map::verifyObservationConsistency() const {
    std::map<std::pair<KeyframeId, KeyframeId>, size_t> expected_covisibility;
    for (const auto& [id, mp] : map_points_) {
        if (!mp || mp->id != id) return false;
        for (const auto& observation : mp->observations()) {
            auto kf_it = keyframes_.find(observation.keyframe_id);
            if (kf_it == keyframes_.end() ||
                observation.feature_index >= kf_it->second->keypoints.size() ||
                observation.feature_index >= kf_it->second->map_points.size() ||
                kf_it->second->map_points[observation.feature_index].get() != mp.get())
                return false;
        }
        std::vector<KeyframeId> ids;
        for (const auto& observation : mp->observations()) ids.push_back(observation.keyframe_id);
        for (size_t i = 0; i < ids.size(); i++)
            for (size_t j = i + 1; j < ids.size(); j++)
                expected_covisibility[std::minmax(ids[i], ids[j])]++;
    }
    for (const auto& [id, keyframe] : keyframes_) {
        if (!keyframe || keyframe->id != id) return false;
        for (size_t i = 0; i < keyframe->map_points.size(); i++) {
            const auto& mp = keyframe->map_points[i];
            if (!mp) continue;
            if (i >= keyframe->keypoints.size()) return false;
            if (map_points_.find(mp->id) == map_points_.end() ||
                !mp->hasObservation({id, static_cast<FeatureIndex>(i)})) return false;
        }
    }
    return expected_covisibility == covisibility_;
}

void Map::insertKeyFrame(Frame::Ptr kf) {
    if (!kf) return;
    if (keyframes_.find(kf->id) == keyframes_.end()) {
        kf->is_keyframe = true;
        keyframes_[kf->id] = kf;
        if (kf->id >= next_kf_id_) next_kf_id_ = kf->id + 1;
        kf_count_.fetch_add(1, std::memory_order_relaxed);
        // 新 KF 按定义不存在旧反向观测，只注册当前 slots；不能调用面向
        // 历史数据修复的 syncKeyframeObservations 全扫整张地图。
        registerKeyframeObservationsUnlocked(kf);
        bumpTopology();
    }
}

Frame::Ptr Map::getKeyFrame(unsigned long id) const {
    auto it = keyframes_.find(id);
    return (it != keyframes_.end()) ? it->second : nullptr;
}

std::vector<Frame::Ptr> Map::getAllKeyFrames() const {
    std::vector<Frame::Ptr> result;
    for (const auto& kv : keyframes_) {
        result.push_back(kv.second);
    }
    return result;
}

std::vector<MapPoint::Ptr> Map::getAllMapPoints() const {
    std::vector<MapPoint::Ptr> result;
    for (const auto& kv : map_points_) {
        result.push_back(kv.second);
    }
    return result;
}

void Map::clear() {
    for (auto& [id, keyframe] : keyframes_) {
        (void)id;
        for (auto& slot : keyframe->map_points) slot.reset();
    }
    for (auto& [id, map_point] : map_points_) {
        (void)id;
        const auto observations = map_point->observations();
        for (const auto& observation : observations)
            map_point->removeObservation(observation);
    }
    map_points_.clear();
    keyframes_.clear();
    covisibility_.clear();
    last_hit_kf_.clear();
    mp_count_.store(0, std::memory_order_relaxed);
    kf_count_.store(0, std::memory_order_relaxed);
    bumpTopology();
    bumpGeometry();
}

} // namespace vslam
