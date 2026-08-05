#include "vslam/map.h"

#include <set>

namespace vslam {

void Map::insertMapPoint(MapPoint::Ptr mp) {
    if (map_points_.find(mp->id) == map_points_.end()) {
        map_points_[mp->id] = mp;
    }
}

MapPoint::Ptr Map::getMapPoint(unsigned long id) const {
    auto it = map_points_.find(id);
    return (it != map_points_.end()) ? it->second : nullptr;
}

void Map::cullMapPoints(int min_observations) {
    // 按全局观测数剔除弱观测点，并同步清空关键帧引用。
    // 历史教训（2026-08-05 两版实验均失败）：
    // ① "只删孤儿点" → 地图无限膨胀，长序列内存失控（perf7 OOM/崩溃）；
    // ② "保留最近 30 KF 引用的单观测点" → 单观测点里垃圾点（错误深度/三角化）
    //    比例高，保留后 PnP 内点率骤降、跟踪 LOST 增 9 倍、子地图重建 37 次
    //    （perf8 ATE 174.8m）。观测数 < 2 的点对跟踪是净负资产，直接删除。
    std::set<unsigned long> removed;
    for (auto it = map_points_.begin(); it != map_points_.end(); ) {
        if (it->second->observed_count < min_observations) {
            removed.insert(it->first);
            it = map_points_.erase(it);
        } else {
            ++it;
        }
    }
    // 同步清空关键帧中的引用，让内存真正释放
    if (!removed.empty()) {
        for (auto& [id, kf] : keyframes_) {
            for (auto& mp : kf->map_points)
                if (mp && removed.count(mp->id)) mp.reset();
        }
    }
}

void Map::insertKeyFrame(Frame::Ptr kf) {
    kf->is_keyframe = true;
    keyframes_[kf->id] = kf;
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
    map_points_.clear();
    keyframes_.clear();
}

} // namespace vslam
