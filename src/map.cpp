#include "vslam/map.h"

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
    for (auto it = map_points_.begin(); it != map_points_.end(); ) {
        if (it->second->observed_count < min_observations) {
            it = map_points_.erase(it);
        } else {
            ++it;
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
