#include "vslam/atlas.h"

namespace vslam {

Submap& Atlas::createSubmap(const SE3& origin_Twc) {
    if (auto* active = activeSubmap()) active->frozen = true;

    Submap submap;
    submap.id = next_id_++;
    submap.map = std::make_shared<Map>();
    submap.origin_Twc = origin_Twc;
    submaps_.push_back(std::move(submap));
    active_id_ = submaps_.back().id;
    LOG_INFO("Atlas: created submap " << active_id_
             << " at " << origin_Twc.t.transpose());
    return submaps_.back();
}

bool Atlas::activate(unsigned long id) {
    for (const auto& submap : submaps_) {
        if (submap.id == id) {
            active_id_ = id;
            return true;
        }
    }
    return false;
}

Submap* Atlas::activeSubmap() {
    for (auto& submap : submaps_)
        if (submap.id == active_id_) return &submap;
    return nullptr;
}

const Submap* Atlas::activeSubmap() const {
    for (const auto& submap : submaps_)
        if (submap.id == active_id_) return &submap;
    return nullptr;
}

Map::Ptr Atlas::activeMap() const {
    const auto* submap = activeSubmap();
    return submap ? submap->map : nullptr;
}

} // namespace vslam
