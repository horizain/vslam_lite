#include "vslam/atlas.h"

namespace vslam {

Submap& Atlas::createSubmap(const SE3& T_ws, bool connected) {
    if (auto* active = activeSubmap()) active->frozen = true;

    Submap submap;
    submap.id = next_id_++;
    submap.map = std::make_shared<Map>();
    submap.T_ws = T_ws;
    submap.connected = connected;
    submaps_.push_back(std::move(submap));
    active_id_ = submaps_.back().id;
    LOG_INFO("Atlas: created submap " << active_id_
             << " at " << T_ws.t.transpose()
             << (connected ? " (connected)" : " (disconnected)"));
    return submaps_.back();
}

bool Atlas::activate(unsigned long id) {
    for (auto& submap : submaps_) {
        if (submap.id == id) {
            active_id_ = id;
            submap.frozen = false;
            return true;
        }
    }
    return false;
}

Submap* Atlas::getSubmap(unsigned long id) {
    for (auto& submap : submaps_)
        if (submap.id == id) return &submap;
    return nullptr;
}

const Submap* Atlas::getSubmap(unsigned long id) const {
    for (const auto& submap : submaps_)
        if (submap.id == id) return &submap;
    return nullptr;
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

std::vector<AtlasConstraint> Atlas::constraintsOf(unsigned long submap_id) const {
    std::vector<AtlasConstraint> out;
    for (const auto& c : constraints_) {
        if (c.a == submap_id || c.b == submap_id) out.push_back(c);
    }
    return out;
}

} // namespace vslam
