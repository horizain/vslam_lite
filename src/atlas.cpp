#include "vslam/atlas.h"

#include <algorithm>
#include <cmath>

namespace vslam {

AtlasConstraint makeCrossSubmapLoopConstraint(
    SubmapId current_submap_id, SubmapId loop_submap_id,
    const SE3& current_pose_cs, const SE3& loop_pose_cs,
    const SE3& T_loop_curr, double weight) {
    // P = Z^-1 * T_cs_loop，T_rel = T_cs_current^-1 * P。
    AtlasConstraint constraint;
    constraint.a = current_submap_id;
    constraint.b = loop_submap_id;
    constraint.T_rel = current_pose_cs.inverse() *
                       T_loop_curr.inverse() * loop_pose_cs;
    constraint.weight = weight;
    constraint.type = AtlasConstraintType::LoopClosure;
    return constraint;
}

SE3 rebaseWorldPoseForSubmapAnchor(
    const SE3& T_cw_old, const SE3& T_ws_old, const SE3& T_ws_new) {
    return T_cw_old * T_ws_old * T_ws_new.inverse();
}

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

bool Atlas::loopConstraintsConsistent(
    double max_translation, double max_rotation,
    double* max_translation_seen, double* max_rotation_seen) const {
    double max_t = 0.0;
    double max_r = 0.0;
    bool ok = true;
    for (const auto& edge : constraints_) {
        if (edge.type != AtlasConstraintType::LoopClosure) continue;
        const auto* a = getSubmap(edge.a);
        const auto* b = getSubmap(edge.b);
        if (!a || !b) {
            ok = false;
            break;
        }
        const SE3 predicted = a->T_ws.inverse() * b->T_ws;
        const SE3 residual = edge.T_rel.inverse() * predicted;
        const double trans = residual.t.norm();
        const double rot = 2.0 * std::acos(std::clamp(
            std::abs(residual.q.normalized().w()), 0.0, 1.0));
        max_t = std::max(max_t, trans);
        max_r = std::max(max_r, rot);
        if (!std::isfinite(trans) || !std::isfinite(rot) ||
            trans > max_translation || rot > max_rotation) {
            ok = false;
            break;
        }
    }
    if (max_translation_seen) *max_translation_seen = max_t;
    if (max_rotation_seen) *max_rotation_seen = max_r;
    return ok;
}

} // namespace vslam
