#include "vslam/loop_graph.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace vslam {

GlobalCameraPoseSplit splitGlobalCameraPose(
    const SE3& T_wc, const SE3& T_oc) {
    return {T_oc, T_wc * T_oc.inverse()};
}

SE3 rebaseTailPose(
    const SE3& tail_pose_cs,
    const SE3& old_endpoint_pose_cs,
    const SE3& new_endpoint_pose_cs) {
    return tail_pose_cs * old_endpoint_pose_cs.inverse() * new_endpoint_pose_cs;
}

Vec3 rebaseTailPoint(
    const Vec3& point_s,
    const SE3& old_reference_pose_cs,
    const SE3& new_reference_pose_cs) {
    return new_reference_pose_cs.inverse() * old_reference_pose_cs * point_s;
}

namespace {

struct IndexedKeyframe {
    size_t index = 0;
    KeyframeState state;
};

std::vector<size_t> selectAnchorIndices(
    const OptimizationSnapshot& full,
    const EssentialGraphConfig& config) {
    const size_t n = full.keyframes.size();
    if (n == 0) return {};
    if (n == 1) return {0};

    // 首尾 + 最新回环两端最多需要 4 个槽位。
    const size_t max_anchors = std::max<size_t>(4, config.max_anchors);
    const size_t stride = std::max<size_t>(1, config.preferred_stride);
    std::unordered_map<KeyframeId, size_t> index_by_id;
    index_by_id.reserve(n);
    for (size_t i = 0; i < n; ++i)
        index_by_id.emplace(full.keyframes[i].id, i);

    std::unordered_set<size_t> selected{0, n - 1};
    // 新回环优先。若累计回环端点本身超过预算，只保留最近提交边；已发布的
    // 旧回环校正已经固化在冻结位姿中，不允许突破本次图的硬上限。
    for (auto it = full.constraints.rbegin(); it != full.constraints.rend(); ++it) {
        if (!it->is_loop) continue;
        const auto a = index_by_id.find(it->a);
        const auto b = index_by_id.find(it->b);
        if (a == index_by_id.end() || b == index_by_id.end()) continue;
        size_t needed = 0;
        if (!selected.contains(a->second)) ++needed;
        if (!selected.contains(b->second)) ++needed;
        if (selected.size() + needed > max_anchors) continue;
        selected.insert(a->second);
        selected.insert(b->second);
    }

    std::vector<size_t> optional;
    for (size_t i = stride; i + 1 < n; i += stride)
        if (!selected.contains(i)) optional.push_back(i);
    const size_t slots = max_anchors > selected.size()
        ? max_anchors - selected.size() : 0;
    if (optional.size() <= slots) {
        selected.insert(optional.begin(), optional.end());
    } else if (slots > 0) {
        // 在首选候选中覆盖完整路径；不连续取前 slots，避免尾段失去锚点。
        for (size_t j = 0; j < slots; ++j) {
            const size_t pos = static_cast<size_t>(std::floor(
                (static_cast<double>(j) + 0.5) * optional.size() / slots));
            selected.insert(optional[std::min(pos, optional.size() - 1)]);
        }
    }

    std::vector<size_t> indices(selected.begin(), selected.end());
    std::ranges::sort(indices);
    return indices;
}

SE3 interpolateCorrection(const SE3& a, const SE3& b, double alpha) {
    alpha = std::clamp(alpha, 0.0, 1.0);
    Eigen::Quaterniond qb = b.q;
    // 显式统一半球，保证 q/-q 不会让中间关键帧绕长弧旋转。
    if (a.q.coeffs().dot(qb.coeffs()) < 0.0) qb.coeffs() *= -1.0;
    Eigen::Quaterniond q = a.q.slerp(alpha, qb);
    q.normalize();
    return {q, (1.0 - alpha) * a.t + alpha * b.t};
}

} // namespace

EssentialAnchorGraph buildEssentialAnchorGraph(
    const OptimizationSnapshot& full,
    const EssentialGraphConfig& config) {
    EssentialAnchorGraph graph;
    graph.full_keyframes = full.keyframes;
    graph.snapshot.submap_id = full.submap_id;
    graph.snapshot.topology_revision = full.topology_revision;
    graph.snapshot.geometry_revision = full.geometry_revision;

    const auto indices = selectAnchorIndices(full, config);
    std::unordered_map<KeyframeId, size_t> full_index;
    full_index.reserve(full.keyframes.size());
    for (size_t i = 0; i < full.keyframes.size(); ++i)
        full_index.emplace(full.keyframes[i].id, i);
    std::unordered_set<KeyframeId> anchor_ids;
    anchor_ids.reserve(indices.size());
    for (const size_t index : indices) {
        graph.snapshot.keyframes.push_back(full.keyframes[index]);
        anchor_ids.insert(full.keyframes[index].id);
    }
    for (const auto id : full.fixed_kf_ids)
        if (anchor_ids.contains(id)) graph.snapshot.fixed_kf_ids.push_back(id);

    // 相邻锚点间只保留一条冻结里程计约束。位姿图变量从逐 KF 降为锚点，
    // 但测量仍严格满足 X_b = X_a * T_rel（X=T_wc）。
    for (size_t i = 1; i < graph.snapshot.keyframes.size(); ++i) {
        const auto& a = graph.snapshot.keyframes[i - 1];
        const auto& b = graph.snapshot.keyframes[i];
        double weight_sum = 0.0;
        size_t weight_count = 0;
        for (const auto& edge : full.constraints) {
            if (edge.is_loop) continue;
            const auto ia = full_index.find(edge.a);
            const auto ib = full_index.find(edge.b);
            if (ia == full_index.end() || ib == full_index.end()) continue;
            const size_t pa = ia->second;
            const size_t pb = ib->second;
            if (pa >= indices[i - 1] && pb <= indices[i]) {
                weight_sum += edge.weight;
                ++weight_count;
            }
        }
        const double weight = weight_count > 0
            ? weight_sum / static_cast<double>(weight_count) : 1.0;
        graph.snapshot.constraints.push_back(
            {a.id, b.id, a.pose_cs * b.pose_cs.inverse(), weight, false});
    }
    for (const auto& edge : full.constraints) {
        if (edge.is_loop && anchor_ids.contains(edge.a) && anchor_ids.contains(edge.b))
            graph.snapshot.constraints.push_back(edge);
    }
    return graph;
}

std::vector<PoseUpdate> propagateAnchorCorrections(
    const EssentialAnchorGraph& graph,
    const OptimizationResult& anchor_result) {
    std::vector<PoseUpdate> updates;
    if (graph.full_keyframes.empty() || graph.snapshot.keyframes.empty() ||
        !anchor_result.valid) return updates;

    std::unordered_map<KeyframeId, SE3> optimized;
    optimized.reserve(anchor_result.poses.size());
    for (const auto& pose : anchor_result.poses)
        optimized.emplace(pose.id, pose.pose_cs);

    std::unordered_map<KeyframeId, size_t> full_index;
    full_index.reserve(graph.full_keyframes.size());
    for (size_t i = 0; i < graph.full_keyframes.size(); ++i)
        full_index.emplace(graph.full_keyframes[i].id, i);

    std::vector<IndexedKeyframe> anchors;
    anchors.reserve(graph.snapshot.keyframes.size());
    for (const auto& old_anchor : graph.snapshot.keyframes) {
        const auto index = full_index.find(old_anchor.id);
        if (index == full_index.end()) continue;
        KeyframeState state = old_anchor;
        if (const auto it = optimized.find(old_anchor.id); it != optimized.end())
            state.pose_cs = it->second;
        anchors.push_back({index->second, state});
    }
    if (anchors.empty()) return updates;

    updates.reserve(graph.full_keyframes.size());
    size_t right = anchors.size() > 1 ? 1 : 0;
    for (size_t i = 0; i < graph.full_keyframes.size(); ++i) {
        while (right + 1 < anchors.size() && i > anchors[right].index) ++right;
        const size_t left = right == 0 ? 0 : right - 1;
        const auto& old = graph.full_keyframes[i];
        const auto& left_old = graph.full_keyframes[anchors[left].index];
        const auto& right_old = graph.full_keyframes[anchors[right].index];
        const SE3 left_correction =
            anchors[left].state.pose_cs.inverse() * left_old.pose_cs;
        const SE3 right_correction =
            anchors[right].state.pose_cs.inverse() * right_old.pose_cs;
        const double denominator = static_cast<double>(
            anchors[right].index - anchors[left].index);
        const double alpha = denominator > 0.0
            ? static_cast<double>(i - anchors[left].index) / denominator : 0.0;
        const SE3 correction = interpolateCorrection(
            left_correction, right_correction, alpha);
        const SE3 old_wc = old.pose_cs.inverse();
        updates.push_back({old.id, (correction * old_wc).inverse()});
    }
    return updates;
}

OptimizationSnapshot buildSubmapGraph(const Atlas& atlas) {
    OptimizationSnapshot snapshot;
    const auto& submaps = atlas.submaps();
    snapshot.keyframes.reserve(submaps.size());
    for (const auto& submap : submaps) {
        snapshot.keyframes.push_back({submap.id, submap.T_ws.inverse()});
    }
    if (!submaps.empty()) snapshot.fixed_kf_ids.push_back(submaps.front().id);

    snapshot.constraints.reserve(atlas.constraints().size());
    for (const auto& constraint : atlas.constraints()) {
        snapshot.constraints.push_back({
            constraint.a, constraint.b, constraint.T_rel,
            constraint.weight, true});
    }
    return snapshot;
}

} // namespace vslam
