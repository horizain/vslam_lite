#include "vslam/loop_region.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vslam {
namespace {

struct PointKey {
    MapPointId id = 0;
    KeyframeId keyframe = 0;
    FeatureIndex feature = 0;
    bool formal_map_point = false;
    bool operator==(const PointKey& other) const {
        return id == other.id && keyframe == other.keyframe &&
               feature == other.feature && formal_map_point == other.formal_map_point;
    }
};

struct PointKeyHash {
    size_t operator()(const PointKey& k) const {
        size_t h = std::hash<MapPointId>{}(k.id);
        h ^= std::hash<KeyframeId>{}(k.keyframe) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        h ^= std::hash<FeatureIndex>{}(k.feature) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        h ^= std::hash<bool>{}(k.formal_map_point) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        return h;
    }
};

bool validDescriptor(const cv::Mat& d) {
    return !d.empty() && d.rows == 1 && d.type() == CV_8U;
}

void addFramePoints(const Frame::Ptr& frame, std::vector<LoopRegionPoint>& points,
                    std::unordered_set<PointKey, PointKeyHash>& seen,
                    size_t max_points, size_t max_add) {
    if (!frame || points.size() >= max_points || max_add == 0) return;
    const size_t target_size = std::min(max_points, points.size() + max_add);
    const size_t n = std::max(frame->map_points.size(), frame->pts_c.size());
    for (size_t i = 0; i < n && points.size() < target_size; ++i) {
        auto mp = i < frame->map_points.size() ? frame->map_points[i] : nullptr;
        const bool has_mp = mp && mp->pos_s.allFinite();
        const bool has_depth = i < frame->pts_c.size() &&
                               frame->pts_c[i].allFinite() &&
                               frame->pts_c[i].z() > 0.0;
        if (!has_mp && !has_depth) continue;

        const PointKey key{has_mp ? mp->id : 0, frame->id,
                           static_cast<FeatureIndex>(i), has_mp};
        // 真正地图点以 MapPoint id 去重；fallback 点以来源观测去重。
        const PointKey dedup_key = has_mp ? PointKey{mp->id, 0, 0, true} : key;
        if (seen.contains(dedup_key)) {
            continue;
        }

        cv::Mat descriptor;
        if (has_mp && validDescriptor(mp->descriptor))
            descriptor = mp->descriptor.clone();
        else if (i < static_cast<size_t>(frame->descriptors.rows))
            descriptor = frame->descriptors.row(static_cast<int>(i)).clone();
        if (!validDescriptor(descriptor)) continue;
        seen.insert(dedup_key);

        LoopRegionPoint p;
        p.id = has_mp ? mp->id : 0;
        p.pos_s = has_mp ? mp->pos_s : frame->pose_cs.inverse() * frame->pts_c[i];
        p.descriptor = std::move(descriptor);
        p.from_pts_c = !has_mp;
        p.source_keyframe_id = frame->id;
        p.source_feature_index = static_cast<FeatureIndex>(i);
        points.push_back(std::move(p));
    }
}

SE3 matToSE3(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = rmat.at<double>(r, c);
    return SE3(Eigen::Quaterniond(R),
               Vec3(tvec.at<double>(0), tvec.at<double>(1),
                    tvec.at<double>(2)));
}

} // namespace

bool LoopRegionVerifier::build(const std::vector<Frame::Ptr>& keyframes,
                               const Frame::Ptr& anchor, SubmapId submap_id,
                               uint64_t topology_revision,
                               uint64_t geometry_revision,
                               const LoopRegionConfig& config,
                               LoopRegionSnapshot& out) {
    out = LoopRegionSnapshot{};
    if (!anchor || config.max_keyframes == 0 || config.max_points == 0)
        return false;

    std::vector<Frame::Ptr> all;
    all.reserve(keyframes.size() + 1);
    all.push_back(anchor);
    for (const auto& frame : keyframes) {
        if (frame && frame->id != anchor->id) all.push_back(frame);
    }

    // 共视由正式 MapPoint id 计算；没有正式点的双目 pts_c 不参与共视排序，
    // 但仍会作为时间邻居的 fallback 几何点进入快照。
    std::unordered_set<MapPointId> anchor_points;
    for (const auto& mp : anchor->map_points)
        if (mp) anchor_points.insert(mp->id);
    struct Ranked {
        Frame::Ptr frame;
        size_t shared = 0;
        size_t temporal_distance = 0;
    };
    std::vector<Ranked> covisible, temporal;
    for (const auto& frame : all) {
        if (!frame || frame->id == anchor->id) continue;
        size_t shared = 0;
        std::unordered_set<MapPointId> unique;
        for (const auto& mp : frame->map_points) {
            if (mp && anchor_points.contains(mp->id) && unique.insert(mp->id).second)
                ++shared;
        }
        const size_t distance = frame->id >= anchor->id
            ? frame->id - anchor->id : anchor->id - frame->id;
        if (shared > 0) covisible.push_back({frame, shared, distance});
        if (distance <= config.temporal_window) temporal.push_back({frame, shared, distance});
    }
    std::ranges::sort(covisible, [](const Ranked& a, const Ranked& b) {
        if (a.shared != b.shared) return a.shared > b.shared;
        return a.frame->id < b.frame->id;
    });
    std::ranges::sort(temporal, [](const Ranked& a, const Ranked& b) {
        if (a.temporal_distance != b.temporal_distance)
            return a.temporal_distance < b.temporal_distance;
        return a.frame->id < b.frame->id;
    });

    std::vector<Frame::Ptr> selected;
    selected.reserve(config.max_keyframes);
    selected.push_back(anchor);
    std::unordered_set<KeyframeId> selected_ids{anchor->id};
    auto select = [&](const std::vector<Ranked>& candidates, size_t limit) {
        size_t added = 0;
        for (const auto& candidate : candidates) {
            if (added >= limit || selected.size() >= config.max_keyframes) break;
            if (selected_ids.insert(candidate.frame->id).second) {
                selected.push_back(candidate.frame);
                ++added;
            }
        }
    };
    select(covisible, config.max_covisible_keyframes);
    select(temporal, config.max_temporal_neighbors);

    // 若上述两类候选不足，补入确定性排序的剩余历史帧，避免稀疏地图只有
    // 一个 anchor 时区域退化为单帧，同时仍严格服从 max_keyframes。
    std::ranges::sort(all, [&](const Frame::Ptr& a, const Frame::Ptr& b) {
        const size_t da = a->id >= anchor->id ? a->id - anchor->id
                                               : anchor->id - a->id;
        const size_t db = b->id >= anchor->id ? b->id - anchor->id
                                               : anchor->id - b->id;
        if (da != db) return da < db;
        return a->id < b->id;
    });
    select([&] {
        std::vector<Ranked> rest;
        rest.reserve(all.size());
        for (const auto& f : all) {
            if (f && f->id != anchor->id)
                rest.push_back({f, 0, f->id >= anchor->id ? f->id - anchor->id
                                                            : anchor->id - f->id});
        }
        return rest;
    }(), config.max_keyframes);

    out.submap_id = submap_id;
    out.topology_revision = topology_revision;
    out.geometry_revision = geometry_revision;
    out.anchor_id = anchor->id;
    out.anchor_pose_cs = anchor->pose_cs;
    out.keyframes.reserve(selected.size());
    std::unordered_set<PointKey, PointKeyHash> seen;
    const size_t fair_share = std::max<size_t>(
        1, config.max_points / std::max<size_t>(1, selected.size()));
    for (const auto& frame : selected) {
        // 快照只记录不可变身份；不能把 live Frame::Ptr 带出 map 读锁。
        out.keyframes.push_back({frame->id});
        // anchor 往往单帧就有上千点；若按帧顺序填满总预算，所谓“区域”
        // 实际仍退化为 anchor。首轮给每个支持 KF 公平份额，再用第二轮
        // 确定性填满余量，使稀疏邻居和视角变化都有真实进入 PnP 的机会。
        addFramePoints(frame, out.points, seen, config.max_points, fair_share);
    }
    for (const auto& frame : selected) {
        if (out.points.size() >= config.max_points) break;
        addFramePoints(frame, out.points, seen, config.max_points,
                       config.max_points - out.points.size());
    }
    return !out.keyframes.empty() && !out.points.empty();
}

bool LoopRegionVerifier::build(const Map& map, const Frame::Ptr& anchor,
                               SubmapId submap_id, const LoopRegionConfig& config,
                               LoopRegionSnapshot& out) {
    if (!anchor || config.max_keyframes == 0) return false;

    // 不能把整张图直接交给通用 build：后者会为共视排序扫描每个 KF 的
    // 全部点，每个 DBoW 候选都执行会退化为 O(candidates*KFs*points)。
    // 先用 Map 的持久共视计数选有限支持集，再补按 id 邻近的历史帧；
    // 通用 build 最终只扫描至多 max_keyframes 个帧。
    std::vector<Frame::Ptr> support;
    support.reserve(config.max_keyframes);
    support.push_back(anchor);
    std::unordered_set<KeyframeId> selected{anchor->id};
    size_t covisible_added = 0;
    for (const auto& cov : map.covisibleKeyframes(anchor->id)) {
        if (support.size() >= config.max_keyframes ||
            covisible_added >= config.max_covisible_keyframes) break;
        if (auto frame = map.getKeyFrame(cov.keyframe_id);
            frame && selected.insert(frame->id).second) {
            support.push_back(std::move(frame));
            ++covisible_added;
        }
    }

    const auto all = map.getAllKeyFrames();  // Map 保证按 id 排序。
    auto center = std::ranges::lower_bound(
        all, anchor->id, {}, [](const Frame::Ptr& frame) { return frame->id; });
    auto left = center;
    auto right = center;
    if (right != all.end() && (*right)->id == anchor->id) ++right;
    size_t temporal_added = 0;
    while (support.size() < config.max_keyframes &&
           temporal_added < config.max_temporal_neighbors &&
           (left != all.begin() || right != all.end())) {
        const Frame::Ptr before = left != all.begin() ? *std::prev(left) : nullptr;
        const Frame::Ptr after = right != all.end() ? *right : nullptr;
        const size_t before_dist = before ? anchor->id - before->id
                                          : std::numeric_limits<size_t>::max();
        const size_t after_dist = after ? after->id - anchor->id
                                        : std::numeric_limits<size_t>::max();
        const bool take_before = before_dist <= after_dist;
        const auto frame = take_before ? before : after;
        const size_t distance = take_before ? before_dist : after_dist;
        if (take_before) --left; else ++right;
        if (!frame || distance > config.temporal_window) continue;
        if (selected.insert(frame->id).second) {
            support.push_back(frame);
            ++temporal_added;
        }
    }

    return build(support, anchor, submap_id,
                 map.topologyRevision(), map.geometryRevision(), config, out);
}

bool LoopRegionVerifier::verify(const LoopRegionSnapshot& region,
                                const Frame::Ptr& current, const Camera& camera,
                                const LoopRegionConfig& config,
                                SE3& T_loop_curr, LoopRegionResult* result) {
    LoopRegionResult local;
    if (result) *result = local;
    if (!current || !camera || region.points.empty() ||
        current->descriptors.empty() || current->descriptors.type() != CV_8U ||
        current->keypoints.empty())
        return false;

    std::vector<LoopRegionPoint> points;
    points.reserve(region.points.size());
    for (const auto& p : region.points)
        if (validDescriptor(p.descriptor) && p.pos_s.allFinite()) points.push_back(p);
    if (points.empty()) return false;

    cv::Mat region_descriptors;
    for (const auto& p : points) region_descriptors.push_back(p.descriptor);
    if (region_descriptors.empty() || region_descriptors.type() != CV_8U) return false;

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> forward_knn, reverse_knn;
    matcher.knnMatch(current->descriptors, region_descriptors, forward_knn, 2);
    matcher.knnMatch(region_descriptors, current->descriptors, reverse_knn, 2);
    std::vector<cv::DMatch> matches;
    std::unordered_set<int> used_region, used_current;
    for (int qi = 0; qi < static_cast<int>(forward_knn.size()); ++qi) {
        const auto& f = forward_knn[qi];
        if (f.size() < 2 || f[0].distance >=
                               config.descriptor_ratio * f[1].distance)
            continue;
        if (f[0].distance > config.max_hamming_distance) continue;
        const int ti = f[0].trainIdx;
        if (ti < 0 || ti >= static_cast<int>(reverse_knn.size())) continue;
        const auto& r = reverse_knn[ti];
        if (r.size() < 2 || r[0].trainIdx != qi ||
            r[0].distance >= config.descriptor_ratio * r[1].distance)
            continue;
        if (!used_current.insert(qi).second || !used_region.insert(ti).second) continue;
        matches.push_back(cv::DMatch(qi, ti, f[0].distance));
    }
    local.matches = static_cast<int>(matches.size());
    if (local.matches < config.min_matches) {
        if (result) *result = local;
        return false;
    }

    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    pts3d.reserve(matches.size());
    pts2d.reserve(matches.size());
    for (const auto& match : matches) {
        if (match.queryIdx < 0 || match.queryIdx >= static_cast<int>(current->keypoints.size()) ||
            match.trainIdx < 0 || match.trainIdx >= static_cast<int>(points.size())) continue;
        const Vec3& p = points[match.trainIdx].pos_s;
        pts3d.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()),
                           static_cast<float>(p.z()));
        pts2d.push_back(current->keypoints[match.queryIdx].pt);
    }
    if (pts3d.size() < 4) return false;

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    const bool pnp_ok = cv::solvePnPRansac(
        pts3d, pts2d, camera->K(), cv::Mat(), rvec, tvec, false,
        std::max(1, config.ransac_iterations), config.ransac_pixel_threshold,
        config.ransac_confidence, inliers, cv::SOLVEPNP_EPNP);
    local.inliers = static_cast<int>(inliers.size());
    local.inlier_ratio = pts3d.empty() ? 0.0
                                       : inliers.size() / static_cast<double>(pts3d.size());
    if (pnp_ok && !inliers.empty()) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(pts3d, rvec, tvec, camera->K(), cv::Mat(), projected);
        double sum_sq = 0.0;
        for (int idx : inliers) {
            const double dx = projected[idx].x - pts2d[idx].x;
            const double dy = projected[idx].y - pts2d[idx].y;
            sum_sq += dx * dx + dy * dy;
        }
        local.reprojection_rmse = std::sqrt(sum_sq / inliers.size());
        const SE3 T_cw = matToSE3(rvec, tvec);
        int positive_depth = 0;
        std::unordered_set<int> occupied_cells;
        const double image_width = camera->img_width > 0
            ? camera->img_width : std::max(1.0, camera->cx * 2.0);
        const double image_height = camera->img_height > 0
            ? camera->img_height : std::max(1.0, camera->cy * 2.0);
        std::unordered_map<KeyframeId, size_t> source_support;
        for (const int idx : inliers) {
            if (idx < 0 || idx >= static_cast<int>(matches.size())) continue;
            const int point_index = matches[idx].trainIdx;
            if (point_index < 0 || point_index >= static_cast<int>(points.size()))
                continue;
            if ((T_cw * points[point_index].pos_s).z() > 0.0) ++positive_depth;
            const int columns = std::max(1, config.grid_columns);
            const int rows = std::max(1, config.grid_rows);
            const int gx = std::clamp(static_cast<int>(
                pts2d[idx].x * columns / image_width), 0, columns - 1);
            const int gy = std::clamp(static_cast<int>(
                pts2d[idx].y * rows / image_height), 0, rows - 1);
            occupied_cells.insert(gy * columns + gx);
            ++source_support[points[point_index].source_keyframe_id];
        }
        local.positive_depth_ratio = positive_depth /
            static_cast<double>(std::max<size_t>(1, inliers.size()));
        local.grid_cells = static_cast<int>(occupied_cells.size());
        size_t best_support = 0;
        for (const auto& [id, support] : source_support) {
            if (support > best_support ||
                (support == best_support && id < local.supporting_keyframe_id)) {
                local.supporting_keyframe_id = id;
                best_support = support;
            }
        }
    }
    if (!pnp_ok || local.inliers < config.min_inliers ||
        local.inlier_ratio < config.min_inlier_ratio ||
        local.reprojection_rmse > config.max_reprojection_rmse ||
        local.positive_depth_ratio < config.min_positive_depth_ratio ||
        local.grid_cells < config.min_grid_cells) {
        if (result) *result = local;
        return false;
    }

    local.T_cw_curr_in_loop = matToSE3(rvec, tvec);
    T_loop_curr = region.anchor_pose_cs * local.T_cw_curr_in_loop.inverse();
    local.accepted = true;
    if (result) *result = local;
    return true;
}

} // namespace vslam
