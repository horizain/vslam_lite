#include "vslam/local_mapper.h"
#include "vslam/mappoint.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace vslam {

bool includeLocalBALandmark(int observation_count, int min_observed) {
    // M1.5：从 vo.h 迁移，公式不变——只保留观测数达到 min_observed 的点
    //（Local BA 有效边要求同一点 ≥3 个正式关键帧观测）。
    return observation_count >= min_observed;
}

LocalMapper::LocalMapper(const Camera& camera) : camera_(camera) {}

void LocalMapper::createMapPointsFromStereo(const Map::Ptr& map,
                                            const Frame::Ptr& frame,
                                            size_t max_map_points) const {
    if (!camera_->hasPerFrameDepth()) return;

    int cnt = 0;
    const bool registered = map->getKeyFrame(frame->id) == frame;
    for (size_t i = 0; i < frame->keypoints.size(); i++) {
        // 建点预算是逐点检查的绝对上限。关键帧仍可插入，但达到上限后
        // 本轮及后续轮次只保留跟踪/观测，不再批量越过 MapPoint 配额。
        if (map->mapPointCount() >= max_map_points) break;
        if (i >= frame->pts_c.size()) continue;
        if (frame->pts_c[i].z() > 0 && frame->map_points[i] == nullptr) {
            // M3：p_s = T_sc * p_c（pose_cs 的逆把相机系点转到子地图局部系）
            Vec3 p_s = frame->pose_cs.inverse() * frame->pts_c[i];
            auto mp = std::make_shared<MapPoint>(map->nextMapPointId());
            mp->pos_s = p_s;
            if (!frame->descriptors.empty())
                mp->descriptor = frame->descriptors.row((int)i).clone();
            map->insertMapPoint(mp);
            // 已注册 KF 必须通过 Map 维护双向正式观测；初始化阶段的
            // 未注册帧先写 slot，随后 insertKeyFrame 会统一 sync。
            if (registered) {
                if (!map->setObservation(frame, static_cast<FeatureIndex>(i), mp)) {
                    map->removeMapPoint(mp->id);
                    continue;
                }
            } else {
                frame->map_points[i] = mp;
            }
            ++cnt;
        }
    }
    if (cnt > 0) LOG_INFO("Stereo map points created: " << cnt);
}

void LocalMapper::triangulateNewPoints(
    const Map::Ptr& map, const Frame::Ptr& f1, const Frame::Ptr& f2,
    const std::vector<cv::DMatch>& matches, size_t max_map_points) const {
    cv::Mat K = camera_->K();
    int cnt = 0;
    const bool f1_registered = map->getKeyFrame(f1->id) == f1;
    const bool f2_registered = map->getKeyFrame(f2->id) == f2;
    for (auto& m : matches) {
        // 不使用“先收集、后批量插入”的方式，避免剩余配额小于匹配数时
        // 一次性越过上限；每次成功插入后重新读取原子计数。
        if (map->mapPointCount() >= max_map_points) break;
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(f1->map_points.size()) ||
            m.trainIdx >= static_cast<int>(f2->map_points.size()) ||
            m.queryIdx >= static_cast<int>(f1->keypoints.size()) ||
            m.trainIdx >= static_cast<int>(f2->keypoints.size()) ||
            f1->map_points[m.queryIdx] != nullptr ||
            f2->map_points[m.trainIdx] != nullptr) {
            // f2 已绑定其它点时禁止覆盖，避免一个正式 slot 被静默重绑。
            continue;
        }
        {
            auto mp = MapPoint::create(
                map->nextMapPointId(),
                Vec2(f1->keypoints[m.queryIdx].pt.x, f1->keypoints[m.queryIdx].pt.y),
                Vec2(f2->keypoints[m.trainIdx].pt.x, f2->keypoints[m.trainIdx].pt.y),
                f1->pose_cs, f2->pose_cs, K);
            if (!mp) continue;  // B2：退化三角化（视差角过小）拒绝
            Vec3 pc = f1->pose_cs * mp->pos_s;
            if (pc.z() > 0) {
                map->insertMapPoint(mp);
                // 已注册帧通过 Map 建立双向正式观测；初始化阶段的两个
                // 未注册帧先写 slot，随后 insertKeyFrame 会统一 sync。
                const bool f1_ok = !f1_registered || map->setObservation(
                    f1, static_cast<FeatureIndex>(m.queryIdx), mp);
                const bool f2_ok = !f2_registered || map->setObservation(
                    f2, static_cast<FeatureIndex>(m.trainIdx), mp);
                if (f1_ok && f2_ok) {
                    if (!f1_registered) f1->map_points[m.queryIdx] = mp;
                    if (!f2_registered) f2->map_points[m.trainIdx] = mp;
                    ++cnt;
                } else {
                    // 新点必须是全有或全无；若第一侧已注册成功而第二侧
                    // 失败，先撤销正式 observation，再删除点，禁止半连接。
                    if (f1_registered && f1_ok)
                        map->clearObservation(
                            f1->id, static_cast<FeatureIndex>(m.queryIdx));
                    if (f2_registered && f2_ok)
                        map->clearObservation(
                            f2->id, static_cast<FeatureIndex>(m.trainIdx));
                    map->removeMapPoint(mp->id);
                }
            }
        }
    }
    LOG_INFO("Triangulated " << cnt << " points");
}

std::vector<Frame::Ptr> LocalMapper::selectLocalWindow(
    const Map::Ptr& map, const Frame::Ptr& curr_frame, int n) const {
    std::vector<Frame::Ptr> window;
    if (n <= 0 || !curr_frame) return window;
    auto all_kfs = map->getAllKeyFrames();
    if (all_kfs.empty()) return window;

    struct Candidate { Frame::Ptr kf; int cov; };
    std::vector<Candidate> cands;
    const auto registered_curr = map->getKeyFrame(curr_frame->id);
    if (registered_curr && registered_curr.get() == curr_frame.get()) {
        // 已注册 KF 使用 Map 的持久共视计数（正式 Observation 的唯一来源），
        // 不再从 transient map_points 指针重复统计。
        for (const auto& covisible : map->covisibleKeyframes(
                 curr_frame->id, 1)) {
            auto kf = map->getKeyFrame(covisible.keyframe_id);
            if (kf) cands.push_back({kf, static_cast<int>(covisible.shared_points)});
        }
    } else {
        // 初始化/尚未注册帧仍只能使用临时指针作安全兜底。
        std::set<MapPointId> curr_mps;
        for (const auto& mp : curr_frame->map_points)
            if (mp) curr_mps.insert(mp->id);
        for (const auto& kf : all_kfs) {
            if (kf->id == curr_frame->id) continue;
            int cov = 0;
            for (const auto& mp : kf->map_points)
                if (mp && curr_mps.contains(mp->id)) ++cov;
            cands.push_back({kf, cov});
        }
    }
    // C++23 ranges：按共视点数量降序（投影 &Candidate::cov，免手写比较器）
    std::ranges::sort(cands, std::greater<>{}, &Candidate::cov);

    // 窗口 = 当前帧 + 共视最多的候选，最多填满 n 个 KF。
    window.push_back(curr_frame);
    for (auto& c : cands) {
        if ((int)window.size() >= n) break;
        if (c.cov >= 2) window.push_back(c.kf);
    }

    // 兜底：共视不足时退化为按时间取最近 n 帧
    if (window.size() < 2) {
        window.clear();
        int start = std::max(0, (int)all_kfs.size() - n);
        for (int i = start; i < (int)all_kfs.size(); i++)
            window.push_back(all_kfs[i]);
    }

    // 按 id 升序；BA 快照随后按实际观测连通分量选择局部锚，不能把
    // 与当前窗口断开的全局 kf#0/#1 强行带入优化。
    std::ranges::sort(window, {}, [](const Frame::Ptr& f) { return f->id; });
    return window;
}

OptimizationSnapshot LocalMapper::buildLocalBASnapshot(
    const Map::Ptr& map, const Atlas::Ptr& atlas,
    const std::vector<Frame::Ptr>& window,
    KeyframeId anchor_kf_id, int min_observed,
    size_t max_landmarks) const {
    OptimizationSnapshot snap;
    const auto* active_submap = atlas->activeSubmap();
    snap.submap_id = active_submap ? active_submap->id : 0;
    snap.topology_revision = map->topologyRevision();
    snap.geometry_revision = map->geometryRevision();

    const int required_observations = std::max(0, min_observed);
    for (const auto& kf : window) {
        const auto live_kf = map->getKeyFrame(kf->id);
        // 不只按 id 判断：不同 Map 的关键帧 id 会从 0 重新分配，旧任务
        // 可能与新子地图恰好撞号，必须绑定对象身份。
        if (!live_kf || live_kf.get() != kf.get()) continue;
        KeyframeState ks;
        ks.id = kf->id;
        ks.pose_cs = kf->pose_cs;
        for (size_t i = 0; i < kf->map_points.size(); i++) {
            const auto& mp = kf->map_points[i];
            if (!mp || i > std::numeric_limits<FeatureIndex>::max() ||
                !includeLocalBALandmark(
                    static_cast<int>(mp->observationCount()),
                    required_observations)) continue;
            const Observation observation{
                kf->id, static_cast<FeatureIndex>(i)};
            if (!mp->hasObservation(observation)) continue;
            ObservationState state;
            state.keyframe_id = kf->id;
            state.feature_index = static_cast<FeatureIndex>(i);
            state.map_point_id = mp->id;
            if (i < kf->keypoints.size()) {
                state.pixel = Vec2(kf->keypoints[i].pt.x,
                                   kf->keypoints[i].pt.y);
            }
            if (i < kf->pts_c.size() && kf->pts_c[i].allFinite() &&
                kf->pts_c[i].z() > 0.0) {
                state.camera_point = kf->pts_c[i];
            }
            snap.observations.push_back(std::move(state));
        }
        snap.keyframes.push_back(std::move(ks));
    }

    // 只保留显式 anchor KF 所在的观测连通分量。共视排序的兜底窗口可能
    // 混入没有共同地图点的远端帧；把它们送进 BA 会产生未锚定的自由度，
    // 也会让"窗口最早帧"错误地落在全局旧坐标上。
    if (snap.keyframes.size() > 1) {
        std::vector<size_t> parent(snap.keyframes.size());
        std::iota(parent.begin(), parent.end(), 0);
        auto find_root = [&](size_t value) {
            size_t root = value;
            while (parent[root] != root) root = parent[root];
            return root;
        };
        auto unite = [&](size_t a, size_t b) {
            a = find_root(a);
            b = find_root(b);
            if (a != b) parent[b] = a;
        };
        std::unordered_map<MapPointId, std::vector<size_t>> observations;
        for (const auto& observation : snap.observations) {
            auto it = std::ranges::find_if(
                snap.keyframes,
                [&](const KeyframeState& kf) {
                    return kf.id == observation.keyframe_id;
                });
            if (it == snap.keyframes.end()) continue;
            observations[observation.map_point_id].push_back(
                static_cast<size_t>(std::distance(snap.keyframes.begin(), it)));
        }
        for (const auto& [id, frames] : observations) {
            (void)id;
            if (frames.size() < 3) continue;  // 与 Optimizer 的 BA 门槛一致
            for (size_t i = 1; i < frames.size(); i++) unite(frames[0], frames[i]);
        }

        size_t current_idx = snap.keyframes.size();
        for (size_t i = 0; i < snap.keyframes.size(); i++) {
            if (snap.keyframes[i].id == anchor_kf_id) {
                current_idx = i;
                break;
            }
        }
        if (current_idx < snap.keyframes.size()) {
            const size_t current_root = find_root(current_idx);
            std::vector<KeyframeState> connected;
            connected.reserve(snap.keyframes.size());
            for (size_t i = 0; i < snap.keyframes.size(); i++)
                if (find_root(i) == current_root)
                    connected.push_back(std::move(snap.keyframes[i]));
            snap.keyframes = std::move(connected);
        }
    }

    std::unordered_set<KeyframeId> kept_keyframes;
    for (const auto& kf : snap.keyframes) kept_keyframes.insert(kf.id);
    std::erase_if(snap.observations, [&](const ObservationState& observation) {
        return !kept_keyframes.contains(observation.keyframe_id);
    });

    std::unordered_set<MapPointId> seen_mp;
    for (const auto& observation : snap.observations)
        seen_mp.insert(observation.map_point_id);
    // 只收集窗口正式观测且达到 min_observed 的点；id=0 是合法地图点，
    // 因此这里不能用 0 作为"无点"哨兵。
    snap.landmarks.reserve(seen_mp.size());
    for (const auto id : seen_mp) {
        auto mp = map->getMapPoint(id);
        if (mp && includeLocalBALandmark(
                static_cast<int>(mp->observationCount()), required_observations)) {
            snap.landmarks.push_back({id, mp->pos_s,
                                      static_cast<int>(mp->observationCount())});
        }
    }
    if (max_landmarks > 0 && snap.landmarks.size() > max_landmarks) {
        std::ranges::partial_sort(
            snap.landmarks,
            snap.landmarks.begin() + static_cast<std::ptrdiff_t>(max_landmarks),
            [](const LandmarkState& a, const LandmarkState& b) {
                if (a.observations != b.observations)
                    return a.observations > b.observations;
                return a.id < b.id;
            });
        snap.landmarks.resize(max_landmarks);
        std::unordered_set<MapPointId> selected;
        selected.reserve(snap.landmarks.size());
        for (const auto& landmark : snap.landmarks)
            selected.insert(landmark.id);
        std::erase_if(snap.observations, [&](const ObservationState& observation) {
            return !selected.contains(observation.map_point_id);
        });
    }
    // 锚定局部观测分量最早 2 帧（固定 = 局部基线长度，避免全局 kf#0/#1
    // 与当前窗口断开时造成 gauge 漂移）。Optimizer 还会按实际 BA 边再次
    // 校验/修正固定集。
    if (snap.keyframes.size() >= 2)
        snap.fixed_kf_ids = {snap.keyframes[0].id, snap.keyframes[1].id};
    return snap;
}

} // namespace vslam
