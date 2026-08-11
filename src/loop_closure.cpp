#include "vslam/loop_closure.h"
#include "vslam/mappoint.h"
#include "perf_monitor.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstdint>
#include <unordered_set>

#ifdef HAS_DBOW3
#include <DBoW3/DBoW3.h>
#include <unordered_map>
#endif

namespace vslam {

// cv::Mat(R, t) → SE3（T_cw 语义，供 PnP 输出转换）
namespace {
SE3 matToSE3(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat rmat = R;
    if (R.rows == 3 && R.cols == 1) cv::Rodrigues(R, rmat);  // rvec → 旋转矩阵
    Eigen::Matrix3d Re;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Re(i, j) = rmat.at<double>(i, j);
    return SE3(Eigen::Quaterniond(Re),
               Vec3(t.at<double>(0), t.at<double>(1), t.at<double>(2)));
}
} // namespace

#ifdef HAS_DBOW3
// ============================================================
// PIMPL：DBoW3 具体状态（与 Map 关键帧生命周期同步）
// ============================================================
class LoopClosure::Impl {
public:
    struct CachedKeyFrame {
        Frame::Ptr frame;
        SubmapId submap_id = 0;
        SE3 T_ws;
        Vec3 camera_position_local = Vec3::Zero();
    };

    std::unique_ptr<DBoW3::Vocabulary> vocab;   // 词典（加载后只读）
    std::unique_ptr<DBoW3::Database>   db;      // 数据库（随关键帧增长）
    // 关键帧 id ↔ 数据库条目 id 双向映射（查询结果反查关键帧用）
    std::unordered_map<unsigned long, DBoW3::EntryId> kf_db_id;
    std::unordered_map<DBoW3::EntryId, unsigned long> db_id_kf;
    std::unordered_map<unsigned long, CachedKeyFrame> kf_cache;  // 候选帧+位置快照
    std::unordered_map<unsigned long, DBoW3::BowVector> kf_bow; // 词袋缓存

    // 一个地点不是单个 KF：相邻历史 KF 形成一个 place cluster。该状态只保留
    // 最近出现过的有限个假设，用跨查询重复观测给候选排序，不能替代 PnP 几何门。
    struct PlaceHypothesis {
        Frame::Ptr representative;
        SubmapId submap_id = 0;
        unsigned long anchor_id = 0;
        unsigned long last_query_id = 0;
        std::uint64_t last_seen = 0;
        int support = 0;
        double best_score = 0.0;
    };
    std::vector<PlaceHypothesis> place_hypotheses;
    std::uint64_t query_serial = 0;

    void rebuildDatabase() {
        if (!vocab) {
            db.reset();
            kf_db_id.clear();
            db_id_kf.clear();
            return;
        }
        db = std::make_unique<DBoW3::Database>(*vocab, false, 0);
        kf_db_id.clear();
        db_id_kf.clear();
        std::vector<unsigned long> ids;
        ids.reserve(kf_bow.size());
        for (const auto& [id, bow] : kf_bow) {
            (void)bow;
            ids.push_back(id);
        }
        std::ranges::sort(ids);
        for (const auto id : ids) {
            const auto bow_it = kf_bow.find(id);
            if (bow_it == kf_bow.end()) continue;
            const DBoW3::EntryId eid = db->add(bow_it->second);
            kf_db_id[id] = eid;
            db_id_kf[eid] = id;
        }
    }
};
#endif

LoopClosure::LoopClosure() {
#ifdef HAS_DBOW3
    impl_ = std::make_unique<Impl>();
#endif
}

LoopClosure::~LoopClosure() = default;

void LoopClosure::setParams(double min_score, int temporal_window,
                            int min_loop_inliers, double pnp_inlier_ratio,
                            double ransac_pixel_threshold, const Camera& camera,
                            int top_candidates,
                            double pos_prior_dist_m,
                            int pos_prior_gap,
                            double max_reprojection_rmse,
                            double min_positive_depth_ratio,
                            int min_grid_cells) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_score_             = min_score;
    temporal_window_       = temporal_window;
    min_loop_inliers_      = min_loop_inliers;
    pnp_inlier_ratio_      = pnp_inlier_ratio;
    ransac_pixel_threshold_ = ransac_pixel_threshold;
    camera_                = camera;
    top_candidates_        = std::max(5, top_candidates);
    position_prior_dist_m_ = pos_prior_dist_m;
    position_prior_gap_    = std::max(temporal_window_, pos_prior_gap);
    max_reprojection_rmse_ = std::max(0.0, max_reprojection_rmse);
    min_positive_depth_ratio_ = std::clamp(min_positive_depth_ratio, 0.0, 1.0);
    min_grid_cells_ = std::max(1, min_grid_cells);
}

bool LoopClosure::loadVocabulary(const std::string& vocab_path) {
#ifndef HAS_DBOW3
    (void)vocab_path;
    LOG_WARN("LoopClosure: built without DBoW3, loop closure disabled");
    return false;
#else
    if (vocab_path.empty()) {
        LOG_WARN("LoopClosure: vocab_path is empty, loop closure disabled");
        return false;
    }
    auto vocab = std::make_unique<DBoW3::Vocabulary>();
    try {
        vocab->load(vocab_path);  // 支持 .txt（DBoW2 文本）与 .dbow3（二进制）
    } catch (const std::exception& e) {
        LOG_ERROR("LoopClosure: failed to load vocabulary (" << e.what() << ")");
        return false;
    }
    if (vocab->empty()) {
        LOG_ERROR("LoopClosure: vocabulary is empty");
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->vocab = std::move(vocab);
    impl_->kf_db_id.clear();
    impl_->db_id_kf.clear();
    impl_->kf_cache.clear();
    impl_->kf_bow.clear();
    impl_->place_hypotheses.clear();
    impl_->query_serial = 0;
    impl_->rebuildDatabase();
    LOG_INFO("LoopClosure: vocabulary loaded (" << vocab_path << "), "
             << impl_->vocab->size() << " words");
    return true;
#endif
}

void LoopClosure::removeKeyFrames(
    const std::vector<KeyframeId>& keyframe_ids) {
#ifndef HAS_DBOW3
    (void)keyframe_ids;
#else
    if (keyframe_ids.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_set<KeyframeId> removed;
    removed.reserve(keyframe_ids.size());
    bool changed = false;
    for (const auto id : keyframe_ids) {
        removed.insert(id);
        changed = impl_->kf_cache.erase(id) > 0 || changed;
        changed = impl_->kf_bow.erase(id) > 0 || changed;
    }
    if (!changed) return;
    std::erase_if(impl_->place_hypotheses, [&](const auto& hypothesis) {
        return removed.contains(hypothesis.anchor_id) ||
               (hypothesis.representative &&
                removed.contains(hypothesis.representative->id));
    });
    // DBoW3 只有 clear()，没有单条删除；一次批量回收只重建一次，既释放
    // 倒排表内存，也保证旧条目不再消耗 query 的 Top-N 名额。
    impl_->rebuildDatabase();
#endif
}

size_t LoopClosure::indexedKeyFrameCount() const {
#ifndef HAS_DBOW3
    return 0;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->kf_cache.size();
#endif
}

void LoopClosure::addKeyFrame(Frame::Ptr kf, SubmapId submap_id,
                              const SE3& T_ws) {
#ifndef HAS_DBOW3
    (void)kf; (void)submap_id; (void)T_ws;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    if (!impl_->vocab || !impl_->db || !kf || kf->descriptors.empty()) return;
    const Vec3 camera_position_local = kf->pose_cs.inverse().t;
    // 异步任务重排时同一 KF 可能重复提交；数据库和保护集合保持幂等。
    if (impl_->kf_db_id.contains(kf->id)) return;
    // 计算词袋向量并入库（复用已算过的，避免重复 transform）
    auto it = impl_->kf_bow.find(kf->id);
    DBoW3::BowVector bow;
    if (it != impl_->kf_bow.end()) {
        bow = it->second;
    } else {
        impl_->vocab->transform(kf->descriptors, bow);
        impl_->kf_bow[kf->id] = bow;
    }
    DBoW3::EntryId eid = impl_->db->add(bow);
    impl_->kf_db_id[kf->id] = eid;
    impl_->db_id_kf[eid]    = kf->id;
    impl_->kf_cache[kf->id] = {kf, submap_id, T_ws, camera_position_local};
#endif
}

std::vector<LoopClosure::LoopCandidate> LoopClosure::detectLoop(
    Frame::Ptr kf, SubmapId query_submap_id, const SE3& query_T_ws,
    const SubmapPoses& latest_submap_poses) {
#ifndef HAS_DBOW3
    (void)kf; (void)query_submap_id; (void)query_T_ws;
    (void)latest_submap_poses;
    return {};
#else
    std::lock_guard<std::mutex> lock(mutex_);
    PERF_SCOPE("lc.detect");
    std::vector<LoopCandidate> candidates;
    // 计数所有检测调用，使中间没有视觉命中的查询也能形成真实断档。
    ++impl_->query_serial;
    if (!impl_->vocab || !impl_->db || !kf || kf->descriptors.empty()) return candidates;

    // 1. 词袋查询 Top-N（召回扩宽；准确性交给下方时间窗/分数过滤 + PnP 验证）
    DBoW3::BowVector bow;
    impl_->vocab->transform(kf->descriptors, bow);
    DBoW3::QueryResults results;
    impl_->db->query(bow, results, top_candidates_);

    // 2. 时间过滤 + 3. 分数过滤：保留整个 Top-N，后面按历史邻域聚成地点。
    //    Top-N 是 DBoW 查询的内存/计算上限，不是几何验证前的 Top-3 截断。
    std::vector<LoopCandidate> scored;
    for (const auto& r : results) {
        auto it = impl_->db_id_kf.find(r.Id);
        if (it == impl_->db_id_kf.end()) continue;
        const unsigned long cand_id = it->second;
        // 分数阈值：DBoW3 归一化分数（0~1）
        if (r.Score < min_score_) continue;
        auto kf_it = impl_->kf_cache.find(cand_id);
        if (kf_it == impl_->kf_cache.end()) continue;
        const auto& cached = kf_it->second;
        // Frame id 只在同一子地图内可作为时间顺序。跨子地图的全局 id
        // 可能连续，不能因刚创建子地图而误杀真正的跨图回环。
        if (cached.submap_id == query_submap_id &&
            (cand_id >= kf->id ||
             kf->id - cand_id < (unsigned long)temporal_window_)) continue;
        scored.push_back({cached.frame, cached.submap_id, r.Score});
    }

    // 4. 历史时间邻域聚类。只依赖候选帧自身的子地图身份和关键帧序号；
    //    跨子地图不借用全局 id 的时间顺序，因此每个跨图地点至少以单 KF
    //    为一个 cluster，避免把不同子地图的相同 id 错合。
    struct PlaceCluster {
        LoopCandidate representative;
        unsigned long min_id = 0;
        unsigned long max_id = 0;
    };
    const unsigned long cluster_gap = static_cast<unsigned long>(
        std::max(1, temporal_window_));
    std::ranges::sort(scored, [](const LoopCandidate& a, const LoopCandidate& b) {
        if (a.submap_id != b.submap_id) return a.submap_id < b.submap_id;
        return a.frame->id < b.frame->id;
    });
    std::vector<PlaceCluster> clusters;
    for (const auto& candidate : scored) {
        if (clusters.empty() ||
            clusters.back().representative.submap_id != candidate.submap_id ||
            (candidate.frame->id > clusters.back().max_id &&
             candidate.frame->id - clusters.back().max_id > cluster_gap)) {
            PlaceCluster cluster{candidate, candidate.frame->id, candidate.frame->id};
            cluster.representative.cluster_members.push_back(candidate.frame);
            clusters.push_back(std::move(cluster));
            continue;
        }
        auto& cluster = clusters.back();
        cluster.min_id = std::min(cluster.min_id, candidate.frame->id);
        cluster.max_id = std::max(cluster.max_id, candidate.frame->id);
        cluster.representative.cluster_members.push_back(candidate.frame);
        if (candidate.score > cluster.representative.score ||
            (candidate.score == cluster.representative.score &&
             candidate.frame->id < cluster.representative.frame->id)) {
            auto members = std::move(cluster.representative.cluster_members);
            cluster.representative = candidate;
            cluster.representative.cluster_members = std::move(members);
        }
    }

    // 5. 更新有界的跨查询假设。一次 detectLoop 可能被并发重复调用；同一
    //    query KF 不重复增加 support，避免并发重排伪造“连续观测”。
    auto same_cluster = [cluster_gap](const Impl::PlaceHypothesis& h,
                                      const LoopCandidate& c) {
        if (h.submap_id != c.submap_id || !h.representative || !c.frame) return false;
        const unsigned long a = h.anchor_id;
        const unsigned long b = c.frame->id;
        return (a >= b) ? (a - b <= cluster_gap) : (b - a <= cluster_gap);
    };
    for (const auto& cluster : clusters) {
        auto hit = std::ranges::find_if(impl_->place_hypotheses,
                                        [&](const auto& h) {
                                            return same_cluster(h, cluster.representative);
                                        });
        if (hit == impl_->place_hypotheses.end()) {
            impl_->place_hypotheses.push_back({
                cluster.representative.frame, cluster.representative.submap_id,
                cluster.representative.frame->id, kf->id,
                impl_->query_serial, 1, cluster.representative.score});
        } else {
            if (hit->last_query_id != kf->id) {
                // 允许至多一次缺测；更长断档不应把偶发重复误认为连续
                // 地点证据，重新从当前查询建立假设。
                if (impl_->query_serial <= hit->last_seen + 2)
                    ++hit->support;
                else
                    hit->support = 1;
            }
            hit->last_query_id = kf->id;
            hit->last_seen = impl_->query_serial;
            if (cluster.representative.score >= hit->best_score) {
                hit->representative = cluster.representative.frame;
                hit->anchor_id = cluster.representative.frame->id;
                hit->best_score = cluster.representative.score;
            }
        }
    }
    // 假设是排序辅助而非地图数据；限制数量并老化，保证异常长序列不因
    // 多帧跟踪额外产生无界内存。当前查询未观测到的假设不输出。
    constexpr std::size_t kMaxPlaceHypotheses = 256;
    constexpr std::uint64_t kHypothesisLifetime = 128;
    std::erase_if(impl_->place_hypotheses, [&](const auto& h) {
        return impl_->query_serial > h.last_seen &&
               impl_->query_serial - h.last_seen > kHypothesisLifetime;
    });
    if (impl_->place_hypotheses.size() > kMaxPlaceHypotheses) {
        std::ranges::sort(impl_->place_hypotheses,
                          [](const auto& a, const auto& b) {
                              if (a.support != b.support) return a.support > b.support;
                              return a.last_seen > b.last_seen;
                          });
        impl_->place_hypotheses.resize(kMaxPlaceHypotheses);
    }

    // 将当前查询的各个地点代表按视觉分数和时序支持排序。成熟假设优先，
    // 但首次观测的 cluster 仍返回，保持旧 API 的单次查询兼容性。
    struct RankedCluster {
        LoopCandidate candidate;
        int support = 1;
    };
    std::vector<RankedCluster> ranked;
    ranked.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        int support = 1;
        if (auto hit = std::ranges::find_if(
                impl_->place_hypotheses, [&](const auto& h) {
                    return same_cluster(h, cluster.representative);
                }); hit != impl_->place_hypotheses.end())
            support = hit->support;
        ranked.push_back({cluster.representative, support});
    }
    std::ranges::sort(ranked, [](const RankedCluster& a, const RankedCluster& b) {
        if (a.support != b.support) return a.support > b.support;
        if (a.candidate.score != b.candidate.score)
            return a.candidate.score > b.candidate.score;
        if (a.candidate.submap_id != b.candidate.submap_id)
            return a.candidate.submap_id < b.candidate.submap_id;
        return a.candidate.frame->id < b.candidate.frame->id;
    });
    // 长期运行中同一外观会在多个历史路段形成 cluster；固定 8 个地点在
    // 跨子图尾段会把真实旧地点挤掉。仍保持严格有界，但给 BoW/位置先验
    // 合计 12 个地点槽，几何门负责最终验收。
    constexpr std::size_t kMaxClustersPerQuery = 12;
    for (const auto& entry : ranked) {
        if (candidates.size() >= kMaxClustersPerQuery) break;
        LoopCandidate candidate = entry.candidate;
        candidate.temporal_support = entry.support;
        candidate.mature = entry.support >= 2;
        candidates.push_back(std::move(candidate));
    }

    // 6. 位置先验：估计轨迹自交区域词袋分数常偏低，用世界系距离补召回。
    //    与词袋候选并列返回（放在后面），由调用方的 PnP 双门槛验证兜底。
    if (position_prior_dist_m_ > 0.0) {
        const Vec3 curr_pos = query_T_ws * kf->pose_cs.inverse().t;
        struct PositionPrior {
            double distance = 0.0;
            unsigned long id = 0;
            const Impl::CachedKeyFrame* cached = nullptr;
        };
        std::vector<PositionPrior> position_priors;
        for (const auto& [cand_id, cached] : impl_->kf_cache) {
            if (!cached.frame) continue;
            if (cached.submap_id == query_submap_id &&
                (cand_id >= kf->id ||
                 kf->id - cand_id < (unsigned long)position_prior_gap_)) continue;
            // pose_cs.inverse().t 是子地图局部光心。只有先乘最新的
            // T_ws 才能进入世界系；加入时的快照仅作缺少 Atlas 条目时的回退。
            SE3 T_ws = cached.T_ws;
            if (const auto pose_it = latest_submap_poses.find(cached.submap_id);
                pose_it != latest_submap_poses.end()) {
                T_ws = pose_it->second;
            }
            const Vec3 cand_pos = T_ws * cached.camera_position_local;
            const double dist = (cand_pos - curr_pos).norm();
            if (dist < position_prior_dist_m_)
                position_priors.push_back({dist, cand_id, &cached});
        }
        // 先用与 BoW 完全相同的“连续间隔单链”语义建历史地点。
        // 例如 gap=30 时 0/25/50 是一个连通 cluster，不能因 0↔50
        // 超过 gap 而浪费第二个 prior 槽。每簇只保留世界距离最近者。
        std::ranges::sort(position_priors, [](const auto& a, const auto& b) {
            if (a.cached->submap_id != b.cached->submap_id)
                return a.cached->submap_id < b.cached->submap_id;
            return a.id < b.id;
        });
        struct PositionPlace {
            PositionPrior best;
            SubmapId submap_id = 0;
            unsigned long min_id = 0;
            unsigned long max_id = 0;
        };
        std::vector<PositionPlace> position_places;
        if (!position_priors.empty()) {
            PositionPrior best = position_priors.front();
            SubmapId cluster_submap = best.cached->submap_id;
            unsigned long min_id = best.id;
            unsigned long last_id = best.id;
            for (size_t i = 1; i < position_priors.size(); ++i) {
                const auto& prior = position_priors[i];
                const bool connected = prior.cached->submap_id == cluster_submap &&
                    prior.id >= last_id && prior.id - last_id <= cluster_gap;
                if (!connected) {
                    position_places.push_back(
                        {best, cluster_submap, min_id, last_id});
                    best = prior;
                    cluster_submap = prior.cached->submap_id;
                    min_id = prior.id;
                } else if (prior.distance < best.distance ||
                           (prior.distance == best.distance && prior.id < best.id)) {
                    best = prior;
                }
                last_id = prior.id;
            }
            position_places.push_back({best, cluster_submap, min_id, last_id});
        }
        std::ranges::sort(position_places, [](const auto& a, const auto& b) {
            if (a.best.distance != b.best.distance)
                return a.best.distance < b.best.distance;
            if (a.submap_id != b.submap_id) return a.submap_id < b.submap_id;
            return a.min_id < b.min_id;
        });
        // 位置先验本身也是多模态的：有累积漂移时“最近的单 KF”
        // 未必是真实重访地点。从不同历史时间 cluster 中保留最多 4 个，
        // 且与 BoW 合计仍不超过 12 个几何验证槽。
        constexpr std::size_t kMaxPositionPriorPlaces = 4;
        std::vector<LoopCandidate> prior_candidates;
        for (const auto& place : position_places) {
            if (prior_candidates.size() >= kMaxPositionPriorPlaces) break;
            const auto& prior = place.best;
            if (!prior.cached || !prior.cached->frame) continue;
            // 跨模态按“子图 + 历史时间簇”去重，不只比较代表 KF id。
            // 否则 BoW kf#100 和 prior kf#101 会浪费两个几何槽。
            const bool dup = std::ranges::any_of(candidates, [&](const auto& c) {
                if (!c.frame || c.submap_id != prior.cached->submap_id) return false;
                unsigned long bow_min = c.frame->id;
                unsigned long bow_max = c.frame->id;
                for (const auto& member : c.cluster_members) {
                    if (!member) continue;
                    bow_min = std::min(bow_min, member->id);
                    bow_max = std::max(bow_max, member->id);
                }
                // 两个按相同 gap 形成的单链地点，只要区间相交或端点距离
                // 不超过 gap，合并后仍是同一连通分量，必须跨模态去重。
                return place.min_id <= bow_max + cluster_gap &&
                       bow_min <= place.max_id + cluster_gap;
            });
            if (dup) continue;
            prior_candidates.push_back(
                {prior.cached->frame, prior.cached->submap_id, 0.0});
            LOG_INFO("LoopClosure: position prior candidate kf#"
                     << prior.id << " submap#" << prior.cached->submap_id
                     << " (dist=" << prior.distance << "m)");
        }
        // 一次性从最低优先级 BoW 尾部预留 N 个槽后再 append；不能
        // 在循环内 pop/push，否则后来的 prior 会弹掉刚加入的更近 prior。
        const size_t bow_keep = std::min(
            candidates.size(), kMaxClustersPerQuery - prior_candidates.size());
        candidates.resize(bow_keep);
        for (auto& prior : prior_candidates)
            candidates.push_back(std::move(prior));
        }

    if (!candidates.empty())
        LOG_INFO("LoopClosure: " << candidates.size() << " candidates, top kf#"
                 << candidates.front().frame->id << " submap#"
                 << candidates.front().submap_id << " (score="
                 << candidates.front().score << ")");
    return candidates;
#endif
}

bool LoopClosure::verifyLoop(Frame::Ptr kf_curr, Frame::Ptr kf_loop,
                             SE3& T_loop_curr) {
#ifndef HAS_DBOW3
    (void)kf_curr; (void)kf_loop; (void)T_loop_curr;
    return false;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    PERF_SCOPE("lc.verify");
    if (!camera_ || !kf_curr || !kf_loop) return false;

    // 1. ORB 匹配：knn + ratio（queryIdx 属于 kf_curr，trainIdx 属于 kf_loop）
    // A2：验证匹配 ratio 放宽（0.7->0.8）——回环场景尺度/光照差异大，
    // 严格比率漏掉真对应；内点判定仍由下方 PnP 双门槛把关
    auto matches = matcher_.match(kf_curr, kf_loop, 0.8, false);
    if (matches.size() < 8) {
        LOG_WARN("LoopClosure: too few matches (" << matches.size() << ")");
        return false;
    }

    // 2. 收集 3D-2D 对应：3D = kf_loop 侧已关联的地图点（回环尺度），
    //    2D = kf_curr 侧特征点像素
    //    A1：地图点被 cull 断链的候选 KF 特征，用其双目观测 pts_c
    //    （相机系，保留在 KF 上）转局部系补 3D——回环召回不再依赖
    //    cull 保留点（实测候选 KF 空点 → "too few 3D-2D" 直接失败）。
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    for (size_t i = 0; i < matches.size(); i++) {
        const size_t ti = matches[i].trainIdx;
        auto mp = (ti < kf_loop->map_points.size()) ? kf_loop->map_points[ti] : nullptr;
        if (!mp && ti < kf_loop->pts_c.size() && kf_loop->pts_c[ti].z() > 0) {
            const Vec3 p_s = kf_loop->pose_cs.inverse() * kf_loop->pts_c[ti];
            const auto& kp = kf_curr->keypoints[matches[i].queryIdx];
            pts3d.emplace_back((float)p_s.x(), (float)p_s.y(), (float)p_s.z());
            pts2d.emplace_back(kp.pt.x, kp.pt.y);
            continue;
        }
        if (!mp) continue;
        const auto& kp = kf_curr->keypoints[matches[i].queryIdx];
        pts3d.emplace_back(mp->pos_s.x(), mp->pos_s.y(), mp->pos_s.z());
        pts2d.emplace_back(kp.pt.x, kp.pt.y);
    }
    if (pts3d.size() < 8) {
        LOG_WARN("LoopClosure: too few 3D-2D correspondences (" << pts3d.size() << ")");
        return false;
    }

    // 3. PnP RANSAC：求 kf_curr 在回环尺度下的位姿（3D 点是回环尺度世界点）
    cv::Mat K = camera_->K();
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    bool ok = cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec,
                                 false, 200, ransac_pixel_threshold_, 0.99,
                                 inliers, cv::SOLVEPNP_ITERATIVE);
    if (!ok || inliers.empty()) return false;

    // 4. 内点判定：数量/比例 + RMSE/正深度/空间覆盖。
    double ratio = (double)inliers.size() / (double)pts3d.size();
    std::vector<cv::Point2f> projected;
    cv::projectPoints(pts3d, rvec, tvec, K, cv::Mat(), projected);
    double sum_sq = 0.0;
    int positive_depth = 0;
    std::unordered_set<int> grid_cells;
    const SE3 T_cw_quality = matToSE3(rvec, tvec);
    const double image_width = camera_->img_width > 0
        ? camera_->img_width : std::max(1.0, camera_->cx * 2.0);
    const double image_height = camera_->img_height > 0
        ? camera_->img_height : std::max(1.0, camera_->cy * 2.0);
    for (const int idx : inliers) {
        if (idx < 0 || idx >= static_cast<int>(pts3d.size())) continue;
        const double dx = projected[idx].x - pts2d[idx].x;
        const double dy = projected[idx].y - pts2d[idx].y;
        sum_sq += dx * dx + dy * dy;
        const Vec3 p(pts3d[idx].x, pts3d[idx].y, pts3d[idx].z);
        if ((T_cw_quality * p).z() > 0.0) ++positive_depth;
        const int gx = std::clamp(static_cast<int>(
            pts2d[idx].x * 8.0 / image_width), 0, 7);
        const int gy = std::clamp(static_cast<int>(
            pts2d[idx].y * 6.0 / image_height), 0, 5);
        grid_cells.insert(gy * 8 + gx);
    }
    const double rmse = std::sqrt(sum_sq / std::max<size_t>(1, inliers.size()));
    const double positive_ratio = positive_depth /
        static_cast<double>(std::max<size_t>(1, inliers.size()));
    if ((int)inliers.size() < min_loop_inliers_ || ratio < pnp_inlier_ratio_ ||
        rmse > max_reprojection_rmse_ ||
        positive_ratio < min_positive_depth_ratio_ ||
        static_cast<int>(grid_cells.size()) < min_grid_cells_) {
        LOG_WARN("LoopClosure: PnP rejected (" << inliers.size() << " inliers, "
                 << ratio << " ratio, " << rmse << "px, "
                 << positive_ratio << " positive, " << grid_cells.size()
                 << " grid cells)");
        return false;
    }

    // 5. solvePnP 的 rvec/tvec 满足 p_c = R·p_w + t（世界→相机），即
    //    matToSE3(rvec, tvec) 是当前帧的 T_cw（与 trackFrame 中直接把
    //    solvePnP 结果存为 pose_cs 的用法一致）。位姿图边的测量约定为
    //    Z = X_loop⁻¹ · X_curr（X 为 T_wc，见 loop_closure.h 与
    //    test_vo.cpp 位姿图测试），故需要 T_cw_curr 的逆：
    //      Z = kf_loop->pose_cs * T_wc_curr = T_cw_loop * T_cw_curr⁻¹。
    //    注意：6c311d7 曾误判 solvePnP 输出为 T_wc 而删掉此逆，导致
    //    回环约束与里程计矛盾、每次校正把轨迹拉向错误方向（ATE 恶化），
    //    已在 2026-08-04 修正回 T_cw 语义。
    const SE3 T_cw_curr_in_loop = matToSE3(rvec, tvec);
    T_loop_curr = kf_loop->pose_cs * T_cw_curr_in_loop.inverse();
    LOG_INFO("LoopClosure: verified! kf#" << kf_loop->id << " -> kf#" << kf_curr->id
             << " inliers=" << inliers.size());
    return true;
#endif
}

} // namespace vslam
