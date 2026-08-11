#include "vslam/loop_closure.h"
#include "vslam/mappoint.h"
#include "perf_monitor.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifdef HAS_DBOW3
#include <DBoW3/DBoW3.h>
#include <quicklz.h>
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

namespace {
constexpr size_t kCompactSignatureBlocks = 5;  // global + 2x2
constexpr size_t kCompactBitsPerBlock = 256;
constexpr size_t kCompactSignatureBytes =
    kCompactSignatureBlocks * kCompactBitsPerBlock;

struct CompactSignature {
    std::array<std::int8_t, kCompactSignatureBytes> values{};
    std::array<std::uint16_t, kCompactSignatureBlocks> counts{};
};

#ifdef HAS_DBOW3
using FlatBow = std::vector<std::pair<std::uint32_t, double>>;

class FlatVocabulary {
public:
    bool load(const std::string& path) {
        if constexpr (std::endian::native != std::endian::little)
            throw std::runtime_error("flat DBoW3 only supports little-endian files");
        std::ifstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot open vocabulary");

        std::uint64_t signature = 0;
        bool compressed = false;
        std::uint32_t node_count = 0;
        readStream(stream, signature);
        readStream(stream, compressed);
        readStream(stream, node_count);
        if (signature != 88877711233ULL || node_count == 0)
            throw std::runtime_error("invalid DBoW3 binary header");

        std::vector<unsigned char> payload;
        if (compressed) {
            std::uint32_t chunks = 0;
            readStream(stream, chunks);
            payload.reserve(static_cast<size_t>(chunks) * 10000);
            qlz_state_decompress state{};
            std::array<char, 10400> input{};
            std::array<unsigned char, 10000> output{};
            for (std::uint32_t i = 0; i < chunks; ++i) {
                stream.read(input.data(), 9);
                if (!stream) throw std::runtime_error("truncated QuickLZ header");
                const size_t compressed_size = qlz_size_compressed(input.data());
                if (compressed_size < 9 || compressed_size > input.size())
                    throw std::runtime_error("invalid QuickLZ chunk size");
                stream.read(input.data() + 9,
                            static_cast<std::streamsize>(compressed_size - 9));
                if (!stream) throw std::runtime_error("truncated QuickLZ chunk");
                const size_t decompressed_size = qlz_decompress(
                    input.data(), output.data(), &state);
                if (decompressed_size > output.size())
                    throw std::runtime_error("invalid QuickLZ output size");
                payload.insert(payload.end(), output.begin(),
                               output.begin() + static_cast<ptrdiff_t>(decompressed_size));
            }
        } else {
            payload.assign(std::istreambuf_iterator<char>(stream), {});
        }

        size_t cursor = 0;
        readPayload(payload, cursor, branching_factor_);
        readPayload(payload, cursor, depth_levels_);
        readPayload(payload, cursor, scoring_);
        readPayload(payload, cursor, weighting_);
        if (branching_factor_ <= 0 || depth_levels_ <= 0 ||
            scoring_ != 0 || weighting_ != 0)
            throw std::runtime_error(
                "flat DBoW3 currently requires L1_NORM + TF_IDF");

        descriptors_.assign(node_count, {});
        weights_.assign(node_count, 0.0);
        first_children_.assign(node_count, 0);
        child_counts_.assign(node_count, 0);
        word_ids_.assign(node_count, kInvalidWord);
        for (std::uint32_t i = 1; i < node_count; ++i) {
            std::uint32_t id = 0;
            std::uint32_t parent = 0;
            readPayload(payload, cursor, id);
            readPayload(payload, cursor, parent);
            if (id >= node_count || parent >= node_count)
                throw std::runtime_error("flat DBoW3 node id out of range");
            readPayload(payload, cursor, weights_[id]);
            int cols = 0;
            int rows = 0;
            int type = 0;
            readPayload(payload, cursor, cols);
            readPayload(payload, cursor, rows);
            readPayload(payload, cursor, type);
            if (cols != 32 || rows != 1 || type != CV_8UC1)
                throw std::runtime_error("flat DBoW3 requires 32-byte ORB nodes");
            readBytes(payload, cursor, descriptors_[id].data(),
                      descriptors_[id].size());

            if (child_counts_[parent] == 0)
                first_children_[parent] = id;
            else if (id != first_children_[parent] + child_counts_[parent])
                throw std::runtime_error("non-contiguous DBoW3 children");
            if (child_counts_[parent] ==
                std::numeric_limits<std::uint16_t>::max())
                throw std::runtime_error("DBoW3 child count overflow");
            ++child_counts_[parent];
        }

        std::uint32_t word_count = 0;
        readPayload(payload, cursor, word_count);
        for (std::uint32_t i = 0; i < word_count; ++i) {
            std::uint32_t word = 0;
            std::uint32_t node = 0;
            readPayload(payload, cursor, word);
            readPayload(payload, cursor, node);
            if (node >= word_ids_.size())
                throw std::runtime_error("flat DBoW3 word node out of range");
            word_ids_[node] = word;
        }
        word_count_ = word_count;
        return true;
    }

    [[nodiscard]] FlatBow transform(const cv::Mat& descriptors) const {
        FlatBow bow;
        if (descriptors_.empty() || descriptors.empty() ||
            descriptors.type() != CV_8U || descriptors.cols < 32) return bow;
        bow.reserve(static_cast<size_t>(descriptors.rows));
        for (int row = 0; row < descriptors.rows; ++row) {
            std::uint32_t node_id = 0;
            const auto* descriptor = descriptors.ptr<unsigned char>(row);
            while (child_counts_[node_id] > 0) {
                const std::uint32_t first_child = first_children_[node_id];
                const std::uint16_t child_count = child_counts_[node_id];
                std::uint32_t best_id = first_child;
                unsigned best_distance = std::numeric_limits<unsigned>::max();
                for (std::uint32_t offset = 0; offset < child_count; ++offset) {
                    const std::uint32_t child_id = first_child + offset;
                    const unsigned distance = hamming(
                        descriptor, descriptors_[child_id].data());
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_id = child_id;
                    }
                }
                node_id = best_id;
            }
            if (word_ids_[node_id] != kInvalidWord && weights_[node_id] > 0.0)
                bow.emplace_back(word_ids_[node_id], weights_[node_id]);
        }
        std::ranges::sort(bow, [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        size_t output = 0;
        for (const auto& item : bow) {
            if (output > 0 && bow[output - 1].first == item.first) {
                bow[output - 1].second += item.second;
            } else {
                bow[output++] = item;
            }
        }
        bow.resize(output);
        double norm = 0.0;
        for (const auto& [word, value] : bow) {
            (void)word;
            norm += std::abs(value);
        }
        if (norm > 0.0)
            for (auto& [word, value] : bow) {
                (void)word;
                value /= norm;
            }
        return bow;
    }

    [[nodiscard]] static double score(const FlatBow& a, const FlatBow& b) {
        double result = 0.0;
        size_t ia = 0;
        size_t ib = 0;
        while (ia < a.size() && ib < b.size()) {
            if (a[ia].first == b[ib].first) {
                result += std::min(a[ia].second, b[ib].second);
                ++ia;
                ++ib;
            } else if (a[ia].first < b[ib].first) {
                ++ia;
            } else {
                ++ib;
            }
        }
        return std::clamp(result, 0.0, 1.0);
    }

    [[nodiscard]] size_t size() const { return word_count_; }
    [[nodiscard]] size_t memoryBytes() const {
        return descriptors_.capacity() * sizeof(Descriptor) +
               weights_.capacity() * sizeof(double) +
               first_children_.capacity() * sizeof(std::uint32_t) +
               child_counts_.capacity() * sizeof(std::uint16_t) +
               word_ids_.capacity() * sizeof(std::uint32_t);
    }

private:
    static constexpr std::uint32_t kInvalidWord =
        std::numeric_limits<std::uint32_t>::max();
    template <typename T>
    static void readStream(std::istream& stream, T& value) {
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!stream) throw std::runtime_error("truncated DBoW3 stream");
    }

    template <typename T>
    static void readPayload(const std::vector<unsigned char>& payload,
                            size_t& cursor, T& value) {
        readBytes(payload, cursor, &value, sizeof(T));
    }

    static void readBytes(const std::vector<unsigned char>& payload,
                          size_t& cursor, void* output, size_t bytes) {
        if (cursor > payload.size() || bytes > payload.size() - cursor)
            throw std::runtime_error("truncated DBoW3 payload");
        std::memcpy(output, payload.data() + cursor, bytes);
        cursor += bytes;
    }

    static unsigned hamming(const unsigned char* a, const unsigned char* b) {
        unsigned distance = 0;
        for (size_t offset = 0; offset < 32; offset += sizeof(std::uint64_t)) {
            std::uint64_t av = 0;
            std::uint64_t bv = 0;
            std::memcpy(&av, a + offset, sizeof(av));
            std::memcpy(&bv, b + offset, sizeof(bv));
            distance += std::popcount(av ^ bv);
        }
        return distance;
    }

    int branching_factor_ = 0;
    int depth_levels_ = 0;
    int scoring_ = 0;
    int weighting_ = 0;
    size_t word_count_ = 0;
    using Descriptor = std::array<unsigned char, 32>;
    std::vector<Descriptor> descriptors_;
    std::vector<double> weights_;
    std::vector<std::uint32_t> first_children_;
    std::vector<std::uint16_t> child_counts_;
    std::vector<std::uint32_t> word_ids_;
};
#endif
} // namespace

// ============================================================
// PIMPL：DBoW3 / compact_binary 状态（与 Map 关键帧生命周期同步）
// ============================================================
class LoopClosure::Impl {
public:
    enum class Backend { None, DBoW3, FlatDBoW3, CompactBinary };
    struct CachedKeyFrame {
        Frame::Ptr frame;
        SubmapId submap_id = 0;
        SE3 T_ws;
        Vec3 camera_position_local = Vec3::Zero();
        std::uint64_t insertion_serial = 0;
    };

    Backend backend = Backend::None;
    size_t compact_max_keyframes = 256;
    std::uint64_t insertion_serial = 0;
    std::unordered_map<unsigned long, CachedKeyFrame> kf_cache;
    std::unordered_map<unsigned long, CompactSignature> compact_signatures;

#ifdef HAS_DBOW3
    std::unique_ptr<FlatVocabulary> flat_vocab;
    std::unordered_map<unsigned long, FlatBow> flat_bow;
    std::unique_ptr<DBoW3::Vocabulary> vocab;   // 词典（加载后只读）
    std::unique_ptr<DBoW3::Database>   db;      // 数据库（随关键帧增长）
    // 关键帧 id ↔ 数据库条目 id 双向映射（查询结果反查关键帧用）
    std::unordered_map<unsigned long, DBoW3::EntryId> kf_db_id;
    std::unordered_map<DBoW3::EntryId, unsigned long> db_id_kf;
    std::unordered_map<unsigned long, DBoW3::BowVector> kf_bow; // 词袋缓存
#endif

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
#ifdef HAS_DBOW3
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
#endif
    }
};

namespace {

CompactSignature makeCompactSignature(
    const Frame& frame, const Camera& camera) {
    CompactSignature signature;
    if (frame.descriptors.empty() || frame.descriptors.type() != CV_8U ||
        frame.descriptors.cols < 32) return signature;

    std::array<std::uint16_t, kCompactSignatureBytes> ones{};
    const double width = camera && camera->img_width > 0
        ? static_cast<double>(camera->img_width) : 1.0;
    const double height = camera && camera->img_height > 0
        ? static_cast<double>(camera->img_height) : 1.0;
    for (int row = 0; row < frame.descriptors.rows; ++row) {
        size_t cell = 1 + static_cast<size_t>(row & 3);
        if (row < static_cast<int>(frame.keypoints.size()) &&
            width > 1.0 && height > 1.0) {
            const auto& point = frame.keypoints[static_cast<size_t>(row)].pt;
            const size_t x = point.x >= width * 0.5 ? 1 : 0;
            const size_t y = point.y >= height * 0.5 ? 1 : 0;
            cell = 1 + y * 2 + x;
        }
        ++signature.counts[0];
        ++signature.counts[cell];
        const auto* descriptor = frame.descriptors.ptr<unsigned char>(row);
        for (size_t byte = 0; byte < 32; ++byte) {
            const unsigned char value = descriptor[byte];
            for (size_t bit = 0; bit < 8; ++bit) {
                if ((value & (1U << bit)) == 0) continue;
                const size_t offset = byte * 8 + bit;
                ++ones[offset];
                ++ones[cell * kCompactBitsPerBlock + offset];
            }
        }
    }

    for (size_t block = 0; block < kCompactSignatureBlocks; ++block) {
        const double count = signature.counts[block];
        if (count <= 0.0) continue;
        const size_t base = block * kCompactBitsPerBlock;
        for (size_t bit = 0; bit < kCompactBitsPerBlock; ++bit) {
            const double centered = 2.0 * ones[base + bit] / count - 1.0;
            signature.values[base + bit] = static_cast<std::int8_t>(
                std::clamp(std::lround(centered * 127.0), -127L, 127L));
        }
    }
    return signature;
}

double compactSimilarity(const CompactSignature& a,
                         const CompactSignature& b) {
    double weighted_score = 0.0;
    double total_weight = 0.0;
    for (size_t block = 0; block < kCompactSignatureBlocks; ++block) {
        if (a.counts[block] == 0 || b.counts[block] == 0) continue;
        const size_t base = block * kCompactBitsPerBlock;
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        for (size_t bit = 0; bit < kCompactBitsPerBlock; ++bit) {
            const double av = a.values[base + bit];
            const double bv = b.values[base + bit];
            dot += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
        }
        if (norm_a <= 0.0 || norm_b <= 0.0) continue;
        const double cosine = std::clamp(
            dot / std::sqrt(norm_a * norm_b), -1.0, 1.0);
        const double weight = block == 0 ? 0.6 : 0.1;
        weighted_score += weight * (cosine + 1.0) * 0.5;
        total_weight += weight;
    }
    return total_weight > 0.0 ? weighted_score / total_weight : 0.0;
}

} // namespace

LoopClosure::LoopClosure() : impl_(std::make_unique<Impl>()) {}

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
    impl_->backend = Impl::Backend::DBoW3;
    impl_->vocab = std::move(vocab);
    impl_->flat_vocab.reset();
    impl_->flat_bow.clear();
    impl_->kf_db_id.clear();
    impl_->db_id_kf.clear();
    impl_->kf_cache.clear();
    impl_->kf_bow.clear();
    impl_->compact_signatures.clear();
    impl_->place_hypotheses.clear();
    impl_->insertion_serial = 0;
    impl_->query_serial = 0;
    impl_->rebuildDatabase();
    LOG_INFO("LoopClosure: vocabulary loaded (" << vocab_path << "), "
             << impl_->vocab->size() << " words");
    return true;
#endif
}

bool LoopClosure::loadFlatVocabulary(const std::string& vocab_path,
                                     size_t max_keyframes) {
#ifndef HAS_DBOW3
    (void)vocab_path;
    (void)max_keyframes;
    LOG_WARN("LoopClosure: flat DBoW3 parser requires DBoW3/QuickLZ support");
    return false;
#else
    auto vocabulary = std::make_unique<FlatVocabulary>();
    try {
        vocabulary->load(vocab_path);
    } catch (const std::exception& error) {
        LOG_ERROR("LoopClosure: failed to load flat vocabulary ("
                  << error.what() << ")");
        return false;
    }
    const size_t words = vocabulary->size();
    const size_t bytes = vocabulary->memoryBytes();
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->backend = Impl::Backend::FlatDBoW3;
    impl_->compact_max_keyframes = std::max<size_t>(1, max_keyframes);
    impl_->insertion_serial = 0;
    impl_->flat_vocab = std::move(vocabulary);
    impl_->flat_bow.clear();
    impl_->kf_cache.clear();
    impl_->compact_signatures.clear();
    impl_->place_hypotheses.clear();
    impl_->query_serial = 0;
    impl_->vocab.reset();
    impl_->db.reset();
    impl_->kf_db_id.clear();
    impl_->db_id_kf.clear();
    impl_->kf_bow.clear();
    LOG_INFO("LoopClosure: flat DBoW3 loaded (" << words << " words, "
             << bytes / (1024.0 * 1024.0) << " MiB nodes, max_kf="
             << impl_->compact_max_keyframes << ")");
    return true;
#endif
}

bool LoopClosure::enableCompactRetrieval(size_t max_keyframes) {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->backend = Impl::Backend::CompactBinary;
    impl_->compact_max_keyframes = std::max<size_t>(1, max_keyframes);
    impl_->insertion_serial = 0;
    impl_->kf_cache.clear();
    impl_->compact_signatures.clear();
    impl_->place_hypotheses.clear();
    impl_->query_serial = 0;
#ifdef HAS_DBOW3
    impl_->flat_vocab.reset();
    impl_->flat_bow.clear();
    impl_->vocab.reset();
    impl_->db.reset();
    impl_->kf_db_id.clear();
    impl_->db_id_kf.clear();
    impl_->kf_bow.clear();
#endif
    LOG_INFO("LoopClosure: compact binary retrieval enabled (max_kf="
             << impl_->compact_max_keyframes << ", signature="
             << kCompactSignatureBytes << "B/KF)");
    return true;
}

void LoopClosure::removeKeyFrames(
    const std::vector<KeyframeId>& keyframe_ids) {
    if (keyframe_ids.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_set<KeyframeId> removed;
    removed.reserve(keyframe_ids.size());
    bool changed = false;
    for (const auto id : keyframe_ids) {
        removed.insert(id);
        changed = impl_->kf_cache.erase(id) > 0 || changed;
        changed = impl_->compact_signatures.erase(id) > 0 || changed;
#ifdef HAS_DBOW3
        changed = impl_->flat_bow.erase(id) > 0 || changed;
        changed = impl_->kf_bow.erase(id) > 0 || changed;
#endif
    }
    if (!changed) return;
    std::erase_if(impl_->place_hypotheses, [&](const auto& hypothesis) {
        return removed.contains(hypothesis.anchor_id) ||
               (hypothesis.representative &&
                removed.contains(hypothesis.representative->id));
    });
    // DBoW3 只有 clear()，没有单条删除；一次批量回收只重建一次。紧凑
    // 后端的 map 可直接删除，无需全量重建。
    if (impl_->backend == Impl::Backend::DBoW3) impl_->rebuildDatabase();
}

size_t LoopClosure::indexedKeyFrameCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->kf_cache.size();
}

size_t LoopClosure::retrievalIndexBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (impl_->backend == Impl::Backend::CompactBinary)
        return impl_->compact_signatures.size() * sizeof(CompactSignature);
#ifdef HAS_DBOW3
    if (impl_->backend == Impl::Backend::FlatDBoW3) {
        size_t bytes = impl_->flat_vocab ? impl_->flat_vocab->memoryBytes() : 0;
        for (const auto& [id, bow] : impl_->flat_bow) {
            (void)id;
            bytes += bow.capacity() * sizeof(FlatBow::value_type);
        }
        return bytes;
    }
    size_t bytes = 0;
    for (const auto& [id, bow] : impl_->kf_bow) {
        (void)id;
        bytes += bow.size() * (sizeof(DBoW3::WordId) + sizeof(DBoW3::WordValue));
    }
    return bytes;
#else
    return 0;
#endif
}

void LoopClosure::addKeyFrame(Frame::Ptr kf, SubmapId submap_id,
                              const SE3& T_ws) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (impl_->backend == Impl::Backend::None || !kf ||
        kf->descriptors.empty()) return;
    const Vec3 camera_position_local = kf->pose_cs.inverse().t;
    // 异步任务重排时同一 KF 可能重复提交；数据库和保护集合保持幂等。
    if (impl_->kf_cache.contains(kf->id)) return;
    auto trim_bounded_index = [&]() {
        if (impl_->kf_cache.size() <= impl_->compact_max_keyframes) return;
        std::vector<std::pair<std::uint64_t, KeyframeId>> timeline;
        timeline.reserve(impl_->kf_cache.size());
        for (const auto& [id, cached] : impl_->kf_cache)
            timeline.emplace_back(cached.insertion_serial, id);
        std::ranges::sort(timeline);
        // 首地点固定，其余槽使用确定性的 reservoir-style 全程采样，让历史
        // 各阶段都有保留机会。FIFO 会必然删除首圈地点，
        // “最密相邻对”在小容量下又会偏删早期连续地点。
        std::uint64_t random = impl_->insertion_serial + 0x9e3779b97f4a7c15ULL;
        random = (random ^ (random >> 30)) * 0xbf58476d1ce4e5b9ULL;
        random = (random ^ (random >> 27)) * 0x94d049bb133111ebULL;
        random ^= random >> 31;
        const size_t reservoir_slots = impl_->compact_max_keyframes - 1;
        const size_t selected = static_cast<size_t>(
            random % std::max<std::uint64_t>(1, impl_->insertion_serial));
        size_t victim_index = timeline.size() - 1;  // 默认拒绝新条目
        if (selected < reservoir_slots)
            victim_index = 1 + selected;  // timeline[0] 首锚永不删除
        const KeyframeId removed_id = timeline[victim_index].second;
        impl_->kf_cache.erase(removed_id);
        impl_->compact_signatures.erase(removed_id);
#ifdef HAS_DBOW3
        impl_->flat_bow.erase(removed_id);
#endif
        std::erase_if(impl_->place_hypotheses, [&](const auto& h) {
            return h.anchor_id == removed_id ||
                   (h.representative && h.representative->id == removed_id);
        });
    };
    if (impl_->backend == Impl::Backend::CompactBinary) {
        impl_->compact_signatures[kf->id] = makeCompactSignature(*kf, camera_);
        impl_->kf_cache[kf->id] = {
            kf, submap_id, T_ws, camera_position_local, ++impl_->insertion_serial};
        trim_bounded_index();
        return;
    }
#ifdef HAS_DBOW3
    if (impl_->backend == Impl::Backend::FlatDBoW3) {
        if (!impl_->flat_vocab) return;
        impl_->flat_bow[kf->id] = impl_->flat_vocab->transform(kf->descriptors);
        impl_->kf_cache[kf->id] = {
            kf, submap_id, T_ws, camera_position_local, ++impl_->insertion_serial};
        trim_bounded_index();
        return;
    }
    if (!impl_->vocab || !impl_->db) return;
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
    impl_->kf_cache[kf->id] = {
        kf, submap_id, T_ws, camera_position_local, ++impl_->insertion_serial};
#endif
}

std::vector<LoopClosure::LoopCandidate> LoopClosure::detectLoop(
    Frame::Ptr kf, SubmapId query_submap_id, const SE3& query_T_ws,
    const SubmapPoses& latest_submap_poses) {
    std::lock_guard<std::mutex> lock(mutex_);
    PERF_SCOPE("lc.detect");
    std::vector<LoopCandidate> candidates;
    // 计数所有检测调用，使中间没有视觉命中的查询也能形成真实断档。
    ++impl_->query_serial;
    if (impl_->backend == Impl::Backend::None || !kf ||
        kf->descriptors.empty()) return candidates;

    struct RetrievalResult {
        unsigned long id = 0;
        double score = 0.0;
    };
    auto retrieval_eligible = [&](unsigned long id) {
        const auto cached = impl_->kf_cache.find(id);
        if (cached == impl_->kf_cache.end()) return false;
        if (cached->second.submap_id != query_submap_id) return true;
        return id < kf->id &&
               kf->id - id >= static_cast<unsigned long>(temporal_window_);
    };
    std::vector<RetrievalResult> results;
    if (impl_->backend == Impl::Backend::CompactBinary) {
        const auto query = makeCompactSignature(*kf, camera_);
        results.reserve(impl_->compact_signatures.size());
        for (const auto& [id, signature] : impl_->compact_signatures) {
            if (retrieval_eligible(id))
                results.push_back({id, compactSimilarity(query, signature)});
        }
        std::ranges::sort(results, [](const auto& a, const auto& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.id < b.id;
        });
        if (results.size() > static_cast<size_t>(top_candidates_))
            results.resize(static_cast<size_t>(top_candidates_));
    } else if (impl_->backend == Impl::Backend::FlatDBoW3) {
#ifdef HAS_DBOW3
        if (!impl_->flat_vocab) return candidates;
        const FlatBow query = impl_->flat_vocab->transform(kf->descriptors);
        results.reserve(impl_->flat_bow.size());
        for (const auto& [id, bow] : impl_->flat_bow) {
            if (retrieval_eligible(id))
                results.push_back({id, FlatVocabulary::score(query, bow)});
        }
        std::ranges::sort(results, [](const auto& a, const auto& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.id < b.id;
        });
        if (results.size() > static_cast<size_t>(top_candidates_))
            results.resize(static_cast<size_t>(top_candidates_));
#else
        return candidates;
#endif
    } else {
#ifdef HAS_DBOW3
        if (!impl_->vocab || !impl_->db) return candidates;
        DBoW3::BowVector bow;
        impl_->vocab->transform(kf->descriptors, bow);
        DBoW3::QueryResults dbow_results;
        const int retrieval_pool = std::max(
            top_candidates_ * 4, top_candidates_ + temporal_window_);
        impl_->db->query(bow, dbow_results, retrieval_pool);
        results.reserve(dbow_results.size());
        for (const auto& result : dbow_results) {
            const auto id = impl_->db_id_kf.find(result.Id);
            if (id != impl_->db_id_kf.end())
                results.push_back({id->second, result.Score});
        }
#else
        return candidates;
#endif
    }

    // 2. 时间过滤 + 3. 分数过滤：保留整个 Top-N，后面按历史邻域聚成地点。
    //    Top-N 是 DBoW 查询的内存/计算上限，不是几何验证前的 Top-3 截断。
    std::vector<LoopCandidate> scored;
    for (const auto& r : results) {
        const unsigned long cand_id = r.id;
        // 两种后端都输出归一化 0~1 相似度；只负责召回，几何门不变。
        if (r.score < min_score_) continue;
        auto kf_it = impl_->kf_cache.find(cand_id);
        if (kf_it == impl_->kf_cache.end()) continue;
        const auto& cached = kf_it->second;
        // Frame id 只在同一子地图内可作为时间顺序。跨子地图的全局 id
        // 可能连续，不能因刚创建子地图而误杀真正的跨图回环。
        if (cached.submap_id == query_submap_id &&
            (cand_id >= kf->id ||
             kf->id - cand_id < (unsigned long)temporal_window_)) continue;
        scored.push_back({cached.frame, cached.submap_id, r.score});
        if (scored.size() >= static_cast<size_t>(top_candidates_)) break;
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
}

bool LoopClosure::verifyLoop(Frame::Ptr kf_curr, Frame::Ptr kf_loop,
                             SE3& T_loop_curr) {
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
}

} // namespace vslam
