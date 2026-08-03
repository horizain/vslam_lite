#include "vslam/loop_closure.h"
#include "vslam/mappoint.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

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
// PIMPL：DBoW3 具体状态（词袋数据库只在关键帧增长时追加，量级与关键帧数线性）
// ============================================================
class LoopClosure::Impl {
public:
    std::unique_ptr<DBoW3::Vocabulary> vocab;   // 词典（加载后只读）
    std::unique_ptr<DBoW3::Database>   db;      // 数据库（随关键帧增长）
    // 关键帧 id ↔ 数据库条目 id 双向映射（查询结果反查关键帧用）
    std::unordered_map<unsigned long, DBoW3::EntryId> kf_db_id;
    std::unordered_map<DBoW3::EntryId, unsigned long> db_id_kf;
    std::unordered_map<unsigned long, Frame::Ptr>     kf_cache;  // 候选帧回查
    std::unordered_map<unsigned long, DBoW3::BowVector> kf_bow; // 词袋缓存
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
                            double ransac_pixel_threshold, const Camera& camera) {
    min_score_             = min_score;
    temporal_window_       = temporal_window;
    min_loop_inliers_      = min_loop_inliers;
    pnp_inlier_ratio_      = pnp_inlier_ratio;
    ransac_pixel_threshold_ = ransac_pixel_threshold;
    camera_                = camera;
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
    impl_->vocab = std::move(vocab);
    impl_->db = std::make_unique<DBoW3::Database>(*impl_->vocab, false, 0);
    LOG_INFO("LoopClosure: vocabulary loaded (" << vocab_path << "), "
             << impl_->vocab->size() << " words");
    return true;
#endif
}

void LoopClosure::addKeyFrame(Frame::Ptr kf) {
#ifndef HAS_DBOW3
    (void)kf;
#else
    if (!impl_->vocab || !impl_->db || !kf || kf->descriptors.empty()) return;
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
    impl_->kf_cache[kf->id] = kf;
#endif
}

Frame::Ptr LoopClosure::detectLoop(Frame::Ptr kf) {
#ifndef HAS_DBOW3
    (void)kf;
    return nullptr;
#else
    if (!impl_->vocab || !impl_->db || !kf || kf->descriptors.empty()) return nullptr;

    // 1. 词袋查询 Top-5
    DBoW3::BowVector bow;
    impl_->vocab->transform(kf->descriptors, bow);
    DBoW3::QueryResults results;
    impl_->db->query(bow, results, 5);

    // 2. 时间过滤 + 3. 分数过滤：取通过过滤的最高分候选
    Frame::Ptr best;
    double best_score = min_score_;
    for (const auto& r : results) {
        auto it = impl_->db_id_kf.find(r.Id);
        if (it == impl_->db_id_kf.end()) continue;
        const unsigned long cand_id = it->second;
        // 时间窗：跳过"刚走过的路"（id 差过小不算回环）
        if (cand_id >= kf->id ||
            kf->id - cand_id < (unsigned long)temporal_window_) continue;
        // 分数阈值：DBoW3 归一化分数（0~1）
        if (r.Score < best_score) continue;
        auto kf_it = impl_->kf_cache.find(cand_id);
        if (kf_it == impl_->kf_cache.end()) continue;
        best = kf_it->second;
        best_score = r.Score;
    }
    if (best) {
        LOG_INFO("LoopClosure: candidate kf#" << best->id
                 << " (score=" << best_score << ")");
    }
    return best;
#endif
}

bool LoopClosure::verifyLoop(Frame::Ptr kf_curr, Frame::Ptr kf_loop,
                             SE3& T_loop_curr) {
#ifndef HAS_DBOW3
    (void)kf_curr; (void)kf_loop; (void)T_loop_curr;
    return false;
#else
    if (!camera_ || !kf_curr || !kf_loop) return false;

    // 1. ORB 匹配：knn + ratio（queryIdx 属于 kf_curr，trainIdx 属于 kf_loop）
    auto matches = matcher_.match(kf_curr, kf_loop, 0.7, false);
    if (matches.size() < 8) {
        LOG_WARN("LoopClosure: too few matches (" << matches.size() << ")");
        return false;
    }

    // 2. 收集 3D-2D 对应：3D = kf_loop 侧已关联的地图点（回环尺度），
    //    2D = kf_curr 侧特征点像素
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    for (size_t i = 0; i < matches.size(); i++) {
        auto mp = kf_loop->map_points[matches[i].trainIdx];
        if (!mp) continue;
        const auto& kp = kf_curr->keypoints[matches[i].queryIdx];
        pts3d.emplace_back(mp->pos_w.x(), mp->pos_w.y(), mp->pos_w.z());
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
                                 false, 100, ransac_pixel_threshold_, 0.99,
                                 inliers, cv::SOLVEPNP_ITERATIVE);
    if (!ok || inliers.empty()) return false;

    // 4. 内点判定：数量 + 比例双门槛
    double ratio = (double)inliers.size() / (double)pts3d.size();
    if ((int)inliers.size() < min_loop_inliers_ || ratio < pnp_inlier_ratio_) {
        LOG_WARN("LoopClosure: PnP rejected (" << inliers.size() << " inliers, "
                 << ratio << " ratio)");
        return false;
    }

    // 5. solvePnP 的 rvec/tvec 把世界点变换到相机系（T_wc 语义），
    //    matToSE3(rvec, tvec) 的矩阵正是当前帧在回环地图坐标中的 T_wc。
    //    由此构造位姿图测量 T_loop_curr = X_loop⁻¹ · X_curr（X 为 T_wc，
    //    满足 X_curr = X_loop · T_loop_curr，见 loop_closure.h）。旧实现用
    //    同一批世界点做 Umeyama 求 Sim3，尺度恒为 1 无法观测单目尺度，
    //    还引入了错误的全图 gauge 变换。
    const SE3 T_wc_curr_in_loop = matToSE3(rvec, tvec);
    T_loop_curr = kf_loop->pose_cw * T_wc_curr_in_loop;
    LOG_INFO("LoopClosure: verified! kf#" << kf_loop->id << " -> kf#" << kf_curr->id
             << " inliers=" << inliers.size());
    return true;
#endif
}

} // namespace vslam
