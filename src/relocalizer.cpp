#include "vslam/relocalizer.h"
#include "vslam/pose_gate.h"

#include <opencv2/calib3d.hpp>

#include <cmath>

namespace vslam {

Relocalizer::Relocalizer(const Camera& camera,
                         int num_features, double scale_factor,
                         int pyramid_levels, int orb_max_bands)
    : camera_(camera) {
    matcher_.setParams(num_features, scale_factor, pyramid_levels, orb_max_bands);
}

SE3 Relocalizer::matToSE3(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat rmat = R;
    if (R.rows == 3 && R.cols == 1)  // rvec input
        cv::Rodrigues(R, rmat);
    Eigen::Matrix3d Re;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Re(i, j) = rmat.at<double>(i, j);
    return SE3(Eigen::Quaterniond(Re),
               Vec3(t.at<double>(0), t.at<double>(1), t.at<double>(2)));
}

RelocalizationResult Relocalizer::verifyCandidate(
    unsigned long submap_id, const Frame::Ptr& kf,
    const Query& query) {
    RelocalizationResult result;

    // 重定位不做 F 矩阵 RANSAC：外点由下方 solvePnPRansac 自己剔除
    //（与 vo.cpp 原 try_kf 注释一致；F 矩阵在重定位场景是纯冗余开销）
    auto matches = matcher_.match(kf, query.curr_frame, query.match_ratio, false);
    if ((int)matches.size() < query.min_matches) return result;

    // 锁内供应 3D-2D 对应与身份/版本（回调返回 false = 候选失效，跳过）
    RelocalizationPointSet pts;
    if (!query.supply_points || !query.supply_points(submap_id, kf, matches, pts))
        return result;
    if (pts.pts3d.size() < (size_t)query.min_pts3d) return result;

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    const bool ok = cv::solvePnPRansac(pts.pts3d, pts.pts2d, camera_->K(), cv::Mat(),
                                       rvec, tvec, false, 200,
                                       query.ransac_pixel_threshold, 0.99, inliers);
    const double ratio = pts.pts3d.empty() ? 0.0
        : (double)inliers.size() / pts.pts3d.size();
    const double rmse = ok
        ? PoseGate::pnpReprojectionRmse(pts.pts3d, pts.pts2d, rvec, tvec, inliers, camera_->K())
        : std::numeric_limits<double>::infinity();

    if (!ok || (int)inliers.size() < query.min_inliers
        || ratio < query.min_ratio || rmse > query.max_rmse) {
        return result;
    }
    result.accepted = true;
    result.inliers = (int)inliers.size();
    result.total = pts.pts3d.size();
    result.T_cs = matToSE3(cv::Mat(rvec), tvec);
    result.kf = kf;
    result.map = pts.map;
    result.geometry_revision = pts.geometry_revision;
    result.submap_id = submap_id;
    result.rmse = rmse;
    return result;
}

RelocalizationResult Relocalizer::relocalize(const Query& query) {
    RelocalizationResult result;

    // 最近关键帧优先（候选已由调用方排序）；粗筛先行：256 描述子子集 BF
    // 匹配数不足的直接跳过（~0.5ms/候选），只对相似候选做全量 BF + PnP。
    for (const auto& [submap_id, kf] : query.candidates) {
        result.quick_checked++;
        if (matcher_.quickMatchCount(kf->descriptors, query.curr_frame->descriptors,
                                     query.quick_subset, query.quick_dist_thresh)
            < query.quick_threshold) {
            continue;
        }
        result.quick_passed++;
        const RelocalizationResult cand = verifyCandidate(submap_id, kf, query);
        if (cand.accepted && cand.inliers > result.inliers) {
            result.inliers = cand.inliers;
            result.total = cand.total;
            result.T_cs = cand.T_cs;
            result.kf = cand.kf;
            result.map = cand.map;
            result.geometry_revision = cand.geometry_revision;
            result.submap_id = cand.submap_id;
            result.rmse = cand.rmse;
        }
        if (result.inliers >= query.min_inliers) break;
    }
    result.accepted = result.inliers >= query.min_inliers;
    return result;
}

} // namespace vslam
