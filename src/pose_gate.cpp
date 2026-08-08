#include "vslam/pose_gate.h"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace vslam {

double PoseGate::pnpReprojectionRmse(const std::vector<cv::Point3f>& pts3d,
                                     const std::vector<cv::Point2f>& pts2d,
                                     const cv::Mat& rvec, const cv::Mat& tvec,
                                     const std::vector<int>& inliers,
                                     const cv::Mat& K) {
    if (inliers.empty()) return std::numeric_limits<double>::infinity();
    std::vector<cv::Point2f> projected;
    cv::projectPoints(pts3d, rvec, tvec, K, cv::Mat(), projected);
    double squared_error = 0.0;
    int count = 0;
    for (int idx : inliers) {
        if (idx < 0 || idx >= (int)projected.size() || idx >= (int)pts2d.size()) continue;
        const cv::Point2f error = projected[idx] - pts2d[idx];
        squared_error += error.dot(error);
        count++;
    }
    return count > 0 ? std::sqrt(squared_error / count)
                     : std::numeric_limits<double>::infinity();
}

bool PoseGate::checkMotionContinuity(const SE3& candidate_pose_cs,
                                     const SE3& baseline_twc,
                                     double max_translation, double max_rotation,
                                     double& translation, double& rotation) {
    const SE3 Twc_cand = candidate_pose_cs.inverse();
    translation = (Twc_cand.t - baseline_twc.t).norm();
    const Eigen::Quaterniond q_rel = Twc_cand.q * baseline_twc.q.inverse();
    rotation = 2.0 * std::acos(std::clamp(std::abs(q_rel.w()), 0.0, 1.0));
    return translation <= max_translation && rotation <= max_rotation;
}

bool PoseGate::acceptPoseCandidate(const SE3& candidate_pose_cs,
                                   int inliers, size_t total, double rmse,
                                   int min_inliers, double min_ratio, double max_rmse,
                                   const std::optional<SE3>& baseline_twc,
                                   double max_translation, double max_rotation,
                                   PoseQuality& quality) {
    quality.inlier_ratio = total > 0
        ? static_cast<double>(inliers) / static_cast<double>(total) : 0.0;
    quality.pose_rmse = rmse;
    quality.translation = 0.0;
    quality.rotation = 0.0;
    quality.geometric_ok = inliers >= min_inliers
        && quality.inlier_ratio >= min_ratio && rmse <= max_rmse;
    if (!quality.geometric_ok) {
        quality.motion_ok = false;
        return false;
    }
    if (!baseline_twc) {  // 无运动基线 → 几何验收即最终验收
        quality.motion_ok = true;
        return true;
    }
    quality.motion_ok = checkMotionContinuity(
        candidate_pose_cs, *baseline_twc, max_translation, max_rotation,
        quality.translation, quality.rotation);
    return quality.motion_ok;
}

PoseDecision PoseGate::evaluate(const PoseCandidate& candidate,
                                const MotionPrediction& prediction,
                                const TrackingQuality& quality,
                                double dt) const {
    (void)dt;  // M3 §7.4 时间归一化连续性使用；M1.1 不参与
    PoseDecision decision;
    decision.accepted = acceptPoseCandidate(
        candidate.pose_cs, quality.inliers, quality.total, quality.rmse,
        candidate.min_inliers, candidate.min_ratio, candidate.max_rmse,
        prediction.baseline_twc, prediction.max_translation, prediction.max_rotation,
        decision.quality);
    return decision;
}

} // namespace vslam
