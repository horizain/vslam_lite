#include "vslam/frontend_tracker.h"
#include "vslam/pose_gate.h"
#include "vslam/relocalizer.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>

namespace vslam {

FrontendTracker::FrontendTracker(const Camera& camera, const TrackerConfig& config)
    : camera_(camera), cfg_(config) {
    matcher_.setParams(cfg_.num_features, cfg_.scale_factor,
                       cfg_.pyramid_levels, cfg_.orb_max_bands);
}

StereoStats FrontendTracker::computeStereoDepths(const Frame::Ptr& frame) const {
    StereoStats stats;
    frame->pts_c.clear();
    frame->pts_c.resize(frame->keypoints.size(), Vec3::Zero());
    // 单目：无单帧深度，pts_c 全部无效（走多帧三角化）
    if (!camera_->hasPerFrameDepth() || frame->image_right.empty()) return stats;

    // 双目匹配用原始灰度（非 CLAHE）：CLAHE 是内容相关的非线性增强，
    // 左右目同一 3D 点的局部直方图不同 → 灰度不一致 → 破坏光度一致性，
    // 显著降低 LK 左右目匹配质量。
    cv::Mat left_raw, right_raw;
    if (frame->image.channels() == 3)
        cv::cvtColor(frame->image, left_raw, cv::COLOR_BGR2GRAY);
    else
        left_raw = frame->image;
    if (frame->image_right.channels() == 3)
        cv::cvtColor(frame->image_right, right_raw, cv::COLOR_BGR2GRAY);
    else
        right_raw = frame->image_right;

    std::vector<cv::Point2f> right_pts;
    auto status = matcher_.matchStereo(left_raw, right_raw, frame->keypoints, right_pts);

    std::vector<double> disparities;
    std::vector<double> depths;
    disparities.reserve(status.size());
    depths.reserve(status.size());
    for (size_t i = 0; i < status.size() && i < frame->keypoints.size(); i++) {
        if (!status[i]) continue;
        double disparity = frame->keypoints[i].pt.x - right_pts[i].x;
        const double min_disparity = camera_->fx * camera_->baseline() / cfg_.stereo_max_depth;
        const double max_disparity = camera_->fx * camera_->baseline() / cfg_.stereo_min_depth;
        if (disparity < min_disparity || disparity > max_disparity) continue;
        double depth = camera_->fx * camera_->baseline() / disparity;  // z = fx*b/d
        if (depth < cfg_.stereo_min_depth || depth > cfg_.stereo_max_depth) continue;
        frame->pts_c[i] = camera_->pixel2camera(
            Vec2(frame->keypoints[i].pt.x, frame->keypoints[i].pt.y), depth);
        disparities.push_back(disparity);
        depths.push_back(depth);
    }

    stats.stereo_points = (int)depths.size();
    if (!depths.empty()) {
        std::ranges::sort(disparities);
        std::ranges::sort(depths);
        stats.median_disparity = disparities[disparities.size() / 2];
        stats.median_depth = depths[depths.size() / 2];
    }
    return stats;
}

TrackingResult FrontendTracker::trackPnP(
    const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
    const SE3& T_ws, const MotionBaseline& motion,
    int min_inliers, double min_ratio, double max_rmse) const {
    TrackingResult r;
    if (pts3d.size() < 6) return r;

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    bool ok = cv::solvePnPRansac(pts3d, pts2d, camera_->K(), cv::Mat(), rvec, tvec,
                                 false, 200, cfg_.ransac_pixel_threshold, 0.99, inliers);
    if (!ok) return r;

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    const SE3 candidate_pose = Relocalizer::matToSE3(R, tvec);
    const double rmse = PoseGate::pnpReprojectionRmse(pts3d, pts2d, rvec, tvec, inliers, camera_->K());

    // M0：统一位姿验收（几何 + 连续性；正常跟踪基线由调用方传入）
    PoseQuality quality;
    const bool accepted = PoseGate::acceptPoseCandidate(
        candidate_pose * T_ws.inverse(), (int)inliers.size(), pts3d.size(), rmse,
        min_inliers, min_ratio, max_rmse,
        motion.baseline_twc, motion.max_translation, motion.max_rotation, quality);

    // 无论通过与否都携带质量信息（原 trackFrame 先写 status_ 再判断）
    r.inlier_ratio = quality.inlier_ratio;
    r.pose_rmse = quality.pose_rmse;
    r.translation_delta = quality.translation;
    r.rotation_delta = quality.rotation;
    r.inliers = (int)inliers.size();   // 原始内点数（供拒绝日志/验收）
    if (accepted) {
        r.pose_cs = candidate_pose;
        r.pnp_inlier_indices = inliers;
        r.valid = true;
    }
    return r;
}

RigidResult FrontendTracker::estimateRigid3D3D(
    const std::vector<cv::Point3f>& pts_w, const std::vector<cv::Point3f>& pts_c,
    int min_inliers, double min_ratio) const {
    RigidResult r;
    r.total = pts_w.size();
    if ((int)pts_w.size() < std::max(20, min_inliers)) return r;

    cv::Mat affine, inliers;
    // RANSAC 3D-3D：返回 3x4 [R|t] 满足 dst = R*src + t → 即 T_cw（世界→相机）
    bool ok = cv::estimateAffine3D(pts_w, pts_c, affine, inliers,
                                   cfg_.rigid_ransac_threshold, 0.99);
    if (!ok) return r;

    r.inlier_mask.resize(pts_w.size(), 0);
    for (size_t i = 0; i < pts_w.size() && i < inliers.total(); i++)
        r.inlier_mask[i] = inliers.at<uchar>((int)i);

    // estimateAffine3D 只用于 RANSAC 选内点。不能把其旋转投影回 SO(3) 后仍沿用
    // 原仿射平移：旋转、缩放和剪切被改变后，原 t 已不属于同一个变换。
    // 在内点上重新做 Kabsch 刚体拟合，统一求解 R、t（dst = R * src + t）。
    Vec3 mean_w = Vec3::Zero();
    Vec3 mean_c = Vec3::Zero();
    int rigid_inliers = 0;
    for (size_t i = 0; i < r.inlier_mask.size(); i++) {
        if (!r.inlier_mask[i]) continue;
        mean_w += Vec3(pts_w[i].x, pts_w[i].y, pts_w[i].z);
        mean_c += Vec3(pts_c[i].x, pts_c[i].y, pts_c[i].z);
        rigid_inliers++;
    }
    r.inliers = rigid_inliers;
    r.ratio = (double)rigid_inliers / pts_w.size();
    if (rigid_inliers < min_inliers || r.ratio < min_ratio) {
        LOG_WARN("3D-3D rejected: inliers=" << rigid_inliers << " ratio=" << r.ratio);
        return r;
    }
    mean_w /= rigid_inliers;
    mean_c /= rigid_inliers;

    Mat33 covariance = Mat33::Zero();
    for (size_t i = 0; i < r.inlier_mask.size(); i++) {
        if (!r.inlier_mask[i]) continue;
        Vec3 pw(pts_w[i].x, pts_w[i].y, pts_w[i].z);
        Vec3 pc(pts_c[i].x, pts_c[i].y, pts_c[i].z);
        covariance += (pc - mean_c) * (pw - mean_w).transpose();
    }
    Eigen::JacobiSVD<Mat33> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Vec3 singular = svd.singularValues();
    if (singular.x() <= 1e-9 || singular.y() / singular.x() < 1e-3) {
        LOG_WARN("3D-3D rejected: degenerate point distribution");
        return r;
    }
    Mat33 U = svd.matrixU();
    const Mat33 V = svd.matrixV();
    Mat33 R_rigid = U * V.transpose();
    if (R_rigid.determinant() < 0) {
        U.col(2) *= -1;
        R_rigid = U * V.transpose();
    }
    Vec3 t_rigid = mean_c - R_rigid * mean_w;
    r.pose_cs = SE3(Eigen::Quaterniond(R_rigid), t_rigid);

    double squared_error = 0.0;
    for (size_t i = 0; i < r.inlier_mask.size(); i++) {
        if (!r.inlier_mask[i]) continue;
        const Vec3 pw(pts_w[i].x, pts_w[i].y, pts_w[i].z);
        const Vec3 pc(pts_c[i].x, pts_c[i].y, pts_c[i].z);
        squared_error += (R_rigid * pw + t_rigid - pc).squaredNorm();
    }
    r.rmse = std::sqrt(squared_error / rigid_inliers);
    r.ok = true;
    return r;
}

TrackingResult FrontendTracker::trackOrb(
    const Frame::Ptr& curr_frame, const Frame::Ptr& ref_frame,
    const RefView& ref, const MotionBaseline& motion,
    const StereoStats& stereo) const {
    TrackingResult r;

    // 跟踪匹配不做基础矩阵 RANSAC（省时，且避免共面场景 F 矩阵退化误剔）：
    // 外点交给下方 solvePnPRansac 自己剔除；仅初始化/回退分支保留 F 矩阵 RANSAC
    auto matches = matcher_.match(ref_frame, curr_frame, cfg_.match_ratio, false);

    // 收集 3D-2D 对应（保留 pts3d[i] 与 matches 的映射，供内点观测计数）
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<int> match_idx;
    if (ref.has_ref) {
        for (auto [k, m] : matches | std::views::enumerate) {
            if (m.queryIdx >= (int)ref.ref_points_s.size()) continue;
            if (!ref.ref_mps[m.queryIdx]) continue;
            const Vec3& p = ref.ref_points_s[m.queryIdx];
            pts3d.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
            pts2d.push_back(curr_frame->keypoints[m.trainIdx].pt);
            match_idx.push_back((int)k);
        }
    }
    r.matches = (int)matches.size();

    // PnP (3D-2D) —— solvePnPRansac 返回的 rvec/tvec 即 T_cw（世界→相机）
    if (pts3d.size() >= 6) {
        const TrackingResult pnp = trackPnP(pts3d, pts2d, ref.T_ws, motion,
                                            cfg_.pnp_min_inliers,
                                            cfg_.pnp_min_inlier_ratio,
                                            cfg_.pnp_max_rmse);
        r.inlier_ratio = pnp.inlier_ratio;
        r.pose_rmse = pnp.pose_rmse;
        r.translation_delta = pnp.translation_delta;
        r.rotation_delta = pnp.rotation_delta;
        if (pnp.valid) {
            r.pose_cs = pnp.pose_cs;
            r.inliers = pnp.inliers;
            r.valid = true;
            r.method = "PNP";
            // 位姿通过全部质量检查后才关联地图点，避免被拒绝的解污染共视统计。
            for (int idx : pnp.pnp_inlier_indices) {
                if (idx < 0 || idx >= (int)match_idx.size()) continue;
                auto& mp = ref.ref_mps[matches[match_idx[idx]].queryIdx];
                if (mp) r.associations.emplace_back(matches[match_idx[idx]].trainIdx, mp);
            }
        } else {
            r.inliers = 0;
            LOG_WARN("PnP rejected: inliers=" << pnp.inliers
                     << " ratio=" << pnp.inlier_ratio << " rmse=" << pnp.pose_rmse
                     << " dtrans=" << pnp.translation_delta
                     << " drot=" << pnp.rotation_delta);
        }
    }

    // 双目/RGB-D：PnP 失败后走 3D-3D 位姿估计（绝对尺度、旋转鲁棒）。
    if (!r.valid && camera_->hasPerFrameDepth() && (int)matches.size() >= 20
        && stereo.stereo_points >= cfg_.stereo_min_points) {
        std::vector<cv::Point3f> pts_w;   // ref 帧世界系 3D 点（快照坐标）
        std::vector<cv::Point3f> pts_c;   // 当前帧相机系 3D 点（双目视差）
        std::vector<int> idx3;
        if (ref.has_ref) {
            for (auto [k, m] : matches | std::views::enumerate) {
                if (m.queryIdx >= (int)ref.ref_points_s.size()) continue;
                if (ref.ref_mps[m.queryIdx] && curr_frame->pts_c[m.trainIdx].z() > 0) {
                    const Vec3& p = ref.ref_points_s[m.queryIdx];
                    pts_w.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
                    pts_c.emplace_back((float)curr_frame->pts_c[m.trainIdx].x(),
                                       (float)curr_frame->pts_c[m.trainIdx].y(),
                                       (float)curr_frame->pts_c[m.trainIdx].z());
                    idx3.push_back((int)k);
                }
            }
        }
        if ((int)pts_w.size() >= std::max(20, cfg_.rigid_min_inliers)) {
            const RigidResult rigid = estimateRigid3D3D(
                pts_w, pts_c, cfg_.rigid_min_inliers, cfg_.rigid_min_inlier_ratio);
            if (rigid.ok) {
                // M0：统一位姿验收（3D-3D 几何门限 + 正常跟踪连续性门限）
                PoseQuality quality;
                const bool accepted = PoseGate::acceptPoseCandidate(
                    rigid.pose_cs * ref.T_ws.inverse(), rigid.inliers, rigid.total,
                    rigid.rmse, cfg_.rigid_min_inliers, cfg_.rigid_min_inlier_ratio,
                    cfg_.rigid_max_rmse, motion.baseline_twc,
                    motion.max_translation, motion.max_rotation, quality);
                if (!accepted) {
                    LOG_WARN("3D-3D rejected: inliers=" << rigid.inliers
                             << " ratio=" << quality.inlier_ratio << " rmse=" << rigid.rmse
                             << " dtrans=" << quality.translation
                             << " drot=" << quality.rotation);
                } else {
                    r.pose_cs = rigid.pose_cs;
                    r.inliers = rigid.inliers;
                    r.valid = true;
                    r.method = "3D3D";
                    r.inlier_ratio = quality.inlier_ratio;
                    r.pose_rmse = quality.pose_rmse;
                    r.translation_delta = quality.translation;
                    r.rotation_delta = quality.rotation;
                    // 关联内点：普通跟踪帧只保留临时 map_points 指针，不产生正式观测。
                    for (size_t i = 0; i < rigid.inlier_mask.size() && i < idx3.size(); i++) {
                        if (!rigid.inlier_mask[i]) continue;
                        const int k = idx3[i];
                        auto& mp = ref.ref_mps[matches[k].queryIdx];
                        if (mp) r.associations.emplace_back(matches[k].trainIdx, mp);
                    }
                }
            }
        }
    }

    if (!r.valid) {
        // 对极几何回退（仅单目：recoverPose 的 t 归一化，双目有绝对尺度不可用）
        if (!camera_->hasPerFrameDepth() && (int)matches.size() >= cfg_.min_matches_track) {
            const SE3 ref_pose_cs = ref.ref_pose_cs;
            std::vector<cv::Point2f> pts1, pts2;
            FeatureMatcher::getMatchedPoints(ref_frame, curr_frame, matches, pts1, pts2);
            cv::Mat E = cv::findEssentialMat(pts1, pts2, camera_->K(), cv::RANSAC, 0.999, 1.0);
            cv::Mat R, t;
            cv::recoverPose(E, pts1, pts2, camera_->K(), R, t);
            SE3 T_rel = Relocalizer::matToSE3(R, t);
            const SE3 pose = T_rel * ref_pose_cs;
            // 对极恢复的 t 只有方向无尺度，组合后可能跳变 → 同样做跳变保护
            const SE3 Twc_new = pose.inverse();
            const SE3 Twc_ref = ref_pose_cs.inverse();
            if ((Twc_new.t - Twc_ref.t).norm() > 30.0) {
                LOG_WARN("Epipolar fallback pose jump (" << (Twc_new.t - Twc_ref.t).norm()
                         << "m), tracking lost");
                r.pose_cs = ref_pose_cs;
                r.recovering = true;
            } else {
                r.pose_cs = pose;
                r.inliers = (int)matches.size();
                r.valid = true;
                r.method = "EPIPOLAR";
            }
        } else {
            // 匹配太少 → LOST
            r.pose_cs = ref.ref_pose_cs;
            r.recovering = true;
            LOG_WARN("Tracking lost! matches=" << matches.size()
                     << " pts3d=" << pts3d.size()
                     << " kf_ref=" << (ref_frame ? (long long)ref_frame->id : -1)
                     << " mp_ref=" << (ref_frame ? ref_frame->map_points.size() : 0));
        }
    }
    return r;
}

KeyframeProposal FrontendTracker::proposeKeyFrame(const KeyframeInput& input) const {
    KeyframeProposal p;
    if (!input.curr_frame) return p;
    SE3 Twc_cur = input.curr_frame->pose_cs.inverse();
    SE3 Twc_ref = input.ref_pose_cs.inverse();
    p.translation = (Twc_cur.t - Twc_ref.t).norm();
    // 相对旋转角：q_rel = q_cur * q_ref^-1，最小表示 = 2*acos(|w|)，处理 q 与 -q 等价
    Eigen::Quaterniond q_rel = input.curr_frame->pose_cs.q * input.ref_pose_cs.q.inverse();
    p.rotation = 2.0 * std::acos(std::clamp(std::abs(q_rel.w()), 0.0, 1.0));
    // 运动阈值 + 匹配衰减阈值：内点过少说明地图不足/视角变化大，强制补充关键帧
    p.weak_match = input.inliers < cfg_.keyframe_min_inliers;
    // 冷却：weak_match 触发需与上一关键帧间隔足够帧数，防止"关键帧风暴"
    if (p.weak_match &&
        input.curr_frame->id - input.last_kf_frame_id < (unsigned long)cfg_.min_keyframe_interval)
        p.weak_match = false;
    // 平移阈值按传感器类型分派：双目/RGB-D 有绝对尺度（真实帧间位移大），
    // 单目尺度归一化后位移小——共用阈值会导致双目每帧插 KF
    double kf_trans = camera_->hasPerFrameDepth()
        ? cfg_.keyframe_translation_stereo : cfg_.keyframe_translation;
    // 规模硬顶：关键帧数超过上限后放大平移阈值，压缩后续冗余帧
    if (cfg_.keyframe_max_count > 0 &&
        input.map_keyframe_count > (unsigned long)cfg_.keyframe_max_count)
        kf_trans *= 1.5;
    p.max_interval = input.curr_frame->id - input.last_kf_frame_id
        >= (unsigned long)cfg_.max_keyframe_interval;
    p.need = p.translation > kf_trans || p.rotation > cfg_.keyframe_rotation
        || p.weak_match || p.max_interval;
    return p;
}

} // namespace vslam
