#include "vslam/frontend_tracker.h"
#include "vslam/pose_gate.h"
#include "vslam/relocalizer.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

namespace vslam {

FrontendTracker::FrontendTracker(const Camera& camera, const TrackerConfig& config)
    : camera_(camera), cfg_(config) {
    matcher_.setParams(cfg_.num_features, cfg_.scale_factor,
                       cfg_.pyramid_levels, cfg_.orb_max_bands,
                       cfg_.stereo_reverse_prune);
}

QualityVerdict FrontendTracker::assessFrameQuality(
    const cv::Mat& raw_gray,
    const std::vector<cv::KeyPoint>& keypoints,
    int image_cols, int image_rows) const {
    // M3.1（§7.2）：图像统计 + 特征网格占用 → 三档判定。纯函数聚合，
    // 无 RNG/全局状态；开关关闭时 classifyTrackingQuality 直接 Full 旁路。
    const ImageQualityStats stats = assessImageQuality(raw_gray);
    const int occupied = countOccupiedGridCells(
        keypoints, image_cols, image_rows, cfg_.quality.grid_cols,
        cfg_.quality.grid_rows);
    return classifyTrackingQuality(stats, static_cast<int>(keypoints.size()),
                                   occupied, cfg_.quality);
}

PoseCovarianceResult FrontendTracker::estimatePnPCovariance(
    const SE3& pose_cs,
    const std::vector<cv::Point3f>& points_s,
    const std::vector<cv::Point2f>& pixels,
    const std::vector<int>& inlier_indices) const {
    // M3.2（§7.5）：中心有限差分数值协方差。纯函数聚合，无 RNG、不触碰
    // 全局状态；退化判定在 pnpPoseCovariance 内部完成。
    const cv::Mat K_cv = camera_->K();
    Mat33 K = Mat33::Identity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            K(r, c) = K_cv.at<double>(r, c);
    return pnpPoseCovariance(pose_cs, points_s, pixels, inlier_indices, K);
}

StereoStats FrontendTracker::computeStereoDepths(const Frame::Ptr& frame) const {
    return computeStereoDepths(frame, cv::Mat(), cv::Mat());
}

StereoStats FrontendTracker::computeStereoDepths(
    const Frame::Ptr& frame, const cv::Mat& left_gray_input,
    const cv::Mat& right_gray_input) const {
    StereoStats stats;
    frame->pts_c.clear();
    frame->pts_c.resize(frame->keypoints.size(), Vec3::Zero());
    // 单目：无单帧深度，pts_c 全部无效（走多帧三角化）
    if (!camera_->hasPerFrameDepth() || frame->image_right.empty()) return stats;

    // 双目匹配用原始灰度（非 CLAHE）：CLAHE 是内容相关的非线性增强，
    // 左右目同一 3D 点的局部直方图不同 → 灰度不一致 → 破坏光度一致性，
    // 显著降低 LK 左右目匹配质量。
    cv::Mat left_raw = left_gray_input;
    cv::Mat right_raw = right_gray_input;
    if (left_raw.empty()) {
        if (frame->image.channels() == 3)
            cv::cvtColor(frame->image, left_raw, cv::COLOR_BGR2GRAY);
        else
            left_raw = frame->image;
    }
    if (right_raw.empty()) {
        if (frame->image_right.channels() == 3)
            cv::cvtColor(frame->image_right, right_raw, cv::COLOR_BGR2GRAY);
        else
            right_raw = frame->image_right;
    }

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
    if (pts3d.size() < 6 || pts3d.size() != pts2d.size()) return r;

    auto solve = [&](bool use_guess, const std::optional<SE3>& guess,
                     cv::Mat& rvec, cv::Mat& tvec,
                     std::vector<int>& inliers) {
        if (use_guess && guess) {
            const Mat33 R_guess = guess->q.toRotationMatrix();
            cv::Mat R_cv(3, 3, CV_64F);
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    R_cv.at<double>(row, col) = R_guess(row, col);
            cv::Rodrigues(R_cv, rvec);
            tvec = (cv::Mat_<double>(3, 1) <<
                guess->t.x(), guess->t.y(), guess->t.z());
        }
        return cv::solvePnPRansac(
            pts3d, pts2d, camera_->K(), cv::Mat(), rvec, tvec,
            use_guess, 200, cfg_.ransac_pixel_threshold, 0.99, inliers);
    };

    auto assess = [&](const cv::Mat& rvec, const cv::Mat& tvec,
                      const std::vector<int>& inliers) {
        TrackingResult candidate;
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        const SE3 pose_cs = Relocalizer::matToSE3(R, tvec);
        const double rmse = PoseGate::pnpReprojectionRmse(
            pts3d, pts2d, rvec, tvec, inliers, camera_->K());
        PoseQuality quality;
        const bool accepted = PoseGate::acceptPoseCandidate(
            pose_cs * T_ws.inverse(), (int)inliers.size(), pts3d.size(), rmse,
            min_inliers, min_ratio, max_rmse,
            motion.baseline_twc, motion.max_translation, motion.max_rotation,
            quality);
        candidate.inlier_ratio = quality.inlier_ratio;
        candidate.pose_rmse = quality.pose_rmse;
        candidate.translation_delta = quality.translation;
        candidate.rotation_delta = quality.rotation;
        candidate.inliers = (int)inliers.size();
        if (accepted) {
            candidate.pose_cs = pose_cs;
            candidate.pnp_inlier_indices = inliers;
            candidate.valid = true;
            // M3.2（§7.5）：接受位姿的数值协方差（相机系左扰动切空间）。
            // 退化时 valid=false，调用方回退保守占位。
            const PoseCovarianceResult cov =
                estimatePnPCovariance(pose_cs, pts3d, pts2d, inliers);
            candidate.pose_covariance = cov.covariance_cs;
            candidate.pose_covariance_valid = cov.valid;
        }
        return candidate;
    };

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    if (solve(false, std::nullopt, rvec, tvec, inliers))
        r = assess(rvec, tvec, inliers);
    if (r.valid || !motion.predicted_pose_cs) return r;

    // RANSAC 在稀疏/近退化窗口中偶尔会返回“内点数看似足够、RMSE 却上千
    // 像素”的数值坏解。不要降低任何几何门；以匀速预测为初值独立重试
    // 同一组 3D-2D 和同一 RANSAC 门。只有重试结果也通过原内点/比例/RMSE/
    // 连续性验收才接受，避免一次随机坏解把前端推入 20 帧 LOST 链。
    cv::Mat retry_rvec, retry_tvec;
    std::vector<int> retry_inliers;
    // OpenCV RANSAC 使用线程局部全局 RNG。重试是数值容错分支，不应因为
    // “是否触发过”改变后续帧的随机序列，否则相同输入会产生级联轨迹分叉。
    const uint64_t rng_state_before_retry = cv::theRNG().state;
    const bool retry_ok = solve(true, motion.predicted_pose_cs,
                                retry_rvec, retry_tvec, retry_inliers);
    cv::theRNG().state = rng_state_before_retry;
    if (!retry_ok) return r;
    TrackingResult retry = assess(retry_rvec, retry_tvec, retry_inliers);
    if (retry.valid) return retry;
    // 两次都拒绝时保留更有诊断意义的一组质量数据。
    if (retry.inliers > r.inliers ||
        (retry.inliers == r.inliers && retry.pose_rmse < r.pose_rmse))
        return retry;
    return r;
}

TrackingResult FrontendTracker::refinePnP(
    const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
    const SE3& initial_pose_cs, const SE3& T_ws,
    const MotionBaseline& motion,
    int min_inliers, double min_ratio, double max_rmse) const {
    TrackingResult r;
    if (pts3d.size() < 6 || pts3d.size() != pts2d.size()) return r;

    // 输入排序：消除双实例交替驱动下局部地图点/关联的容器遍历顺序差异
    // （M0 §4.4 等价性验收：raw2/raw3/est 三实例在 09d2a35 方案 B 引入后
    // 出现 float 精度级（2^-25）分歧——solvePnP 的 LM 累加对输入顺序敏感，
    // 相同点集不同顺序收敛到不同浮点解；按 3D 坐标字典序稳定排序后输入
    // 逐位一致，精修输出恢复确定性）。
    std::vector<cv::Point3f> sorted_3d;
    std::vector<cv::Point2f> sorted_2d;
    {
        std::vector<size_t> order(pts3d.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::ranges::stable_sort(order, [&](size_t a, size_t b) {
            const auto& pa = pts3d[a];
            const auto& pb = pts3d[b];
            if (pa.x != pb.x) return pa.x < pb.x;
            if (pa.y != pb.y) return pa.y < pb.y;
            return pa.z < pb.z;
        });
        sorted_3d.resize(pts3d.size());
        sorted_2d.resize(pts2d.size());
        for (size_t i = 0; i < order.size(); i++) {
            sorted_3d[i] = pts3d[order[i]];
            sorted_2d[i] = pts2d[order[i]];
        }
    }

    // 确定性精修：不重跑 RANSAC（避免消耗全局 RNG，破坏双实例交替驱动的
    // 确定性等价），而是用首轮位姿作初值做纯迭代优化。solvePnP 的
    // rvec/tvec 语义与 trackPnP 一致：p_c = R·p_s + t，即 T_cs。
    const Mat33 R_e = initial_pose_cs.q.toRotationMatrix();
    cv::Mat R_init(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_init.at<double>(i, j) = R_e(i, j);
    cv::Mat rvec_init, tvec_init;
    cv::Rodrigues(R_init, rvec_init);
    tvec_init = (cv::Mat_<double>(3, 1) << initial_pose_cs.t.x(),
                 initial_pose_cs.t.y(), initial_pose_cs.t.z());

    cv::Mat rvec = rvec_init.clone(), tvec = tvec_init.clone();
    const bool ok = cv::solvePnP(sorted_3d, sorted_2d, camera_->K(), cv::Mat(),
                                 rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    if (!ok) return r;

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    const SE3 candidate_pose = Relocalizer::matToSE3(R, tvec);
    std::vector<int> all_indices(sorted_3d.size());
    for (size_t i = 0; i < all_indices.size(); i++)
        all_indices[i] = static_cast<int>(i);
    const double rmse = PoseGate::pnpReprojectionRmse(
        sorted_3d, sorted_2d, rvec, tvec, all_indices, camera_->K());

    // 与 trackPnP 相同的统一验收（几何 + 连续性）
    PoseQuality quality;
    const bool accepted = PoseGate::acceptPoseCandidate(
        candidate_pose * T_ws.inverse(), (int)sorted_3d.size(), sorted_3d.size(), rmse,
        min_inliers, min_ratio, max_rmse,
        motion.baseline_twc, motion.max_translation, motion.max_rotation, quality);

    r.inlier_ratio = quality.inlier_ratio;
    r.pose_rmse = quality.pose_rmse;
    r.translation_delta = quality.translation;
    r.rotation_delta = quality.rotation;
    r.inliers = (int)sorted_3d.size();   // 确定性精修不筛外点，内点集由调用方提供
    if (accepted) {
        r.pose_cs = candidate_pose;
        r.valid = true;
        // M3.2（§7.5）：精修接受位姿的数值协方差（输入已按 3D 字典序稳定
        // 排序，协方差与排序无关，但复用同一数组避免二次拷贝）。
        const PoseCovarianceResult cov = estimatePnPCovariance(
            candidate_pose, sorted_3d, sorted_2d, all_indices);
        r.pose_covariance = cov.covariance_cs;
        r.pose_covariance_valid = cov.valid;
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

// ============================================================
// 方案 A/B 公共：像素网格索引 + 窗口描述子比率匹配
// ============================================================
std::vector<std::vector<int>> FrontendTracker::buildKeypointGrid(
    const Frame::Ptr& frame, double cell_size_px) const {
    std::vector<std::vector<int>> grid;
    if (!frame) return grid;
    const int cell_w = std::max(1, (int)std::ceil(cell_size_px));
    const int cols = std::max(1, camera_->width() / cell_w + 1);
    const int rows = std::max(1, camera_->height() / cell_w + 1);
    grid.resize((size_t)cols * rows);
    for (size_t i = 0; i < frame->keypoints.size(); i++) {
        const int cx = (int)frame->keypoints[i].pt.x / cell_w;
        const int cy = (int)frame->keypoints[i].pt.y / cell_w;
        if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) continue;
        grid[(size_t)cy * cols + cx].push_back((int)i);
    }
    return grid;
}

bool FrontendTracker::windowRatioMatch(
    const cv::Mat& query_desc, const cv::Mat& train_desc,
    const std::vector<int>& candidates,
    double ratio_thresh, float max_dist, int& best_train_idx, float& best_dist,
    float& second_dist) const {
    best_train_idx = -1;
    best_dist = std::numeric_limits<float>::max();
    second_dist = std::numeric_limits<float>::max();
    for (int ti : candidates) {
        if (ti < 0 || ti >= train_desc.rows) continue;
        const float d = cv::norm(query_desc, train_desc.row(ti), cv::NORM_HAMMING);
        if (d < best_dist) {
            second_dist = best_dist;
            best_dist = d;
            best_train_idx = ti;
        } else if (d < second_dist) {
            second_dist = d;
        }
    }
    if (best_train_idx < 0) return false;
    // 绝对距离上限：ORB 256 位描述子的典型匹配距离 < 64（与 quickMatchCount
    // 默认一致）。窗口匹配不能只靠比率——单候选时无条件接受会把窗口内的
    // 随机近邻当作匹配，而局部地图精修（refinePnP）不做 RANSAC 外点过滤，
    // 这类错误对应会直接拉偏位姿。
    if (best_dist > max_dist) return false;
    // 候选不足 2 个时退化为最近邻直取（窗口约束 + 距离上限已过滤误匹配）
    return second_dist == std::numeric_limits<float>::max()
        || best_dist < ratio_thresh * second_dist;
}

// ============================================================
// 方案 A：运动模型引导匹配
// 用预测位姿把参考帧有 3D 点的特征投影到当前帧，只在投影点邻域内做
// 描述子比率匹配。相比全图 BF：少误匹配（空间先验）+ 少计算（窗口小）。
// ============================================================
std::vector<cv::DMatch> FrontendTracker::matchGuided(
    const Frame::Ptr& ref_frame, const Frame::Ptr& curr_frame,
    const std::vector<Vec3>& ref_points_s,
    const SE3& T_cs_pred, double search_radius_px, double ratio_thresh) const {
    std::vector<cv::DMatch> matches;
    if (!ref_frame || !curr_frame || ref_frame->descriptors.empty()
        || curr_frame->descriptors.empty()) return matches;

    const auto grid = buildKeypointGrid(curr_frame, search_radius_px);
    const int cell_w = std::max(1, (int)std::ceil(search_radius_px));
    const int cols = std::max(1, camera_->width() / cell_w + 1);
    const int rows = std::max(1, camera_->height() / cell_w + 1);
    const int radius_cells = std::max(1, (int)std::ceil(search_radius_px / cell_w));

    const size_t n = std::min(ref_frame->keypoints.size(), ref_points_s.size());
    for (size_t qi = 0; qi < n; qi++) {
        // 只对有 3D 点的特征做引导匹配（无点无法投影，交给全图匹配回退）
        if (ref_points_s[qi].isZero(1e-12)) continue;
        const Vec3 p_c = T_cs_pred * ref_points_s[qi];
        if (p_c.z() <= 0.01) continue;   // 预测位姿下点在相机后方 → 跳过
        const Vec2 px = camera_->camera2pixel(p_c);
        if (px.x() < 0 || px.y() < 0 || px.x() >= camera_->width()
            || px.y() >= camera_->height()) continue;
        const int gx = (int)px.x() / cell_w;
        const int gy = (int)px.y() / cell_w;

        // 收集投影点邻域所有格子的候选特征
        std::vector<int> candidates;
        candidates.reserve(64);
        for (int dy = -radius_cells; dy <= radius_cells; dy++) {
            for (int dx = -radius_cells; dx <= radius_cells; dx++) {
                const int cx = gx + dx, cy = gy + dy;
                if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) continue;
                const auto& cell = grid[(size_t)cy * cols + cx];
                candidates.insert(candidates.end(), cell.begin(), cell.end());
            }
        }
        if (candidates.empty()) continue;

        int best_ti = -1;
        float best_d = 0.0f, second_d = 0.0f;
        if (!windowRatioMatch(ref_frame->descriptors.row((int)qi),
                              curr_frame->descriptors, candidates,
                              ratio_thresh, 64.0f, best_ti, best_d, second_d))
            continue;
        matches.emplace_back((int)qi, best_ti, best_d);
    }
    return matches;
}

// ============================================================
// 方案 B：共视图局部地图投影匹配
// 首轮 PnP 成功后，把参考 KF 共视的地图点投影进当前帧补匹配，扩大
// 3D-2D 对应集（特征稀疏/视角变化大时保住 PnP）。
// ============================================================
LocalMapTrackResult FrontendTracker::trackLocalMap(
    const Frame::Ptr& curr_frame,
    const std::vector<Vec3>& local_points_s,
    const std::vector<cv::Mat>& local_descs,
    const std::vector<MapPoint::Ptr>& local_mps,
    const std::vector<int>& occupied_features,
    const SE3& T_cs, double search_radius_px, double ratio_thresh) const {
    LocalMapTrackResult result;
    if (!curr_frame || curr_frame->descriptors.empty()
        || local_points_s.size() != local_descs.size()
        || local_points_s.size() != local_mps.size()) return result;

    std::unordered_set<int> occupied(occupied_features.begin(), occupied_features.end());
    const auto grid = buildKeypointGrid(curr_frame, search_radius_px);
    const int cell_w = std::max(1, (int)std::ceil(search_radius_px));
    const int cols = std::max(1, camera_->width() / cell_w + 1);
    const int rows = std::max(1, camera_->height() / cell_w + 1);
    const int radius_cells = std::max(1, (int)std::ceil(search_radius_px / cell_w));

    for (size_t pi = 0; pi < local_points_s.size(); pi++) {
        const Vec3 p_c = T_cs * local_points_s[pi];
        if (p_c.z() <= 0.01) continue;
        const Vec2 px = camera_->camera2pixel(p_c);
        if (px.x() < 0 || px.y() < 0 || px.x() >= camera_->width()
            || px.y() >= camera_->height()) continue;
        const int gx = (int)px.x() / cell_w;
        const int gy = (int)px.y() / cell_w;

        std::vector<int> candidates;
        candidates.reserve(64);
        for (int dy = -radius_cells; dy <= radius_cells; dy++) {
            for (int dx = -radius_cells; dx <= radius_cells; dx++) {
                const int cx = gx + dx, cy = gy + dy;
                if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) continue;
                const auto& cell = grid[(size_t)cy * cols + cx];
                candidates.insert(candidates.end(), cell.begin(), cell.end());
            }
        }
        // 剔除已被首轮匹配占用的特征（同一特征不能关联两个 3D 点）
        std::erase_if(candidates, [&](int ti) { return occupied.contains(ti); });
        if (candidates.empty()) continue;

        int best_ti = -1;
        float best_d = 0.0f, second_d = 0.0f;
        if (!windowRatioMatch(local_descs[pi], curr_frame->descriptors,
                              candidates, ratio_thresh, 64.0f,
                              best_ti, best_d, second_d))
            continue;
        const Vec3& p = local_points_s[pi];
        result.pts3d.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
        result.pts2d.emplace_back(curr_frame->keypoints[best_ti].pt);
        result.mps.push_back(local_mps[pi]);
        result.curr_feature_indices.push_back(best_ti);
        occupied.insert(best_ti);
        result.added++;
    }
    return result;
}

TrackingResult FrontendTracker::trackOrb(
    const Frame::Ptr& curr_frame, const Frame::Ptr& ref_frame,
    const RefView& ref, const MotionBaseline& motion,
    const StereoStats& stereo) const {
    TrackingResult r;

    // 跟踪匹配不做基础矩阵 RANSAC（省时，且避免共面场景 F 矩阵退化误剔）：
    // 外点交给下方 solvePnPRansac 自己剔除；仅初始化/回退分支保留 F 矩阵 RANSAC
    // 方案 A：有预测位姿且配置开启时先走运动模型引导匹配（投影邻域内搜索），
    // 匹配数不足说明预测失效/视角变化大，回退全图 BF（与旧行为一致）。
    std::vector<cv::DMatch> matches;
    if (cfg_.guided_match && ref.has_ref && motion.predicted_pose_cs) {
        matches = matchGuided(ref_frame, curr_frame, ref.ref_points_s,
                              *motion.predicted_pose_cs,
                              cfg_.guided_search_radius_px, cfg_.match_ratio);
        if ((int)matches.size() < cfg_.min_matches_track) {
            LOG_WARN("Guided match degraded (" << matches.size()
                     << "), fallback to full BF");
            matches = matcher_.match(ref_frame, curr_frame, cfg_.match_ratio, false);
        }
    } else {
        matches = matcher_.match(ref_frame, curr_frame, cfg_.match_ratio, false);
    }

    // 收集 3D-2D 对应（保留 pts3d[i] 与 matches 的映射，供内点观测计数）
    r.match_pairs = matches;
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
            // M3.2（§7.5）：数值协方差随最终结果传播（refine 覆盖时同步更新）
            r.pose_covariance = pnp.pose_covariance;
            r.pose_covariance_valid = pnp.pose_covariance_valid;
            // 位姿通过全部质量检查后才关联地图点，避免被拒绝的解污染共视统计。
            for (int idx : pnp.pnp_inlier_indices) {
                if (idx < 0 || idx >= (int)match_idx.size()) continue;
                const auto& m = matches[match_idx[idx]];
                auto& mp = ref.ref_mps[m.queryIdx];
                if (mp) {
                    r.associations.emplace_back(m.trainIdx, mp);
                    // 快照坐标与关联逐位对齐（refine 等后续步骤必须用快照，
                    // 不能读 live mp->pos_s——后端 BA 可能已改写）
                    r.association_points_s.push_back(ref.ref_points_s[m.queryIdx]);
                }
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
                        const auto& m = matches[k];
                        auto& mp = ref.ref_mps[m.queryIdx];
                        if (mp) {
                            r.associations.emplace_back(m.trainIdx, mp);
                            r.association_points_s.push_back(
                                ref.ref_points_s[m.queryIdx]);
                        }
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
