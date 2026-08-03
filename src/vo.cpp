#include "vslam/vo.h"
#include "vslam/optimizer.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <Eigen/SVD>

#include <set>
#include <unordered_map>
#include <algorithm>
#include <ranges>
#include <limits>

namespace vslam {

VOConfig VOConfig::fromYaml(const std::string& path) {
    VOConfig cfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (auto f = root["Feature"]) {
            if (f["num_features"])    cfg.num_features           = f["num_features"].as<int>();
            if (f["scale_factor"])    cfg.scale_factor           = f["scale_factor"].as<double>();
            if (f["pyramid_levels"])  cfg.pyramid_levels         = f["pyramid_levels"].as<int>();
            if (f["match_ratio"])     cfg.match_ratio            = f["match_ratio"].as<double>();
            if (f["ransac_threshold"]) cfg.ransac_pixel_threshold = f["ransac_threshold"].as<double>();
        }
        if (auto v = root["VO"]) {
            if (v["method"])              cfg.feature_method       = v["method"].as<int>();
            if (v["min_matches_init"])    cfg.min_matches_init     = v["min_matches_init"].as<int>();
            if (v["min_matches_track"])   cfg.min_matches_track    = v["min_matches_track"].as<int>();
            if (v["pnp_min_inliers"])     cfg.pnp_min_inliers      = v["pnp_min_inliers"].as<int>();
            if (v["pnp_min_inlier_ratio"]) cfg.pnp_min_inlier_ratio = v["pnp_min_inlier_ratio"].as<double>();
            if (v["pnp_max_rmse"])        cfg.pnp_max_rmse         = v["pnp_max_rmse"].as<double>();
            if (v["max_tracking_failures"]) cfg.max_tracking_failures = v["max_tracking_failures"].as<int>();
            if (v["max_relocalize_frames"]) cfg.max_relocalize_frames = v["max_relocalize_frames"].as<int>();
            if (v["max_frame_translation"]) cfg.max_frame_translation = v["max_frame_translation"].as<double>();
            if (v["max_frame_rotation"]) cfg.max_frame_rotation = v["max_frame_rotation"].as<double>();
            if (v["keyframe_translation"]) cfg.keyframe_translation = v["keyframe_translation"].as<double>();
            if (v["keyframe_rotation"])   cfg.keyframe_rotation    = v["keyframe_rotation"].as<double>();
            if (v["keyframe_min_inliers"]) cfg.keyframe_min_inliers = v["keyframe_min_inliers"].as<int>();
            if (v["min_keyframe_interval"]) cfg.min_keyframe_interval = v["min_keyframe_interval"].as<int>();
            if (v["max_keyframe_interval"]) cfg.max_keyframe_interval = v["max_keyframe_interval"].as<int>();
        }
        if (auto o = root["Optimizer"]) {
            if (o["local_window_size"])   cfg.local_window_size   = o["local_window_size"].as<int>();
            if (o["local_ba_iterations"]) cfg.local_ba_iterations = o["local_ba_iterations"].as<int>();
            if (o["enable_local_ba"])     cfg.enable_local_ba     = o["enable_local_ba"].as<bool>();
        }
        if (auto s = root["Stereo"]) {
            if (s["min_depth"]) cfg.stereo_min_depth = s["min_depth"].as<double>();
            if (s["max_depth"]) cfg.stereo_max_depth = s["max_depth"].as<double>();
            if (s["min_points"]) cfg.stereo_min_points = s["min_points"].as<int>();
            if (s["rigid_min_inliers"]) cfg.rigid_min_inliers = s["rigid_min_inliers"].as<int>();
            if (s["rigid_min_inlier_ratio"]) cfg.rigid_min_inlier_ratio = s["rigid_min_inlier_ratio"].as<double>();
            if (s["rigid_ransac_threshold"]) cfg.rigid_ransac_threshold = s["rigid_ransac_threshold"].as<double>();
            if (s["rigid_max_rmse"]) cfg.rigid_max_rmse = s["rigid_max_rmse"].as<double>();
            if (s["keyframe_translation"])
                cfg.keyframe_translation_stereo = s["keyframe_translation"].as<double>();
        }
        if (auto lc = root["LoopClosure"]) {
            if (lc["enable_loop_closure"]) cfg.enable_loop_closure = lc["enable_loop_closure"].as<bool>();
            if (lc["vocab_path"])          cfg.vocab_path = lc["vocab_path"].as<std::string>();
            if (lc["min_score"])           cfg.min_score = lc["min_score"].as<double>();
            if (lc["temporal_window"])     cfg.temporal_window = lc["temporal_window"].as<int>();
            if (lc["detection_interval"])  cfg.detection_interval = lc["detection_interval"].as<int>();
            if (lc["pnp_inlier_ratio"])    cfg.pnp_inlier_ratio = lc["pnp_inlier_ratio"].as<double>();
            if (lc["min_loop_inliers"])    cfg.min_loop_inliers = lc["min_loop_inliers"].as<int>();
            if (lc["loop_cooldown_frames"]) cfg.loop_cooldown_frames = lc["loop_cooldown_frames"].as<int>();
        }
        if (auto o = root["Optimizer"]) {
            if (o["global_ba_iterations"]) cfg.global_ba_iterations = o["global_ba_iterations"].as<int>();
        }
        LOG_INFO("VO config loaded from: " << path);
    } catch (const std::exception& e) {
        LOG_WARN("VOConfig::fromYaml failed (" << e.what() << "), using defaults");
    }
    return cfg;
}

VisualOdometry::VisualOdometry(const Camera& camera, const VOConfig& cfg)
    : camera_(camera), cfg_(cfg), atlas_(std::make_shared<Atlas>()) {
    map_ = atlas_->createSubmap(SE3()).map;
    feature_matcher_.setParams(cfg_.num_features, cfg_.scale_factor, cfg_.pyramid_levels);
    if (cfg_.feature_method != 0) {
        LOG_INFO("feature_method=" << cfg_.feature_method << " (LK 光流)");
    }
    // Phase 2：配置中启用回环且给出词典路径时自动加载
    if (cfg_.enable_loop_closure && !cfg_.vocab_path.empty())
        enableLoopClosure(cfg_.vocab_path);
}

bool VisualOdometry::enableLoopClosure(const std::string& vocab_path) {
    if (!loop_closure_) loop_closure_ = std::make_unique<LoopClosure>();
    loop_closure_->setParams(cfg_.min_score, cfg_.temporal_window,
                             cfg_.min_loop_inliers, cfg_.pnp_inlier_ratio,
                             cfg_.ransac_pixel_threshold, camera_);
    loop_closure_enabled_ = loop_closure_->loadVocabulary(vocab_path);
    if (loop_closure_enabled_) {
        LOG_INFO("Loop closure ENABLED (vocab=" << vocab_path
                 << ", interval=" << cfg_.detection_interval << ")");
    }
    return loop_closure_enabled_;
}

SE3 VisualOdometry::addFrame(const cv::Mat& image, double timestamp) {
    return addFrameImpl(image, cv::Mat(), timestamp);
}

SE3 VisualOdometry::addFrame(const cv::Mat& left, const cv::Mat& right, double timestamp) {
    return addFrameImpl(left, right, timestamp);
}

SE3 VisualOdometry::addFrameImpl(const cv::Mat& left, const cv::Mat& right, double timestamp) {
    unsigned long frame_id = frame_count_++;
    LOG_INFO("--- Frame " << frame_id << " ---");

    // 1. 创建当前帧 + CLAHE 增强
    curr_frame_ = std::make_shared<Frame>(frame_id, timestamp);
    status_.tracking_valid = false;
    status_.pose_valid = false;
    status_.pose_method = "NONE";
    status_.stereo_points = 0;
    status_.median_disparity = 0.0;
    status_.median_depth = 0.0;
    status_.inlier_ratio = 0.0;
    status_.pose_rmse = 0.0;
    status_.translation_delta = 0.0;
    status_.rotation_delta = 0.0;
    if (left.channels() == 3)
        cv::cvtColor(left, curr_frame_->image_gray, cv::COLOR_BGR2GRAY);
    else
        curr_frame_->image_gray = left;
    curr_frame_->image = left;

    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(curr_frame_->image_gray, curr_frame_->image_gray);

    // 双目：右目图做同样的灰度化 + 增强（左右目一致，保证双目匹配质量）
    if (!right.empty()) {
        curr_frame_->image_right = right;
        if (right.channels() == 3)
            cv::cvtColor(right, curr_frame_->image_right_gray, cv::COLOR_BGR2GRAY);
        else
            curr_frame_->image_right_gray = right;
        clahe->apply(curr_frame_->image_right_gray, curr_frame_->image_right_gray);
    }

    // 2. 提取/跟踪特征
    // LK 模式（feature_method=1）：TRACKING 阶段用光流跟踪上一帧，不重新提取 ORB
    const bool use_lk = (cfg_.feature_method == 1 && state_ == State::TRACKING
                         && prev_frame_ && !prev_frame_->keypoints.empty());
    if (use_lk) {
        feature_matcher_.trackLK(prev_frame_, curr_frame_);
        if (curr_frame_->keypoints.size() < (size_t)cfg_.min_matches_track) {
            LOG_WARN("LK track degraded (" << curr_frame_->keypoints.size()
                     << " pts), fallback to ORB extraction");
            feature_matcher_.extract(curr_frame_);
        }
    } else {
        feature_matcher_.extract(curr_frame_);
    }
    if (curr_frame_->keypoints.empty()) {
        LOG_WARN("No features extracted, skipping frame");
        if (state_ != State::INITIALIZING && ref_frame_) {
            tracking_failures_++;
            state_ = tracking_failures_ >= cfg_.max_tracking_failures
                ? State::LOST : State::RECOVERING;
            if (state_ == State::LOST) relocalization_frames_++;
            curr_frame_->pose_cw = has_last_valid_pose_ ? last_valid_pose_cw_
                                                        : ref_frame_->pose_cw;
        }
        updateStatus(0, 0, 0.0);
        return (state_ != State::INITIALIZING) ? curr_frame_->pose_cw : SE3();
    }

    // 2.5 双目/RGB-D：视差（或深度）→ 每特征点的相机系 3D 观测 pts_c
    computeStereoDepths();

    // 3. 状态机处理
    if (state_ == State::INITIALIZING) {
        bool ok = tryInitialize();
        if (ok) {
            state_ = State::TRACKING;
            tracking_failures_ = 0;
            relocalization_frames_ = 0;
            status_.tracking_valid = true;
            status_.pose_method = "INIT";
        }
    } else if (state_ == State::TRACKING) {
        // LK 模式：描述子为空说明走的是光流路径，用索引对齐的 PnP 跟踪
        if (cfg_.feature_method == 1 && curr_frame_->descriptors.empty())
            trackFrameLK();
        else
            trackFrame();
        // 跟踪可能把状态置为 RECOVERING（跳变保护），此时不能再插入关键帧
        if (state_ == State::TRACKING && needNewKeyFrame()) insertKeyFrame();
    }

    if (state_ == State::RECOVERING || state_ == State::LOST) {
        if (state_ == State::RECOVERING && tracking_failures_ < cfg_.max_tracking_failures) {
            ++tracking_failures_;
        } else {
            state_ = State::LOST;
        }

        if (tryRelocalize()) {
            state_ = State::TRACKING;
            tracking_failures_ = 0;
            relocalization_frames_ = 0;
            status_.tracking_valid = true;
            status_.pose_method = "RELOCALIZE";
        } else {
            ++relocalization_frames_;
            if (relocalization_frames_ >= cfg_.max_relocalize_frames) {
                createSubmap();
                if (tryInitialize()) {
                    LOG_WARN((camera_->hasPerFrameDepth()
                              ? "Stereo re-init in anchored submap"
                              : "Monocular re-init in anchored submap"));
                    state_ = State::TRACKING;
                    tracking_failures_ = 0;
                    relocalization_frames_ = 0;
                    status_.tracking_valid = true;
                    status_.pose_method = "SUBMAP_INIT";
                }
            }
        }
    }

    prev_frame_ = curr_frame_;
    if (state_ == State::TRACKING && status_.tracking_valid) {
        last_valid_pose_cw_ = curr_frame_->pose_cw;
        has_last_valid_pose_ = true;
        tracking_failures_ = 0;
    }
    updateStatus(status_.matches, status_.inliers, status_.parallax);
    status_.pose_valid = status_.tracking_valid && status_.map_connected;
    // pose_cw.t 是世界原点在相机系中的坐标，不是相机位置。
    // 原地旋转时它会随 R_cw 绕圈；轨迹必须记录相机光心 C_w = -R_cw^T t_cw。
    if (status_.pose_valid) {
        pose_trajectory_.push_back(curr_frame_->pose_cw);
        traj_frame_ids_.push_back(curr_frame_->id);  // 回环校正定位漂移段用
    }
    return curr_frame_->pose_cw;
}

// ============================================================
// 双目/RGB-D 深度计算：视差（或深度）→ pts_c
// ============================================================
void VisualOdometry::computeStereoDepths() {
    curr_frame_->pts_c.clear();
    curr_frame_->pts_c.resize(curr_frame_->keypoints.size(), Vec3::Zero());
    // 单目：无单帧深度，pts_c 全部无效（走多帧三角化）
    if (!camera_->hasPerFrameDepth() || curr_frame_->image_right.empty()) return;

    // 双目匹配用原始灰度（非 CLAHE）：CLAHE 是内容相关的非线性增强，
    // 左右目同一 3D 点的局部直方图不同 → 灰度不一致 → 破坏光度一致性，
    // 显著降低 LK 左右目匹配质量。
    cv::Mat left_raw, right_raw;
    if (curr_frame_->image.channels() == 3)
        cv::cvtColor(curr_frame_->image, left_raw, cv::COLOR_BGR2GRAY);
    else
        left_raw = curr_frame_->image;
    if (curr_frame_->image_right.channels() == 3)
        cv::cvtColor(curr_frame_->image_right, right_raw, cv::COLOR_BGR2GRAY);
    else
        right_raw = curr_frame_->image_right;

    std::vector<cv::Point2f> right_pts;
    auto status = feature_matcher_.matchStereo(
        left_raw, right_raw,
        curr_frame_->keypoints, right_pts);

    std::vector<double> disparities;
    std::vector<double> depths;
    disparities.reserve(status.size());
    depths.reserve(status.size());
    for (size_t i = 0; i < status.size() && i < curr_frame_->keypoints.size(); i++) {
        if (!status[i]) continue;
        double disparity = curr_frame_->keypoints[i].pt.x - right_pts[i].x;
        const double min_disparity = camera_->fx * camera_->baseline() / cfg_.stereo_max_depth;
        const double max_disparity = camera_->fx * camera_->baseline() / cfg_.stereo_min_depth;
        if (disparity < min_disparity || disparity > max_disparity) continue;
        double depth = camera_->fx * camera_->baseline() / disparity;  // z = fx*b/d
        if (depth < cfg_.stereo_min_depth || depth > cfg_.stereo_max_depth) continue;
        curr_frame_->pts_c[i] = camera_->pixel2camera(
            Vec2(curr_frame_->keypoints[i].pt.x, curr_frame_->keypoints[i].pt.y), depth);
        disparities.push_back(disparity);
        depths.push_back(depth);
    }

    status_.stereo_points = (int)depths.size();
    if (!depths.empty()) {
        std::ranges::sort(disparities);
        std::ranges::sort(depths);
        status_.median_disparity = disparities[disparities.size() / 2];
        status_.median_depth = depths[depths.size() / 2];
    }
}

// ============================================================
// 双目/RGB-D 单帧建点：pts_c 有效 → 世界系 3D 地图点
// ============================================================
void VisualOdometry::createMapPointsFromStereo(const Frame::Ptr& frame) {
    if (!camera_->hasPerFrameDepth()) return;

    int cnt = 0;
    for (size_t i = 0; i < frame->keypoints.size(); i++) {
        if (frame->pts_c[i].z() > 0 && frame->map_points[i] == nullptr) {
            // p_w = T_wc * p_c（pose_cw 的逆把相机系点转到世界系）
            Vec3 p_w = frame->pose_cw.inverse() * frame->pts_c[i];
            auto mp = std::make_shared<MapPoint>(map_->nextMapPointId());
            mp->pos_w = p_w;
            if (!frame->descriptors.empty())
                mp->descriptor = frame->descriptors.row((int)i).clone();
            mp->observed_count = 1;  // 仅当前帧观测，后续跟踪帧会累加
            map_->insertMapPoint(mp);
            frame->map_points[i] = mp;
            cnt++;
        }
    }
    if (cnt > 0) LOG_INFO("Stereo map points created: " << cnt);
}

// ============================================================
// 状态更新辅助
// ============================================================
void VisualOdometry::updateStatus(int matches, int inliers, double parallax) {
    status_.state      = state_;
    status_.matches    = matches;
    status_.inliers    = inliers;
    status_.parallax   = parallax;
    status_.map_points = map_->mapPointCount();
    status_.keyframes  = map_->keyFrameCount();
    const auto* active_submap = atlas_->activeSubmap();
    status_.submap_id  = active_submap ? active_submap->id : 0;
    status_.map_connected = active_submap ? active_submap->connected : false;
    status_.lost_frames = tracking_failures_ + relocalization_frames_;
}

void VisualOdometry::createSubmap() {
    // 新子地图的原点必须落在上一段全局轨迹附近。短时丢失时使用最后
    // 有效位姿；如果还没有有效位姿，则退化为当前参考帧位姿。
    const SE3 anchor_Twc = has_last_valid_pose_
        ? last_valid_pose_cw_.inverse()
        : (ref_frame_ ? ref_frame_->pose_cw.inverse() : SE3());

    // 丢失期间的真实位移未知，anchor 只用于 Viewer 连续显示，不能声称
    // 新子地图已经连接到全局世界系。后续重定位到旧地图后才恢复全局有效轨迹。
    auto& submap = atlas_->createSubmap(anchor_Twc, false);
    map_ = submap.map;
    ref_frame_.reset();
    prev_frame_.reset();
    last_kf_frame_id_ = 0;
    curr_frame_->pose_cw = anchor_Twc.inverse();
    state_ = State::INITIALIZING;
    tracking_failures_ = 0;
    relocalization_frames_ = 0;
    LOG_WARN("Tracking lost for too long; creating anchored submap " << submap.id);
}

// ============================================================
// 初始化
// ============================================================
bool VisualOdometry::tryInitialize() {
    // 双目/RGB-D：首帧即有绝对尺度深度，直接建图进入 TRACKING
    // （无需对极初始化——这是双目相对单目的本质区别：尺度可观测）
    if (camera_->hasPerFrameDepth()) {
        if (!ref_frame_) {
            if (status_.stereo_points < cfg_.stereo_min_points) {
                LOG_WARN("Stereo init postponed: only " << status_.stereo_points
                         << " valid depth points");
                updateStatus(0, 0, 0.0);
                return false;
            }
            ref_frame_ = curr_frame_;
            createMapPointsFromStereo(ref_frame_);
            map_->insertKeyFrame(ref_frame_);
            last_kf_frame_id_ = curr_frame_->id;
            LOG_INFO("Stereo init OK! (first frame, absolute scale) mp="
                     << map_->mapPointCount());
            updateStatus(0, 0, 0.0);
            return true;
        }
        return false;
    }

    // ---- 单目：两帧对极几何初始化（尺度归一化）----
    if (!ref_frame_) {
        ref_frame_ = curr_frame_;
        updateStatus(0, 0, 0.0);
        return false;
    }

    auto matches = feature_matcher_.match(ref_frame_, curr_frame_, cfg_.match_ratio, true);
    if (matches.size() < (size_t)cfg_.min_matches_init) {
        LOG_INFO("Init: too few matches (" << matches.size() << ")");
        ref_frame_ = curr_frame_;
        updateStatus((int)matches.size(), 0, 0.0);
        return false;
    }

    std::vector<cv::Point2f> pts1, pts2;
    FeatureMatcher::getMatchedPoints(ref_frame_, curr_frame_, matches, pts1, pts2);

    cv::Mat E = cv::findEssentialMat(pts1, pts2, camera_->K(),
                                     cv::RANSAC, 0.999, 1.0);
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts1, pts2, camera_->K(), R, t);
    double parallax = cv::norm(t);

    if (parallax < 0.1) {
        LOG_INFO("Init: insufficient parallax (" << parallax << ")");
        ref_frame_ = curr_frame_;
        updateStatus((int)matches.size(), inliers, parallax);
        return false;
    }

    // 第一帧使用当前子地图的锚定位姿。初始子地图锚点为单位位姿，
    // 后续子地图则继承全局位姿，避免重建后轨迹跳回原点。
    // recoverPose 返回的相对变换 T_rel 满足 p_c2 = T_rel * p_c1。
    SE3 anchor_cw = ref_frame_->pose_cw;
    Eigen::Matrix3d R_eigen;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_eigen(i, j) = R.at<double>(i, j);
    SE3 T_cw2(Eigen::Quaterniond(R_eigen),
              Vec3(t.at<double>(0), t.at<double>(1), t.at<double>(2)));
    curr_frame_->pose_cw = T_cw2 * anchor_cw;

    triangulateNewPoints(ref_frame_, curr_frame_, matches);

    map_->insertKeyFrame(ref_frame_);
    map_->insertKeyFrame(curr_frame_);
    odometry_edges_.push_back({
        ref_frame_->id, curr_frame_->id,
        ref_frame_->pose_cw * curr_frame_->pose_cw.inverse(), 1.0});
    last_kf_frame_id_ = curr_frame_->id;   // 初始化插入的两个关键帧也参与冷却

    LOG_INFO("Init OK! parallax=" << parallax << " inliers=" << inliers
             << " mp=" << map_->mapPointCount());
    updateStatus((int)matches.size(), inliers, parallax);
    return true;
}

// ============================================================
// 跟踪
// ============================================================
double VisualOdometry::pnpReprojectionRmse(
    const std::vector<cv::Point3f>& pts3d,
    const std::vector<cv::Point2f>& pts2d,
    const cv::Mat& rvec, const cv::Mat& tvec,
    const std::vector<int>& inliers) const {
    if (inliers.empty()) return std::numeric_limits<double>::infinity();
    std::vector<cv::Point2f> projected;
    cv::projectPoints(pts3d, rvec, tvec, camera_->K(), cv::Mat(), projected);
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

bool VisualOdometry::validateMotion(
    const SE3& pose_cw, double& translation, double& rotation) const {
    translation = 0.0;
    rotation = 0.0;
    if (!camera_->hasPerFrameDepth() || !has_last_valid_pose_) return true;

    const SE3 Twc_new = pose_cw.inverse();
    const SE3 Twc_last = last_valid_pose_cw_.inverse();
    translation = (Twc_new.t - Twc_last.t).norm();
    const Eigen::Quaterniond q_rel = Twc_new.q * Twc_last.q.inverse();
    rotation = 2.0 * std::acos(std::clamp(std::abs(q_rel.w()), 0.0, 1.0));
    return translation <= cfg_.max_frame_translation
        && rotation <= cfg_.max_frame_rotation;
}

SE3 VisualOdometry::trackFrame() {
    if (!ref_frame_ || !curr_frame_) return SE3();

    // 跟踪匹配不做基础矩阵 RANSAC（省时，且避免共面场景 F 矩阵退化误剔）：
    // 外点交给下方 solvePnPRansac 自己剔除；仅初始化/回退分支保留 F 矩阵 RANSAC
    auto matches = feature_matcher_.match(ref_frame_, curr_frame_, cfg_.match_ratio, false);

    // 收集 3D-2D 对应（保留 pts3d[i] 与 matches 的映射，供内点观测计数）
    // C++23 的 views::enumerate 同时给出索引与元素
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<int> match_idx;
    for (auto [k, m] : matches | std::views::enumerate) {
        auto& mp = ref_frame_->map_points[m.queryIdx];
        if (mp) {
            pts3d.emplace_back((float)mp->pos_w.x(), (float)mp->pos_w.y(), (float)mp->pos_w.z());
            pts2d.push_back(curr_frame_->keypoints[m.trainIdx].pt);
            match_idx.push_back((int)k);
        }
    }

    int inliers_cnt = 0;

    // PnP (3D-2D) —— solvePnPRansac 返回的 rvec/tvec 即 T_cw（世界→相机），直接存入 pose_cw
    if (pts3d.size() >= 6) {
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        bool ok = cv::solvePnPRansac(pts3d, pts2d, camera_->K(),
                                     cv::Mat(), rvec, tvec,
                                     false, 100, cfg_.ransac_pixel_threshold, 0.99, inliers);
        if (ok) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            const SE3 candidate_pose = matToSE3(R, tvec);
            const double inlier_ratio = (double)inliers.size() / pts3d.size();
            const double rmse = pnpReprojectionRmse(pts3d, pts2d, rvec, tvec, inliers);
            double translation = 0.0;
            double rotation = 0.0;
            const bool motion_ok = validateMotion(candidate_pose, translation, rotation);
            const bool quality_ok = (int)inliers.size() >= cfg_.pnp_min_inliers
                && inlier_ratio >= cfg_.pnp_min_inlier_ratio
                && rmse <= cfg_.pnp_max_rmse && motion_ok;

            status_.inlier_ratio = inlier_ratio;
            status_.pose_rmse = rmse;
            status_.translation_delta = translation;
            status_.rotation_delta = rotation;
            if (quality_ok) {
                curr_frame_->pose_cw = candidate_pose;
                inliers_cnt = (int)inliers.size();
                status_.tracking_valid = true;
                status_.pose_method = "PNP";
                // 位姿通过全部质量检查后才关联地图点，避免被拒绝的解污染共视统计。
                for (int idx : inliers) {
                    if (idx < 0 || idx >= (int)match_idx.size()) continue;
                    auto& mp = ref_frame_->map_points[matches[match_idx[idx]].queryIdx];
                    if (mp) {
                        mp->observed_count++;
                        curr_frame_->map_points[matches[match_idx[idx]].trainIdx] = mp;
                    }
                }
                updateStatus((int)matches.size(), inliers_cnt, 0.0);
                return curr_frame_->pose_cw;
            }
            LOG_WARN("PnP rejected: inliers=" << inliers.size()
                     << " ratio=" << inlier_ratio << " rmse=" << rmse
                     << " dtrans=" << translation << " drot=" << rotation);
        }
    }

    // 双目/RGB-D：PnP 失败后走 3D-3D 位姿估计（绝对尺度、旋转鲁棒）。
    // 对极几何是单目尺度估计的手段（recoverPose 的 t 归一化、旋转主导时退化），
    // 双目有绝对尺度（当前帧 pts_c），3D-3D 天然保持尺度且对旋转鲁棒。
    if (tryTrack3D3D(matches)) return curr_frame_->pose_cw;

    // 对极几何回退（仅单目：recoverPose 的 t 归一化，双目有绝对尺度不可用）
    // recoverPose 返回 T_rel 满足 p_c2 = T_rel * p_c1 → T_cw2 = T_rel * T_cw1
    if (!camera_->hasPerFrameDepth() && matches.size() >= (size_t)cfg_.min_matches_track) {
        std::vector<cv::Point2f> pts1, pts2;
        FeatureMatcher::getMatchedPoints(ref_frame_, curr_frame_, matches, pts1, pts2);
        cv::Mat E = cv::findEssentialMat(pts1, pts2, camera_->K(), cv::RANSAC, 0.999, 1.0);
        cv::Mat R, t;
        cv::recoverPose(E, pts1, pts2, camera_->K(), R, t);
        SE3 T_rel = matToSE3(R, t);
        curr_frame_->pose_cw = T_rel * ref_frame_->pose_cw;
        inliers_cnt = (int)matches.size();
        // 对极恢复的 t 只有方向无尺度，组合后可能跳变 → 同样做跳变保护
        SE3 Twc_new = curr_frame_->pose_cw.inverse();
        SE3 Twc_ref = ref_frame_->pose_cw.inverse();
        if ((Twc_new.t - Twc_ref.t).norm() > 30.0) {
            LOG_WARN("Epipolar fallback pose jump (" << (Twc_new.t - Twc_ref.t).norm()
                     << "m), tracking lost");
            curr_frame_->pose_cw = ref_frame_->pose_cw;
            state_ = State::RECOVERING;
            updateStatus((int)matches.size(), 0, 0.0);
            return ref_frame_->pose_cw;
        }
        status_.tracking_valid = true;
        status_.pose_method = "EPIPOLAR";
    } else {
        // 匹配太少 → LOST
        curr_frame_->pose_cw = ref_frame_->pose_cw;
        state_ = State::RECOVERING;
        LOG_WARN("Tracking lost! matches=" << matches.size()
                 << " pts3d=" << pts3d.size()
                 << " kf_ref=" << (ref_frame_ ? ref_frame_->id : -1)
                 << " mp_ref=" << (ref_frame_ ? ref_frame_->map_points.size() : 0));
    }

    updateStatus((int)matches.size(), inliers_cnt, 0.0);
    return curr_frame_->pose_cw;
}

// ============================================================
// 双目/RGB-D 3D-3D 位姿估计
// ref 帧世界系点 ↔ 当前帧相机系点（双目视差），RANSAC 求解 T_cw。
// 用真实深度测量：天然保持绝对尺度、对旋转-平移歧义鲁棒
// （对极几何的 t 归一化 + 旋转主导退化在双目下不可用）。
// ============================================================
bool VisualOdometry::tryTrack3D3D(const std::vector<cv::DMatch>& matches) {
    if (!camera_->hasPerFrameDepth() || matches.size() < 20
        || status_.stereo_points < cfg_.stereo_min_points) return false;

    std::vector<cv::Point3f> pts_w;   // ref 帧世界系 3D 点
    std::vector<cv::Point3f> pts_c;   // 当前帧相机系 3D 点（双目视差）
    std::vector<int> idx3;
    for (auto [k, m] : matches | std::views::enumerate) {
        auto& mp = ref_frame_->map_points[m.queryIdx];
        if (mp && curr_frame_->pts_c[m.trainIdx].z() > 0) {
            pts_w.emplace_back((float)mp->pos_w.x(), (float)mp->pos_w.y(), (float)mp->pos_w.z());
            pts_c.emplace_back((float)curr_frame_->pts_c[m.trainIdx].x(),
                               (float)curr_frame_->pts_c[m.trainIdx].y(),
                               (float)curr_frame_->pts_c[m.trainIdx].z());
            idx3.push_back((int)k);
        }
    }
    if ((int)pts_w.size() < std::max(20, cfg_.rigid_min_inliers)) return false;

    cv::Mat affine, inliers;
    // RANSAC 3D-3D：返回 3x4 [R|t] 满足 dst = R*src + t → 即 T_cw（世界→相机）
    bool ok = cv::estimateAffine3D(pts_w, pts_c, affine, inliers,
                                   cfg_.rigid_ransac_threshold, 0.99);
    if (!ok) return false;

    // estimateAffine3D 只用于 RANSAC 选内点。不能把其旋转投影回 SO(3) 后仍沿用
    // 原仿射平移：旋转、缩放和剪切被改变后，原 t 已不属于同一个变换。
    // 在内点上重新做 Kabsch 刚体拟合，统一求解 R、t（dst = R * src + t）。
    Vec3 mean_w = Vec3::Zero();
    Vec3 mean_c = Vec3::Zero();
    int rigid_inliers = 0;
    for (size_t i = 0; i < inliers.total(); i++) {
        if (!inliers.at<uchar>((int)i)) continue;
        mean_w += Vec3(pts_w[i].x, pts_w[i].y, pts_w[i].z);
        mean_c += Vec3(pts_c[i].x, pts_c[i].y, pts_c[i].z);
        rigid_inliers++;
    }
    const double inlier_ratio = (double)rigid_inliers / pts_w.size();
    if (rigid_inliers < cfg_.rigid_min_inliers
        || inlier_ratio < cfg_.rigid_min_inlier_ratio) {
        LOG_WARN("3D-3D rejected: inliers=" << rigid_inliers
                 << " ratio=" << inlier_ratio);
        return false;
    }
    mean_w /= rigid_inliers;
    mean_c /= rigid_inliers;

    Mat33 covariance = Mat33::Zero();
    for (size_t i = 0; i < inliers.total(); i++) {
        if (!inliers.at<uchar>((int)i)) continue;
        Vec3 pw(pts_w[i].x, pts_w[i].y, pts_w[i].z);
        Vec3 pc(pts_c[i].x, pts_c[i].y, pts_c[i].z);
        covariance += (pc - mean_c) * (pw - mean_w).transpose();
    }
    Eigen::JacobiSVD<Mat33> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Vec3 singular = svd.singularValues();
    if (singular.x() <= 1e-9 || singular.y() / singular.x() < 1e-3) {
        LOG_WARN("3D-3D rejected: degenerate point distribution");
        return false;
    }
    Mat33 U = svd.matrixU();
    const Mat33 V = svd.matrixV();
    Mat33 R_rigid = U * V.transpose();
    if (R_rigid.determinant() < 0) {
        U.col(2) *= -1;
        R_rigid = U * V.transpose();
    }
    Vec3 t_rigid = mean_c - R_rigid * mean_w;
    SE3 pose_cw(Eigen::Quaterniond(R_rigid), t_rigid);

    double squared_error = 0.0;
    for (size_t i = 0; i < inliers.total(); i++) {
        if (!inliers.at<uchar>((int)i)) continue;
        const Vec3 pw(pts_w[i].x, pts_w[i].y, pts_w[i].z);
        const Vec3 pc(pts_c[i].x, pts_c[i].y, pts_c[i].z);
        squared_error += (R_rigid * pw + t_rigid - pc).squaredNorm();
    }
    const double rmse = std::sqrt(squared_error / rigid_inliers);
    double translation = 0.0;
    double rotation = 0.0;
    const bool motion_ok = validateMotion(pose_cw, translation, rotation);
    if (rmse > cfg_.rigid_max_rmse || !motion_ok) {
        LOG_WARN("3D-3D rejected: rmse=" << rmse
                 << " dtrans=" << translation << " drot=" << rotation);
        return false;
    }

    curr_frame_->pose_cw = pose_cw;
    status_.tracking_valid = true;
    status_.pose_method = "3D3D";
    status_.inlier_ratio = inlier_ratio;
    status_.pose_rmse = rmse;
    status_.translation_delta = translation;
    status_.rotation_delta = rotation;
    // 关联内点（共视统计 + observed_count）
    int inl_cnt = 0;
    for (size_t i = 0; i < inliers.total() && i < idx3.size(); i++) {
        if (!inliers.at<uchar>(i)) continue;
        inl_cnt++;
        int k = idx3[i];
        auto& mp = ref_frame_->map_points[matches[k].queryIdx];
        if (mp) {
            mp->observed_count++;
            curr_frame_->map_points[matches[k].trainIdx] = mp;
        }
    }
    updateStatus((int)matches.size(), inl_cnt, 0.0);
    return true;
}

// ============================================================
// LK 光流跟踪（feature_method=1）
// 光流后 map_points 与关键点索引对齐（继承自上一帧），直接做 PnP
// ============================================================
SE3 VisualOdometry::trackFrameLK() {
    if (!curr_frame_) return SE3();

    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<int> kp_idx;
    for (size_t i = 0; i < curr_frame_->keypoints.size(); i++) {
        auto& mp = curr_frame_->map_points[i];
        if (mp) {
            pts3d.emplace_back((float)mp->pos_w.x(), (float)mp->pos_w.y(), (float)mp->pos_w.z());
            pts2d.push_back(curr_frame_->keypoints[i].pt);
            kp_idx.push_back((int)i);
        }
    }

    int inliers_cnt = 0;
    if (pts3d.size() >= 6) {
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        bool ok = cv::solvePnPRansac(pts3d, pts2d, camera_->K(),
                                     cv::Mat(), rvec, tvec,
                                     false, 100, cfg_.ransac_pixel_threshold, 0.99, inliers);
        if (ok) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            const SE3 candidate_pose = matToSE3(R, tvec);
            const double ratio = (double)inliers.size() / pts3d.size();
            const double rmse = pnpReprojectionRmse(pts3d, pts2d, rvec, tvec, inliers);
            double translation = 0.0;
            double rotation = 0.0;
            const bool motion_ok = validateMotion(candidate_pose, translation, rotation);
            if ((int)inliers.size() >= cfg_.pnp_min_inliers
                && ratio >= cfg_.pnp_min_inlier_ratio
                && rmse <= cfg_.pnp_max_rmse && motion_ok) {
                curr_frame_->pose_cw = candidate_pose;
                inliers_cnt = (int)inliers.size();
                status_.tracking_valid = true;
                status_.pose_method = "LK_PNP";
                status_.inlier_ratio = ratio;
                status_.pose_rmse = rmse;
                status_.translation_delta = translation;
                status_.rotation_delta = rotation;
                for (int idx : inliers) {
                    if (idx >= 0 && idx < (int)kp_idx.size()) {
                        auto& mp = curr_frame_->map_points[kp_idx[idx]];
                        if (mp) mp->observed_count++;
                    }
                }
                updateStatus((int)pts3d.size(), inliers_cnt, 0.0);
                return curr_frame_->pose_cw;
            }
        }
    }

    // LK PnP 失败 → 重新提取 ORB 特征，回退到 ORB 匹配跟踪
    LOG_WARN("LK PnP failed (" << pts3d.size() << " 3D pts), fallback to ORB track");
    feature_matcher_.extract(curr_frame_);
    return trackFrame();
}

// ============================================================
// 重定位（LOST 状态下尝试匹配所有关键帧恢复跟踪）
// ============================================================
bool VisualOdometry::tryRelocalize() {
    // 先搜索当前子地图，再按新旧顺序搜索历史子地图。候选总数设上限，
    // 防止连续丢失时 Atlas 越大、单帧重定位开销越高。
    std::vector<std::pair<unsigned long, Frame::Ptr>> candidates;
    constexpr int kMaxRelocTries = 60;
    auto append_submap = [&](const Submap& submap) {
        if ((int)candidates.size() >= kMaxRelocTries) return;
        auto kfs = submap.map->getAllKeyFrames();
        int per_map = 0;
        for (auto kf_it = kfs.rbegin();
             kf_it != kfs.rend() && per_map < 30 && (int)candidates.size() < kMaxRelocTries;
             ++kf_it, ++per_map) {
            candidates.emplace_back(submap.id, *kf_it);
        }
    };
    const auto* active = atlas_->activeSubmap();
    if (active) append_submap(*active);
    // 断开的子地图优先尝试回到已有全局地图，恢复可用于 ATE 的全局位姿。
    for (auto it = atlas_->submaps().rbegin(); it != atlas_->submaps().rend(); ++it) {
        if ((!active || it->id != active->id) && it->connected) append_submap(*it);
    }
    for (auto it = atlas_->submaps().rbegin(); it != atlas_->submaps().rend(); ++it) {
        if ((!active || it->id != active->id) && !it->connected) append_submap(*it);
    }
    if (candidates.empty()) return false;

    // LK 模式：当前帧可能无描述子，重定位前先提取 ORB
    if (curr_frame_->descriptors.empty())
        feature_matcher_.extract(curr_frame_);

    int best_inliers = 0;
    SE3 best_pose;
    Frame::Ptr best_kf;
    unsigned long best_submap_id = 0;

    // 对单个关键帧做 PnP 匹配，内点达标(20)即返回 true
    double best_ratio = 0.0;
    double best_rmse = std::numeric_limits<double>::infinity();
    auto try_kf = [&](unsigned long submap_id, const Frame::Ptr& kf) -> bool {
        auto matches = feature_matcher_.match(kf, curr_frame_, cfg_.match_ratio, true);
        // 重定位用较低门槛（min_matches_init=100 是初始化专用，RANSAC 后常达不到）
        if ((int)matches.size() < 30) return false;

        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        for (auto& m : matches) {
            auto& mp = kf->map_points[m.queryIdx];
            if (mp) {
                pts3d.emplace_back((float)mp->pos_w.x(), (float)mp->pos_w.y(), (float)mp->pos_w.z());
                pts2d.push_back(curr_frame_->keypoints[m.trainIdx].pt);
            }
        }
        if (pts3d.size() < 10) return false;

        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        bool ok = cv::solvePnPRansac(pts3d, pts2d, camera_->K(),
                                     cv::Mat(), rvec, tvec,
                                     false, 100, cfg_.ransac_pixel_threshold, 0.99, inliers);
        const double ratio = pts3d.empty() ? 0.0 : (double)inliers.size() / pts3d.size();
        const double rmse = ok
            ? pnpReprojectionRmse(pts3d, pts2d, rvec, tvec, inliers)
            : std::numeric_limits<double>::infinity();
        const int reloc_min_inliers = std::max(20, cfg_.pnp_min_inliers);
        if (ok && (int)inliers.size() >= reloc_min_inliers
            && ratio >= std::max(0.4, cfg_.pnp_min_inlier_ratio)
            && rmse <= cfg_.pnp_max_rmse
            && (int)inliers.size() > best_inliers) {
            best_inliers = (int)inliers.size();
            best_pose = matToSE3(cv::Mat(rvec), tvec);
            best_kf = kf;
            best_submap_id = submap_id;
            best_ratio = ratio;
            best_rmse = rmse;
        }
        return best_inliers >= reloc_min_inliers;
    };

    // 最近关键帧优先；如果当前子地图失效，继续尝试历史子地图。
    for (const auto& [submap_id, kf] : candidates) {
        if (try_kf(submap_id, kf)) break;
    }

    if (best_inliers >= 20) {
        atlas_->activate(best_submap_id);
        map_ = atlas_->activeMap();
        curr_frame_->pose_cw = best_pose;
        ref_frame_ = best_kf;
        status_.tracking_valid = true;
        status_.pose_method = "RELOCALIZE";
        status_.inlier_ratio = best_ratio;
        status_.pose_rmse = best_rmse;
        LOG_INFO("Relocalized in submap " << best_submap_id
                 << "! inliers=" << best_inliers);
        updateStatus(0, best_inliers, 0.0);
        return true;
    }

    LOG_INFO("Reloc failed, still LOST");
    updateStatus(0, 0, 0.0);
    return false;
}

// ============================================================
// 关键帧插入 + 三角化 + Local BA
// ============================================================
void VisualOdometry::insertKeyFrame() {
    const Frame::Ptr prev_kf = ref_frame_;
    map_->insertKeyFrame(curr_frame_);
    // 双目/RGB-D：当前帧有视差/深度的特征直接建点（绝对尺度）
    createMapPointsFromStereo(curr_frame_);
    // LK 模式：关键帧用干净的 ORB 特征重建（LK 关键点无方向，描述子无法与
    // 历史关键帧匹配），保证与上一关键帧的 ORB 匹配/三角化可靠；
    // 普通帧仍用 LK 光流跟踪（从关键帧 ORB 特征出发）
    if (cfg_.feature_method == 1)
        feature_matcher_.extract(curr_frame_);
    triangulateNewPoints(ref_frame_, curr_frame_,
        feature_matcher_.match(ref_frame_, curr_frame_, cfg_.match_ratio, true));
    ref_frame_ = curr_frame_;
    last_kf_frame_id_ = curr_frame_->id;   // 更新关键帧冷却基准

    // 定期清理观测不足的地图点（每 20 个关键帧一次），防止地图无限增长
    if (map_->keyFrameCount() % 20 == 0)
        map_->cullMapPoints(2);

    LOG_INFO("New KF. mp=" << map_->mapPointCount());

    // 共视图滑动窗口（按共视地图点数选帧）+ Local BA
    std::vector<Frame::Ptr> window = selectLocalWindow(cfg_.local_window_size);
    if (cfg_.enable_local_ba)
        Optimizer::localBundleAdjustment(camera_, map_, window, cfg_.local_ba_iterations);

    // Local BA 和新点关联完成后冻结相邻 KF 测量；后续位姿图不得从已优化
    // 轨迹重算它，否则会把上一次闭环结果误当成新的里程计观测。
    if (prev_kf) {
        std::set<unsigned long> prev_mp_ids;
        for (const auto& mp : prev_kf->map_points)
            if (mp) prev_mp_ids.insert(mp->id);
        int covisibility = 0;
        for (const auto& mp : curr_frame_->map_points)
            if (mp && prev_mp_ids.count(mp->id)) covisibility++;
        odometry_edges_.push_back({
            prev_kf->id, curr_frame_->id,
            prev_kf->pose_cw * curr_frame_->pose_cw.inverse(),
            1.0 + std::log2(1.0 + covisibility)});
    }

    // ============================================================
    // Phase 2 回环钩子：新关键帧入词袋数据库；每 N 个关键帧检测一次。
    // 放在 Local BA 之后，避免两处位姿修改互相干扰。
    // ============================================================
    if (loop_closure_enabled_ && loop_closure_) {
        loop_closure_->addKeyFrame(curr_frame_);
        if (map_->keyFrameCount() % (unsigned long)std::max(1, cfg_.detection_interval) == 0) {
            // 回环校正冷却：上次校正后至少间隔 N 帧才允许再次校正。
            // 同一区域会被词袋反复命中（分数高），连续校正会让 S_global
            // 叠加冲突、轨迹被反复拉扯变形。
            const bool cooled = curr_frame_->id - last_loop_kf_id_
                >= (unsigned long)cfg_.loop_cooldown_frames;
            if (cooled) {
                auto cand = loop_closure_->detectLoop(curr_frame_);
                if (cand) {
                    // DBoW3 数据库跨 Atlas 子地图缓存；当前尚未实现跨尺度
                    // 子地图融合，不能把另一张 map 的候选伪装成本图闭环。
                    if (!map_->getKeyFrame(cand->id)) {
                        LOG_WARN("LoopClosure: cross-submap candidate kf#"
                                 << cand->id << " rejected");
                        updateStatus(status_.matches, status_.inliers, status_.parallax);
                        return;
                    }
                    SE3 T_loop_curr;
                    if (loop_closure_->verifyLoop(curr_frame_, cand, T_loop_curr))
                        handleLoopCorrection(T_loop_curr, curr_frame_, cand);
                }
            }
        }
    }

    updateStatus(status_.matches, status_.inliers, status_.parallax);
}

// ============================================================
// Phase 2 回环校正：位姿图优化 → 地图点同步 → 全局 BA → 逐帧轨迹同步
// ============================================================
void VisualOdometry::handleLoopCorrection(const SE3& T_loop_curr,
                                          const Frame::Ptr& kf_curr,
                                          const Frame::Ptr& kf_loop) {
    // 1. 保存优化前位姿。旧实现先对全部 KF/MP 施加同一 Sim3，这只会更换
    //    全局坐标系，完全不改变 loop ↔ curr 的相对残差；这里直接让位姿图
    //    在固定首帧的前提下沿轨迹分配闭环误差。
    auto all_kfs = map_->getAllKeyFrames();
    std::unordered_map<unsigned long, SE3> old_pose_cw;
    for (const auto& kf : all_kfs) old_pose_cw.emplace(kf->id, kf->pose_cw);

    // 2. 累积保留历史回环边，避免后一次闭环丢掉前一次约束。
    LoopEdge le;
    le.a = kf_loop->id;
    le.b = kf_curr->id;
    le.T_rel = T_loop_curr;
    le.weight = 10.0;  // 回环约束高置信
    loop_edges_.push_back(le);
    if (!Optimizer::poseGraphOptimization(map_, odometry_edges_, loop_edges_)) {
        loop_edges_.pop_back();
        LOG_WARN("Loop correction skipped: pose graph backend unavailable or constraints invalid");
        return;
    }

    // 3. 位姿图只优化关键帧。地图点按其最早观测关键帧的位姿增量同步，
    //    保持该关键帧中的局部坐标不变，再交给全局 BA 做小量精修。
    std::unordered_map<unsigned long, unsigned long> mp_reference_kf;
    for (const auto& kf : all_kfs) {
        for (const auto& mp : kf->map_points) {
            if (mp) mp_reference_kf.emplace(mp->id, kf->id);
        }
    }
    for (auto& mp : map_->getAllMapPoints()) {
        auto ref = mp_reference_kf.find(mp->id);
        if (ref == mp_reference_kf.end()) continue;
        auto old = old_pose_cw.find(ref->second);
        auto kf = map_->getKeyFrame(ref->second);
        if (old == old_pose_cw.end() || !kf) continue;
        const SE3 correction = kf->pose_cw.inverse() * old->second;
        mp->pos_w = correction * mp->pos_w;
    }

    // 4. 全局 BA：地图点固定、优化关键帧位姿；使用配置的迭代次数。
    Optimizer::globalBundleAdjustment(camera_, map_, cfg_.global_ba_iterations);

    // 5. 非关键帧不在位姿图中。将相邻关键帧的最终校正插值后施加到完整
    //    T_cw 轨迹，既保留朝向供 TUM 输出，也避免整段只用一个刚体变换。
    std::vector<std::pair<unsigned long, SE3>> corrections;
    corrections.reserve(all_kfs.size());
    for (const auto& kf : all_kfs) {
        auto old = old_pose_cw.find(kf->id);
        if (old == old_pose_cw.end()) continue;
        corrections.emplace_back(kf->id, kf->pose_cw.inverse() * old->second);
    }
    for (size_t i = 0; i < pose_trajectory_.size() && i < traj_frame_ids_.size(); i++) {
        const unsigned long frame_id = traj_frame_ids_[i];
        auto upper = std::lower_bound(
            corrections.begin(), corrections.end(), frame_id,
            [](const auto& item, unsigned long id) { return item.first < id; });

        SE3 correction;
        if (upper != corrections.end() && upper->first == frame_id) {
            // 关键帧直接采用最终优化结果，避免历史 Local BA 已经让缓存轨迹
            // 与 old_pose_cw 存在微小偏差。
            auto kf = map_->getKeyFrame(frame_id);
            if (kf) {
                pose_trajectory_[i] = kf->pose_cw;
                continue;
            }
            correction = upper->second;
        } else if (upper == corrections.begin()) {
            correction = upper->second;
        } else if (upper == corrections.end()) {
            correction = corrections.back().second;
        } else {
            const auto& [id1, c1] = *upper;
            const auto& [id0, c0] = *(upper - 1);
            const double alpha = static_cast<double>(frame_id - id0) /
                                 static_cast<double>(id1 - id0);
            correction.q = c0.q.slerp(alpha, c1.q).normalized();
            correction.t = (1.0 - alpha) * c0.t + alpha * c1.t;
        }
        const SE3 Twc_new = correction * pose_trajectory_[i].inverse();
        pose_trajectory_[i] = Twc_new.inverse();
    }

    loop_closure_count_++;
    last_loop_kf_id_ = kf_curr->id;  // 更新回环校正冷却基准
    LOG_INFO("Loop closed! kf#" << kf_loop->id << " -> kf#" << kf_curr->id
             << " (total " << loop_closure_count_ << ")");
}

// ============================================================
// 共视图滑动窗口：与当前关键帧共视地图点最多的帧 + 当前帧
// ============================================================
std::vector<Frame::Ptr> VisualOdometry::selectLocalWindow(int n) const {
    std::vector<Frame::Ptr> window;
    auto all_kfs = map_->getAllKeyFrames();
    if (all_kfs.empty() || !curr_frame_) return window;

    // 当前帧引用的地图点集合
    std::set<unsigned long> curr_mps;
    for (auto& mp : curr_frame_->map_points)
        if (mp) curr_mps.insert(mp->id);

    // 统计每个关键帧与当前帧的共视点数量
    struct Candidate { Frame::Ptr kf; int cov; };
    std::vector<Candidate> cands;
    for (auto& kf : all_kfs) {
        if (kf->id == curr_frame_->id) continue;
        int cov = 0;
        for (auto& mp : kf->map_points)
            if (mp && curr_mps.count(mp->id)) cov++;
        cands.push_back({kf, cov});
    }
    // C++23 ranges：按共视点数量降序（投影 &Candidate::cov，免手写比较器）
    std::ranges::sort(cands, std::greater<>{}, &Candidate::cov);

    // 窗口 = 当前帧 + 共视最多的前 n-3 帧（预留 2 个尺度锚位）
    window.push_back(curr_frame_);
    for (auto& c : cands) {
        if ((int)window.size() >= n - 2) break;
        if (c.cov >= 2) window.push_back(c.kf);
    }

    // 兜底：共视不足时退化为按时间取最近 n 帧
    if (window.size() < 2) {
        window.clear();
        int start = std::max(0, (int)all_kfs.size() - n);
        for (int i = start; i < (int)all_kfs.size(); i++)
            window.push_back(all_kfs[i]);
    }

    // 强制加入全局最早的两个关键帧（id 0/1，初始化尺度基准）：
    // BA 固定它们后，每个滑动窗口共用同一基线，避免窗口间锚定帧不同
    // 导致的尺度漂移。
    for (auto& kf : all_kfs) {
        if (kf->id == 0 || kf->id == 1) {
            if (std::find(window.begin(), window.end(), kf) == window.end())
                window.push_back(kf);
        }
    }

    // 按 id 升序，最早帧在 index 0/1（BA 中固定，尺度锚定）
    std::ranges::sort(window, {}, [](const Frame::Ptr& f) { return f->id; });
    return window;
}

// ============================================================
// 辅助
// ============================================================
SE3 VisualOdometry::matToSE3(const cv::Mat& R, const cv::Mat& t) {
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

void VisualOdometry::triangulateNewPoints(
    const Frame::Ptr& f1, const Frame::Ptr& f2,
    const std::vector<cv::DMatch>& matches) {

    cv::Mat K = camera_->K();
    int cnt = 0;
    for (auto& m : matches) {
        if (f1->map_points[m.queryIdx] == nullptr) {
            auto mp = MapPoint::create(
                map_->nextMapPointId(),
                Vec2(f1->keypoints[m.queryIdx].pt.x, f1->keypoints[m.queryIdx].pt.y),
                Vec2(f2->keypoints[m.trainIdx].pt.x, f2->keypoints[m.trainIdx].pt.y),
                f1->pose_cw, f2->pose_cw, K);
            Vec3 pc = f1->pose_cw * mp->pos_w;
            if (pc.z() > 0) {
                mp->observed_count = 2;  // 初始被 f1、f2 两个关键帧观测
                map_->insertMapPoint(mp);
                f1->map_points[m.queryIdx] = mp;
                f2->map_points[m.trainIdx] = mp;
                cnt++;
            }
        }
    }
    LOG_INFO("Triangulated " << cnt << " points");
}

bool VisualOdometry::needNewKeyFrame() const {
    if (!ref_frame_ || !curr_frame_) return false;
    // T_cw 的平移没有可比性，必须用相机在世界系中的位姿 T_wc = T_cw^-1 计算位移
    SE3 Twc_cur = curr_frame_->pose_cw.inverse();
    SE3 Twc_ref = ref_frame_->pose_cw.inverse();
    double dtrans = (Twc_cur.t - Twc_ref.t).norm();
    // 相对旋转角：q_rel = q_cur * q_ref^-1，最小表示 = 2*acos(|w|)，处理 q 与 -q 等价
    Eigen::Quaterniond q_rel = curr_frame_->pose_cw.q * ref_frame_->pose_cw.q.inverse();
    double drot = 2.0 * std::acos(
        std::clamp(std::abs(q_rel.w()), 0.0, 1.0));
    // 运动阈值 + 匹配衰减阈值：内点过少说明地图不足/视角变化大，强制补充关键帧
    bool weak_match = status_.inliers < cfg_.keyframe_min_inliers;
    // 冷却：weak_match 触发需与上一关键帧间隔足够帧数，防止"关键帧风暴"
    // （一旦地图质量差，无间隔限制会每帧插关键帧 → BA/重定位越来越慢 → 卡死）
    if (weak_match &&
        curr_frame_->id - last_kf_frame_id_ < (unsigned long)cfg_.min_keyframe_interval)
        weak_match = false;
    // 平移阈值按传感器类型分派：双目/RGB-D 有绝对尺度（真实帧间位移大），
    // 单目尺度归一化后位移小——共用阈值会导致双目每帧插 KF
    double kf_trans = camera_->hasPerFrameDepth()
        ? cfg_.keyframe_translation_stereo : cfg_.keyframe_translation;
    const bool max_interval = curr_frame_->id - last_kf_frame_id_
        >= (unsigned long)cfg_.max_keyframe_interval;
    return dtrans > kf_trans || drot > cfg_.keyframe_rotation
        || weak_match || max_interval;
}

std::vector<Vec3> VisualOdometry::getTrajectory() const {
    std::vector<Vec3> trajectory;
    trajectory.reserve(pose_trajectory_.size());
    for (const auto& pose_cw : pose_trajectory_)
        trajectory.push_back(pose_cw.camera_position());
    return trajectory;
}

} // namespace vslam
