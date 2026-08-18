#include "vslam/vo.h"
#include "vslam/pose_gate.h"
#include "vslam/optimizer.h"
#include "perf_monitor.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <Eigen/SVD>

#include <set>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <ranges>
#include <limits>
#include <numeric>

namespace vslam {

namespace {

/// §6.4（M2.3 遗留清理）：BackendCommitter 结果 → 调度统计结果
TaskOutcome toTaskOutcome(CommitStatus s) {
    switch (s) {
        case CommitStatus::COMMITTED: return TaskOutcome::Committed;
        case CommitStatus::STALE:     return TaskOutcome::Stale;
        case CommitStatus::INVALID:   return TaskOutcome::Invalid;
        case CommitStatus::NOT_FOUND: return TaskOutcome::NotFound;
    }
    return TaskOutcome::Invalid;
}

/// M1.4：从 VOConfig 快照跟踪相关字段到 FrontendTracker 配置
///（构造函数调用一次；运行期 cfg_ 的跟踪字段不改变，快照一致）
TrackerConfig buildTrackerConfig(const VOConfig& cfg) {
    TrackerConfig tc;
    tc.num_features = cfg.num_features;
    tc.scale_factor = cfg.scale_factor;
    tc.pyramid_levels = cfg.pyramid_levels;
    tc.orb_max_bands = cfg.orb_max_bands;
    tc.stereo_reverse_prune = cfg.stereo_reverse_prune;
    tc.match_ratio = cfg.match_ratio;
    tc.ransac_pixel_threshold = cfg.ransac_pixel_threshold;
    tc.min_matches_track = cfg.min_matches_track;
    tc.pnp_min_inliers = cfg.pnp_min_inliers;
    tc.pnp_min_inlier_ratio = cfg.pnp_min_inlier_ratio;
    tc.pnp_max_rmse = cfg.pnp_max_rmse;
    tc.max_frame_translation = cfg.max_frame_translation;
    tc.max_frame_rotation = cfg.max_frame_rotation;
    tc.stereo_min_depth = cfg.stereo_min_depth;
    tc.stereo_max_depth = cfg.stereo_max_depth;
    tc.stereo_min_points = cfg.stereo_min_points;
    tc.rigid_min_inliers = cfg.rigid_min_inliers;
    tc.rigid_min_inlier_ratio = cfg.rigid_min_inlier_ratio;
    tc.rigid_ransac_threshold = cfg.rigid_ransac_threshold;
    tc.rigid_max_rmse = cfg.rigid_max_rmse;
    tc.guided_match = cfg.guided_match;
    tc.guided_search_radius_px = cfg.guided_search_radius_px;
    tc.local_map_tracking = cfg.local_map_tracking;
    tc.local_map_min_shared = cfg.local_map_min_shared;
    tc.local_map_max_points = cfg.local_map_max_points;
    tc.local_map_search_radius_px = cfg.local_map_search_radius_px;
    tc.keyframe_translation = cfg.keyframe_translation;
    tc.keyframe_translation_stereo = cfg.keyframe_translation_stereo;
    tc.keyframe_max_count = cfg.keyframe_max_count;
    tc.keyframe_rotation = cfg.keyframe_rotation;
    tc.keyframe_min_inliers = cfg.keyframe_min_inliers;
    tc.min_keyframe_interval = cfg.min_keyframe_interval;
    tc.max_keyframe_interval = cfg.max_keyframe_interval;
    return tc;
}

bool triangulateNormalizedPoint(const Vec2& p1, const Vec2& p2,
                                const Mat33& R, const Vec3& t,
                                Vec3& point_c1) {
    Eigen::Matrix4d A;
    const Eigen::Vector4d P1_r0(1.0, 0.0, 0.0, 0.0);
    const Eigen::Vector4d P1_r1(0.0, 1.0, 0.0, 0.0);
    const Eigen::Vector4d P1_r2(0.0, 0.0, 1.0, 0.0);
    Eigen::Matrix<double, 1, 4> P2[3];
    P2[0].head<3>() = R.row(0);
    P2[0](3) = t.x();
    P2[1].head<3>() = R.row(1);
    P2[1](3) = t.y();
    P2[2].head<3>() = R.row(2);
    P2[2](3) = t.z();

    A.row(0) = p1.x() * P1_r2.transpose() - P1_r0.transpose();
    A.row(1) = p1.y() * P1_r2.transpose() - P1_r1.transpose();
    A.row(2) = p2.x() * P2[2] - P2[0];
    A.row(3) = p2.y() * P2[2] - P2[1];

    const Eigen::JacobiSVD<Eigen::Matrix4d> svd(
        A, Eigen::ComputeFullV);
    const Eigen::Vector4d X_h = svd.matrixV().col(3);
    if (!X_h.allFinite() || std::abs(X_h.w()) < 1e-12) return false;
    point_c1 = X_h.head<3>() / X_h.w();
    return point_c1.allFinite();
}

bool sampleRgb(const cv::Mat& image, const cv::Point2f& pixel,
               uint8_t& r, uint8_t& g, uint8_t& b) {
    if (image.empty() || image.depth() != CV_8U ||
        !std::isfinite(pixel.x) || !std::isfinite(pixel.y))
        return false;
    const int x = cvRound(pixel.x);
    const int y = cvRound(pixel.y);
    if (x < 0 || y < 0 || x >= image.cols || y >= image.rows) return false;

    if (image.channels() == 1) {
        const uint8_t value = image.at<uint8_t>(y, x);
        r = value; g = value; b = value;
        return true;
    }
    if (image.channels() == 3) {
        const cv::Vec3b value = image.at<cv::Vec3b>(y, x);  // BGR
        r = value[2]; g = value[1]; b = value[0];
        return true;
    }
    if (image.channels() == 4) {
        const cv::Vec4b value = image.at<cv::Vec4b>(y, x);  // BGRA
        r = value[2]; g = value[1]; b = value[0];
        return true;
    }
    return false;
}

} // namespace

MonocularInitializationQuality assessMonocularInitialization(
    const std::vector<Vec2>& normalized_ref,
    const std::vector<Vec2>& normalized_curr,
    const Mat33& relative_rotation,
    const Vec3& relative_translation,
    double min_parallax_rad,
    double min_positive_depth_ratio) {
    MonocularInitializationQuality quality;
    if (normalized_ref.size() != normalized_curr.size() ||
        normalized_ref.empty() || !relative_rotation.allFinite() ||
        !relative_translation.allFinite() ||
        // recoverPose 通常将 t 归一化为 1；这里仍拒绝明确的近零输入，
        // 以保护纯函数调用和异常求解结果。
        relative_translation.norm() <= 1e-6) {
        return quality;
    }

    std::vector<double> parallax_angles;
    parallax_angles.reserve(normalized_ref.size());
    int positive_depths = 0;
    for (size_t i = 0; i < normalized_ref.size(); ++i) {
        const Vec3 ray1(normalized_ref[i].x(), normalized_ref[i].y(), 1.0);
        const Vec3 ray2_c2(normalized_curr[i].x(), normalized_curr[i].y(), 1.0);
        if (!ray1.allFinite() || !ray2_c2.allFinite()) continue;

        // 两条光线统一表达在第一帧坐标系：d1 与 R^T d2 的夹角就是
        // 真实三角化角。纯旋转/近零基线时该角接近 0。
        const Vec3 d1 = ray1.normalized();
        const Vec3 d2 = (relative_rotation.transpose() * ray2_c2).normalized();
        const double cos_angle = std::clamp(d1.dot(d2), -1.0, 1.0);
        const double angle = std::acos(cos_angle);
        if (!std::isfinite(angle)) continue;
        Vec3 point_c1;
        if (!triangulateNormalizedPoint(normalized_ref[i], normalized_curr[i],
                                        relative_rotation, relative_translation,
                                        point_c1)) {
            continue;
        }
        parallax_angles.push_back(angle);
        const Vec3 point_c2 = relative_rotation * point_c1 + relative_translation;
        if (point_c1.z() > 1e-9 && point_c2.z() > 1e-9) positive_depths++;
    }

    if (parallax_angles.empty()) return quality;
    const auto middle = parallax_angles.begin() + parallax_angles.size() / 2;
    std::nth_element(parallax_angles.begin(), middle, parallax_angles.end());
    quality.parallax_rad = *middle;
    quality.triangulated_points = static_cast<int>(parallax_angles.size());
    quality.positive_depth_ratio = quality.triangulated_points > 0
        ? static_cast<double>(positive_depths) / quality.triangulated_points : 0.0;
    quality.accepted = quality.parallax_rad >= min_parallax_rad &&
                       quality.positive_depth_ratio >= min_positive_depth_ratio;
    return quality;
}

SE3 rebaseAnchoredFramePose(
    const SE3& frame_pose_cs,
    const SE3& old_ref_pose_cs,
    const SE3& new_ref_pose_cs) {
    const SE3 T_ca = frame_pose_cs * old_ref_pose_cs.inverse();
    return T_ca * new_ref_pose_cs;
}

SE3 rebaseTrajectoryAnchor(
    const SE3& T_ca_old,
    const SE3& old_anchor_pose_cs,
    const SE3& new_anchor_pose_cs) {
    return T_ca_old * old_anchor_pose_cs * new_anchor_pose_cs.inverse();
}

SE3 rebaseTrajectoryForSubmapAnchor(
    const SE3& T_ca_old,
    const SE3& anchor_pose_cs,
    const SE3& old_T_ws,
    const SE3& new_T_ws,
    double alpha) {
    alpha = std::clamp(alpha, 0.0, 1.0);
    Eigen::Quaterniond q = old_T_ws.q.slerp(alpha, new_T_ws.q).normalized();
    const Vec3 t = (1.0 - alpha) * old_T_ws.t + alpha * new_T_ws.t;
    const SE3 interpolated_T_ws(q, t);
    const SE3 local_pose_cw = T_ca_old * anchor_pose_cs;
    const SE3 desired_world_cw = local_pose_cw * interpolated_T_ws.inverse();
    return desired_world_cw * new_T_ws * anchor_pose_cs.inverse();
}

double submapTrajectoryCorrectionAlpha(
    unsigned long frame_id,
    unsigned long segment_start_frame_id,
    unsigned long first_frame_id,
    unsigned long endpoint_frame_id) {
    if (frame_id < segment_start_frame_id) return 0.0;
    if (endpoint_frame_id <= first_frame_id) return 1.0;
    const double span = static_cast<double>(
        endpoint_frame_id - first_frame_id);
    return std::clamp(static_cast<double>(frame_id - first_frame_id) / span,
                      0.0, 1.0);
}

SE3 composeAnchoredWorldPose(
    const SE3& T_ca,
    const SE3& anchor_pose_cs,
    const SE3& T_ws) {
    return T_ca * (anchor_pose_cs * T_ws.inverse());
}

void appendFormalObservations(
    const Frame::Ptr& keyframe,
    std::vector<ObservationState>& observations,
    const std::unordered_set<MapPointId>* allowed_points = nullptr) {
    if (!keyframe) return;
    for (size_t i = 0; i < keyframe->map_points.size(); ++i) {
        const auto& map_point = keyframe->map_points[i];
        if (!map_point || i > std::numeric_limits<FeatureIndex>::max()) continue;
        const Observation observation{
            keyframe->id, static_cast<FeatureIndex>(i)};
        if (!map_point->hasObservation(observation)) continue;
        if (allowed_points && !allowed_points->contains(map_point->id)) continue;

        ObservationState state;
        state.keyframe_id = keyframe->id;
        state.feature_index = static_cast<FeatureIndex>(i);
        state.map_point_id = map_point->id;
        if (i < keyframe->keypoints.size()) {
            state.pixel = Vec2(keyframe->keypoints[i].pt.x,
                               keyframe->keypoints[i].pt.y);
        }
        if (i < keyframe->pts_c.size() && keyframe->pts_c[i].allFinite() &&
            keyframe->pts_c[i].z() > 0.0) {
            state.camera_point = keyframe->pts_c[i];
        }
        observations.push_back(std::move(state));
    }
}

VOConfig VOConfig::fromYaml(const std::string& path) {
    VOConfig cfg;
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (auto f = root["Feature"]) {
            if (f["num_features"])    cfg.num_features           = f["num_features"].as<int>();
            if (f["scale_factor"])    cfg.scale_factor           = f["scale_factor"].as<double>();
            if (f["pyramid_levels"])  cfg.pyramid_levels         = f["pyramid_levels"].as<int>();
            if (f["orb_max_bands"])   cfg.orb_max_bands          = f["orb_max_bands"].as<int>();
            if (f["match_ratio"])     cfg.match_ratio            = f["match_ratio"].as<double>();
            if (f["ransac_threshold"]) cfg.ransac_pixel_threshold = f["ransac_threshold"].as<double>();
        }
        if (auto r = root["Runtime"]) {
            if (r["opencv_threads"]) cfg.opencv_threads = r["opencv_threads"].as<int>();
            if (r["async_backend"])  cfg.async_backend  = r["async_backend"].as<bool>();
            if (r["rng_seed"])       cfg.rng_seed       = r["rng_seed"].as<int>();
            if (r["max_cpu_cores"])
                cfg.runtime_resources.max_cpu_cores = r["max_cpu_cores"].as<size_t>();
            if (r["max_rss_mb"])
                cfg.runtime_resources.max_rss_mb = r["max_rss_mb"].as<size_t>();
            if (r["backend_reserved_cores"])
                cfg.runtime_resources.backend_reserved_cores =
                    r["backend_reserved_cores"].as<size_t>();
            if (r["enforce_cpu_affinity"])
                cfg.runtime_resources.enforce_cpu_affinity =
                    r["enforce_cpu_affinity"].as<bool>();
            if (r["pin_backend_worker"])
                cfg.runtime_resources.pin_backend_worker =
                    r["pin_backend_worker"].as<bool>();
            if (r["max_backend_task_age_ms"])
                cfg.max_backend_task_age_ms =
                    r["max_backend_task_age_ms"].as<int>();
            if (r["reuse_keyframe_matches"])
                cfg.reuse_keyframe_matches = r["reuse_keyframe_matches"].as<bool>();
            if (r["stereo_reverse_prune"])
                cfg.stereo_reverse_prune = r["stereo_reverse_prune"].as<bool>();
            if (r["use_raw_stereo_gray"])
                cfg.use_raw_stereo_gray = r["use_raw_stereo_gray"].as<bool>();
            if (r["copy_gray_before_clahe"])
                cfg.copy_gray_before_clahe = r["copy_gray_before_clahe"].as<bool>();
        }
        // M2.2 遗留清理：§6.2 MapBudget 段（首版参数；缺省保持默认值）
        if (auto b = root["MapBudget"]) {
            if (b["max_active_keyframes"])    cfg.map_budget.max_active_keyframes = b["max_active_keyframes"].as<size_t>();
            if (b["max_active_points"])       cfg.map_budget.max_active_points    = b["max_active_points"].as<size_t>();
            if (b["max_descriptor_mb"])       cfg.map_budget.max_descriptor_mb    = b["max_descriptor_mb"].as<size_t>();
            if (b["max_snapshot_mb"])         cfg.map_budget.max_snapshot_mb      = b["max_snapshot_mb"].as<size_t>();
            if (b["max_total_estimated_mb"])  cfg.map_budget.max_total_estimated_mb = b["max_total_estimated_mb"].as<size_t>();
            if (b["active_dense_keyframes"]) cfg.map_budget.active_dense_keyframes = b["active_dense_keyframes"].as<size_t>();
            if (b["max_historical_anchors"]) cfg.map_budget.max_historical_anchors = b["max_historical_anchors"].as<size_t>();
            if (b["historical_anchor_stride"]) cfg.map_budget.historical_anchor_stride = b["historical_anchor_stride"].as<size_t>();
        }
        if (auto v = root["VO"]) {
            if (v["method"])              cfg.feature_method       = v["method"].as<int>();
            if (v["min_parallax"])        cfg.min_parallax           = v["min_parallax"].as<double>();
            if (v["min_matches_init"])    cfg.min_matches_init     = v["min_matches_init"].as<int>();
            if (v["min_init_inliers"])    cfg.min_init_inliers     = v["min_init_inliers"].as<int>();
            if (v["min_matches_track"])   cfg.min_matches_track    = v["min_matches_track"].as<int>();
            if (v["pnp_min_inliers"])     cfg.pnp_min_inliers      = v["pnp_min_inliers"].as<int>();
            if (v["pnp_min_inlier_ratio"]) cfg.pnp_min_inlier_ratio = v["pnp_min_inlier_ratio"].as<double>();
            if (v["pnp_max_rmse"])        cfg.pnp_max_rmse         = v["pnp_max_rmse"].as<double>();
            if (v["max_tracking_failures"]) cfg.max_tracking_failures = v["max_tracking_failures"].as<int>();
            if (v["max_relocalize_frames"]) cfg.max_relocalize_frames = v["max_relocalize_frames"].as<int>();
            if (v["max_relocalization_candidates"])
                cfg.max_relocalization_candidates =
                    v["max_relocalization_candidates"].as<int>();
            if (v["max_frame_translation"]) cfg.max_frame_translation = v["max_frame_translation"].as<double>();
            if (v["max_frame_rotation"]) cfg.max_frame_rotation = v["max_frame_rotation"].as<double>();
            if (v["keyframe_translation"]) cfg.keyframe_translation = v["keyframe_translation"].as<double>();
            if (v["keyframe_rotation"])   cfg.keyframe_rotation    = v["keyframe_rotation"].as<double>();
            if (v["keyframe_min_inliers"]) cfg.keyframe_min_inliers = v["keyframe_min_inliers"].as<int>();
            if (v["keyframe_max_count"])   cfg.keyframe_max_count   = v["keyframe_max_count"].as<int>();
            if (v["min_keyframe_interval"]) cfg.min_keyframe_interval = v["min_keyframe_interval"].as<int>();
            if (v["max_keyframe_interval"]) cfg.max_keyframe_interval = v["max_keyframe_interval"].as<int>();
            if (v["guided_match"])             cfg.guided_match            = v["guided_match"].as<bool>();
            if (v["guided_search_radius_px"])  cfg.guided_search_radius_px = v["guided_search_radius_px"].as<double>();
            if (v["local_map_tracking"])       cfg.local_map_tracking      = v["local_map_tracking"].as<bool>();
            if (v["local_map_min_shared"])     cfg.local_map_min_shared    = v["local_map_min_shared"].as<int>();
            if (v["local_map_max_points"])     cfg.local_map_max_points    = v["local_map_max_points"].as<int>();
            if (v["local_map_search_radius_px"]) cfg.local_map_search_radius_px = v["local_map_search_radius_px"].as<double>();
            if (v["local_map_rescue"]) cfg.local_map_rescue = v["local_map_rescue"].as<bool>();
            if (v["local_map_rescue_max_points"])
                cfg.local_map_rescue_max_points = v["local_map_rescue_max_points"].as<int>();
            if (v["local_map_rescue_radius_px"])
                cfg.local_map_rescue_radius_px = v["local_map_rescue_radius_px"].as<double>();
        }
        if (auto o = root["Optimizer"]) {
            if (o["local_window_size"])   cfg.local_window_size   = o["local_window_size"].as<int>();
            if (o["local_ba_iterations"]) cfg.local_ba_iterations = o["local_ba_iterations"].as<int>();
            if (o["local_ba_passes"]) cfg.local_ba_passes = o["local_ba_passes"].as<int>();
            if (o["local_ba_max_points"])
                cfg.local_ba_max_points = o["local_ba_max_points"].as<size_t>();
            if (o["local_ba_max_correction"])
                cfg.local_ba_max_correction = o["local_ba_max_correction"].as<double>();
            if (o["enable_local_ba"])     cfg.enable_local_ba     = o["enable_local_ba"].as<bool>();
            if (o["pose_graph_iterations"])
                cfg.pose_graph_iterations = o["pose_graph_iterations"].as<int>();
            if (o["pose_graph_max_anchors"])
                cfg.pose_graph_max_anchors = o["pose_graph_max_anchors"].as<int>();
            if (o["pose_graph_anchor_stride"])
                cfg.pose_graph_anchor_stride = o["pose_graph_anchor_stride"].as<int>();
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
            if (lc["retrieval_backend"])
                cfg.loop_retrieval_backend = lc["retrieval_backend"].as<std::string>();
            if (lc["compact_max_keyframes"])
                cfg.loop_compact_max_keyframes = lc["compact_max_keyframes"].as<size_t>();
            if (lc["vocab_path"])          cfg.vocab_path = lc["vocab_path"].as<std::string>();
            if (lc["min_score"])           cfg.min_score = lc["min_score"].as<double>();
            if (lc["temporal_window"])     cfg.temporal_window = lc["temporal_window"].as<int>();
            if (lc["detection_interval"])  cfg.detection_interval = lc["detection_interval"].as<int>();
            if (lc["pnp_inlier_ratio"])    cfg.pnp_inlier_ratio = lc["pnp_inlier_ratio"].as<double>();
            if (lc["min_loop_inliers"])    cfg.min_loop_inliers = lc["min_loop_inliers"].as<int>();
            if (lc["loop_cooldown_kfs"])   cfg.loop_cooldown_kfs = lc["loop_cooldown_kfs"].as<int>();
            // 兼容旧字段名 loop_cooldown_frames（帧计数）→ 按关键帧数换算（约 1/10）
            if (!lc["loop_cooldown_kfs"] && lc["loop_cooldown_frames"])
                cfg.loop_cooldown_kfs = lc["loop_cooldown_frames"].as<int>() / 10;
            if (lc["top_candidates"])      cfg.loop_top_candidates = lc["top_candidates"].as<int>();
            if (lc["mature_verification_limit"])
                cfg.loop_mature_verification_limit =
                    lc["mature_verification_limit"].as<int>();
            if (lc["verification_limit"])
                cfg.loop_verification_limit = lc["verification_limit"].as<int>();
            if (lc["position_prior_dist"]) cfg.loop_position_prior_dist = lc["position_prior_dist"].as<double>();
            if (lc["position_prior_gap"])  cfg.loop_position_prior_gap = lc["position_prior_gap"].as<int>();
            if (lc["region_max_keyframes"]) cfg.loop_region.max_keyframes = lc["region_max_keyframes"].as<size_t>();
            if (lc["region_max_points"]) cfg.loop_region.max_points = lc["region_max_points"].as<size_t>();
            if (lc["region_max_covisible"]) cfg.loop_region.max_covisible_keyframes = lc["region_max_covisible"].as<size_t>();
            if (lc["region_max_temporal"]) cfg.loop_region.max_temporal_neighbors = lc["region_max_temporal"].as<size_t>();
            if (lc["pnp_max_rmse"]) cfg.loop_region.max_reprojection_rmse = lc["pnp_max_rmse"].as<double>();
            if (lc["pnp_min_positive_depth_ratio"]) cfg.loop_region.min_positive_depth_ratio = lc["pnp_min_positive_depth_ratio"].as<double>();
            if (lc["pnp_min_grid_cells"]) cfg.loop_region.min_grid_cells = lc["pnp_min_grid_cells"].as<int>();
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
    : camera_(camera), cfg_(cfg), atlas_(std::make_shared<Atlas>()),
      relocalizer_(camera, cfg.num_features, cfg.scale_factor,
                   cfg.pyramid_levels, cfg.orb_max_bands),
      frontend_tracker_(camera, buildTrackerConfig(cfg)),
      local_mapper_(camera),
      map_budget_(cfg.map_budget),  // M2.2 遗留清理：§6.3 预算引擎配置绑定
      backend_scheduler_([this](BackendTask& task) { runBackendTask(task); }) {
    // 区域验证扩大的是历史几何覆盖范围，不得拥有一套更宽松的接受门。
    // 数量/比例/RANSAC 阈值始终绑定现有回环配置，YAML 只允许调资源上限。
    cfg_.loop_region.min_inliers = cfg_.min_loop_inliers;
    cfg_.loop_region.min_inlier_ratio = cfg_.pnp_inlier_ratio;
    cfg_.loop_region.ransac_pixel_threshold = cfg_.ransac_pixel_threshold;
    cfg_.loop_region.temporal_window = static_cast<size_t>(
        std::max(1, cfg_.temporal_window));
    const bool affinity_ok = RuntimeResources::configure(
        cfg_.runtime_resources, cfg_.opencv_threads, cfg_.async_backend);
    if (cfg_.runtime_resources.enforce_cpu_affinity && !affinity_ok) {
        LOG_WARN("Runtime CPU affinity could not be narrowed; continuing with OpenCV limit");
    }
    backend_scheduler_.setMaxTaskAgeMs(
        static_cast<double>(cfg_.max_backend_task_age_ms));
    if (cfg_.async_backend && cfg_.runtime_resources.pin_backend_worker) {
        const size_t cpus = RuntimeResources::allowedCpuCount();
        if (cpus > 1) backend_scheduler_.setWorkerCpuOrdinal(cpus - 1);
    }
    map_ = atlas_->createSubmap(SE3()).map;
    if (cfg_.opencv_threads > 0) {
        LOG_INFO("OpenCV threads limited to " << cv::getNumThreads());
    }
    // M1 确定性：固定全局 RNG（solvePnPRansac 等内部 RNG），配 deterministic.yaml
    if (cfg_.rng_seed != 0) cv::setRNGSeed((unsigned)cfg_.rng_seed);
    feature_matcher_.setParams(cfg_.num_features, cfg_.scale_factor,
                               cfg_.pyramid_levels, cfg_.orb_max_bands,
                               cfg_.stereo_reverse_prune);
    if (cfg_.feature_method != 0) {
        LOG_INFO("feature_method=" << cfg_.feature_method << " (LK 光流)");
    }
    // Phase 2：配置中启用回环且给出词典路径时自动加载
    if (cfg_.enable_loop_closure && !cfg_.vocab_path.empty())
        enableLoopClosure(cfg_.vocab_path);
    if (cfg_.async_backend) {
        backend_scheduler_.start();
        LOG_INFO("Async backend ENABLED (BA/loop on background thread)");
    }
}

VisualOdometry::~VisualOdometry() {
    if (cfg_.async_backend) backend_scheduler_.stop();
}

BudgetStatus VisualOdometry::mapBudgetStatus() const {
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    return map_budget_.evaluate(
        map_, static_cast<size_t>(std::max<long long>(0, mapSnapshotBytes())));
}

bool VisualOdometry::enableLoopClosure(const std::string& vocab_path) {
    if (!loop_closure_) loop_closure_ = std::make_unique<LoopClosure>();
    loop_closure_->setParams(cfg_.min_score, cfg_.temporal_window,
                             cfg_.min_loop_inliers, cfg_.pnp_inlier_ratio,
                             cfg_.ransac_pixel_threshold, camera_,
                             cfg_.loop_top_candidates,
                             cfg_.loop_position_prior_dist,
                             cfg_.loop_position_prior_gap,
                             cfg_.loop_region.max_reprojection_rmse,
                             cfg_.loop_region.min_positive_depth_ratio,
                             cfg_.loop_region.min_grid_cells);
    if (cfg_.loop_retrieval_backend == "compact_binary") {
        loop_closure_enabled_ = loop_closure_->enableCompactRetrieval(
            cfg_.loop_compact_max_keyframes);
    } else if (cfg_.loop_retrieval_backend == "flat_dbow3") {
        loop_closure_enabled_ = loop_closure_->loadFlatVocabulary(
            vocab_path, cfg_.loop_compact_max_keyframes);
    } else if (cfg_.loop_retrieval_backend == "dbow3") {
        loop_closure_enabled_ = loop_closure_->loadVocabulary(vocab_path);
    } else {
        LOG_ERROR("Unknown loop retrieval backend: "
                  << cfg_.loop_retrieval_backend);
        loop_closure_enabled_ = false;
    }
    if (loop_closure_enabled_) {
        LOG_INFO("Loop closure ENABLED (retrieval="
                 << cfg_.loop_retrieval_backend << ", vocab=" << vocab_path
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
    PERF_SCOPE("vo.frame_total");
    unsigned long frame_id = frame_count_++;

    // 先复制 Mat 头，兼容调用方把 currentFrame()->image 直接作为下一帧输入；
    // 随后的 releaseImages 只移除旧 Frame 的引用，不会令本轮输入失效。
    const cv::Mat left_input = left;
    const cv::Mat right_input = right;

    // 上次 addFrame 返回后 Viewer 已完成取帧。旧当前帧若仍是 LK 的上一帧，
    // 只保留左灰度图到本轮光流结束；其余像素缓冲现在即可释放。异常帧可能
    // 未被设为 prev_frame_，这种帧在进入下一轮时可以直接释放全部图像。
    Frame::Ptr old_current = curr_frame_;
    if (old_current) old_current->releaseImages(old_current == prev_frame_);

    // 1. 创建当前帧 + CLAHE 增强。原始灰度同时供双目 LK 使用，避免
    // 在 computeStereoDepths() 中对同一左右图再次做 BGR→灰度转换。
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
    cv::Mat left_gray_raw;
    cv::Mat right_gray_raw;
    if (left_input.channels() == 3)
        cv::cvtColor(left_input, left_gray_raw, cv::COLOR_BGR2GRAY);
    else
        left_gray_raw = left_input;
    curr_frame_->image_gray = cfg_.copy_gray_before_clahe
        ? left_gray_raw.clone() : left_gray_raw;
    curr_frame_->image = left_input;

    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(curr_frame_->image_gray, curr_frame_->image_gray);

    // 双目：右目图做同样的灰度化 + 增强（左右目一致，保证双目匹配质量）
    if (!right_input.empty()) {
        curr_frame_->image_right = right_input;
        if (right_input.channels() == 3)
            cv::cvtColor(right_input, right_gray_raw, cv::COLOR_BGR2GRAY);
        else
            right_gray_raw = right_input;
        curr_frame_->image_right_gray = cfg_.copy_gray_before_clahe
            ? right_gray_raw.clone() : right_gray_raw;
        clahe->apply(curr_frame_->image_right_gray, curr_frame_->image_right_gray);
    }

    // 2. 提取/跟踪特征
    // LK 模式（feature_method=1）：TRACKING 阶段用光流跟踪上一帧，不重新提取 ORB
    {
        PERF_SCOPE("vo.extract");
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
    }
    if (curr_frame_->keypoints.empty()) {
        LOG_WARN("No features extracted, skipping frame");
        // 即使本帧没有特征，也要先在地图读锁内捕获一致的参考位姿/锚点；
        // 早退路径不得直接读取可能被回环提交改写的 live ref_frame_。
        {
            std::shared_lock<std::shared_mutex> lock(map_mutex_);
            const TrackingSnapshot next = captureTrackingSnapshot();
            syncFrontendGeometry(snap_, next);
            syncFrontendAnchor(next.submap_id, next.T_ws);
            snap_ = next;
        }
        if (state_ != State::INITIALIZING && snap_.has_ref) {
            tracking_failures_++;
            state_ = tracking_failures_ >= cfg_.max_tracking_failures
                ? State::LOST : State::RECOVERING;
            if (state_ == State::LOST) relocalization_frames_++;
            // M3：last_valid_pose_world_ 是世界系，存回局部系需组合 T_ws
            curr_frame_->pose_cs = has_last_valid_pose_
                ? last_valid_pose_world_ * snap_.T_ws
                : (snap_.has_ref ? snap_.ref_pose_cs : SE3());
        }
        updateStatus(0, 0, 0.0);
        return (state_ != State::INITIALIZING)
            ? curr_frame_->pose_cs * snap_.T_ws.inverse() : SE3();
    }
    // 时序 LK 已完成，旧上一帧的左灰度图不再被前端使用。若它同时是历史
    // 关键帧，仅释放像素数据；描述子/关键点/map_points/pts_c 继续留在地图中。
    // 当前帧提取失败时不能释放：prev_frame_ 仍会作为下一轮 LK 的输入。
    if (prev_frame_) prev_frame_->releaseImages();

    // 2.5 双目/RGB-D：视差（或深度）→ 每特征点的相机系 3D 观测 pts_c
    {
        PERF_SCOPE("vo.stereo_depth");
        if (right_input.empty() || !cfg_.use_raw_stereo_gray)
            computeStereoDepths();
        else
            computeStereoDepths(left_gray_raw, right_gray_raw);
    }

    // 2.6 M2：每帧捕获一次只读快照（版本 + 参考帧位姿/点坐标拷贝）。
    // 整帧跟踪只消费本快照，不跨版本读实时地图；后端提交在锁内进行，
    // 前端与后端之间通过快照实现帧级一致。
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        const TrackingSnapshot next = captureTrackingSnapshot();
        syncFrontendGeometry(snap_, next);
        syncFrontendAnchor(next.submap_id, next.T_ws);
        snap_ = next;
    }

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
        {
            PERF_SCOPE("vo.track");
            if (cfg_.feature_method == 1 && curr_frame_->descriptors.empty())
                trackFrameLK();
            else
                trackFrame();
        }
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
                {
                    PERF_SCOPE("vo.submap_reinit");
                    createSubmap();
                }
                if (tryInitialize()) {
                    LOG_WARN((camera_->hasPerFrameDepth()
                              ? "Stereo re-init in anchored submap"
                              : "Monocular re-init in anchored submap"));
                    state_ = State::TRACKING;
                    tracking_failures_ = 0;
                    relocalization_frames_ = 0;
                    status_.tracking_valid = true;
                    status_.pose_method = "SUBMAP_INIT";
                    // 双目：子地图积累 ≥3 个关键帧后与历史轨迹刚体对齐，
                    // 消除外推锚点的残留偏差（单目尺度不同源，不做）。
                    submap_needs_alignment_ = camera_->hasPerFrameDepth();
                }
            }
        }
    }

    prev_frame_ = curr_frame_;
    updateStatus(status_.matches, status_.inliers, status_.parallax);
    status_.pose_valid = status_.tracking_valid && status_.map_connected;
    SE3 output_pose;
    {
        // 回环提交与前端收尾共享同一地图快照；锁序固定为 map → traj，
        // 且此临界区不调用会再次获取 map_mutex_ 的函数。
        std::shared_lock<std::shared_mutex> map_lock(map_mutex_);
        // 异步回环可能在本帧跟踪期间更新活动 T_ws。后台不触碰任何前端
        // 成员；在这里把上一有效世界位姿与本帧快照一起重基到新锚点，
        // 保证连续性门、轨迹记录和下一帧预测看到同一 Atlas 版本。
        const TrackingSnapshot frame_snapshot = snap_;
        if (const auto* active = atlas_->activeSubmap(); active &&
            active->id == snap_.submap_id && active->map == snap_.map) {
            syncFrontendAnchor(active->id, active->T_ws);
            TrackingSnapshot live_snapshot = snap_;
            live_snapshot.geometry_revision = map_->geometryRevision();
            if (snap_.has_ref) {
                const auto live_anchor = map_->getKeyFrame(snap_.ref_kf_id);
                if (live_anchor) live_snapshot.ref_pose_cs = live_anchor->pose_cs;
            }
            syncFrontendGeometry(frame_snapshot, live_snapshot);
        }
        // 先构造与持久轨迹完全相同的锚定记录，再从 live anchor 组合本帧
        // 世界位姿。后端若在普通帧跟踪期间校正参考 KF，不能继续用旧
        // snap_.ref_pose_cs 更新 last_valid_pose_world_，否则下一帧会把整次
        // 回环校正误判成相邻运动并进入 LOST。
        FramePoseRecord rec;
        bool has_record = false;
        if (status_.pose_valid) {
            const bool self_anchor = !snap_.has_ref || ref_frame_ == curr_frame_;
            rec.frame_id = curr_frame_->id;
            rec.submap_id = snap_.submap_id;
            rec.anchor_kf_id = self_anchor ? curr_frame_->id : snap_.ref_kf_id;
            SE3 record_anchor_pose = snap_.ref_pose_cs;
            if (!self_anchor && snap_.map == last_local_ba_commit_map_ &&
                snap_.submap_id == last_local_ba_commit_submap_id_ &&
                snap_.geometry_revision != map_->geometryRevision() &&
                map_->geometryRevision() ==
                    last_local_ba_commit_geometry_revision_) {
                // 本帧在旧几何上完成跟踪，而 Local BA 恰好在帧内改写了
                // reference。Local BA 不应移动已经估计出的当前世界位姿，
                // 因此直接绑定 live anchor；回环 PGO 不设置该标记，仍按
                // 旧 T_ca 传播全局闭环校正。
                const auto live_anchor = map_->getKeyFrame(snap_.ref_kf_id);
                if (live_anchor) record_anchor_pose = live_anchor->pose_cs;
            }
            rec.T_ca = self_anchor ? SE3()
                : curr_frame_->pose_cs * record_anchor_pose.inverse();
            rec.anchor_pose_cs = self_anchor
                ? curr_frame_->pose_cs : record_anchor_pose;
            rec.valid = true;
            has_record = true;
        }
        const SE3 frame_pose_world = has_record
            ? composeRecordWorld(rec)
            : curr_frame_->pose_cs * snap_.T_ws.inverse();
        // 连续里程计必须跟随“对外发布”语义，而不是内部状态名。
        // RECOVERING 边界帧只要 pose_valid=true，T_ob 就必须积分该帧；
        // 否则会出现 T_wb 前进而 T_ob 停在上帧的分层断裂。
        if (status_.pose_valid) {
            if (has_last_valid_pose_) {
                // 记录逐帧相对运动（世界系 Twc 语义，M3：由局部位姿组合 T_ws 得到），
                // LOST 期匀速外推锚点用——基线必须与世界系轨迹一致
                const SE3 X_cur = frame_pose_world.inverse();
                const SE3 X_last = last_valid_pose_world_.inverse();
                per_frame_motion_ = X_last.inverse() * X_cur;
                has_per_frame_motion_ = true;
            }
            const SE3 current_wc = frame_pose_world.inverse();
            {
                std::lock_guard<std::mutex> output_lock(output_pose_mutex_);
                if (!has_continuous_pose_) {
                    continuous_pose_oc_ = current_wc;
                    global_correction_T_wo_ = SE3();
                    has_continuous_pose_ = true;
                } else if (has_per_frame_motion_) {
                    continuous_pose_oc_ = continuous_pose_oc_ * per_frame_motion_;
                    const SE3 next_T_wo = splitGlobalCameraPose(
                        current_wc, continuous_pose_oc_).T_wo;
                    if ((next_T_wo.t - global_correction_T_wo_.t).norm() > 1e-6 ||
                        next_T_wo.q.angularDistance(global_correction_T_wo_.q) > 1e-8) {
                        global_correction_generation_.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    global_correction_T_wo_ = next_T_wo;
                }
            }
            // 世界系 T_cw（轨迹/验收基线与轨迹条目同一坐标系）
            last_valid_pose_world_ = frame_pose_world;
            last_valid_geometry_revision_ = map_->geometryRevision();
            has_last_valid_pose_ = true;
            tracking_failures_ = 0;
        }
        // pose_cs.t 是子地图原点在相机系的坐标，不是相机位置；轨迹必须
        // 在同一 map 读快照下记录锚定信息并计算输出世界位姿。
        if (has_record) {
            // M4：只记录锚定关键帧 + 局部运动；世界位姿由
            // composePoseTrajectory 读时组合（回环/对齐自动跟随锚点）。
            std::lock_guard<std::mutex> traj_lock(traj_mutex_);
            pose_records_.push_back(rec);
        }
        output_pose = frame_pose_world;
    }
    return output_pose;
}

// ============================================================
// 双目/RGB-D 深度计算：视差（或深度）→ pts_c
// ============================================================
void VisualOdometry::computeStereoDepths(const cv::Mat& left_gray,
                                         const cv::Mat& right_gray) {
    // M1.4：深度计算与统计迁移至 FrontendTracker（frontend_tracker.cpp），
    // 公式与默认值保持不变；这里只应用返回的深度统计到 status_。
    const StereoStats stats = left_gray.empty() || right_gray.empty()
        ? frontend_tracker_.computeStereoDepths(curr_frame_)
        : frontend_tracker_.computeStereoDepths(
              curr_frame_, left_gray, right_gray);
    status_.stereo_points = stats.stereo_points;
    status_.median_disparity = stats.median_disparity;
    status_.median_depth = stats.median_depth;
}

// ============================================================
// 双目/RGB-D 单帧建点：pts_c 有效 → 世界系 3D 地图点
// ============================================================
/// 双目/RGB-D 单帧建点：pts_c 有效 → 子地图局部系 3D 地图点。
/// M1.5：算法迁移至 LocalMapper::createMapPointsFromStereo（local_mapper.cpp），
/// 公式与 Map API 调用不变；这里只负责转调（调用方须持 map_mutex_ 独占锁）。
void VisualOdometry::createMapPointsFromStereo(const Frame::Ptr& frame) {
    local_mapper_.createMapPointsFromStereo(
        map_, frame, map_budget_.config().max_active_points);
}

// ============================================================
// 状态更新辅助
// ============================================================
void VisualOdometry::updateStatus(int matches, int inliers, double parallax) {
    status_.state      = state_;
    status_.matches    = matches;
    status_.inliers    = inliers;
    status_.parallax   = parallax;
    // 计数为原子量（P0-3），atlas 只被前端线程读写（createSubmap/activate/
    // activeSubmap 均在前端路径），故此处无需持 map_mutex_——每帧省一次锁。
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
    SE3 anchor_Twc = has_last_valid_pose_
        ? last_valid_pose_world_.inverse()
        : (ref_frame_ ? ref_frame_->pose_cs.inverse() : SE3());

    // 匀速模型外推：丢失期间（relocalization_frames_ 帧）用丢失前的逐帧
    // 相对运动继续推进锚点，避免重初始化轨迹在丢失处出现静止/回跳
    // （KITTI 高速段 20 帧丢失 ≈ 20m，不外推则锚点明显滞后）。
    const int lost_frames = std::max(relocalization_frames_, 1);
    if (has_last_valid_pose_ && has_per_frame_motion_) {
        SE3 extrapolated = anchor_Twc;
        // 单步位移限幅：丢失前的瞬时速度可能被 PnP 高估（off-road 段
        // 出现过 20 帧外推 41m 的情况），限制每步 ≤ 2.5m 防锚点跑偏
        const double step_cap = 2.5;
        for (int i = 0; i < std::min(lost_frames, 60); i++) {
            SE3 step = per_frame_motion_;
            if (step.t.norm() > step_cap) {
                step.t = step.t.normalized() * step_cap;
                step.q = Eigen::Quaterniond::Identity();
            }
            extrapolated = extrapolated * step;
        }
        // 外推距离防爆：超过 60m 视为异常（KITTI ~1m/帧），退回不外推
        if ((extrapolated.t - anchor_Twc.t).norm() <= 60.0)
            anchor_Twc = extrapolated;
        LOG_WARN("Submap anchor extrapolated by " << lost_frames
                 << " frames (" << (anchor_Twc.t - (has_last_valid_pose_
                     ? last_valid_pose_world_.inverse().t : Vec3::Zero())).norm()
                 << "m)");
    }

    // anchor 继承最后有效全局位姿，新子地图实际已锚定到全局世界系。
    // 若标为 disconnected，此后所有帧 pose_valid=false（tracking_valid &&
    // map_connected），轨迹将永久冻结、只剩旋转箭头转动。标记 connected
    // 让重初始化后轨迹继续记录（丢失期间的真实位移仍未知，会有短暂停顿，
    // 但不再彻底冻结）。
    // M3：新子地图原点 = 相机当前位置（T_ws = anchor），首帧局部位姿为单位。
    // Atlas 写与 map_ 交换必须在同一独占锁内（M4：getter 锁序 map→traj，
    // Atlas 写路径全部收敛到 map_mutex_ 独占，Viewer 线程读安全）。
    const auto* prev_sub = atlas_->activeSubmap();  // 创建前 = 旧活动子地图
    Submap* submap = nullptr;
    {
        std::unique_lock<std::shared_mutex> lock(map_mutex_);
        submap = &atlas_->createSubmap(anchor_Twc, true);
        // M5：TrackingBridge 约束——丢失期外推锚定低置信，可被后续
        // Relocalization/LoopClosure 约束修正（§14.5）。
        if (prev_sub && prev_sub->id != submap->id) {
            AtlasConstraint bridge;
            bridge.a = prev_sub->id;
            bridge.b = submap->id;
            bridge.T_rel = prev_sub->T_ws.inverse() * anchor_Twc;
            bridge.weight = 0.3;
            bridge.type = AtlasConstraintType::TrackingBridge;
            atlas_->addConstraint(bridge);
        }
        map_ = submap->map;
        curr_frame_->pose_cs = SE3();
        snap_.map = map_;
        snap_.T_ws = submap->T_ws;      // 刷新帧内快照锚（轨迹推/返回用）
        snap_.submap_id = submap->id;
        snap_.has_ref = false;          // 新子地图无参考帧
        snap_.ref_kf_id = 0;
    }
    ref_frame_.reset();
    prev_frame_.reset();
    active_trajectory_segment_start_frame_id_ = curr_frame_
        ? curr_frame_->id : frame_count_;
    last_kf_frame_id_ = 0;
    active_keyframe_serial_ = 0;
    last_loop_keyframe_serial_ = 0;
    state_ = State::INITIALIZING;
    tracking_failures_ = 0;
    relocalization_frames_ = 0;
    LOG_WARN("Tracking lost for too long; creating anchored submap " << submap->id);
}

// ============================================================
// M5：Atlas 子地图约束图求解（§14.5）
// 节点 = 子地图 T_ws，边 = TrackingBridge/Relocalization/LoopClosure。
// 首个子地图固定（世界系锚），其余 T_ws 由约束图优化对齐——跨子地图
// 重定位只连坐标、不融合地图，连接处不跳变。
// ============================================================
bool VisualOdometry::solveAtlasConstraints() {
    // 调用方必须持有 map_mutex_ 独占锁；这里故意不再获取该锁，避免
    // 跨子地图重定位事务发生 shared_mutex 重入。
    const auto& subs = atlas_->submaps();
    if (subs.size() < 2) return false;

    // 只有最早子地图是 gauge。历史节点不能全部固定，否则多次 LOST/预算
    // 边界形成的 yaw 漂移无法被后续跨图回环沿整条链重新分配，只会集中
    // 到活动尾节点。图构建独立成纯函数，非单位旋转/平移由回归覆盖。
    OptimizationSnapshot snap = buildSubmapGraph(*atlas_);
    auto result = Optimizer::solvePoseGraph(snap);
    if (!result.valid) return false;

    bool updated = false;
    for (const auto& u : result.poses) {
        if (auto* sub = atlas_->getSubmap(u.id)) {
            // 优化结果仍遵循 pose_cs/T_cw 契约，写回 Atlas 时恢复为 T_ws。
            sub->T_ws = u.pose_cs.inverse();
            updated = true;
        }
    }
    if (updated) {
        LOG_INFO("Atlas constraint graph solved: " << subs.size() << " submaps, "
                 << atlas_->constraints().size() << " constraints");
    }
    return updated;
}

// ============================================================
// 子地图-历史轨迹刚体对齐：新子地图关键帧相机位置 vs 历史轨迹最近点，
// Umeyama 求解 Sim3（双目下尺度≈1，仅校正旋转/平移），
// 让重初始化后的轨迹平滑接回丢失点，不出现断点跳变。
// M3：只更新 Submap::T_ws（子地图→世界唯一权威）——不再遍历移动
// KF/地图点；地图局部坐标与 geometry revision 完全不变，后端已排队
// 的 BA/回环快照不失效。
// ============================================================
void VisualOdometry::alignSubmapToTrajectory() {
    auto* active = atlas_->activeSubmap();
    if (!active || active->map != map_) return;
    auto kfs = map_->getAllKeyFrames();
    if (kfs.size() < 3) return;

    // M4：组合锚定轨迹 → 世界系相机位置（调用方已持 map_mutex_ 独占，勿重入）
    std::vector<Vec3> traj_pos;
    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        traj_pos.reserve(pose_records_.size());
        for (const auto& rec : pose_records_) {
            if (!rec.valid) continue;
            traj_pos.push_back(composeRecordWorld(rec).camera_position());
        }
    }
    if (traj_pos.empty()) return;

    // 只搜索轨迹末端附近（丢失点必然在当前轨迹末尾，最多向前取 100 帧）：
    // 新子地图锚在丢失点附近，最近的旧位置就在这段里；限制搜索窗避免
    // 误配到无关的早期区域（轨迹自交时最近邻会找错位置）。
    const size_t search_from = traj_pos.size() > 100
        ? traj_pos.size() - 100 : 0;

    const SE3 T_ws = active->T_ws;

    std::vector<Vec3> src, dst;
    src.reserve(kfs.size());
    dst.reserve(kfs.size());
    const double max_match_dist = 50.0;  // 外推锚点可能偏离丢失点 20-40m，半径放宽；
                                         // 误配风险由"末端搜索窗 + scale≈1 检查"兜底
    for (const auto& kf : kfs) {
        // M3：局部相机位置 → 世界系（p_w = T_ws · p_s）
        const Vec3 pos = T_ws * kf->pose_cs.inverse().t;
        double best = max_match_dist;
        Vec3 best_p = Vec3::Zero();
        bool found = false;
        for (size_t i = search_from; i < traj_pos.size(); i++) {
            const Vec3 c = traj_pos[i];
            const double d = (c - pos).norm();
            if (d < best) {
                best = d;
                best_p = c;
                found = true;
            }
        }
        if (found) {
            src.push_back(pos);
            dst.push_back(best_p);
        }
    }
    if (src.size() < 4) {
        LOG_WARN("Submap alignment skipped: only " << src.size() << " matches");
        return;
    }

    Sim3 S;
    if (!Sim3::estimate(src, dst, S)) {
        LOG_WARN("Submap alignment failed (degenerate point set)");
        return;
    }
    // 双目有绝对尺度，只接受近似刚体对齐（尺度偏离 >15% 视为误配）
    if (std::abs(S.s - 1.0) > 0.15) {
        LOG_WARN("Submap alignment rejected: scale=" << S.s);
        return;
    }

    // M3：T_ws' = S ∘ T_ws（点变换复合）——KF/点局部坐标零移动
    const SE3 T_ws_new(S.q * T_ws.q, S * T_ws.t);
    active->T_ws = T_ws_new;
    snap_.T_ws = T_ws_new;  // 本帧轨迹推/返回立即生效
    LOG_INFO("Submap aligned to trajectory: " << src.size()
             << " matches, scale=" << S.s
             << ", T_ws shift=" << (T_ws_new.t - T_ws.t).norm() << "m");
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
            {
                std::unique_lock<std::shared_mutex> lock(map_mutex_);
                ref_frame_ = curr_frame_;
                createMapPointsFromStereo(ref_frame_);
                map_->insertKeyFrame(ref_frame_);
                active_keyframe_serial_ = 1;
            }
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

    cv::Mat essential_mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, camera_->K(),
                                     cv::RANSAC, 0.999, 1.0,
                                     essential_mask);
    cv::Mat R, t;
    int inliers = E.empty() ? 0 : cv::recoverPose(
        E, pts1, pts2, camera_->K(), R, t, essential_mask);

    std::vector<Vec2> normalized_ref, normalized_curr;
    std::vector<cv::DMatch> inlier_matches;
    normalized_ref.reserve(pts1.size());
    normalized_curr.reserve(pts2.size());
    inlier_matches.reserve(matches.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        bool is_inlier = essential_mask.empty();
        if (!essential_mask.empty()) {
            is_inlier = essential_mask.rows == 1
                ? essential_mask.at<uchar>(0, static_cast<int>(i)) != 0
                : essential_mask.at<uchar>(static_cast<int>(i), 0) != 0;
        }
        if (!is_inlier) continue;
        inlier_matches.push_back(matches[i]);
        normalized_ref.emplace_back(
            (pts1[i].x - camera_->cx) / camera_->fx,
            (pts1[i].y - camera_->cy) / camera_->fy);
        normalized_curr.emplace_back(
            (pts2[i].x - camera_->cx) / camera_->fx,
            (pts2[i].y - camera_->cy) / camera_->fy);
    }

    const int geometry_inliers = static_cast<int>(inlier_matches.size());
    if (geometry_inliers < cfg_.min_init_inliers) {
        LOG_INFO("Init: too few geometric inliers (" << geometry_inliers
                 << ", need " << cfg_.min_init_inliers << ")");
        ref_frame_ = curr_frame_;
        updateStatus((int)matches.size(), inliers, 0.0);
        return false;
    }

    Mat33 R_eigen = Mat33::Identity();
    Vec3 t_eigen = Vec3::Zero();
    if (!R.empty() && !t.empty()) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R_eigen(i, j) = R.at<double>(i, j);
        t_eigen = Vec3(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    }
    const auto init_quality = assessMonocularInitialization(
        normalized_ref, normalized_curr, R_eigen, t_eigen,
        cfg_.min_parallax);
    const double parallax = init_quality.parallax_rad;

    if (!init_quality.accepted) {
        LOG_INFO("Init: degenerate geometry (parallax=" << parallax
                 << " rad, positive_depth_ratio="
                 << init_quality.positive_depth_ratio << ", points="
                 << init_quality.triangulated_points << ")");
        ref_frame_ = curr_frame_;
        updateStatus((int)matches.size(), inliers, parallax);
        return false;
    }

    // 第一帧使用当前子地图的锚定位姿。初始子地图锚点为单位位姿，
    // 后续子地图则继承全局位姿，避免重建后轨迹跳回原点。
    // recoverPose 返回的相对变换 T_rel 满足 p_c2 = T_rel * p_c1。
    SE3 anchor_cw = ref_frame_->pose_cs;
    SE3 T_cw2(Eigen::Quaterniond(R_eigen), t_eigen);
    curr_frame_->pose_cs = T_cw2 * anchor_cw;

    {
        std::unique_lock<std::shared_mutex> lock(map_mutex_);
        triangulateNewPoints(ref_frame_, curr_frame_, inlier_matches);
        map_->insertKeyFrame(ref_frame_);
        map_->insertKeyFrame(curr_frame_);
        active_keyframe_serial_ = 2;
        const auto shared_points = map_->sharedObservationCount(
            ref_frame_->id, curr_frame_->id);
        odometry_edges_.push_back({
            ref_frame_->id, curr_frame_->id,
            ref_frame_->pose_cs * curr_frame_->pose_cs.inverse(),
            1.0 + std::log2(1.0 + static_cast<double>(shared_points))});
    }
    last_kf_frame_id_ = curr_frame_->id;   // 初始化插入的两个关键帧也参与冷却

    LOG_INFO("Init OK! parallax=" << parallax << " inliers=" << inliers
             << " mp=" << map_->mapPointCount());
    updateStatus((int)matches.size(), inliers, parallax);
    return true;
}

// ============================================================
// 跟踪
// M1.1：pnpReprojectionRmse / acceptPoseCandidate / checkMotionContinuity
// 已迁移至 PoseGate（pose_gate.{h,cpp}），公式与默认值保持不变。
// ============================================================

// ============================================================
// M0：统一位姿验收（正常跟踪与重定位共用同一条通路）
// M1.1：几何质量与连续性逻辑已迁移至 PoseGate；本函数只负责
// 从 VO 状态计算运动基线与门限，再转调 PoseGate::acceptPoseCandidate。
// ============================================================
/// M5：重定位运动基线（世界系 T_wc）——丢失期匀速外推（单步 2.5m 限幅、
/// 总量 60m，与 createSubmap 同一规则）。const：只读成员，不写 snap_。
SE3 VisualOdometry::relocBaselineWorld() const {
    SE3 baseline = last_valid_pose_world_.inverse();
    const int lost = std::max(1, relocalization_frames_ + tracking_failures_);
    if (has_per_frame_motion_) {
        SE3 extrapolated = baseline;
        const double step_cap = 2.5;
        for (int i = 0; i < std::min(lost, 60); i++) {
            SE3 step = per_frame_motion_;
            if (step.t.norm() > step_cap) {
                step.t = step.t.normalized() * step_cap;
                step.q = Eigen::Quaterniond::Identity();
            }
            extrapolated = extrapolated * step;
        }
        if ((extrapolated.t - baseline.t).norm() <= 60.0)
            baseline = extrapolated;
    }
    return baseline;
}

bool VisualOdometry::acceptPose(
    const SE3& candidate_pose_world, int inliers, size_t total, double rmse,
    bool reloc_mode, PoseQuality& quality,
    int min_inliers_override, double min_ratio_override,
    double max_rmse_override) const {
    const int min_inliers = min_inliers_override > 0 ? min_inliers_override
        : (reloc_mode ? std::max(20, cfg_.pnp_min_inliers) : cfg_.pnp_min_inliers);
    const double min_ratio = min_ratio_override > 0 ? min_ratio_override
        : (reloc_mode ? std::max(0.4, cfg_.pnp_min_inlier_ratio)
                      : cfg_.pnp_min_inlier_ratio);
    const double max_rmse = max_rmse_override > 0 ? max_rmse_override
        : cfg_.pnp_max_rmse;

    // 运动基线：
    // - 正常跟踪：上一有效位姿（T_wc），门限 max_frame_translation/rotation；
    // - 重定位：丢失期匀速外推的期望位姿（单步 2.5m 限幅、总量 60m，与
    //   createSubmap 同一外推规则），平移门限放宽至 max(50m, 3×外推位移)、
    //   旋转门限固定 60°；无运动模型时退化为固定门限。
    // - 单目（尺度归一化，位移无物理意义）或无上一有效位姿 → 跳过连续性，
    //   仅几何验收（保持历史行为，避免破坏单目跟踪）。
    std::optional<SE3> baseline;
    double max_translation = 0.0;
    double max_rotation = 0.0;
    if (camera_->hasPerFrameDepth() && has_last_valid_pose_) {
        const SE3 Twc_last = last_valid_pose_world_.inverse();
        if (reloc_mode) {
            // M5：基线提取为 relocBaselineWorld（tryRelocalize 跨子地图
            // 约束路径需要同一外推规则）
            const SE3 baseline_twc = relocBaselineWorld();
            const double expected = (baseline_twc.t - Twc_last.t).norm();
            baseline = baseline_twc;
            max_translation = std::max(50.0, 3.0 * expected);
            max_rotation = 60.0 * M_PI / 180.0;
        } else {
            baseline = Twc_last;
            max_translation = cfg_.max_frame_translation;
            max_rotation = cfg_.max_frame_rotation;
        }
    }
    // M3：候选位姿与基线均为世界系（调用方已组合 T_ws）
    return PoseGate::acceptPoseCandidate(candidate_pose_world, inliers, total, rmse,
                                         min_inliers, min_ratio, max_rmse,
                                         baseline, max_translation, max_rotation, quality);
}

SE3 VisualOdometry::trackFrame() {
    if (!ref_frame_ || !curr_frame_) return SE3();

    // M1.4：跟踪几何计算（匹配 → PnP → 3D-3D → 对极回退 → RECOVERING）
    // 已迁移至 FrontendTracker::trackOrb（frontend_tracker.cpp，公式不变）。
    // 这里只负责：① 构造查询（帧首快照参考数据 + 运动基线）；② 应用结果。
    RefView ref;
    ref.has_ref = snap_.has_ref;
    ref.ref_pose_cs = snap_.ref_pose_cs;
    ref.T_ws = snap_.T_ws;
    ref.ref_points_s = snap_.ref_points_s;
    ref.ref_mps = snap_.ref_mps;
    // 方案 B：共视图局部地图（锁内快照拷贝，前端只消费拷贝）
    ref.local_points_s = snap_.local_points_s;
    ref.local_descs = snap_.local_descs;
    ref.local_mps = snap_.local_mps;
    const StereoStats stereo{status_.stereo_points, status_.median_disparity,
                             status_.median_depth};
    TrackingResult r = frontend_tracker_.trackOrb(
        curr_frame_, ref_frame_, ref, normalMotionBaseline(), stereo);
    last_tracking_matches_ = r.match_pairs;
    last_tracking_ref_id_ = ref_frame_->id;
    last_tracking_curr_id_ = curr_frame_->id;

    // 首轮参考 KF PnP 失败时，ORB 仍可能有大量 2D 匹配，但其中只有少数
    // 特征仍挂着正式 3D 点。先用运动预测把共视局部地图投影到当前帧，再用
    // 同一组 PnP 几何/连续性门验收，避免“有匹配却因为 pts3d 太少直接 LOST”。
    if (!r.valid && cfg_.local_map_rescue &&
        !snap_.ref_descs.empty() && !snap_.ref_points_s.empty()) {
        const MotionBaseline motion = normalMotionBaseline();
        const SE3 seed_pose = motion.predicted_pose_cs
            ? *motion.predicted_pose_cs : snap_.ref_pose_cs;
        const size_t rescue_limit = static_cast<size_t>(std::max(
            8, cfg_.local_map_rescue_max_points));
        std::vector<Vec3> rescue_points;
        std::vector<cv::Mat> rescue_descs;
        std::vector<MapPoint::Ptr> rescue_mps;
        std::unordered_set<MapPointId> rescue_seen;
        rescue_points.reserve(rescue_limit);
        rescue_descs.reserve(rescue_limit);
        rescue_mps.reserve(rescue_limit);
        for (size_t i = 0; i < snap_.ref_descs.size() &&
                            i < snap_.ref_mps.size() &&
                            i < snap_.ref_points_s.size() &&
                            rescue_points.size() < rescue_limit; ++i) {
            const auto& mp = snap_.ref_mps[i];
            if (!mp || snap_.ref_descs[i].empty() ||
                !rescue_seen.insert(mp->id).second) continue;
            rescue_points.push_back(snap_.ref_points_s[i]);
            rescue_descs.push_back(snap_.ref_descs[i]);
            rescue_mps.push_back(mp);
        }
        // 参考 KF 之外再补充少量共视缓存，覆盖参考点被 cull/遮挡的情况。
        for (size_t i = 0; i < snap_.local_points_s.size() &&
                            rescue_points.size() < rescue_limit; ++i) {
            if (i >= snap_.local_descs.size() || i >= snap_.local_mps.size()) break;
            const auto& mp = snap_.local_mps[i];
            if (!mp || snap_.local_descs[i].empty() ||
                !rescue_seen.insert(mp->id).second) continue;
            rescue_points.push_back(snap_.local_points_s[i]);
            rescue_descs.push_back(snap_.local_descs[i]);
            rescue_mps.push_back(mp);
        }
        const LocalMapTrackResult local = frontend_tracker_.trackLocalMap(
            curr_frame_, rescue_points, rescue_descs, rescue_mps, {}, seed_pose,
            cfg_.local_map_rescue_radius_px, cfg_.match_ratio);
        if (local.added >= 8) {
            TrackingResult rescued = frontend_tracker_.trackPnP(
                local.pts3d, local.pts2d, snap_.T_ws, motion,
                cfg_.pnp_min_inliers, cfg_.pnp_min_inlier_ratio,
                cfg_.pnp_max_rmse);
            if (rescued.valid) {
                rescued.matches = std::max(r.matches, local.added);
                rescued.method = "LOCAL_MAP_PNP";
                for (const int index : rescued.pnp_inlier_indices) {
                    if (index < 0 || index >= static_cast<int>(local.mps.size())) continue;
                    rescued.associations.emplace_back(
                        local.curr_feature_indices[index], local.mps[index]);
                    const auto& point = rescue_points[index];
                    rescued.association_points_s.push_back(point);
                }
                r = std::move(rescued);
                LOG_INFO("Local map rescue: " << local.added
                         << " projected correspondences, inliers=" << r.inliers);
            }
        }
    }

    // ---- 方案 B：局部地图投影补匹配 + 精修位姿 ----
    // 首轮 PnP 成功后，把参考 KF 共视的地图点投影进当前帧补充 3D-2D 对应，
    // 合并后重跑 PnP：新增对应 > 阈值且内点数不减时才采纳（外点交给
    // solvePnPRansac 过滤；连续性验收由 PoseGate 复用正常跟踪门限）。
    if (r.valid && cfg_.local_map_tracking
        && !snap_.local_mps.empty() && !snap_.local_points_s.empty()) {
        std::vector<int> occupied;
        occupied.reserve(r.associations.size());
        for (const auto& [train_idx, mp] : r.associations)
            occupied.push_back(train_idx);
        const LocalMapTrackResult local = frontend_tracker_.trackLocalMap(
            curr_frame_, snap_.local_points_s, snap_.local_descs,
            snap_.local_mps, occupied, r.pose_cs,
            cfg_.local_map_search_radius_px, cfg_.match_ratio);
        if (local.added >= 8) {
            // 合并首轮 3D-2D 与局部地图新增对应，做确定性位姿精修。
            // 首轮对应必须用 trackOrb 返回的版本绑定快照坐标（association_points_s），
            // 不能读 live mp->pos_s——后端 BA 可能已改写坐标，违反快照约定。
            // 精修用 refinePnP（solvePnP iterative + useExtrinsicGuess），
            // 不重跑 RANSAC：避免额外消耗全局 RNG 破坏双实例交替驱动的
            // 确定性等价（test_localizer_contract），也避免内点集抖动。
            std::vector<cv::Point3f> pts3d = local.pts3d;
            std::vector<cv::Point2f> pts2d = local.pts2d;
            // 合并集的前缀来自局部地图新增关联，后缀来自首轮跟踪关联。
            // 必须显式保留这份来源标记，精修通过后才能把通过几何筛选的
            // 新增地图点写回当前帧；旧代码初始化为全 0，导致新增关联永远
            // 不会写回。
            std::vector<char> keep_local(pts3d.size(), 1);
            for (size_t k = 0; k < r.associations.size(); k++) {
                const auto& [train_idx, mp] = r.associations[k];
                (void)mp;
                if (k >= r.association_points_s.size()
                    || train_idx < 0
                    || train_idx >= (int)curr_frame_->keypoints.size()) continue;
                const Vec3& p_s = r.association_points_s[k];
                pts3d.emplace_back((float)p_s.x(), (float)p_s.y(), (float)p_s.z());
                pts2d.push_back(curr_frame_->keypoints[train_idx].pt);
                keep_local.push_back(0);
            }
            // 确定性外点筛选：用首轮位姿投影，剔除重投影误差过大的对应。
            // 窗口匹配只保证"描述子相似"，不保证"几何一致"——局部地图点
            // 坐标陈旧或首轮位姿有误差时，投影到错误位置附近碰巧匹配到
            // 相似描述子的概率随地图膨胀上升；refinePnP 是全量最小二乘
            //（无 RANSAC），这类错配会直接拉偏位姿（完整 KITTI 00 实测
            // 无筛选时轨迹长度比 1.04 → 1.12、ATE 恶化）。筛选保持确定性
            //（不消耗 RNG），阈值沿用 PnP RANSAC 像素阈值，非数据集特化。
            std::vector<cv::Point3f> pts3d_f;
            std::vector<cv::Point2f> pts2d_f;
            std::vector<char> keep_all(pts3d.size(), 0);
            for (size_t i = 0; i < pts3d.size(); i++) {
                const Vec3 p_c = r.pose_cs * Vec3(pts3d[i].x, pts3d[i].y, pts3d[i].z);
                if (p_c.z() <= 0.01) continue;
                const Vec2 px = camera_->camera2pixel(p_c);
                const Vec2 obs(pts2d[i].x, pts2d[i].y);
                if ((px - obs).norm() > cfg_.ransac_pixel_threshold) continue;
                pts3d_f.push_back(pts3d[i]);
                pts2d_f.push_back(pts2d[i]);
                keep_all[i] = 1;
            }
            if (pts3d_f.size() < 6) {
                LOG_WARN("Local map refine: reprojection filter too few (" << pts3d_f.size() << ")");
            } else {
                const TrackingResult refined = frontend_tracker_.refinePnP(
                    pts3d_f, pts2d_f, r.pose_cs, snap_.T_ws, normalMotionBaseline(),
                    cfg_.pnp_min_inliers, cfg_.pnp_min_inlier_ratio, cfg_.pnp_max_rmse);
                if (refined.valid && refined.inliers >= r.inliers) {
                    LOG_INFO("Local map refine: +" << local.added
                             << " corr, kept " << pts3d_f.size()
                             << " after filter, inliers " << r.inliers
                             << " -> " << refined.inliers);
                    r.pose_cs = refined.pose_cs;
                    r.inliers = refined.inliers;
                    r.inlier_ratio = refined.inlier_ratio;
                    r.pose_rmse = refined.pose_rmse;
                    r.translation_delta = refined.translation_delta;
                    r.rotation_delta = refined.rotation_delta;
                    // 关联通过重投影筛选的局部地图新增匹配（合并集靠前 = local 部分）
                    for (size_t i = 0; i < keep_local.size(); i++) {
                        if (!keep_local[i] || !keep_all[i]) continue;
                        const int ti = local.curr_feature_indices[i];
                        if (ti >= 0 && ti < (int)curr_frame_->map_points.size())
                            curr_frame_->map_points[ti] = local.mps[i];
                    }
                }
            }
        }
    }

    // 应用结果（与旧内联逻辑逐项等价）
    curr_frame_->pose_cs = r.pose_cs;
    for (const auto& [train_idx, mp] : r.associations) {
        if (train_idx >= 0 && train_idx < (int)curr_frame_->map_points.size())
            curr_frame_->map_points[train_idx] = mp;
    }
    status_.inlier_ratio = r.inlier_ratio;
    status_.pose_rmse = r.pose_rmse;
    status_.translation_delta = r.translation_delta;
    status_.rotation_delta = r.rotation_delta;
    if (r.recovering) state_ = State::RECOVERING;
    if (r.valid) {
        status_.tracking_valid = true;
        status_.pose_method = r.method;
    }
    updateStatus(r.matches, r.inliers, 0.0);
    return curr_frame_->pose_cs;
}

MotionBaseline VisualOdometry::normalMotionBaseline() const {
    // 正常跟踪基线 = 上一有效位姿（世界系 T_wc），门限
    // max_frame_translation/rotation（与 acceptPose 的正常跟踪分支同一规则）。
    MotionBaseline motion;
    if (camera_->hasPerFrameDepth() && has_last_valid_pose_) {
        motion.baseline_twc = last_valid_pose_world_.inverse();
        motion.max_translation = cfg_.max_frame_translation;
        motion.max_rotation = cfg_.max_frame_rotation;
    }
    // 方案 A：匀速模型预测当前帧位姿（子地图局部系 T_cs）。
    // X_cur = X_last · per_frame_motion_（X 为世界系 T_wc）→
    // T_cs_pred = X_cur⁻¹ · T_ws = per_frame_motion_⁻¹ · X_last⁻¹ · T_ws。
    // 单目尺度归一化下 per_frame_motion_ 仍可作投影先验（引导匹配只是
    // 缩小搜索窗口，预测不准时前端自动回退全图 BF），故不限制传感器。
    if (has_per_frame_motion_) {
        const SE3 X_last = last_valid_pose_world_.inverse();
        const SE3 X_pred = X_last * per_frame_motion_;
        motion.predicted_pose_cs = X_pred.inverse() * snap_.T_ws;
    }
    return motion;
}

// ============================================================
// LK 光流跟踪（feature_method=1）
// 光流后 map_points 与关键点索引对齐（继承自上一帧），直接做 PnP
// M1.4：PnP 核心迁移至 FrontendTracker::trackPnP；这里只负责锁内收集
// 3D-2D 对应、转调与应用结果。
// ============================================================
SE3 VisualOdometry::trackFrameLK() {
    if (!curr_frame_) return SE3();

    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：读路径共享锁
        for (size_t i = 0; i < curr_frame_->keypoints.size(); i++) {
            auto& mp = curr_frame_->map_points[i];
            if (mp) {
                pts3d.emplace_back((float)mp->pos_s.x(), (float)mp->pos_s.y(), (float)mp->pos_s.z());
                pts2d.push_back(curr_frame_->keypoints[i].pt);
            }
        }
    }

    if (pts3d.size() >= 6) {
        const TrackingResult pnp = frontend_tracker_.trackPnP(
            pts3d, pts2d, snap_.T_ws, normalMotionBaseline(),
            cfg_.pnp_min_inliers, cfg_.pnp_min_inlier_ratio, cfg_.pnp_max_rmse);
        if (pnp.valid) {
            curr_frame_->pose_cs = pnp.pose_cs;
            status_.tracking_valid = true;
            status_.pose_method = "LK_PNP";
            status_.inlier_ratio = pnp.inlier_ratio;
            status_.pose_rmse = pnp.pose_rmse;
            status_.translation_delta = pnp.translation_delta;
            status_.rotation_delta = pnp.rotation_delta;
            // 普通 LK 跟踪帧只继承临时地图点指针；正式观测仅由
            // 关键帧插入/Map::setObservation 记录。
            updateStatus((int)pts3d.size(), pnp.inliers, 0.0);
            return curr_frame_->pose_cs;
        }
    }

    // LK PnP 失败 → 重新提取 ORB 特征，回退到 ORB 匹配跟踪
    LOG_WARN("LK PnP failed (" << pts3d.size() << " 3D pts), fallback to ORB track");
    feature_matcher_.extract(curr_frame_);
    return trackFrame();
}

// ============================================================
// 双目/RGB-D 3D-3D 位姿估计
// M1.4：estimateAffine3D + Kabsch 刚体拟合已迁移至
// FrontendTracker::estimateRigid3D3D（纯几何）；验收转调 PoseGate。
// ============================================================
// ============================================================
// 重定位（LOST 状态下尝试匹配所有关键帧恢复跟踪）
// ============================================================
bool VisualOdometry::tryRelocalize() {
    // 先搜索当前子地图，再按新旧顺序搜索历史子地图。候选总数设上限，
    // 防止连续丢失时 Atlas 越大、单帧重定位开销越高。
    std::vector<std::pair<unsigned long, Frame::Ptr>> candidates;
    const int max_reloc_candidates = std::max(
        1, cfg_.max_relocalization_candidates);
    const int per_map_limit = std::max(1, max_reloc_candidates / 2);
    auto append_submap = [&](const Submap& submap) {
        if ((int)candidates.size() >= max_reloc_candidates) return;
        auto kfs = submap.map->getAllKeyFrames();
        int per_map = 0;
        for (auto kf_it = kfs.rbegin();
             kf_it != kfs.rend() && per_map < per_map_limit &&
                 (int)candidates.size() < max_reloc_candidates;
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

    // M1.2：候选的粗筛 + ORB 匹配 + PnP 几何验证已迁移至 Relocalizer（§5.3）。
    // 这里只负责构造查询（候选列表 + 锁内 3D-2D 对应供应）并转调；候选排序与
    // "首个内点达标即返回"语义与原 try_kf 循环逐行等价。Relocalizer 只返回结果，
    // 不切换 Atlas、不写 Map、不写轨迹——提交仍在下方独占锁事务中完成。
    const int reloc_min_inliers = std::max(20, cfg_.pnp_min_inliers);
    Relocalizer::Query query;
    query.curr_frame = curr_frame_;
    query.candidates = std::move(candidates);
    query.min_inliers = reloc_min_inliers;
    query.min_ratio = std::max(0.4, cfg_.pnp_min_inlier_ratio);
    query.max_rmse = cfg_.pnp_max_rmse;
    query.match_ratio = cfg_.match_ratio;
    query.ransac_pixel_threshold = cfg_.ransac_pixel_threshold;
    query.supply_points = [this](unsigned long submap_id, const Frame::Ptr& kf,
                                 const std::vector<cv::DMatch>& matches,
                                 RelocalizationPointSet& out) {
        // P1：读路径共享锁。身份/版本检查与点收集与原 try_kf 逐行一致——
        // 同一子地图内必须能按 kf->id 取回同一对象，防止后台 BA/回环在
        // PnP 前替换参考 KF/点。
        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        const auto* submap = atlas_->getSubmap(submap_id);
        if (!submap || !submap->map ||
            submap->map->getKeyFrame(kf->id).get() != kf.get()) {
            return false;
        }
        out.map = submap->map;
        out.geometry_revision = out.map->geometryRevision();
        out.submap_id = submap_id;
        for (auto& m : matches) {
            auto& mp = kf->map_points[m.queryIdx];
            if (mp) {
                out.pts3d.emplace_back((float)mp->pos_s.x(), (float)mp->pos_s.y(), (float)mp->pos_s.z());
                out.pts2d.push_back(curr_frame_->keypoints[m.trainIdx].pt);
            }
        }
        return true;
    };
    const RelocalizationResult reloc = relocalizer_.relocalize(query);
    if (reloc.quick_passed > 0)
        LOG_INFO("Reloc prefilter: " << reloc.quick_passed << "/" << reloc.quick_checked
                 << " candidates passed");

    if (reloc.accepted) {
        const int best_inliers = reloc.inliers;
        const size_t best_total = reloc.total;
        const SE3 best_pose = reloc.T_cs;
        const Frame::Ptr best_kf = reloc.kf;
        const Map::Ptr best_map = reloc.map;
        const uint64_t best_geometry_revision = reloc.geometry_revision;
        const unsigned long best_submap_id = reloc.submap_id;
        const double best_rmse = reloc.rmse;

        // M0：重定位与正常跟踪共用同一验收通路（几何 + 连续性）。
        // §3.19 根因：正常 PnP 已按 3m/0.35rad 门限拒绝 258.8m 等坏解，随后
        // tryRelocalize 只查内点/RMSE 又把相同解接受为重定位 → 单帧大跳变。
        // 现在以"丢失期匀速外推的期望位姿"为基线做连续性验收，远离基线的
        // 候选保持 LOST（不写位姿、不换子地图、状态完全不变）。
        // M3：best_pose 是候选子地图局部系（T_cs），必须组合该子地图 T_ws
        // 到世界系再与基线（世界系）比较——跨子地图坐标差异不再被当作跳变。
        // PnP 在锁外计算；最终接受必须重新绑定同一 Map/Submap/KF 与几何版本。
        // 否则后台 BA/回环可在 PnP 期间更新点/KF，旧 best_pose 会被错误地
        // 绑定到 live 新参考帧。锁保持到 Atlas 求解、activate 和帧写入完成。
        std::unique_lock<std::shared_mutex> reloc_lock(map_mutex_);
        const auto* locked_candidate = atlas_->getSubmap(best_submap_id);
        const auto locked_best_kf = locked_candidate && locked_candidate->map
            ? locked_candidate->map->getKeyFrame(best_kf->id) : nullptr;
        if (!best_map || !locked_candidate || locked_candidate->map != best_map ||
            !locked_best_kf || locked_best_kf.get() != best_kf.get() ||
            best_map->geometryRevision() != best_geometry_revision) {
            LOG_WARN("Relocation candidate geometry became stale, dropped");
            reloc_lock.unlock();
            updateStatus(0, 0, 0.0);
            return false;
        }

        unsigned long cur_sub_id = best_submap_id;
        bool cross_submap = false;
        SE3 best_T_ws;
        bool found_tws = false;
        const auto* cur_sub = atlas_->activeSubmap();
        cur_sub_id = cur_sub ? cur_sub->id : best_submap_id;
        cross_submap = cur_sub && best_submap_id != cur_sub->id;
        if (locked_candidate) {
            best_T_ws = locked_candidate->T_ws;
            found_tws = true;
        }
        // M5：跨子地图重定位——先生成 Relocalization 约束、优化 Atlas 锚点、
        // 再验收（§14.5：不立即 activate + 覆盖位姿）。事务式：失败回滚。
        std::vector<SE3> saved_tws;
        size_t saved_constraints = 0;
        SE3 saved_snap_tws = snap_.T_ws;
        PoseQuality quality;
        bool quality_ok = false;
        if (cross_submap && found_tws && has_last_valid_pose_) {
            // 约束新增、T_ws 快照、求解写回全部在同一 Atlas 写事务中。
            const auto* locked_cur_sub = atlas_->activeSubmap();
            if (!locked_cur_sub || locked_cur_sub->id != cur_sub_id ||
                !locked_candidate) {
                LOG_WARN("Atlas relocation became stale before constraint solve");
                reloc_lock.unlock();
                updateStatus(0, 0, 0.0);
                return false;
            }
            saved_tws.reserve(atlas_->submaps().size());
            for (const auto& sub : atlas_->submaps()) saved_tws.push_back(sub.T_ws);
            saved_constraints = atlas_->constraints().size();
            saved_snap_tws = snap_.T_ws;
            best_T_ws = locked_candidate->T_ws;
            // 约束：T_ws_cand = T_ws_cur · T_rel，T_rel = T_cs_cur⁻¹ · T_cs_cand
            //（相机同一世界位姿：T_cs_cur ∘ T_ws_cur⁻¹ = T_cs_cand ∘ T_ws_cand⁻¹）
            const SE3 T_cs_cur = relocBaselineWorld().inverse()
                * locked_cur_sub->T_ws;
            AtlasConstraint rc;
            rc.a = cur_sub_id;
            rc.b = best_submap_id;
            rc.T_rel = T_cs_cur.inverse() * best_pose;
            rc.weight = 1.0;
            rc.type = AtlasConstraintType::Relocalization;
            atlas_->addConstraint(rc);
            if (!solveAtlasConstraints()) {
                LOG_WARN("Atlas constraint solve failed, rolling back reloc");
                atlas_->removeConstraintsFrom(saved_constraints);
                size_t k = 0;
                for (const auto& sub : atlas_->submaps()) {
                    if (auto* saved_sub = atlas_->getSubmap(sub.id))
                        saved_sub->T_ws = saved_tws[k];
                    ++k;
                }
                snap_.T_ws = saved_snap_tws;
                reloc_lock.unlock();
                updateStatus(0, 0, 0.0);
                return false;
            }
            for (const auto& sub : atlas_->submaps()) {
                if (sub.id == best_submap_id) best_T_ws = sub.T_ws;
                if (sub.id == cur_sub_id) snap_.T_ws = sub.T_ws;
            }
            // 验收也必须在同一写事务中完成，避免失败结果短暂暴露给
            // 轨迹/Viewer 读者；拒绝时在释放锁前恢复全部 Atlas 状态。
            quality_ok = acceptPose(
                best_pose * best_T_ws.inverse(), best_inliers, best_total,
                best_rmse, true /* reloc_mode */, quality);
            if (!quality_ok) {
                LOG_WARN("Reloc rejected after Atlas solve, rolling back reloc");
                atlas_->removeConstraintsFrom(saved_constraints);
                size_t k = 0;
                for (const auto& sub : atlas_->submaps()) {
                    if (auto* saved_sub = atlas_->getSubmap(sub.id))
                        saved_sub->T_ws = saved_tws[k];
                    ++k;
                }
                snap_.T_ws = saved_snap_tws;
                reloc_lock.unlock();
                updateStatus(0, 0, 0.0);
                return false;
            }
        } else {
            quality_ok = acceptPose(
                found_tws ? best_pose * best_T_ws.inverse() : best_pose,
                best_inliers, best_total, best_rmse,
                true /* reloc_mode */, quality);
        }
        if (!quality_ok) {
            LOG_WARN("Reloc rejected by unified acceptance: inliers="
                     << best_inliers << " ratio=" << quality.inlier_ratio
                     << " dtrans=" << quality.translation
                     << "m drot=" << quality.rotation * 180.0 / M_PI << "deg");
            reloc_lock.unlock();
            updateStatus(0, 0, 0.0);
            return false;
        }
        // 同一独占事务中切换地图并绑定 live 参考 KF，旧 PnP 结果不能跨
        // geometry revision 暴露给轨迹或下一帧。
        atlas_->activate(best_submap_id);
        map_ = atlas_->activeMap();
        if (cross_submap) {
            active_keyframe_serial_ = map_ ? map_->keyFrameCount() : 0;
            last_loop_keyframe_serial_ = 0;
        }
        active_trajectory_segment_start_frame_id_ = curr_frame_->id;
        curr_frame_->pose_cs = best_pose;
        ref_frame_ = best_kf;
        // M3：本帧轨迹推/返回与验收基线需要新子地图锚
        snap_.T_ws = best_T_ws;
        snap_.map = map_;
        snap_.submap_id = best_submap_id;
        snap_.topology_revision = map_->topologyRevision();
        snap_.geometry_revision = map_->geometryRevision();
        snap_.has_ref = true;
        snap_.ref_kf_id = best_kf->id;   // M4：轨迹锚点
        snap_.ref_pose_cs = best_kf->pose_cs;
        snap_.ref_mps = best_kf->map_points;
        snap_.ref_points_s.resize(best_kf->map_points.size(), Vec3::Zero());
        for (size_t i = 0; i < best_kf->map_points.size(); i++) {
            if (best_kf->map_points[i])
                snap_.ref_points_s[i] = best_kf->map_points[i]->pos_s;
        }
        reloc_lock.unlock();
        status_.tracking_valid = true;
        status_.pose_method = "RELOCALIZE";
        status_.inlier_ratio = quality.inlier_ratio;
        status_.pose_rmse = quality.pose_rmse;
        status_.translation_delta = quality.translation;
        status_.rotation_delta = quality.rotation;
        LOG_INFO("Relocalized in submap " << best_submap_id
                 << "! inliers=" << best_inliers
                 << " dtrans=" << quality.translation << "m");
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
bool VisualOdometry::insertKeyFrame() {
    const Frame::Ptr prev_kf = ref_frame_;

    // 地图集合/建点/edges 写入统一持独占锁（异步后端与后台线程互斥）
    {
        std::unique_lock<std::shared_mutex> lock(map_mutex_);  // P1：写路径（独占）

        const auto* active_submap = atlas_->activeSubmap();
        const bool same_identity = snap_.map == map_ && active_submap &&
                                   active_submap->id == snap_.submap_id;
        if (!same_identity || snap_.geometry_revision != map_->geometryRevision()) {
            // 后端可能在本帧 PnP 之后、KF 插入之前完成回环。把旧几何帧直接
            // 插入会冻结成不在 PGO 快照内的新锚点，下一条轨迹产生 40m 级跳变。
            // 保留旧 T_ca，把本帧重基到已校正参考 KF；本帧只作为普通帧记录，
            // 下一帧基于新几何重新跟踪并决定是否插 KF。
            if (same_identity && snap_.has_ref) {
                const auto live_ref = map_->getKeyFrame(snap_.ref_kf_id);
                if (live_ref && live_ref.get() == ref_frame_.get()) {
                    const bool local_ba_only =
                        map_ == last_local_ba_commit_map_ &&
                        active_submap->id == last_local_ba_commit_submap_id_ &&
                        map_->geometryRevision() ==
                            last_local_ba_commit_geometry_revision_;
                    if (!local_ba_only) {
                        curr_frame_->pose_cs = rebaseAnchoredFramePose(
                            curr_frame_->pose_cs, snap_.ref_pose_cs,
                            live_ref->pose_cs);
                    }
                    snap_ = captureTrackingSnapshot();
                }
            }
            LOG_WARN("Keyframe insertion skipped: tracking geometry changed in-flight");
            return false;
        }

        // LK 普通帧的 keypoints/map_points 只是临时跟踪索引。关键帧必须先
        // 重提 ORB，再按新索引重算双目深度，之后才能注册 slot/正式观测；
        // 否则先 insertKeyFrame 或先建点都会在 extract 后留下反向 stale obs。
        if (cfg_.feature_method == 1) {
            feature_matcher_.extract(curr_frame_);
            computeStereoDepths();
        }
        map_->insertKeyFrame(curr_frame_);
        active_keyframe_serial_.fetch_add(1, std::memory_order_relaxed);
        // 普通帧的临时关联在注册为 KF 后计入最近命中统计。否则
        // Map::lastHitKeyframeCount 永远停在默认值 0，rolling reclaim 会
        // 把仍被车辆持续观测的强点误判为陈旧点。
        for (const auto& mp : curr_frame_->map_points)
            if (mp) map_->recordTrackingHit(mp->id);

        // 子地图重建后延迟对齐：等新子地图有 ≥3 个关键帧（拟合最低点数），
        // 与丢失点附近的历史轨迹做 Umeyama 刚体对齐
        if (submap_needs_alignment_ && map_->keyFrameCount() >= 3) {
            alignSubmapToTrajectory();
            submap_needs_alignment_ = false;
        }

        // 定期清理正式观测不足的地图点（每 20 个关键帧一次），防止地图无限增长。
        // 必须在建新点之前，避免本轮新点在正式观测尚未补齐时被清理。
        {
            PERF_SCOPE("kf.cull");
            if (map_->keyFrameCount() % 20 == 0)
                map_->cullMapPoints(2);
        }

        // 达到点预算时先回收长期未被正式观测/跟踪命中的旧点，再为当前
        // KF 建新点。当前帧、参考帧和最近窗口中的点受保护，避免把前端
        // 下一帧仍依赖的局部地图一次性清空。
        reclaimStaleMapPoints();

        // 双目/RGB-D：当前帧有视差/深度的特征直接建点（绝对尺度）
        {
            PERF_SCOPE("kf.build_points");
            createMapPointsFromStereo(curr_frame_);
            std::vector<cv::DMatch> keyframe_matches;
            if (cfg_.reuse_keyframe_matches && cfg_.feature_method == 0 &&
                !last_tracking_matches_.empty() &&
                last_tracking_ref_id_ == prev_kf->id &&
                last_tracking_curr_id_ == curr_frame_->id) {
                keyframe_matches = feature_matcher_.filterFundamental(
                    prev_kf, curr_frame_, last_tracking_matches_,
                    cfg_.ransac_pixel_threshold);
            } else {
                keyframe_matches = feature_matcher_.match(
                    prev_kf, curr_frame_, cfg_.match_ratio, true);
            }
            triangulateNewPoints(ref_frame_, curr_frame_, keyframe_matches);
        }

        // Local BA 和新点关联完成后冻结相邻 KF 测量；后续位姿图不得从已优化
        // 轨迹重算它，否则会把上一次闭环结果误当成新的里程计观测。
        if (prev_kf) {
            const auto covisibility = map_->sharedObservationCount(
                prev_kf->id, curr_frame_->id);
            odometry_edges_.push_back({
                prev_kf->id, curr_frame_->id,
                prev_kf->pose_cs * curr_frame_->pose_cs.inverse(),
                1.0 + std::log2(1.0 + static_cast<double>(covisibility))});
        }

        // 预算引擎会遍历/回收地图容器，必须在本独占事务内执行。
        enforceMapBudget();
    }
    ref_frame_ = curr_frame_;
    last_kf_frame_id_ = curr_frame_->id;   // 更新关键帧冷却基准

    LOG_INFO("New KF. mp=" << map_->mapPointCount());

    if (loop_cleanup_pending_.load(std::memory_order_acquire)) {
        if (cfg_.async_backend) {
            BackendTask cleanup;
            cleanup.type = BackendTask::Type::LoopMaintenance;
            submitBackendTask(std::move(cleanup));
        } else {
            drainLoopKeyframeCleanup();
        }
    }

    if (cfg_.async_backend) {
        // ---- 异步路径：Local BA / 回环检测+校正 提交后台线程 ----
        if (cfg_.enable_local_ba) {
            BackendTask task;
            task.type = BackendTask::Type::LocalBA;
            {
                std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：选窗为读路径
                task.window = selectLocalWindow(cfg_.local_window_size);
                task.map = map_;
                const auto* active_submap = atlas_->activeSubmap();
                task.submap_id = active_submap ? active_submap->id : 0;
            }
            task.anchor_kf_id = curr_frame_->id;
            submitBackendTask(std::move(task));
        }
        if (loop_closure_enabled_ && loop_closure_) {
            // LoopClosure 读取 KF 的 pose/descriptor；保持 map → loop 锁序。
            {
                std::shared_lock<std::shared_mutex> lock(map_mutex_);
                const auto* active = atlas_->activeSubmap();
                if (active)
                    loop_closure_->addKeyFrame(
                        curr_frame_, active->id, active->T_ws);
            }
            const auto keyframe_serial = active_keyframe_serial_.load();
            if (keyframe_serial % (unsigned long)std::max(1, cfg_.detection_interval) == 0) {
                const bool cooled = keyframe_serial - last_loop_keyframe_serial_.load()
                    >= (unsigned long)cfg_.loop_cooldown_kfs;
                if (cooled) {
                    BackendTask task;
                    task.type = BackendTask::Type::LoopClosure;
                    task.curr_kf = curr_frame_;
                    submitBackendTask(std::move(task));
                }
            }
        }
    } else {
        // ---- 同步路径（旧行为）：主线程直接执行 ----
        // 共视图滑动窗口（按共视地图点数选帧）+ Local BA
        std::vector<Frame::Ptr> window = selectLocalWindow(cfg_.local_window_size);
        if (cfg_.enable_local_ba) {
            // M1：只读快照 → 纯计算 → 结果提交（stale 检查 + 跳过活动参考帧）。
            // 同步模式下每帧都可能插 KF，BA 若无跳过保护会把刚验收并记入轨迹的
            // 位姿挪动数米（地图质量差时可达 ~10m），连续轨迹条目随即产生
            // 10m 级跳变（KITTI 00 实测 205 次 >10m 跳变的直接来源，M0 修复）。
            runWindowLocalBA(window, curr_frame_->id);
        }

        // ============================================================
        // Phase 2 回环钩子：新关键帧入词袋数据库；每 N 个关键帧检测一次。
        // 放在 Local BA 之后，避免两处位姿修改互相干扰。
        // ============================================================
        if (loop_closure_enabled_ && loop_closure_) {
            // LoopClosure 读取 KF 的 pose/descriptor；保持 map → loop 锁序。
            {
                std::shared_lock<std::shared_mutex> lock(map_mutex_);
                const auto* active = atlas_->activeSubmap();
                if (active)
                    loop_closure_->addKeyFrame(
                        curr_frame_, active->id, active->T_ws);
            }
            const auto keyframe_serial = active_keyframe_serial_.load();
            if (keyframe_serial % (unsigned long)std::max(1, cfg_.detection_interval) == 0) {
                // 回环校正冷却：上次校正后至少间隔 N 个关键帧才允许再次校正。
                // 同一区域会被词袋反复命中（分数高），连续校正会让 S_global
                // 叠加冲突、轨迹被反复拉扯变形。以关键帧数计（与 KF 密度无关）。
                const bool cooled = keyframe_serial - last_loop_keyframe_serial_.load()
                    >= (unsigned long)cfg_.loop_cooldown_kfs;
                if (cooled) {
                    std::vector<LoopClosure::LoopCandidate> cands;
                    {
                        // 保持 map → LoopClosure 锁序；候选发现本身不构成提交，
                        // 最终验证/快照/写回由 handleLoopCorrection 事务重做。
                        std::shared_lock<std::shared_mutex> lock(map_mutex_);
                        const auto* active = atlas_->activeSubmap();
                        if (active) {
                            LoopClosure::SubmapPoses poses;
                            for (const auto& sub : atlas_->submaps())
                                poses.emplace(sub.id, sub.T_ws);
                            cands = loop_closure_->detectLoop(
                                curr_frame_, active->id, active->T_ws, poses);
                        }
                    }
                    const int verification_limit =
                        std::max(1, cfg_.loop_verification_limit);
                    int verification_count = 0;
                    for (const auto& cand : cands) {
                        if (verification_count++ >= verification_limit) break;
                        LoopCorrectionContext context;
                        {
                            std::shared_lock<std::shared_mutex> lock(map_mutex_);
                            const auto live_curr = map_->getKeyFrame(curr_frame_->id);
                            const auto* active_submap = atlas_->activeSubmap();
                            const auto* loop_submap = atlas_->getSubmap(cand.submap_id);
                            const auto loop_map = loop_submap ? loop_submap->map : nullptr;
                            const auto live_loop = loop_map && cand.frame
                                ? loop_map->getKeyFrame(cand.frame->id) : nullptr;
                            if (!live_curr || live_curr.get() != curr_frame_.get() ||
                                !live_loop || live_loop.get() != cand.frame.get() ||
                                !active_submap || !loop_submap) continue;
                            context.map = map_;
                            context.loop_map = loop_map;
                            context.submap_id = active_submap->id;
                            context.loop_submap_id = cand.submap_id;
                            context.topology_revision = map_->topologyRevision();
                            context.geometry_revision = map_->geometryRevision();
                            context.curr_kf_id = curr_frame_->id;
                            context.loop_kf_id = cand.frame->id;
                            context.curr_kf = curr_frame_;
                            context.loop_kf = cand.frame;
                        }
                        if (context.map && handleLoopCorrection(context))
                            break;  // 第一个成功提交的候选即回环
                    }
                }
            }
        }
    }

    updateStatus(status_.matches, status_.inliers, status_.parallax);
    return true;
}

// ============================================================
// Phase 2 回环校正：前缀快照 → Essential Graph 锁外求解 → 尾段重基提交。
// 同子图的验证/快照只持共享锁，前端跟踪读不等待；耗时图求解不持 Map 锁，
// 只在最终身份/revision 复验、尾段合并和原子写回时短持独占锁。
// ============================================================
bool VisualOdometry::handleLoopCorrection(
    const LoopCorrectionContext& context) {
    PERF_SCOPE("loop.transaction");
    // 读阶段与前端 tracking 共享；只暂停 KF/点的拓扑写入。
    // 锁序固定为 map → LoopClosure，禁止反向回调。
    std::shared_lock<std::shared_mutex> map_lock(map_mutex_);

    const auto* active_submap = atlas_->activeSubmap();
    const auto live_curr = context.map
        ? context.map->getKeyFrame(context.curr_kf_id) : nullptr;
    const auto* loop_submap = atlas_->getSubmap(context.loop_submap_id);
    const auto live_loop = context.loop_map
        ? context.loop_map->getKeyFrame(context.loop_kf_id) : nullptr;
    if (!context.map || !context.loop_map || map_ != context.map ||
        !active_submap || !loop_submap || loop_submap->map != context.loop_map ||
        active_submap->id != context.submap_id ||
        !live_curr || live_curr.get() != context.curr_kf.get() ||
        !live_loop || live_loop.get() != context.loop_kf.get()) {
        LOG_WARN("Loop correction identity changed before transaction");
        return false;
    }

    // 候选可能在锁外发现后经历 Local BA/新增 KF；不能复用旧 T_loop_curr。
    // 在同一 map 共享快照下重新验证，LoopClosure 只取得自己的 mutex，故无反向
    // map 锁重入，且验证使用的 KF/地图点状态与下方全图快照完全一致。
    SE3 T_loop_curr;
    double verified_reference_time = live_loop->timestamp;
    bool loop_verified = false;
    if (context.preverified &&
        context.preverified_geometry_revision == context.map->geometryRevision()) {
        T_loop_curr = context.preverified_T_loop_curr;
        verified_reference_time = context.preverified_reference_time;
        loop_verified = true;
    } else if (loop_closure_) {
        loop_verified = loop_closure_->verifyLoop(
            live_curr, live_loop, T_loop_curr);
    }
    if (!loop_verified) {
        // 单个历史 KF 可能因点回收/视角变化而稀疏；只在原验证失败后，
        // 从同一历史 Map 构造有界共视+时间邻域。区域 PnP 复用同一几何门，
        // 不降低内点数量/比例，也不依赖数据集或 GT。
        LoopRegionSnapshot region;
        LoopRegionResult region_result;
        loop_verified = LoopRegionVerifier::build(
                            *context.loop_map, live_loop,
                            context.loop_submap_id, cfg_.loop_region, region) &&
                        LoopRegionVerifier::verify(
                            region, live_curr, camera_, cfg_.loop_region,
                            T_loop_curr, &region_result);
        if (loop_verified) {
            if (const auto support = context.loop_map->getKeyFrame(
                    region_result.supporting_keyframe_id))
                verified_reference_time = support->timestamp;
            LOG_INFO("Loop correction verified by historical region: submap#"
                     << context.loop_submap_id << " anchor kf#"
                     << context.loop_kf_id << " support kf#"
                     << region_result.supporting_keyframe_id
                     << " points=" << region.points.size());
        }
    }
    if (!loop_verified) {
        LOG_WARN("Loop correction verification failed in transaction");
        return false;
    }

    // 跨子地图回环只校正 Atlas 锚点，不把旧图候选伪装成本图 KF，也不搬动
    // 两张图中的局部点/KF。约束和全部 T_ws 作为一个事务提交，求解失败回滚。
    if (context.loop_submap_id != context.submap_id) {
        const uint64_t verified_geometry = context.map->geometryRevision();
        map_lock.unlock();
        std::unique_lock<std::shared_mutex> cross_lock(map_mutex_);
        const auto* cross_active = atlas_->activeSubmap();
        const auto* cross_loop_submap = atlas_->getSubmap(context.loop_submap_id);
        const auto cross_curr = context.map->getKeyFrame(context.curr_kf_id);
        const auto cross_loop = context.loop_map->getKeyFrame(context.loop_kf_id);
        if (map_ != context.map || !cross_active || !cross_loop_submap ||
            cross_active->id != context.submap_id ||
            cross_loop_submap->map != context.loop_map ||
            context.map->geometryRevision() != verified_geometry ||
            !cross_curr || cross_curr.get() != context.curr_kf.get() ||
            !cross_loop || cross_loop.get() != context.loop_kf.get()) {
            LOG_WARN("Cross-submap loop changed before write transaction");
            return false;
        }
        std::vector<std::pair<SubmapId, SE3>> saved_tws;
        saved_tws.reserve(atlas_->submaps().size());
        for (const auto& sub : atlas_->submaps())
            saved_tws.emplace_back(sub.id, sub.T_ws);
        const size_t saved_constraints = atlas_->constraints().size();

        atlas_->addConstraint(makeCrossSubmapLoopConstraint(
            context.submap_id, context.loop_submap_id,
            live_curr->pose_cs, live_loop->pose_cs, T_loop_curr, 10.0));
        if (!solveAtlasConstraints()) {
            atlas_->removeConstraintsFrom(saved_constraints);
            for (const auto& [sid, T_ws] : saved_tws)
                if (auto* sub = atlas_->getSubmap(sid)) sub->T_ws = T_ws;
            LOG_WARN("Cross-submap loop solve failed; Atlas transaction rolled back");
            return false;
        }

        // 总 chi2 下降不代表所有已提交回环仍然自洽。Atlas 只有每子图一个
        // 刚体锚，多段内部漂移会让新边把旧闭环重新拉开；Huber 核又可能
        // 把这种冲突隐藏成“优化成功”。提交前逐条检查高置信 LoopClosure
        // 残差，任一旧/新闭环超过 5m/20deg 就整笔回滚。TrackingBridge 是
        // 低置信连续性先验，不进入此硬门。
        double max_loop_residual_m = 0.0;
        double max_loop_residual_rad = 0.0;
        const bool loop_constraints_consistent = atlas_->loopConstraintsConsistent(
            5.0, 20.0 * M_PI / 180.0,
            &max_loop_residual_m, &max_loop_residual_rad);
        if (!loop_constraints_consistent) {
            atlas_->removeConstraintsFrom(saved_constraints);
            for (const auto& [sid, T_ws] : saved_tws)
                if (auto* sub = atlas_->getSubmap(sid)) sub->T_ws = T_ws;
            LOG_WARN("Cross-submap loop rejected: Atlas loop residual "
                     << max_loop_residual_m << "m/"
                     << max_loop_residual_rad * 180.0 / M_PI
                     << "deg exceeds 5m/20deg; transaction rolled back");
            return false;
        }

        const auto* corrected_active = atlas_->activeSubmap();
        if (!corrected_active || corrected_active->id != context.submap_id) {
            atlas_->removeConstraintsFrom(saved_constraints);
            for (const auto& [sid, T_ws] : saved_tws)
                if (auto* sub = atlas_->getSubmap(sid)) sub->T_ws = T_ws;
            LOG_WARN("Cross-submap loop changed active identity; rolled back");
            return false;
        }

        // Atlas 每个子图只有一个刚体锚。若新回环直接把整个活动子图搬到
        // 新位置，子图首帧也会被同量搬走，从而在 TrackingBridge 边界产生
        // 单帧跳变。地图仍使用优化后的刚体锚；持久轨迹则从子图首帧的
        // 零校正平滑过渡到回环端点的完整校正，等价于把累计漂移沿该段
        // 里程计分布，而不是把误差集中到滚动边界。只依赖拓扑/时间顺序，
        // 不读取 GT，也不包含序列或帧号特例。
        SE3 old_active_T_ws = corrected_active->T_ws;
        for (const auto& [sid, T_ws] : saved_tws) {
            if (sid == context.submap_id) {
                old_active_T_ws = T_ws;
                break;
            }
        }
        const SE3 new_active_T_ws = corrected_active->T_ws;
        if ((old_active_T_ws.matrix() - new_active_T_ws.matrix()).norm() > 1e-12) {
            std::lock_guard<std::mutex> traj_lock(traj_mutex_);
            const unsigned long segment_start =
                active_trajectory_segment_start_frame_id_;
            unsigned long first_frame_id = context.curr_kf_id;
            bool found_record = false;
            for (const auto& rec : pose_records_) {
                if (rec.submap_id != context.submap_id ||
                    rec.frame_id < segment_start) continue;
                first_frame_id = found_record
                    ? std::min(first_frame_id, rec.frame_id) : rec.frame_id;
                found_record = true;
            }
            for (auto& rec : pose_records_) {
                if (rec.submap_id != context.submap_id) continue;
                const auto anchor = context.map->getKeyFrame(rec.anchor_kf_id);
                const SE3 anchor_pose = anchor
                    ? anchor->pose_cs : rec.anchor_pose_cs;
                const double alpha = submapTrajectoryCorrectionAlpha(
                    rec.frame_id, segment_start, first_frame_id,
                    context.curr_kf_id);
                rec.T_ca = rebaseTrajectoryForSubmapAnchor(
                    rec.T_ca, anchor_pose,
                    old_active_T_ws, new_active_T_ws, alpha);
            }
        }

        // 后台事务只提交 Atlas。snap_、curr_frame_、last_valid_pose_world_
        // 都归前端线程所有；直接在 worker 写它们会与 trackFrame/addFrame
        // 数据竞争，并把任务 KF 与已经前进的实时帧混为一谈。前端会在
        // 帧首/帧尾安全点通过 syncFrontendAnchor() 原子重基。
        last_loop_keyframe_serial_ = active_keyframe_serial_.load();
        loop_closure_count_++;
        LOG_INFO("Cross-submap loop closed! submap#" << context.loop_submap_id
                 << " kf#" << context.loop_kf_id << " -> submap#"
                 << context.submap_id << " kf#" << context.curr_kf_id
                 << " reference_time=" << verified_reference_time
                 << " query_time=" << live_curr->timestamp
                 << " detection_time=" << live_curr->timestamp
                 << " (total " << loop_closure_count_.load() << ")");
        return true;
    }

    // ---- 阶段 1：在同一事务锁内收集只读快照（M1 纯数据）----
    std::vector<KeyframeState> kf_states;                 // 全部 KF（id 升序，pose_cs）
    std::unordered_map<unsigned long, SE3> old_pose;      // KF id → 优化前 pose_cs
    std::unordered_map<unsigned long, unsigned long> mp_ref_kf;  // 点 id → 参考 KF id
    std::vector<LandmarkState> points;                    // 全量点快照（坐标 + 观测数）
    std::vector<ObservationState> observations;           // 全量正式观测快照
    std::vector<Constraint> constraints;                  // 里程计边 + 累计回环边 + 新边
    std::unordered_map<KeyframeId, Frame::Ptr> prefix_kf_identity;
    std::unordered_map<MapPointId, MapPoint::Ptr> prefix_point_identity;
    uint64_t snap_topology = 0, snap_geometry = 0;
    unsigned long submap_id = 0;
    KeyframeId prefix_endpoint_id = 0;
    SE3 old_prefix_endpoint_pose;
    Map::Ptr snap_map;
    {
        PERF_SCOPE("loop.snapshot");
        snap_map = map_;
        auto all_kfs = map_->getAllKeyFrames();
        if (all_kfs.size() < 2) return false;
        submap_id = context.submap_id;
        snap_topology = map_->topologyRevision();
        snap_geometry = map_->geometryRevision();
        kf_states.reserve(all_kfs.size());
        prefix_kf_identity.reserve(all_kfs.size());
        for (const auto& kf : all_kfs) {
            old_pose.emplace(kf->id, kf->pose_cs);
            prefix_kf_identity.emplace(kf->id, kf);
            KeyframeState ks;
            ks.id = kf->id;
            ks.pose_cs = kf->pose_cs;
            kf_states.push_back(std::move(ks));
            appendFormalObservations(kf, observations);
        }
        prefix_endpoint_id = kf_states.back().id;
        old_prefix_endpoint_pose = kf_states.back().pose_cs;
        // 点参考 KF 映射（最早观测）+ 全量点快照。observations 已由
        // appendFormalObservations 单遍生成；这里再单遍取最小 KF id，避免
        // 在全图快照中做 KF×Observation 的灾难性嵌套扫描。
        for (const auto& observation : observations) {
            auto [it, inserted] = mp_ref_kf.emplace(
                observation.map_point_id, observation.keyframe_id);
            if (!inserted && observation.keyframe_id < it->second)
                it->second = observation.keyframe_id;
        }
        const auto all_points = map_->getAllMapPoints();
        prefix_point_identity.reserve(all_points.size());
        for (auto& mp : all_points) {
            points.push_back({mp->id, mp->pos_s,
                              static_cast<int>(mp->observationCount())});
            prefix_point_identity.emplace(mp->id, mp);
        }
        // 位姿图边：里程计边（无核）+ 累计回环边 + 本次新边（Huber + 预检）
        constraints.reserve(odometry_edges_.size() + loop_edges_.size() + 1);
        for (const auto& e : odometry_edges_)
            constraints.push_back({e.a, e.b, e.T_rel, e.weight, false});
        for (const auto& e : loop_edges_)
            constraints.push_back({e.a, e.b, e.T_rel, e.weight, true});
        // 同区域回环去重：与已有回环边端点相距 <200 KF 的重叠边拒绝——
        // 同区域连续校正会让 PGO 约束叠加拉扯（abcd7 实测 #2(2420,3367) 后
        // 43 KF 内 #3(2454,3410) 再校正 88m，路径拉长 90m）。
        {
            bool dup = false;
            for (const auto& e : constraints) {
                if (!e.is_loop) continue;
                const bool a_close = std::abs((long long)e.a -
                                               (long long)context.loop_kf_id) < 200;
                const bool b_close = std::abs((long long)e.b -
                                               (long long)context.curr_kf_id) < 200;
                if (a_close && b_close) { dup = true; break; }
            }
            if (dup) {
                LOG_WARN("Loop closure rejected: duplicate region (kf#"
                         << context.loop_kf_id << " -> kf#"
                         << context.curr_kf_id << ")");
                return false;
            }
        }
        constraints.push_back({context.loop_kf_id, context.curr_kf_id,
                               T_loop_curr, 10.0, true});
    }
    // 冻结前缀到此为止。后续图构建/求解/点同步都只消费深拷贝，
    // 必须在 map_mutex_ 之外执行，前端可继续跟踪并向同一子图追加尾段。
    map_lock.unlock();

    // ---- 阶段 2：锁外构建 Essential Anchor Graph 并纯计算 ----
    OptimizationSnapshot pgo_snap;
    pgo_snap.submap_id = submap_id;
    pgo_snap.topology_revision = snap_topology;
    pgo_snap.geometry_revision = snap_geometry;
    pgo_snap.keyframes = kf_states;
    pgo_snap.constraints = constraints;

    // final_pose：KF id → 最终位姿（PGO 结果，GBA 后部分精修）
    std::unordered_map<unsigned long, SE3> final_pose;
    std::unordered_map<unsigned long, Vec3> synced;   // 点 id → 同步后坐标
    OptimizationMetrics pgo_metrics;
    {
        PERF_SCOPE("loop.pose_graph");
        const auto essential = buildEssentialAnchorGraph(
            pgo_snap, {static_cast<size_t>(std::max(4, cfg_.pose_graph_max_anchors)),
                       static_cast<size_t>(std::max(1, cfg_.pose_graph_anchor_stride))});
        auto pgo = Optimizer::solvePoseGraph(
            essential.snapshot, cfg_.pose_graph_iterations);
        if (!pgo.valid) {
            LOG_WARN("Loop correction skipped: pose graph backend unavailable or constraints invalid");
            return false;  // 不保留失败的回环边（未提交）
        }
        pgo_metrics = pgo.metrics;
        const auto propagated = propagateAnchorCorrections(essential, pgo);
        if (propagated.size() != kf_states.size()) {
            LOG_WARN("Loop correction skipped: incomplete anchor propagation");
            return false;
        }
        final_pose.reserve(propagated.size());
        for (const auto& u : propagated)
            final_pose.emplace(u.id, u.pose_cs);

        // 2b. 地图点粗同步：按参考 KF 位姿增量移动（correction = new⁻¹ * old）
        synced.reserve(points.size());
        for (const auto& lm : points) {
            auto ref = mp_ref_kf.find(lm.id);
            if (ref == mp_ref_kf.end()) continue;
            auto old = old_pose.find(ref->second);
            auto np = final_pose.find(ref->second);
            if (old == old_pose.end() || np == final_pose.end()) continue;
            synced.emplace(lm.id, np->second.inverse() * old->second * lm.pos_s);
        }

        // 2c. 全局 BA（快照）：采样每 3 个 KF + 末帧（规模控制）。
        //     §3.18 已知的 sampled-GBA 锯齿缺陷保留旧语义，由 M4 锚定轨迹根治。
        OptimizationSnapshot gba_snap;
        gba_snap.submap_id = submap_id;
        gba_snap.topology_revision = snap_topology;
        gba_snap.geometry_revision = snap_geometry;
        for (size_t i = 0; i < kf_states.size(); i += 3) {
            KeyframeState gs = kf_states[i];
            auto it = final_pose.find(gs.id);
            if (it != final_pose.end()) gs.pose_cs = it->second;
            gba_snap.keyframes.push_back(std::move(gs));
        }
        if (gba_snap.keyframes.empty()
            || gba_snap.keyframes.back().id != kf_states.back().id) {
            KeyframeState gs = kf_states.back();
            auto it = final_pose.find(gs.id);
            if (it != final_pose.end()) gs.pose_cs = it->second;
            gba_snap.keyframes.push_back(std::move(gs));
        }
        std::unordered_set<KeyframeId> gba_keyframes;
        for (const auto& kf : gba_snap.keyframes) gba_keyframes.insert(kf.id);
        for (const auto& observation : observations) {
            if (gba_keyframes.contains(observation.keyframe_id))
                gba_snap.observations.push_back(observation);
        }
        gba_snap.landmarks = points;  // 观测数保留（≥3 过滤）
        for (auto& lm : gba_snap.landmarks) {
            auto it = synced.find(lm.id);
            if (it != synced.end()) lm.pos_s = it->second;
        }
        if (gba_snap.keyframes.size() >= 2)
            gba_snap.fixed_kf_ids = {gba_snap.keyframes[0].id, gba_snap.keyframes[1].id};
        if (cfg_.global_ba_iterations > 0) {
            PERF_SCOPE("loop.global_ba");
            // sampled-GBA（每 3 个 KF 采样）与锚定轨迹不兼容：采样 KF 被 GBA
            // 精修、未采样 KF 保持 PGO 位姿 → 锚定轨迹出现三周期锯齿
            // （§3.18 预测；M4 删除插值后实测 RPE 0.095→0.87 回归）。
            // 全量 GBA 在同步路径不可行（分钟级），M6 COW 异步后端再恢复。
            auto gba = Optimizer::solveLocalBA(camera_, gba_snap,
                                               cfg_.global_ba_iterations,
                                               std::nullopt, 1500);
            if (gba.valid) {
                for (const auto& u : gba.poses) final_pose[u.id] = u.pose_cs;
                for (const auto& p : gba.points) synced[p.id] = p.pos_s;
            } else {
                LOG_WARN("Loop global BA failed, keeping pose graph result");
            }
        }

        // 2d.（M4 删除）轨迹不再全量插值重写：普通帧锚定关键帧，
        // 世界位姿读时组合（composeRecordWorld），回环校正只更新锚点
        // KF 局部位姿，轨迹自动跟随——插值残留/锯齿类缺陷整体消失。
    }

    // ---- 阶段 3：短独占临界区内验证前缀身份、重基尾段、原子提交 ----
        PERF_SCOPE("loop.commit");
        std::unique_lock<std::shared_mutex> commit_lock(map_mutex_);
        const auto* commit_submap = atlas_->activeSubmap();
        if (map_ != snap_map || !commit_submap || commit_submap->id != submap_id) {
            LOG_WARN("Loop correction stale: active submap/map changed, dropped");
            return false;
        }
        if (snap_geometry != map_->geometryRevision()) {
            LOG_WARN("Loop correction stale (snap geo " << snap_geometry
                     << " vs live " << map_->geometryRevision()
                     << "), dropped");
            return false;
        }

        // 拓扑可在锁外求解期间只做尾部追加。冻结前缀中任一 KF/点被
        // 删除或同 id 对象被替换，都会整笔拒绝，不做部分成功。
        for (const auto& [id, expected] : prefix_kf_identity) {
            const auto live = map_->getKeyFrame(id);
            if (!live || live.get() != expected.get()) {
                LOG_WARN("Loop correction stale: prefix keyframe identity changed kf#"
                         << id);
                return false;
            }
        }
        for (const auto& [id, expected] : prefix_point_identity) {
            const auto live = map_->getMapPoint(id);
            if (!live || live.get() != expected.get()) {
                LOG_WARN("Loop correction stale: prefix point identity changed mp#"
                         << id);
                return false;
            }
        }

        const auto endpoint_new = final_pose.find(prefix_endpoint_id);
        if (endpoint_new == final_pose.end()) {
            LOG_WARN("Loop correction stale: prefix endpoint was not propagated");
            return false;
        }
        std::unordered_map<KeyframeId, SE3> live_old_pose;
        const auto live_keyframes = map_->getAllKeyFrames();
        live_old_pose.reserve(live_keyframes.size());
        final_pose.reserve(live_keyframes.size());
        size_t tail_keyframes = 0;
        for (const auto& kf : live_keyframes) {
            live_old_pose.emplace(kf->id, kf->pose_cs);
            if (prefix_kf_identity.contains(kf->id)) continue;
            final_pose.emplace(kf->id, rebaseTailPose(
                kf->pose_cs, old_prefix_endpoint_pose, endpoint_new->second));
            ++tail_keyframes;
        }

        // 快照后新建的点不得留在旧尾段坐标中。优先跟随其最早存活观测 KF；
        // 若点尚无正式观测，则跟随冻结端点的刚体校正。
        size_t tail_points = 0;
        for (const auto& mp : map_->getAllMapPoints()) {
            if (prefix_point_identity.contains(mp->id)) continue;
            const SE3* old_ref = &old_prefix_endpoint_pose;
            const SE3* new_ref = &endpoint_new->second;
            for (const auto& observation : mp->observations()) {
                const auto old_it = live_old_pose.find(observation.keyframe_id);
                const auto new_it = final_pose.find(observation.keyframe_id);
                if (old_it == live_old_pose.end() || new_it == final_pose.end())
                    continue;
                old_ref = &old_it->second;
                new_ref = &new_it->second;
                break;
            }
            synced.emplace(mp->id, rebaseTailPoint(
                mp->pos_s, *old_ref, *new_ref));
            ++tail_points;
        }

        OptimizationResult loop_result;
        loop_result.submap_id = submap_id;
        loop_result.base_topology_revision = snap_topology;
        loop_result.base_geometry_revision = snap_geometry;
        loop_result.metrics = pgo_metrics;
        loop_result.valid = true;
        loop_result.poses.reserve(final_pose.size());
        loop_result.points.reserve(synced.size());
        for (const auto& [kf_id, pose] : final_pose)
            loop_result.poses.push_back({kf_id, pose});
        for (const auto& [mp_id, pos] : synced)
            loop_result.points.push_back({mp_id, pos});

        // 全量写回（无 skip）：位姿与点必须一致写回，否则端点 KF 出现
        // "位姿旧、点新"失配 → 下一帧跟踪的 T_ca 变 44m 级 → 轨迹跳变
        // （abcd4/abcd9 实测：loop_skip 保护端点位姿但点仍被写回）。
        const CommitStatus commit_status =
            BackendCommitter::commit(map_, loop_result, {}, 0.0, snap_map);
        // §6.4（M2.3 遗留清理）：提交结果上报（committed/stale/invalid/not_found）
        backend_scheduler_.recordTaskOutcome(toTaskOutcome(commit_status));
        if (commit_status != CommitStatus::COMMITTED) {
            LOG_WARN("Loop correction commit failed (status "
                     << (int)commit_status << ")");
            return false;
        }
        LOG_INFO("Loop prefix commit: frozen_kf=" << prefix_kf_identity.size()
                 << " tail_kf=" << tail_keyframes
                 << " frozen_points=" << prefix_point_identity.size()
                 << " tail_points=" << tail_points);
        {
            // 位姿图是全局校正：T_ca 保持不变，让历史轨迹跟随新 KF pose；
            // 同时刷新剔除兜底快照，避免该 anchor 日后被预算回收时轨迹
            // 突然退回旧几何。
            std::lock_guard<std::mutex> traj_lock(traj_mutex_);
            for (auto& rec : pose_records_) {
                if (rec.submap_id != submap_id) continue;
                const auto anchor = map_->getKeyFrame(rec.anchor_kf_id);
                if (anchor) rec.anchor_pose_cs = anchor->pose_cs;
            }
        }
        // 保留累积回环边（含本次，优化成功）
        loop_edges_.clear();
        for (const auto& c : constraints)
            if (c.is_loop) loop_edges_.push_back({c.a, c.b, c.T_rel, c.weight});
        // 冷却基准更新留在锁内：锁外解引用 map_ 会与前端 createSubmap 的
        // map_ 成员交换并发（shared_ptr 读写竞争，§3.16）
        last_loop_keyframe_serial_ = active_keyframe_serial_.load();
    loop_closure_count_++;
    LOG_INFO("Loop closed! kf#" << context.loop_kf_id << " -> kf#"
             << context.curr_kf_id
             << " reference_time=" << verified_reference_time
             << " query_time=" << live_curr->timestamp
             << " detection_time=" << live_curr->timestamp
             << " (total " << loop_closure_count_.load() << ")");
    return true;
}

// ============================================================
// 异步后端（P2-1）：后台线程执行 Local BA / 回环检测+校正。
// 数据隔离原则：优化永远在"快照数据"上执行（快照帧 pose 深拷贝、
// 快照点对象深拷贝），不触碰地图集合；写回在 map_mutex_ 保护下进行。
// ============================================================
Frame::Ptr VisualOdometry::snapshotFrame(
    const Frame::Ptr& kf,
    std::unordered_map<unsigned long, MapPoint::Ptr>& snap_cache,
    const std::unordered_set<unsigned long>* keep_points,
    bool with_descriptors) {
    auto snap = std::make_shared<Frame>(kf->id, 0.0);
    snap->pose_cs = kf->pose_cs;
    snap->keypoints = kf->keypoints;
    snap->pts_c = kf->pts_c;
    if (with_descriptors && !kf->descriptors.empty())
        snap->descriptors = kf->descriptors.clone();
    snap->map_points.resize(kf->map_points.size(), nullptr);
    for (size_t i = 0; i < kf->map_points.size(); i++) {
        const auto& mp = kf->map_points[i];
        if (!mp) continue;
        if (keep_points && !keep_points->count(mp->id)) continue;
        auto it = snap_cache.find(mp->id);
        if (it != snap_cache.end()) {
            snap->map_points[i] = it->second;
            continue;
        }
        auto smp = std::make_shared<MapPoint>(mp->id);
        smp->pos_s = mp->pos_s;
        if (with_descriptors && !mp->descriptor.empty())
            smp->descriptor = mp->descriptor.clone();
        snap_cache.emplace(mp->id, smp);
        snap->map_points[i] = smp;
    }
    return snap;
}

// ============================================================
// M2：前端只读快照
// ============================================================
VisualOdometry::TrackingSnapshot VisualOdometry::captureTrackingSnapshot() {
    TrackingSnapshot snap;
    snap.map = map_;
    snap.topology_revision = map_->topologyRevision();
    snap.geometry_revision = map_->geometryRevision();
    const auto* sub = atlas_->activeSubmap();
    snap.submap_id = sub ? sub->id : 0;
    snap.T_ws = sub ? sub->T_ws : SE3();
    if (ref_frame_) {
        snap.has_ref = true;
        snap.ref_kf_id = ref_frame_->id;   // M4：轨迹锚点
        snap.ref_pose_cs = ref_frame_->pose_cs;
        snap.ref_mps = ref_frame_->map_points;  // 指针（关联写引用用）
        snap.ref_points_s.resize(ref_frame_->map_points.size(), Vec3::Zero());
        for (size_t i = 0; i < ref_frame_->map_points.size(); i++) {
            if (ref_frame_->map_points[i])
                snap.ref_points_s[i] = ref_frame_->map_points[i]->pos_s;
        }
        if (snap_ref_desc_kf_id_ != ref_frame_->id ||
            snap_ref_desc_topology_rev_ != snap.topology_revision ||
            snap_ref_desc_geo_rev_ != snap.geometry_revision) {
            snap_ref_desc_kf_id_ = ref_frame_->id;
            snap_ref_desc_topology_rev_ = snap.topology_revision;
            snap_ref_desc_geo_rev_ = snap.geometry_revision;
            snap_ref_descs_.clear();
            snap_ref_descs_.resize(ref_frame_->map_points.size());
            for (size_t i = 0; i < ref_frame_->map_points.size(); ++i) {
                const auto& mp = ref_frame_->map_points[i];
                if (mp && !mp->descriptor.empty())
                    snap_ref_descs_[i] = mp->descriptor.clone();
            }
        }
        snap.ref_descs = snap_ref_descs_;

        // ---- 方案 B：共视图局部地图（按 ref KF + 几何版本缓存）----
        // covisibleKeyframes 是 O(KF²) 全量扫描；只在参考 KF 或几何版本
        // 变化时重收集（后端 BA/回环提交会 bumpGeometry，缓存自动失效）。
        // 点预算按共视降序截断，避免大图上每帧拷贝全部点。
        snap.local_map_kf_id = ref_frame_->id;
        if (cfg_.local_map_tracking &&
            (snap_local_map_kf_id_ != ref_frame_->id ||
             snap_local_map_topology_rev_ != snap.topology_revision ||
             snap_local_map_geo_rev_ != map_->geometryRevision())) {
            snap_local_map_kf_id_ = ref_frame_->id;
            snap_local_map_topology_rev_ = snap.topology_revision;
            snap_local_map_geo_rev_ = map_->geometryRevision();
            snap_local_points_s_.clear();
            snap_local_descs_.clear();
            snap_local_mps_.clear();
            std::unordered_set<MapPointId> seen;
            auto add_local_point = [&](const MapPoint::Ptr& mp) {
                if (!mp || mp->descriptor.empty()) return;
                if (!seen.insert(mp->id).second) return;   // 去重（共视 KF 共享点）
                if ((int)snap_local_mps_.size() >= cfg_.local_map_max_points) return;
                snap_local_points_s_.push_back(mp->pos_s);
                snap_local_descs_.push_back(mp->descriptor.clone());
                snap_local_mps_.push_back(mp);
            };
            // 参考 KF 自身的点优先（与 ref 跟踪同源，最可信）
            for (const auto& mp : ref_frame_->map_points) add_local_point(mp);
            // 再按共视点数降序补充共视 KF 的点
            for (const auto& cov : map_->covisibleKeyframes(
                     ref_frame_->id, (size_t)cfg_.local_map_min_shared)) {
                if ((int)snap_local_mps_.size() >= cfg_.local_map_max_points) break;
                const auto kf = map_->getKeyFrame(cov.keyframe_id);
                if (!kf || kf.get() == ref_frame_.get()) continue;
                for (const auto& mp : kf->map_points) add_local_point(mp);
            }
        }
        if (cfg_.local_map_tracking &&
            snap_local_map_kf_id_ == ref_frame_->id) {
            snap.local_points_s = snap_local_points_s_;
            snap.local_descs = snap_local_descs_;
            snap.local_mps = snap_local_mps_;
        }
    }
    return snap;
}

// ============================================================
// M1：Optimizer 只读快照 / 结果提交
// ============================================================
void VisualOdometry::runWindowLocalBA(
    const std::vector<Frame::Ptr>& window, KeyframeId anchor_kf_id,
    const Map::Ptr& expected_map,
    std::optional<unsigned long> expected_submap_id) {
    if (window.empty() || !cfg_.enable_local_ba) return;
    PERF_SCOPE("kf.local_ba");
    OptimizationSnapshot snap;
    Map::Ptr snapshot_map;
    unsigned long snapshot_submap_id = 0;
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        const auto* active_submap = atlas_->activeSubmap();
        if (!active_submap ||
            (expected_map && map_ != expected_map) ||
            (expected_submap_id && active_submap->id != *expected_submap_id)) {
            LOG_WARN("Local BA task dropped: Map/Submap identity changed before snapshot");
            return;
        }
        snapshot_map = map_;
        snapshot_submap_id = active_submap->id;
        // Optimizer 的有效 BA 边要求同一点至少有 3 个正式关键帧观测；
        // 在快照入口就执行同一门槛，避免先构建无效弱点再原样重算一次。
        snap = buildLocalBASnapshot(window, anchor_kf_id, 3);
    }
    if (snap.keyframes.size() < 2) return;
    // 跳过提交任务捕获的 anchor KF（通常是前端 ref_frame_；§3.16/§M0：
    // 后端写回其位姿会被前端误判为跳变 → LOST 带）。
    const std::unordered_set<unsigned long> skip{anchor_kf_id};
    const size_t ba_max_points = cfg_.local_ba_max_points > 0
        ? cfg_.local_ba_max_points : 4000;
    auto result = Optimizer::solveLocalBA(
        camera_, snap, cfg_.local_ba_iterations, std::nullopt,
        ba_max_points, cfg_.local_ba_passes);
    if (!result.valid) return;
    // INVALID 直接丢弃；STALE 由覆盖式队列里的更新任务自然追赶。
    // 对同一有效图重算不会改变质量验收结果，只会把 BA 成本翻倍。
    const CommitStatus ba_status =
        applyLocalBAResult(result, skip, snapshot_map, snapshot_submap_id);
    backend_scheduler_.recordTaskOutcome(toTaskOutcome(ba_status));
}

OptimizationSnapshot VisualOdometry::buildLocalBASnapshot(
    const std::vector<Frame::Ptr>& window,
    KeyframeId anchor_kf_id,
    int min_observed) const {
    // M1.5：快照构造迁移至 LocalMapper::buildLocalBASnapshot
    //（local_mapper.cpp，min_observed 门槛 + anchor 连通分量裁剪不变）。
    return local_mapper_.buildLocalBASnapshot(
        map_, atlas_, window, anchor_kf_id, min_observed,
        cfg_.local_ba_max_points);
}

CommitStatus VisualOdometry::applyLocalBAResult(
    const OptimizationResult& result,
    const std::unordered_set<unsigned long>& skip_pose,
    const Map::Ptr& expected_map,
    unsigned long expected_submap_id) {
    // M2：唯一提交路径——BackendCommitter 完成 stale 检查、质量验收、
    // 对象存活检查和一次锁内原子写回 + bumpGeometry。
    {
        std::unique_lock<std::shared_mutex> lock(map_mutex_);
        const auto* active_submap = atlas_->activeSubmap();
        if (map_ != expected_map || !active_submap ||
            active_submap->id != expected_submap_id ||
            result.submap_id != expected_submap_id) {
            LOG_WARN("Local BA result dropped: Map/Submap identity changed before commit");
            return CommitStatus::STALE;
        }
        // Local BA 只能承担局部亚米级精修。位姿图闭环允许的大范围全局
        // 校正走 handleLoopCorrection 的独立提交路径；若这里也放行 10m，
        // 稀疏/退化窗口会在闭环后把个别关键帧拉出数米，锚定轨迹随即在
        // 相邻关键帧边界形成锯齿跳变。1m 是米制局部优化的安全上界，
        // 与序列、帧号和 GT 无关；单目内部尺度下同样只是拒绝异常大改写。
        std::unordered_map<KeyframeId, SE3> old_anchor_poses;
        old_anchor_poses.reserve(result.poses.size());
        for (const auto& update : result.poses) {
            if (skip_pose.contains(update.id)) continue;
            const auto live = map_->getKeyFrame(update.id);
            if (live) old_anchor_poses.emplace(update.id, live->pose_cs);
        }

        const CommitStatus status = BackendCommitter::commit(
            map_, result, skip_pose, cfg_.local_ba_max_correction, expected_map);
        if (status != CommitStatus::COMMITTED) return status;

        // Local BA 是局部地图精修，不是全局轨迹校正。它改写 KF anchor 时，
        // 必须在同一 map 独占事务中把所有历史 T_ca 反向重基，否则最终导出
        // 会用 live anchor 二次应用这次修正，在关键帧边界形成锯齿。锁序
        // 固定为 map -> traj，与 composePoseTrajectory/帧尾记录一致。
        std::unordered_map<KeyframeId, std::pair<SE3, SE3>> anchor_changes;
        anchor_changes.reserve(old_anchor_poses.size());
        for (const auto& [id, old_pose] : old_anchor_poses) {
            const auto live = map_->getKeyFrame(id);
            if (!live) continue;
            if ((old_pose.matrix() - live->pose_cs.matrix()).norm() <= 1e-12)
                continue;
            anchor_changes.emplace(id, std::make_pair(old_pose, live->pose_cs));
        }
        if (!anchor_changes.empty()) {
            std::lock_guard<std::mutex> traj_lock(traj_mutex_);
            for (auto& rec : pose_records_) {
                if (rec.submap_id != expected_submap_id) continue;
                const auto it = anchor_changes.find(rec.anchor_kf_id);
                if (it == anchor_changes.end()) continue;
                rec.T_ca = rebaseTrajectoryAnchor(
                    rec.T_ca, it->second.first, it->second.second);
                rec.anchor_pose_cs = it->second.second;
            }
        }
        last_local_ba_commit_map_ = map_;
        last_local_ba_commit_submap_id_ = expected_submap_id;
        last_local_ba_commit_geometry_revision_ = map_->geometryRevision();
        return status;
    }
}

void VisualOdometry::finishPendingBackendWork() {
    // M1.3：排空槽内任务并 join（stop 语义：设置标志 → notify → join，不 detach）
    if (backend_scheduler_.running()) backend_scheduler_.stop();
    // 同步档或停止前恰好没有可执行任务时，也必须释放已边缘化 KF 的
    // LoopClosure 强引用。此时前端已结束，不存在实时锁竞争。
    drainLoopKeyframeCleanup();
}

void VisualOdometry::submitBackendTask(BackendTask task) {
    // M1.3：提交到覆盖式单任务槽（§5.4）。覆盖语义（LoopClosure 优先 / 同类
    // Local BA 覆盖 / 等待槽容量 1）与内存边界由 BackendScheduler 负责；
    // 本函数是 VO 侧兼容 Facade 的转发入口，永不阻塞。
    backend_scheduler_.submit(std::move(task));
}

void VisualOdometry::drainLoopKeyframeCleanup() {
    if (!loop_closure_ || !loop_cleanup_pending_.load(std::memory_order_acquire))
        return;
    std::vector<KeyframeId> ids;
    {
        std::lock_guard<std::mutex> lock(loop_cleanup_mutex_);
        ids.swap(pending_loop_cleanup_ids_);
        loop_cleanup_pending_ = false;
    }
    if (ids.empty()) return;
    std::ranges::sort(ids);
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    loop_closure_->removeKeyFrames(ids);
}

void VisualOdometry::enforceMapBudget() {
    // M2.2 遗留清理（§6.3）：预算触发点。调用方（insertKeyFrame）已持
    // map_mutex_ 独占锁；evaluate 只读零副作用（预算内确定性不变）。
    // 恢复（解除 stopped）只在 needNewKeyFrame 的完整评估通过时发生。
    if (!RuntimeResources::withinRssBudget(cfg_.runtime_resources.max_rss_mb)) {
        map_growth_stopped_ = true;
        LOG_WARN("Runtime RSS budget exceeded; map growth stopped (limit="
                 << cfg_.runtime_resources.max_rss_mb << "MiB)");
        return;
    }
    const BudgetStatus before = map_budget_.evaluate(map_);
    if (before.within_budget) {
        // 点数恰好达到硬配额并不表示建图停止：下一次 KF 插入会先滚动
        // 回收陈旧点，再按逐点配额补充新点。若在这里置 stopped，
        // needNewKeyFrame 会在每个普通帧全量扫描地图预算，造成 O(N) 卡顿。
        map_growth_stopped_ = false;
        return;
    }

    // 保护集：回环约束两端 KF + 当前参考帧/当前帧（§6.3：不得随机删除锚点，
    // 冗余剔除必须跳过回环/子地图锚点；最近窗口 KF 由 reclaim 内部按 id 保护）。
    std::unordered_set<KeyframeId> protected_ids;
    for (const auto& e : loop_edges_) {
        protected_ids.insert(e.a);
        protected_ids.insert(e.b);
    }
    if (ref_frame_) protected_ids.insert(ref_frame_->id);
    if (curr_frame_) protected_ids.insert(curr_frame_->id);

    // 非活动子地图 → 其 KF id 列表（第 5 步冻结/卸载图像用）与 Map 指针
    // （第 5 步弱陈点回收用）
    std::unordered_map<SubmapId, std::vector<KeyframeId>> submap_keyframes;
    std::unordered_map<SubmapId, Map::Ptr> inactive_submaps;
    const Submap* active = atlas_->activeSubmap();
    for (const auto& s : atlas_->submaps()) {
        if (active && s.id == active->id) continue;
        if (!s.map) continue;
        std::vector<KeyframeId> ids;
        for (const auto& kf : s.map->getAllKeyFrames()) ids.push_back(kf->id);
        submap_keyframes[s.id] = std::move(ids);
        inactive_submaps[s.id] = s.map;
    }

    const BudgetReclaimResult r = map_budget_.reclaim(
        map_, protected_ids, submap_keyframes,
        static_cast<size_t>(std::max<long long>(0, mapSnapshotBytes())),
        inactive_submaps,
        [this](const Frame::Ptr& removed, const Frame::Ptr& replacement) {
            if (!removed || !replacement) return;
            std::lock_guard<std::mutex> traj_lock(traj_mutex_);
            const auto* active = atlas_->activeSubmap();
            if (!active) return;
            for (auto& rec : pose_records_) {
                if (rec.submap_id != active->id ||
                    rec.anchor_kf_id != removed->id) continue;
                // 保持局部 T_cw 不变，只替换轨迹锚：
                // T_ca_new * replacement == T_ca_old * removed。
                rec.T_ca = rebaseTrajectoryAnchor(
                    rec.T_ca, removed->pose_cs, replacement->pose_cs);
                rec.anchor_kf_id = replacement->id;
                rec.anchor_pose_cs = replacement->pose_cs;
            }
        });
    // DBoW3 无单条删除 API。这里只在 Map 事务内合并 id；真正的
    // clear/rebuild 必须在锁外（异步档由单后台 worker）执行，避免数百条
    // BoW 重建把前端堵在 Map 独占锁上。
    if (loop_closure_ && !r.culled_keyframe_ids.empty()) {
        std::lock_guard<std::mutex> cleanup_lock(loop_cleanup_mutex_);
        pending_loop_cleanup_ids_.insert(
            pending_loop_cleanup_ids_.end(),
            r.culled_keyframe_ids.begin(), r.culled_keyframe_ids.end());
        loop_cleanup_pending_ = true;
    }

    // 历史 KF 压缩后，逐帧里程计边也必须同步缩成存活锚点之间的冻结边；
    // 否则 edge vector 仍会随全序列增长，并携带已删除端点。当前几何已经
    // 固化了此前校正，重建边只表达相邻存活锚的当前相对测量。
    if (r.compacted_historical_keyframes > 0) {
        const auto anchors = map_->getAllKeyFrames();
        odometry_edges_.clear();
        odometry_edges_.reserve(anchors.size() > 1 ? anchors.size() - 1 : 0);
        for (size_t i = 1; i < anchors.size(); ++i) {
            odometry_edges_.push_back({
                anchors[i - 1]->id, anchors[i]->id,
                anchors[i - 1]->pose_cs * anchors[i]->pose_cs.inverse(), 1.0});
        }
    }
    map_growth_stopped_ = r.stopped_map_growth;
    if (r.stopped_map_growth) {
        LOG_WARN("Map budget exhausted: map growth stopped (KF="
                 << map_->keyFrameCount() << ", pts=" << map_->mapPointCount() << ")");
    } else if (r.removed_zero_obs_points || r.removed_weak_stale_points ||
               r.culled_redundant_keyframes || r.unloaded_kf_images ||
               r.compacted_historical_keyframes ||
               r.removed_frozen_submap_points) {
        LOG_INFO("Map budget reclaimed: zero-obs=" << r.removed_zero_obs_points
                 << " weak-stale=" << r.removed_weak_stale_points
                 << " unloaded-img=" << r.unloaded_kf_images
                 << " culled-kf=" << r.culled_redundant_keyframes
                 << " compacted-history-kf=" << r.compacted_historical_keyframes
                 << " frozen-weak-pts=" << r.removed_frozen_submap_points);
    }
}

void VisualOdometry::reclaimStaleMapPoints() {
    const auto& budget = map_budget_.config();
    if (map_->mapPointCount() < budget.max_active_points) return;

    std::unordered_set<MapPointId> protected_points;
    const auto keyframes = map_->getAllKeyFrames();
    const size_t keep_count = std::max<size_t>(2, budget.kf_image_keep_recent + 1);
    const size_t first_kept = keyframes.size() > keep_count
        ? keyframes.size() - keep_count : 0;
    for (size_t i = first_kept; i < keyframes.size(); i++) {
        for (const auto& mp : keyframes[i]->map_points)
            if (mp) protected_points.insert(mp->id);
    }
    if (ref_frame_) {
        for (const auto& mp : ref_frame_->map_points)
            if (mp) protected_points.insert(mp->id);
    }
    if (curr_frame_) {
        for (const auto& mp : curr_frame_->map_points)
            if (mp) protected_points.insert(mp->id);
    }

    const size_t kf_count = map_->keyFrameCount();
    std::vector<MapPointId> stale;
    for (const auto& mp : map_->getAllMapPoints()) {
        if (!mp || protected_points.contains(mp->id)) continue;
        const size_t last_hit = map_->lastHitKeyframeCount(mp->id);
        if (last_hit < kf_count &&
            kf_count - last_hit > budget.weak_point_stale_kf_window)
            stale.push_back(mp->id);
    }
    const size_t removed = map_->removeMapPoints(stale);
    if (removed > 0) {
        LOG_INFO("Rolling stale points reclaimed: " << removed
                 << " (remaining=" << map_->mapPointCount() << ")");
    }
}

void VisualOdometry::runBackendTask(BackendTask& task) {
    // M1.3：调度器 worker 的任务分发（原 backendLoop 的任务执行部分）。
    // 锁序：本函数在调度器槽锁之外执行；任务内部再按需获取 map/traj 锁，
    // 不嵌套持锁（backend → map/traj 的旧顺序不再需要）。
    // 任意后台任务都先清空合并后的索引维护队列。即使显式 Maintenance
    // 被更高优先级 LoopClosure 覆盖，清理也不会丢失。
    drainLoopKeyframeCleanup();
    if (task.type == BackendTask::Type::LoopMaintenance)
        return;
    if (task.type == BackendTask::Type::LocalBA)
        runBackendLocalBA(task);
    else
        runBackendLoopClosure(task.curr_kf);
}

void VisualOdometry::runBackendLocalBA(const BackendTask& task) {
    // 后台执行窗口 Local BA（统一入口 runWindowLocalBA：快照隔离 +
    // 跳过活动参考帧写回 + 正式弱观测点前置过滤）
    // §6.4（M2.3 遗留清理）：在途快照字节估算——窗口 KF 描述子 +
    // 其引用点描述子（深拷贝的主要字节来源，重复引用近似计一次）
    size_t bytes = 0;
    for (const auto& kf : task.window) {
        bytes += ResourceBudget::matBytes(kf->descriptors);
        for (const auto& mp : kf->map_points)
            if (mp) bytes += ResourceBudget::matBytes(mp->descriptor);
    }
    map_snapshot_bytes_ = static_cast<long long>(bytes);
    try {
        runWindowLocalBA(task.window, task.anchor_kf_id,
                         task.map, task.submap_id);
    } catch (...) {
        map_snapshot_bytes_ = 0;
        throw;
    }
    map_snapshot_bytes_ = 0;
}

void VisualOdometry::runBackendLoopClosure(const Frame::Ptr& curr_kf) {
    if (!loop_closure_enabled_ || !loop_closure_ || !curr_kf) return;
    PERF_SCOPE("lc.detect");

    // 锁内快照当前 KF（detectLoop 只需描述子/位姿；点深拷贝浪费，keep 空集）
    static const std::unordered_set<unsigned long> kNoPoints;
    Frame::Ptr snap_curr;
    LoopCorrectionContext current_context;
    LoopClosure::SubmapPoses atlas_poses;
    SE3 current_T_ws;
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：快照为读路径
        const auto live_curr = map_->getKeyFrame(curr_kf->id);
        const auto* active_submap = atlas_->activeSubmap();
        if (!live_curr || live_curr.get() != curr_kf.get() || !active_submap)
            return;
        current_context.map = map_;
        current_context.loop_map = map_;
        current_context.submap_id = active_submap->id;
        current_context.loop_submap_id = active_submap->id;
        current_context.topology_revision = map_->topologyRevision();
        current_context.geometry_revision = map_->geometryRevision();
        current_context.curr_kf_id = curr_kf->id;
        current_context.curr_kf = curr_kf;
        std::unordered_map<unsigned long, MapPoint::Ptr> snap_cache;
        snap_curr = snapshotFrame(curr_kf, snap_cache, &kNoPoints);
        current_T_ws = active_submap->T_ws;
        for (const auto& sub : atlas_->submaps())
            atlas_poses.emplace(sub.id, sub.T_ws);
    }

    // 检测（LoopClosure 内部互斥；候选可能跨子地图，写回前在锁内过滤）
    auto cands = loop_closure_->detectLoop(
        snap_curr, current_context.submap_id, current_T_ws, atlas_poses);
    // 可配置的简单两级 cascade：DBoW/flat DBoW 仍负责完整召回和跨查询
    // 假设更新；mobile 的昂贵 PnP 只处理有限成熟地点并留一个位置先验。
    // 高精度档默认验证全部候选，最终接受门对所有档都完全不变。
    std::vector<LoopClosure::LoopCandidate> verification_candidates;
    if (cfg_.loop_mature_verification_limit <= 0) {
        // 桌面/高精度档保持历史行为：不因 mobile 优化改变召回面。
        verification_candidates = cands;
    } else {
        const size_t max_mature = static_cast<size_t>(
            cfg_.loop_mature_verification_limit);
        verification_candidates.reserve(max_mature + 1);
        for (const auto& cand : cands) {
            if (!cand.mature || cand.score <= 0.0) continue;
            verification_candidates.push_back(cand);
            if (verification_candidates.size() >= max_mature) break;
        }
        if (const auto prior = std::ranges::find_if(cands, [](const auto& cand) {
                return cand.score == 0.0;
            }); prior != cands.end()) {
            verification_candidates.push_back(*prior);
        }
    }
    const int verification_limit = std::max(1, cfg_.loop_verification_limit);
    int verification_count = 0;
    for (const auto& cand : verification_candidates) {
        if (verification_count++ >= verification_limit) break;
        LoopCorrectionContext context = current_context;
        Frame::Ptr snap_loop;
        LoopRegionSnapshot loop_region;
        bool has_loop_region = false;
        {
            // 先深拷贝候选几何；锁外预验证淘汰绝大多数假候选，避免每个
            // 假候选都持全图独占锁做昂贵 PnP。真正提交前仍在事务内重验。
            std::shared_lock<std::shared_mutex> lock(map_mutex_);
            const auto live_curr = map_->getKeyFrame(context.curr_kf_id);
            const auto* active_submap = atlas_->activeSubmap();
            const auto* loop_submap = atlas_->getSubmap(cand.submap_id);
            const auto loop_map = loop_submap ? loop_submap->map : nullptr;
            const auto live_loop = loop_map && cand.frame
                ? loop_map->getKeyFrame(cand.frame->id) : nullptr;
            if (!live_curr || live_curr.get() != context.curr_kf.get() ||
                !live_loop || live_loop.get() != cand.frame.get() || !active_submap ||
                map_ != context.map || active_submap->id != context.submap_id) continue;
            context.loop_map = loop_map;
            context.loop_submap_id = cand.submap_id;
            context.loop_kf_id = cand.frame->id;
            context.loop_kf = cand.frame;
            std::unordered_map<unsigned long, MapPoint::Ptr> snap_cache;
            snap_loop = snapshotFrame(cand.frame, snap_cache, nullptr);
            has_loop_region = LoopRegionVerifier::build(
                *loop_map, cand.frame, cand.submap_id,
                cfg_.loop_region, loop_region);
        }
        SE3 preverified_measurement;
        bool preverified = loop_closure_->verifyLoop(
            snap_curr, snap_loop, preverified_measurement);
        if (!preverified && has_loop_region) {
            preverified = LoopRegionVerifier::verify(
                loop_region, snap_curr, camera_, cfg_.loop_region,
                preverified_measurement);
        }
        if (!preverified) continue;
        context.preverified = true;
        context.preverified_T_loop_curr = preverified_measurement;
        context.preverified_geometry_revision = current_context.geometry_revision;
        context.preverified_reference_time = cand.frame
            ? cand.frame->timestamp : 0.0;
        // 校正（内部三阶段：收集/计算/写回，自行管理锁）
        if (handleLoopCorrection(context)) break;
    }
}

// ============================================================
// 共视图滑动窗口：与当前关键帧共视地图点最多的帧 + 当前帧
// ============================================================
std::vector<Frame::Ptr> VisualOdometry::selectLocalWindow(int n) const {
    // M1.5：窗口选择迁移至 LocalMapper::selectLocalWindow（local_mapper.cpp，
    // 共视排序 + 兜底最近 n 帧 + id 升序不变）。
    return local_mapper_.selectLocalWindow(map_, curr_frame_, n);
}

// ============================================================
// 辅助
// ============================================================
// M1.2：matToSE3（cv::Mat → SE3）已迁移至 Relocalizer::matToSE3，
// 公式与默认值保持不变。

void VisualOdometry::triangulateNewPoints(
    const Frame::Ptr& f1, const Frame::Ptr& f2,
    const std::vector<cv::DMatch>& matches) {
    // M1.5：三角化建点迁移至 LocalMapper::triangulateNewPoints
    //（local_mapper.cpp，MapPoint::create + 正式观测绑定不变）。
    local_mapper_.triangulateNewPoints(
        map_, f1, f2, matches, map_budget_.config().max_active_points);
}

bool VisualOdometry::needNewKeyFrame() {
    if (!ref_frame_ || !curr_frame_ || !snap_.has_ref) return false;
    // M1.4：关键帧提议必须先完成。正常资源压力由 enforceMapBudget 把
    // 普通历史压成 Essential Anchor 骨架；不得创建新坐标子图来释放内存。
    KeyframeInput input;
    input.curr_frame = curr_frame_;
    input.ref_pose_cs = snap_.ref_pose_cs;
    input.inliers = status_.inliers;
    input.last_kf_frame_id = last_kf_frame_id_;
    input.map_keyframe_count = map_->keyFrameCount();
    const auto proposal = frontend_tracker_.proposeKeyFrame(input);
    if (!proposal.need) return false;

    if (!RuntimeResources::withinRssBudget(cfg_.runtime_resources.max_rss_mb)) {
        map_growth_stopped_ = true;
        return false;
    }

    // M2.2：预算检查必须与地图读锁绑定。MapBudget::evaluate 会遍历
    // keyframe/map-point 容器，不能在无锁的前端提议路径读取。
    if (map_growth_stopped_.load(std::memory_order_relaxed)) {
        BudgetStatus budget;
        {
            std::shared_lock<std::shared_mutex> lock(map_mutex_);
            budget = map_budget_.evaluate(map_);
        }
        // MapPoint 上限只禁止继续建点，不禁止关键帧/描述子继续进入，
        // 否则一次地图点耗尽会把前端变成 LOST。真正禁止 KF 的是 KF、
        // 描述子、快照或总量预算。
        const bool keyframe_blocked = budget.over_keyframes ||
            budget.over_descriptor || budget.over_snapshot || budget.over_total;
        if (keyframe_blocked) {
            LOG_WARN("Map budget remains blocked after Essential history compaction; "
                     "keyframe insertion deferred without Atlas rollover");
            return false;
        }
        const bool points_blocked =
            budget.points >= map_budget_.config().max_active_points;
        if (!points_blocked && budget.within_budget) {
            map_growth_stopped_ = false;
            LOG_INFO("Map budget back within limits: map growth resumed");
        }
    }
    return true;
}

// ============================================================
// M4：锚定轨迹组合（世界系 T_cw）
// ============================================================
void VisualOdometry::syncFrontendAnchor(SubmapId submap_id, const SE3& T_ws) {
    if (!snap_.map || snap_.submap_id != submap_id) return;
    if ((snap_.T_ws.t - T_ws.t).norm() <= 1e-12 &&
        snap_.T_ws.q.angularDistance(T_ws.q) <= 1e-12)
        return;

    // 原世界位姿 T_cw_old = T_cs * T_ws_old^-1；同一局部位姿在新锚下为
    // T_cw_new = T_cw_old * T_ws_old * T_ws_new^-1。
    // 只更新前端运动基线；锚定轨迹会在读取时从 Atlas 自动组合新 T_ws。
    if (has_last_valid_pose_)
        last_valid_pose_world_ = rebaseWorldPoseForSubmapAnchor(
            last_valid_pose_world_, snap_.T_ws, T_ws);
    snap_.T_ws = T_ws;
}

void VisualOdometry::syncFrontendGeometry(
    const TrackingSnapshot& old_snapshot,
    const TrackingSnapshot& new_snapshot) {
    if (!has_last_valid_pose_ || !old_snapshot.map ||
        old_snapshot.map != new_snapshot.map ||
        old_snapshot.submap_id != new_snapshot.submap_id ||
        !old_snapshot.has_ref || !new_snapshot.has_ref ||
        old_snapshot.ref_kf_id != new_snapshot.ref_kf_id ||
        old_snapshot.geometry_revision == new_snapshot.geometry_revision ||
        last_valid_geometry_revision_ != old_snapshot.geometry_revision) {
        return;
    }
    // Local BA 在同一事务内反向重基历史 T_ca，语义是不移动已发布世界位姿；
    // 只有 PGO/全局校正才需要把运动基线跟随新 anchor。
    const bool local_ba_only =
        new_snapshot.map == last_local_ba_commit_map_ &&
        new_snapshot.submap_id == last_local_ba_commit_submap_id_ &&
        new_snapshot.geometry_revision == last_local_ba_commit_geometry_revision_;
    if (!local_ba_only) {
        const SE3 old_local_pose = last_valid_pose_world_ * old_snapshot.T_ws;
        const SE3 new_local_pose = rebaseAnchoredFramePose(
            old_local_pose, old_snapshot.ref_pose_cs, new_snapshot.ref_pose_cs);
        last_valid_pose_world_ = new_local_pose * new_snapshot.T_ws.inverse();
    }
    last_valid_geometry_revision_ = new_snapshot.geometry_revision;
}

SE3 VisualOdometry::continuousCameraPose() const {
    std::lock_guard<std::mutex> lock(output_pose_mutex_);
    return has_continuous_pose_ ? continuous_pose_oc_ : SE3();
}

SE3 VisualOdometry::globalCorrection() const {
    std::lock_guard<std::mutex> lock(output_pose_mutex_);
    return has_continuous_pose_ ? global_correction_T_wo_ : SE3();
}

/// 组合单条锚定记录为世界位姿（调用方必须已持 map_mutex_ 读/写锁；
/// Atlas/地图访问与前端互斥）
SE3 VisualOdometry::composeRecordWorld(const FramePoseRecord& rec) const {
    const Submap* sub = nullptr;
    for (const auto& s : atlas_->submaps()) {
        if (s.id == rec.submap_id) { sub = &s; break; }
    }
    if (!sub || !sub->map) return SE3();
    auto anchor = sub->map->getKeyFrame(rec.anchor_kf_id);
    const SE3 anchor_pose = anchor ? anchor->pose_cs : rec.anchor_pose_cs;
    // 锚点世界位姿：T_aw = pose_cs(anchor) · T_ws⁻¹（当前锚点/子地图状态，
    // 回环校正与子地图对齐自动传播到所有普通帧——M4 删除全量轨迹插值）
    return composeAnchoredWorldPose(rec.T_ca, anchor_pose, sub->T_ws);
}

std::vector<SE3> VisualOdometry::composePoseTrajectory() const {
    std::vector<SE3> result;
    // 锁序：map_mutex_（共享）→ traj_mutex_（叶子锁，无反向持锁路径）
    std::shared_lock<std::shared_mutex> map_lock(map_mutex_);
    std::lock_guard<std::mutex> lock(traj_mutex_);
    result.reserve(pose_records_.size());
    for (const auto& rec : pose_records_) {
        if (!rec.valid) continue;
        result.push_back(composeRecordWorld(rec));
    }
    return result;
}

std::vector<Vec3> VisualOdometry::getTrajectory() const {
    const auto poses = composePoseTrajectory();
    std::vector<Vec3> trajectory;
    trajectory.reserve(poses.size());
    for (const auto& pose_cw : poses)
        trajectory.push_back(pose_cw.camera_position());
    return trajectory;
}

std::vector<Vec3> VisualOdometry::getTrajectory(size_t max_points) const {
    if (max_points == 0) return {};

    std::vector<Vec3> trajectory;
    // 只从尾部读取有效记录，避免实时可视化随着运行时间增长反复组合全轨迹。
    std::shared_lock<std::shared_mutex> map_lock(map_mutex_);
    std::lock_guard<std::mutex> lock(traj_mutex_);
    trajectory.reserve(std::min(max_points, pose_records_.size()));
    for (auto it = pose_records_.rbegin();
         it != pose_records_.rend() && trajectory.size() < max_points; ++it) {
        if (!it->valid) continue;
        trajectory.push_back(composeRecordWorld(*it).camera_position());
    }
    std::reverse(trajectory.begin(), trajectory.end());
    return trajectory;
}

std::vector<Vec3> VisualOdometry::getMapPointsWorld(size_t max_points) const {
    if (max_points == 0) return {};

    // 地图点/子地图 T_ws 均受 map_mutex_ 保护；与 getTrajectory 相同的
    // 共享锁模式，避免可视化读取与后端提交/前端插入交错。
    std::shared_lock<std::shared_mutex> map_lock(map_mutex_);
    std::vector<Vec3> all;
    all.reserve(std::min<size_t>(max_points, 1u << 16));
    for (const auto& submap : atlas_->submaps()) {
        if (!submap.map) continue;
        for (const auto& mp : submap.map->getAllMapPoints()) {
            // M3 几何契约：p_w = T_ws · p_s
            all.push_back(submap.T_ws * mp->pos_s);
        }
    }
    if (all.size() <= max_points) return all;

    // 均匀抽样，避免只保留 id 靠前（最早创建）的点而缺失新子地图点。
    std::vector<Vec3> sampled;
    sampled.reserve(max_points);
    const double stride = static_cast<double>(all.size()) / max_points;
    for (size_t i = 0; i < max_points; ++i) {
        sampled.push_back(all[static_cast<size_t>(i * stride)]);
    }
    return sampled;
}

std::vector<ColoredPoint> VisualOdometry::getCurrentStereoPointCloud(
    size_t max_points) const {
    if (max_points == 0) return {};

    std::shared_lock<std::shared_mutex> map_lock(map_mutex_);
    const auto frame = curr_frame_;
    const auto* active_submap = atlas_ ? atlas_->activeSubmap() : nullptr;
    if (!frame || !active_submap || frame->image.empty() ||
        frame->pts_c.empty() || frame->keypoints.empty())
        return {};

    // pose_cs: 子地图→相机；组合后 T_wc: 相机→世界。
    const SE3 T_wc = active_submap->T_ws * frame->pose_cs.inverse();
    const size_t count = std::min(frame->pts_c.size(), frame->keypoints.size());
    std::vector<ColoredPoint> all;
    all.reserve(std::min(max_points, count));
    for (size_t i = 0; i < count; ++i) {
        const Vec3& p_c = frame->pts_c[i];
        if (!p_c.allFinite() || p_c.z() <= 0.0) continue;

        ColoredPoint point;
        point.position_w = T_wc * p_c;
        if (!point.position_w.allFinite()) continue;
        if (!sampleRgb(frame->image, frame->keypoints[i].pt,
                       point.r, point.g, point.b))
            continue;
        all.push_back(point);
    }
    if (all.size() <= max_points) return all;

    // 超出上限时均匀抽样，避免只显示特征索引前段。
    std::vector<ColoredPoint> sampled;
    sampled.reserve(max_points);
    const double stride = static_cast<double>(all.size()) / max_points;
    for (size_t i = 0; i < max_points; ++i)
        sampled.push_back(all[static_cast<size_t>(i * stride)]);
    return sampled;
}

} // namespace vslam
