#include "vslam/vo.h"
#include "vslam/optimizer.h"
#include "perf_monitor.h"
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
            if (f["orb_max_bands"])   cfg.orb_max_bands          = f["orb_max_bands"].as<int>();
            if (f["match_ratio"])     cfg.match_ratio            = f["match_ratio"].as<double>();
            if (f["ransac_threshold"]) cfg.ransac_pixel_threshold = f["ransac_threshold"].as<double>();
        }
        if (auto r = root["Runtime"]) {
            if (r["opencv_threads"]) cfg.opencv_threads = r["opencv_threads"].as<int>();
            if (r["async_backend"])  cfg.async_backend  = r["async_backend"].as<bool>();
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
            if (v["keyframe_max_count"])   cfg.keyframe_max_count   = v["keyframe_max_count"].as<int>();
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
            if (lc["loop_cooldown_kfs"])   cfg.loop_cooldown_kfs = lc["loop_cooldown_kfs"].as<int>();
            // 兼容旧字段名 loop_cooldown_frames（帧计数）→ 按关键帧数换算（约 1/10）
            if (!lc["loop_cooldown_kfs"] && lc["loop_cooldown_frames"])
                cfg.loop_cooldown_kfs = lc["loop_cooldown_frames"].as<int>() / 10;
            if (lc["top_candidates"])      cfg.loop_top_candidates = lc["top_candidates"].as<int>();
            if (lc["position_prior_dist"]) cfg.loop_position_prior_dist = lc["position_prior_dist"].as<double>();
            if (lc["position_prior_gap"])  cfg.loop_position_prior_gap = lc["position_prior_gap"].as<int>();
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
    if (cfg_.opencv_threads > 0) {
        cv::setNumThreads(cfg_.opencv_threads);
        LOG_INFO("OpenCV threads limited to " << cv::getNumThreads());
    }
    feature_matcher_.setParams(cfg_.num_features, cfg_.scale_factor,
                               cfg_.pyramid_levels, cfg_.orb_max_bands);
    if (cfg_.feature_method != 0) {
        LOG_INFO("feature_method=" << cfg_.feature_method << " (LK 光流)");
    }
    // Phase 2：配置中启用回环且给出词典路径时自动加载
    if (cfg_.enable_loop_closure && !cfg_.vocab_path.empty())
        enableLoopClosure(cfg_.vocab_path);
    if (cfg_.async_backend) {
        startBackend();
        LOG_INFO("Async backend ENABLED (BA/loop on background thread)");
    }
}

VisualOdometry::~VisualOdometry() {
    if (cfg_.async_backend) stopBackend();
}

bool VisualOdometry::enableLoopClosure(const std::string& vocab_path) {
    if (!loop_closure_) loop_closure_ = std::make_unique<LoopClosure>();
    loop_closure_->setParams(cfg_.min_score, cfg_.temporal_window,
                             cfg_.min_loop_inliers, cfg_.pnp_inlier_ratio,
                             cfg_.ransac_pixel_threshold, camera_,
                             cfg_.loop_top_candidates,
                             cfg_.loop_position_prior_dist,
                             cfg_.loop_position_prior_gap);
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
    PERF_SCOPE("vo.frame_total");
    unsigned long frame_id = frame_count_++;
    LOG_INFO("--- Frame " << frame_id << " ---");

    // 先复制 Mat 头，兼容调用方把 currentFrame()->image 直接作为下一帧输入；
    // 随后的 releaseImages 只移除旧 Frame 的引用，不会令本轮输入失效。
    const cv::Mat left_input = left;
    const cv::Mat right_input = right;

    // 上次 addFrame 返回后 Viewer 已完成取帧。旧当前帧若仍是 LK 的上一帧，
    // 只保留左灰度图到本轮光流结束；其余像素缓冲现在即可释放。异常帧可能
    // 未被设为 prev_frame_，这种帧在进入下一轮时可以直接释放全部图像。
    Frame::Ptr old_current = curr_frame_;
    if (old_current) old_current->releaseImages(old_current == prev_frame_);

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
    if (left_input.channels() == 3)
        cv::cvtColor(left_input, curr_frame_->image_gray, cv::COLOR_BGR2GRAY);
    else
        curr_frame_->image_gray = left_input;
    curr_frame_->image = left_input;

    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(curr_frame_->image_gray, curr_frame_->image_gray);

    // 双目：右目图做同样的灰度化 + 增强（左右目一致，保证双目匹配质量）
    if (!right_input.empty()) {
        curr_frame_->image_right = right_input;
        if (right_input.channels() == 3)
            cv::cvtColor(right_input, curr_frame_->image_right_gray, cv::COLOR_BGR2GRAY);
        else
            curr_frame_->image_right_gray = right_input;
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
    // 时序 LK 已完成，旧上一帧的左灰度图不再被前端使用。若它同时是历史
    // 关键帧，仅释放像素数据；描述子/关键点/map_points/pts_c 继续留在地图中。
    // 当前帧提取失败时不能释放：prev_frame_ 仍会作为下一轮 LK 的输入。
    if (prev_frame_) prev_frame_->releaseImages();

    // 2.5 双目/RGB-D：视差（或深度）→ 每特征点的相机系 3D 观测 pts_c
    {
        PERF_SCOPE("vo.stereo_depth");
        computeStereoDepths();
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
    if (state_ == State::TRACKING && status_.tracking_valid) {
        if (has_last_valid_pose_) {
            // 记录逐帧相对运动（Twc 语义），LOST 期匀速外推锚点用
            const SE3 X_cur = curr_frame_->pose_cw.inverse();
            const SE3 X_last = last_valid_pose_cw_.inverse();
            per_frame_motion_ = X_cur * X_last.inverse();
            has_per_frame_motion_ = true;
        }
        last_valid_pose_cw_ = curr_frame_->pose_cw;
        has_last_valid_pose_ = true;
        tracking_failures_ = 0;
    }
    updateStatus(status_.matches, status_.inliers, status_.parallax);
    status_.pose_valid = status_.tracking_valid && status_.map_connected;
    // pose_cw.t 是世界原点在相机系中的坐标，不是相机位置。
    // 原地旋转时它会随 R_cw 绕圈；轨迹必须记录相机光心 C_w = -R_cw^T t_cw。
    if (status_.pose_valid) {
        std::lock_guard<std::mutex> lock(traj_mutex_);  // 后端回环校正重写轨迹
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
        ? last_valid_pose_cw_.inverse()
        : (ref_frame_ ? ref_frame_->pose_cw.inverse() : SE3());

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
                     ? last_valid_pose_cw_.inverse().t : Vec3::Zero())).norm()
                 << "m)");
    }

    // anchor 继承最后有效全局位姿，新子地图实际已锚定到全局世界系。
    // 若标为 disconnected，此后所有帧 pose_valid=false（tracking_valid &&
    // map_connected），轨迹将永久冻结、只剩旋转箭头转动。标记 connected
    // 让重初始化后轨迹继续记录（丢失期间的真实位移仍未知，会有短暂停顿，
    // 但不再彻底冻结）。
    auto& submap = atlas_->createSubmap(anchor_Twc, true);
    // map_ 成员被后端线程（runBackendLocalBA 等）锁内读取；此处切换地图必须
    // 持独占锁，否则 shared_ptr 成员读写与后端并发 → 数据竞争（P1 修复）。
    {
        std::unique_lock<std::shared_mutex> lock(map_mutex_);
        map_ = submap.map;
        curr_frame_->pose_cw = anchor_Twc.inverse();
    }
    ref_frame_.reset();
    prev_frame_.reset();
    last_kf_frame_id_ = 0;
    state_ = State::INITIALIZING;
    tracking_failures_ = 0;
    relocalization_frames_ = 0;
    LOG_WARN("Tracking lost for too long; creating anchored submap " << submap.id);
}

// ============================================================
// 子地图-历史轨迹刚体对齐：新子地图关键帧相机位置 vs 历史轨迹最近点，
// Umeyama 求解 Sim3（双目下尺度≈1，仅校正旋转/平移），
// 让重初始化后的轨迹平滑接回丢失点，不出现断点跳变。
// ============================================================
void VisualOdometry::alignSubmapToTrajectory() {
    if (pose_trajectory_.empty()) return;
    auto kfs = map_->getAllKeyFrames();
    if (kfs.size() < 3) return;

    // 只搜索轨迹末端附近（丢失点必然在当前轨迹末尾，最多向前取 100 帧）：
    // 新子地图锚在丢失点附近，最近的旧位置就在这段里；限制搜索窗避免
    // 误配到无关的早期区域（轨迹自交时最近邻会找错位置）。
    const size_t search_from = pose_trajectory_.size() > 100
        ? pose_trajectory_.size() - 100 : 0;

    std::vector<Vec3> src, dst;
    src.reserve(kfs.size());
    dst.reserve(kfs.size());
    const double max_match_dist = 50.0;  // 外推锚点可能偏离丢失点 20-40m，半径放宽；
                                         // 误配风险由"末端搜索窗 + scale≈1 检查"兜底
    for (const auto& kf : kfs) {
        const Vec3 pos = kf->pose_cw.inverse().t;
        double best = max_match_dist;
        Vec3 best_p = Vec3::Zero();
        bool found = false;
        for (size_t i = search_from; i < pose_trajectory_.size(); i++) {
            const Vec3 c = pose_trajectory_[i].camera_position();
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

    for (const auto& kf : kfs) {
        const SE3 X = kf->pose_cw.inverse();          // T_wc
        const SE3 X2(S.q * X.q, S * X.t);             // X' = S ∘ X（点变换复合）
        kf->pose_cw = X2.inverse();
    }
    for (auto& mp : map_->getAllMapPoints())
        mp->pos_w = S * mp->pos_w;
    LOG_INFO("Submap aligned to trajectory: " << src.size()
             << " matches, scale=" << S.s);
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
    // 锁内读点：异步后端写回 Eigen::Vector3d（24B 非原子）会与裸读竞争，
    // 造成 torn read → 垃圾 3D 点 → PnP 失败 → LOST 爆增（基准实测 114 vs 34）
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<int> match_idx;
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：读路径共享锁
        for (auto [k, m] : matches | std::views::enumerate) {
            auto& mp = ref_frame_->map_points[m.queryIdx];
            if (mp) {
                pts3d.emplace_back((float)mp->pos_w.x(), (float)mp->pos_w.y(), (float)mp->pos_w.z());
                pts2d.push_back(curr_frame_->keypoints[m.trainIdx].pt);
                match_idx.push_back((int)k);
            }
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
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：读路径共享锁
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
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：读路径共享锁
        for (size_t i = 0; i < curr_frame_->keypoints.size(); i++) {
            auto& mp = curr_frame_->map_points[i];
            if (mp) {
                pts3d.emplace_back((float)mp->pos_w.x(), (float)mp->pos_w.y(), (float)mp->pos_w.z());
                pts2d.push_back(curr_frame_->keypoints[i].pt);
                kp_idx.push_back((int)i);
            }
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
        // 重定位不做 F 矩阵 RANSAC：外点由下方 solvePnPRansac 自己剔除
        //（与 trackFrame 的注释一致；F 矩阵在重定位场景是纯冗余开销）
        auto matches = feature_matcher_.match(kf, curr_frame_, cfg_.match_ratio, false);
        // 重定位用较低门槛（min_matches_init=100 是初始化专用，RANSAC 后常达不到）
        if ((int)matches.size() < 30) return false;

        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        {
            std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：读路径共享锁
            for (auto& m : matches) {
                auto& mp = kf->map_points[m.queryIdx];
                if (mp) {
                    pts3d.emplace_back((float)mp->pos_w.x(), (float)mp->pos_w.y(), (float)mp->pos_w.z());
                    pts2d.push_back(curr_frame_->keypoints[m.trainIdx].pt);
                }
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
    // 粗筛先行：256 描述子子集 BF 匹配数不足的直接跳过（~0.5ms/候选），
    // 只对相似候选做全量 BF + PnP。候选最多 60 个，全量匹配+PnP 每个
    // ~10ms+，粗筛把 LOST 帧的单帧开销从秒级降到几十毫秒。
    constexpr int kQuickMatchThreshold = 20;
    int quick_checked = 0, quick_passed = 0;
    for (const auto& [submap_id, kf] : candidates) {
        quick_checked++;
        if (feature_matcher_.quickMatchCount(kf->descriptors,
                                             curr_frame_->descriptors,
                                             256, 64.0) < kQuickMatchThreshold)
            continue;
        quick_passed++;
        if (try_kf(submap_id, kf)) break;
    }
    if (quick_passed > 0)
        LOG_INFO("Reloc prefilter: " << quick_passed << "/" << quick_checked
                 << " candidates passed");

    if (best_inliers >= 20) {
        // 换地图需与后端写回互斥（后端校正期间写的是旧 map_）
        // 注意：块内不得调用 updateStatus（它内部会再次 lock(map_mutex_)，
        // 非递归锁重入会死锁——实测卡死复现点）
        {
            std::unique_lock<std::shared_mutex> lock(map_mutex_);  // P1：换地图为写路径（独占）
            atlas_->activate(best_submap_id);
            map_ = atlas_->activeMap();
            curr_frame_->pose_cw = best_pose;
            ref_frame_ = best_kf;
        }
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

    // 地图集合/建点/edges 写入统一持独占锁（异步后端与后台线程互斥）
    {
        std::unique_lock<std::shared_mutex> lock(map_mutex_);  // P1：写路径（独占）
        map_->insertKeyFrame(curr_frame_);

        // 子地图重建后延迟对齐：等新子地图有 ≥3 个关键帧（拟合最低点数），
        // 与丢失点附近的历史轨迹做 Umeyama 刚体对齐
        if (submap_needs_alignment_ && map_->keyFrameCount() >= 3) {
            alignSubmapToTrajectory();
            submap_needs_alignment_ = false;
        }

        // 定期清理观测不足的地图点（每 20 个关键帧一次），防止地图无限增长。
        // 必须在建新点之前执行：若在建点后剔除，本轮 createMapPointsFromStereo
        // 刚创建的点（observed_count=1）会被 cullMapPoints(2) 当场删除，参考
        // 关键帧瞬间失去全部新点，后续帧 PnP 可用 3D 点骤减 → 垃圾解 → LOST。
        {
            PERF_SCOPE("kf.cull");
            if (map_->keyFrameCount() % 20 == 0)
                map_->cullMapPoints(2);
        }

        // 双目/RGB-D：当前帧有视差/深度的特征直接建点（绝对尺度）
        {
            PERF_SCOPE("kf.build_points");
            createMapPointsFromStereo(curr_frame_);
            // LK 模式：关键帧用干净的 ORB 特征重建（LK 关键点无方向，描述子无法与
            // 历史关键帧匹配），保证与上一关键帧的 ORB 匹配/三角化可靠；
            // 普通帧仍用 LK 光流跟踪（从关键帧 ORB 特征出发）
            if (cfg_.feature_method == 1)
                feature_matcher_.extract(curr_frame_);
            triangulateNewPoints(ref_frame_, curr_frame_,
                feature_matcher_.match(ref_frame_, curr_frame_, cfg_.match_ratio, true));
        }

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
    }
    ref_frame_ = curr_frame_;
    last_kf_frame_id_ = curr_frame_->id;   // 更新关键帧冷却基准

    LOG_INFO("New KF. mp=" << map_->mapPointCount());

    if (cfg_.async_backend) {
        // ---- 异步路径：Local BA / 回环检测+校正 提交后台线程 ----
        if (cfg_.enable_local_ba) {
            std::vector<Frame::Ptr> window;
            {
                std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：选窗为读路径
                window = selectLocalWindow(cfg_.local_window_size);
            }
            BackendTask task;
            task.type = BackendTask::Type::LocalBA;
            task.window = std::move(window);
            submitBackendTask(std::move(task));
        }
        if (loop_closure_enabled_ && loop_closure_) {
            loop_closure_->addKeyFrame(curr_frame_);
            if (map_->keyFrameCount() % (unsigned long)std::max(1, cfg_.detection_interval) == 0) {
                const bool cooled = map_->keyFrameCount() - last_loop_kf_count_.load()
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
            PERF_SCOPE("kf.local_ba");
            Optimizer::localBundleAdjustment(camera_, map_, window, cfg_.local_ba_iterations);
        }

        // ============================================================
        // Phase 2 回环钩子：新关键帧入词袋数据库；每 N 个关键帧检测一次。
        // 放在 Local BA 之后，避免两处位姿修改互相干扰。
        // ============================================================
        if (loop_closure_enabled_ && loop_closure_) {
            loop_closure_->addKeyFrame(curr_frame_);
            if (map_->keyFrameCount() % (unsigned long)std::max(1, cfg_.detection_interval) == 0) {
                // 回环校正冷却：上次校正后至少间隔 N 个关键帧才允许再次校正。
                // 同一区域会被词袋反复命中（分数高），连续校正会让 S_global
                // 叠加冲突、轨迹被反复拉扯变形。以关键帧数计（与 KF 密度无关）。
                const bool cooled = map_->keyFrameCount() - last_loop_kf_count_.load()
                    >= (unsigned long)cfg_.loop_cooldown_kfs;
                if (cooled) {
                    auto cands = loop_closure_->detectLoop(curr_frame_);
                    for (auto& cand : cands) {
                        // DBoW3 数据库跨 Atlas 子地图缓存；当前尚未实现跨尺度
                        // 子地图融合，不能把另一张 map 的候选伪装成本图闭环。
                        if (!map_->getKeyFrame(cand->id)) {
                            LOG_WARN("LoopClosure: cross-submap candidate kf#"
                                     << cand->id << " rejected");
                            continue;
                        }
                        SE3 T_loop_curr;
                        if (loop_closure_->verifyLoop(curr_frame_, cand, T_loop_curr)) {
                            handleLoopCorrection(T_loop_curr, curr_frame_, cand);
                            break;  // 第一个验证通过的候选即回环
                        }
                    }
                }
            }
        }
    }

    updateStatus(status_.matches, status_.inliers, status_.parallax);
}

// ============================================================
// Phase 2 回环校正：位姿图优化 → 地图点同步 → 全局 BA → 逐帧轨迹同步
// 三阶段设计（异步后端 P2-1）：
//   阶段 1 锁内收集工作集（快照 KF/点/edges/轨迹，深拷贝隔离）
//   阶段 2 锁外计算（位姿图 / 点同步 / 全局 BA / 轨迹插值——不触碰地图数据）
//   阶段 3 锁内写回（真 KF 位姿 / 真点坐标 / 轨迹替换）
// 同步路径（async_backend=false）由主线程直接调用同一实现，锁无竞争。
// ============================================================
void VisualOdometry::handleLoopCorrection(const SE3& T_loop_curr,
                                          const Frame::Ptr& kf_curr,
                                          const Frame::Ptr& kf_loop) {
    // ---- 阶段 1：锁内收集工作集 ----
    struct Workset {
        std::vector<Frame::Ptr> snap_kfs;                     // 快照 KF（pose 独立）
        std::vector<Frame::Ptr> real_kfs;                     // 对应真 KF（写回用）
        std::unordered_map<unsigned long, SE3> old_pose;      // KF id → 优化前 pose_cw
        std::unordered_map<unsigned long, unsigned long> mp_ref_kf;  // 点 id → 参考 KF id
        std::vector<LoopEdge> odom_edges, loop_edges;         // 位姿图边（拷贝）
        std::vector<std::pair<MapPoint::Ptr, Vec3>> points;   // 全量点（真指针 + pos，点同步）
        std::vector<SE3> traj;                                // 轨迹快照
        std::vector<unsigned long> traj_ids;
        bool valid = true;
    };

    Workset ws;
    {
        // P1：阶段 1 收集为读路径（共享锁），多读者并发；写回在阶段 3 独占。
        std::shared_lock<std::shared_mutex> lock(map_mutex_);
        auto all_kfs = map_->getAllKeyFrames();
        if (all_kfs.size() < 2) return;

        ws.old_pose.reserve(all_kfs.size());
        for (const auto& kf : all_kfs) {
            ws.old_pose.emplace(kf->id, kf->pose_cw);
            // 快照 KF：pose 深拷贝；map_points 数组按原样深拷贝（快照点对象，
            // 优化器在锁外安全地读写它们，与地图集合完全隔离）。
            // BA/位姿图不消费描述子 → with_descriptors=false 省全图描述子拷贝
            auto snap = snapshotFrame(kf, nullptr, false);
            ws.snap_kfs.push_back(snap);
            ws.real_kfs.push_back(kf);
        }

        // 点参考 KF 映射（最早观测）+ 全量点快照（点同步用）
        for (const auto& kf : all_kfs) {
            for (const auto& mp : kf->map_points)
                if (mp) ws.mp_ref_kf.emplace(mp->id, kf->id);
        }
        for (auto& mp : map_->getAllMapPoints())
            ws.points.emplace_back(mp, mp->pos_w);

        // 位姿图边（拷贝）：累计回环边 + 本次新边
        ws.odom_edges = odometry_edges_;
        ws.loop_edges = loop_edges_;
        LoopEdge le;
        le.a = kf_loop->id;
        le.b = kf_curr->id;
        le.T_rel = T_loop_curr;
        le.weight = 10.0;  // 回环约束高置信
        ws.loop_edges.push_back(le);
    }
    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        ws.traj = pose_trajectory_;
        ws.traj_ids = traj_frame_ids_;
    }

    // ---- 阶段 2：锁外计算 ----
    // 2a. 位姿图：快照 KF 塞入临时 Map（optimizer 只从 KF 列表读数据）
    auto tmp_map = std::make_shared<Map>();
    for (auto& s : ws.snap_kfs) tmp_map->insertKeyFrame(s);
    {
        PERF_SCOPE("loop.pose_graph");
        if (!Optimizer::poseGraphOptimization(tmp_map, ws.odom_edges, ws.loop_edges)) {
            LOG_WARN("Loop correction skipped: pose graph backend unavailable or constraints invalid");
            return;  // 不保留失败的回环边（未写回 ws.loop_edges）
        }
    }

    // 2b. 地图点同步：按参考 KF 的位姿增量移动全量点（锁外，全量快照数据）
    for (auto& [mp, pos] : ws.points) {
        auto ref = ws.mp_ref_kf.find(mp->id);
        if (ref == ws.mp_ref_kf.end()) continue;
        auto old = ws.old_pose.find(ref->second);
        if (old == ws.old_pose.end()) continue;
        // 快照 KF 的 pose 已是位姿图结果：correction = new⁻¹ * old
        // （用真实 KF 查找快照 KF：按 id 在 snap_kfs 中找）
        SE3 correction;
        bool found = false;
        for (auto& s : ws.snap_kfs) {
            if (s->id == ref->second) {
                correction = s->pose_cw.inverse() * old->second;
                found = true;
                break;
            }
        }
        if (found) pos = correction * pos;
    }

    // 2c. 全局 BA（快照上）：优化所有快照 KF 位姿 + 快照点
    {
        PERF_SCOPE("loop.global_ba");
        Optimizer::globalBundleAdjustment(camera_, tmp_map, cfg_.global_ba_iterations,
                                          std::nullopt /* fix_points 按传感器 */,
                                          1500 /* max_points */);
    }

    // 2d. 非关键帧不在位姿图中：将相邻关键帧的最终校正插值后施加到完整轨迹
    std::vector<std::pair<unsigned long, SE3>> corrections;
    corrections.reserve(ws.snap_kfs.size());
    for (const auto& kf : ws.snap_kfs) {
        auto old = ws.old_pose.find(kf->id);
        if (old == ws.old_pose.end()) continue;
        corrections.emplace_back(kf->id, kf->pose_cw.inverse() * old->second);
    }
    std::vector<SE3> new_traj = ws.traj;
    for (size_t i = 0; i < new_traj.size() && i < ws.traj_ids.size(); i++) {
        const unsigned long frame_id = ws.traj_ids[i];
        auto upper = std::lower_bound(
            corrections.begin(), corrections.end(), frame_id,
            [](const auto& item, unsigned long id) { return item.first < id; });

        SE3 correction;
        if (upper != corrections.end() && upper->first == frame_id) {
            // 关键帧直接采用最终优化结果（快照 KF 位姿）
            bool hit = false;
            for (auto& s : ws.snap_kfs) {
                if (s->id == frame_id) {
                    new_traj[i] = s->pose_cw;
                    hit = true;
                    break;
                }
            }
            if (hit) continue;
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
        const SE3 Twc_new = correction * new_traj[i].inverse();
        new_traj[i] = Twc_new.inverse();
    }

    // ---- 阶段 3：锁内写回（独占）----
    {
        std::unique_lock<std::shared_mutex> lock(map_mutex_);  // P1：写回为写路径
        for (size_t i = 0; i < ws.snap_kfs.size() && i < ws.real_kfs.size(); i++) {
            auto real = map_->getKeyFrame(ws.real_kfs[i]->id);
            if (real) real->pose_cw = ws.snap_kfs[i]->pose_cw;
        }
        // 全局 BA 优化的快照点 → 真点（按 id 查，点可能已被 cull 删除则跳过）
        for (const auto& s : ws.snap_kfs) {
            for (const auto& smp : s->map_points) {
                if (!smp) continue;
                auto real = map_->getMapPoint(smp->id);
                if (real) real->pos_w = smp->pos_w;
            }
        }
        // 全量点同步结果 → 真点（快照指针，shared_ptr 保证存活）
        for (const auto& [mp, pos] : ws.points) {
            auto real = map_->getMapPoint(mp->id);
            if (real) real->pos_w = pos;
        }
        loop_edges_ = ws.loop_edges;  // 保留累积回环边（含本次，优化成功）
        // 冷却基准更新留在锁内：锁外解引用 map_ 会与前端 createSubmap 的
        // map_ 成员交换并发（shared_ptr 读写竞争，代码审查发现，§3.16）
        last_loop_kf_count_ = map_->keyFrameCount();
    }
    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        pose_trajectory_ = std::move(new_traj);
        traj_frame_ids_ = ws.traj_ids;
    }

    loop_closure_count_++;
    LOG_INFO("Loop closed! kf#" << kf_loop->id << " -> kf#" << kf_curr->id
             << " (total " << loop_closure_count_.load() << ")");
}

// ============================================================
// 异步后端（P2-1）：后台线程执行 Local BA / 回环检测+校正。
// 数据隔离原则：优化永远在"快照数据"上执行（快照帧 pose 深拷贝、
// 快照点对象深拷贝），不触碰地图集合；写回在 map_mutex_ 保护下进行。
// ============================================================
Frame::Ptr VisualOdometry::snapshotFrame(
    const Frame::Ptr& kf,
    const std::unordered_set<unsigned long>* keep_points,
    bool with_descriptors) {
    auto snap = std::make_shared<Frame>(kf->id, 0.0);
    snap->pose_cw = kf->pose_cw;
    snap->keypoints = kf->keypoints;
    if (with_descriptors && !kf->descriptors.empty())
        snap->descriptors = kf->descriptors.clone();
    snap->map_points.resize(kf->map_points.size(), nullptr);
    for (size_t i = 0; i < kf->map_points.size(); i++) {
        const auto& mp = kf->map_points[i];
        if (!mp) continue;
        if (keep_points && !keep_points->count(mp->id)) continue;
        auto smp = std::make_shared<MapPoint>(mp->id);
        smp->pos_w = mp->pos_w;
        if (with_descriptors && !mp->descriptor.empty())
            smp->descriptor = mp->descriptor.clone();
        smp->observed_count = mp->observed_count;
        snap->map_points[i] = smp;
    }
    return snap;
}

void VisualOdometry::startBackend() {
    if (backend_running_.exchange(true)) return;
    backend_thread_ = std::thread(&VisualOdometry::backendLoop, this);
}

void VisualOdometry::stopBackend() {
    {
        std::lock_guard<std::mutex> lock(backend_mutex_);
        backend_stop_ = true;
    }
    backend_cv_.notify_all();
    if (backend_thread_.joinable()) backend_thread_.join();
    backend_running_ = false;
}

void VisualOdometry::submitBackendTask(BackendTask task) {
    // 覆盖式有界队列（P1）：Local BA 的窗口是"最近 N 个关键帧"，新任务包含
    // 旧任务的全部关键帧 → 旧 LocalBA 必然被超车，直接丢弃（滞后有界 ≤1 窗口，
    // 前端 PnP 恒用最新窗口数据，解决 §3.15 的数据陈旧/滞后无界问题）。
    // 回环任务低频且可被后到的检测覆盖，超限时丢最旧，杜绝"队列满→调用方同步
    // 执行→前端阻塞"的旧路径。
    constexpr size_t kMaxQueued = 4;
    {
        std::lock_guard<std::mutex> lock(backend_mutex_);
        if (task.type == BackendTask::Type::LocalBA) {
            auto it = std::remove_if(backend_tasks_.begin(), backend_tasks_.end(),
                [](const BackendTask& t) {
                    return t.type == BackendTask::Type::LocalBA;
                });
            backend_tasks_.erase(it, backend_tasks_.end());
            // 队首优先：局部 BA 是保新鲜的最高优先级，先于回环长任务执行
            backend_tasks_.push_front(std::move(task));
        } else {
            backend_tasks_.push_back(std::move(task));
        }
        while (backend_tasks_.size() > kMaxQueued)
            backend_tasks_.pop_back();
    }
    backend_cv_.notify_one();
}

void VisualOdometry::backendLoop() {
    while (true) {
        BackendTask task;
        {
            std::unique_lock<std::mutex> lock(backend_mutex_);
            backend_cv_.wait(lock, [&] {
                return backend_stop_.load() || !backend_tasks_.empty();
            });
            if (backend_stop_.load() && backend_tasks_.empty()) break;
            task = std::move(backend_tasks_.front());
            backend_tasks_.pop_front();
        }
        if (task.type == BackendTask::Type::LocalBA)
            runBackendLocalBA(task.window);
        else
            runBackendLoopClosure(task.curr_kf);
    }
}

void VisualOdometry::runBackendLocalBA(const std::vector<Frame::Ptr>& window) {
    if (window.empty() || !cfg_.enable_local_ba) return;
    PERF_SCOPE("kf.local_ba");

    // 锁内快照窗口（pose 深拷贝、点对象深拷贝——与地图集合完全隔离）
    std::vector<Frame::Ptr> snaps, reals;
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：快照为读路径
        for (const auto& kf : window) {
            if (!map_->getKeyFrame(kf->id)) continue;  // 已被清理/地图重建
            reals.push_back(kf);
            snaps.push_back(snapshotFrame(kf, nullptr, false));  // BA 不消费描述子
        }
    }
    if (snaps.size() < 2) return;

    // 锁外优化（快照）
    auto tmp_map = std::make_shared<Map>();
    for (auto& s : snaps) tmp_map->insertKeyFrame(s);
    Optimizer::localBundleAdjustment(camera_, tmp_map, snaps, cfg_.local_ba_iterations);

    // 锁内写回：快照 KF pose → 真 KF；快照点 pos → 真点（按 id，可能已删则跳过）。
    // 跳过"窗口内最新关键帧"（= 前端当前跟踪参考 ref_frame_，由前端独占）：
    // 后端写回它的位姿会让前端 validateMotion 把"参考帧被 BA 挪动"误判成
    // 位姿跳变 → 每 2 帧一次的 LOST 带（异步模式实测 74 vs 同步 31，见 §3.16）。
    // 该 KF 会在下一个窗口任务中作为普通成员被 BA 写回（覆盖式队列保证滞后 ≤1 窗口）。
    if (snaps.empty()) return;
    unsigned long newest_id = snaps[0]->id;
    for (const auto& s : snaps)
        newest_id = std::max(newest_id, s->id);
    {
        std::unique_lock<std::shared_mutex> lock(map_mutex_);  // P1：写回为写路径
        for (size_t i = 0; i < snaps.size() && i < reals.size(); i++) {
            auto real = map_->getKeyFrame(reals[i]->id);
            if (!real || real->id == newest_id) continue;  // 跳过活动参考帧
            real->pose_cw = snaps[i]->pose_cw;
        }
        for (const auto& s : snaps) {
            for (const auto& smp : s->map_points) {
                if (!smp) continue;
                auto real = map_->getMapPoint(smp->id);
                if (real) real->pos_w = smp->pos_w;
            }
        }
    }
}

void VisualOdometry::runBackendLoopClosure(const Frame::Ptr& curr_kf) {
    if (!loop_closure_enabled_ || !loop_closure_ || !curr_kf) return;
    PERF_SCOPE("lc.detect");

    // 锁内快照当前 KF（detectLoop 只需描述子/位姿；点深拷贝浪费，keep 空集）
    static const std::unordered_set<unsigned long> kNoPoints;
    Frame::Ptr snap_curr;
    {
        std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：快照为读路径
        if (!map_->getKeyFrame(curr_kf->id)) return;
        snap_curr = snapshotFrame(curr_kf, &kNoPoints);
    }

    // 检测（LoopClosure 内部互斥；候选可能跨子地图，写回前在锁内过滤）
    auto cands = loop_closure_->detectLoop(snap_curr);
    for (const auto& cand : cands) {
        SE3 T_loop_curr;
        bool verified = false;
        {
            // 锁内验证：候选必须仍在本地图（防跨子地图误闭环），
            // verifyLoop 读取候选 KF 的 map_points（真数组，与前端 cull 互斥）
            std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：验证为读路径
            if (!map_->getKeyFrame(cand->id)) continue;
            verified = loop_closure_->verifyLoop(snap_curr, cand, T_loop_curr);
        }
        if (!verified) continue;
        // 校正（内部三阶段：收集/计算/写回，自行管理锁）
        handleLoopCorrection(T_loop_curr, snap_curr, cand);
        break;
    }
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
    std::shared_lock<std::shared_mutex> lock(map_mutex_);  // P1：读路径共享锁（ref 位姿可能被后台 BA 写回）
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
    // 规模硬顶：关键帧数超过上限后放大平移阈值，压缩后续冗余帧
    //（KITTI 00 全程 2800+ KF → 硬顶后 2200-，-22%）
    if (cfg_.keyframe_max_count > 0 &&
        map_->keyFrameCount() > (unsigned long)cfg_.keyframe_max_count)
        kf_trans *= 1.5;
    const bool max_interval = curr_frame_->id - last_kf_frame_id_
        >= (unsigned long)cfg_.max_keyframe_interval;
    return dtrans > kf_trans || drot > cfg_.keyframe_rotation
        || weak_match || max_interval;
}

std::vector<Vec3> VisualOdometry::getTrajectory() const {
    std::lock_guard<std::mutex> lock(traj_mutex_);  // 后端回环校正重写轨迹
    std::vector<Vec3> trajectory;
    trajectory.reserve(pose_trajectory_.size());
    for (const auto& pose_cw : pose_trajectory_)
        trajectory.push_back(pose_cw.camera_position());
    return trajectory;
}

} // namespace vslam
