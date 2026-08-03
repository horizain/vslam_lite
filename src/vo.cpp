#include "vslam/vo.h"
#include "vslam/optimizer.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <Eigen/SVD>

#include <set>
#include <algorithm>
#include <ranges>

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
            if (v["keyframe_translation"]) cfg.keyframe_translation = v["keyframe_translation"].as<double>();
            if (v["keyframe_rotation"])   cfg.keyframe_rotation    = v["keyframe_rotation"].as<double>();
        }
        if (auto o = root["Optimizer"]) {
            if (o["local_window_size"])   cfg.local_window_size   = o["local_window_size"].as<int>();
            if (o["local_ba_iterations"]) cfg.local_ba_iterations = o["local_ba_iterations"].as<int>();
            if (o["enable_local_ba"])     cfg.enable_local_ba     = o["enable_local_ba"].as<bool>();
        }
        if (auto s = root["Stereo"]) {
            if (s["min_depth"]) cfg.stereo_min_depth = s["min_depth"].as<double>();
            if (s["max_depth"]) cfg.stereo_max_depth = s["max_depth"].as<double>();
            if (s["keyframe_translation"])
                cfg.keyframe_translation_stereo = s["keyframe_translation"].as<double>();
        }
        LOG_INFO("VO config loaded from: " << path);
    } catch (const std::exception& e) {
        LOG_WARN("VOConfig::fromYaml failed (" << e.what() << "), using defaults");
    }
    return cfg;
}

VisualOdometry::VisualOdometry(const Camera& camera, const VOConfig& cfg)
    : camera_(camera), cfg_(cfg), map_(std::make_shared<Map>()) {
    feature_matcher_.setParams(cfg_.num_features, cfg_.scale_factor, cfg_.pyramid_levels);
    if (cfg_.feature_method != 0) {
        LOG_INFO("feature_method=" << cfg_.feature_method << " (LK 光流)");
    }
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
        updateStatus(0, 0, 0.0);
        return (state_ != State::INITIALIZING) ? ref_frame_->pose_cw : SE3();
    }

    // 2.5 双目/RGB-D：视差（或深度）→ 每特征点的相机系 3D 观测 pts_c
    computeStereoDepths();

    // 3. 状态机处理
    if (state_ == State::INITIALIZING) {
        bool ok = tryInitialize();
        if (ok) { state_ = State::TRACKING; }
    } else if (state_ == State::TRACKING) {
        // LK 模式：描述子为空说明走的是光流路径，用索引对齐的 PnP 跟踪
        if (cfg_.feature_method == 1 && curr_frame_->descriptors.empty())
            trackFrameLK();
        else
            trackFrame();
        // 跟踪可能把状态置为 LOST（跳变保护），此时不能再插入关键帧
        if (state_ == State::TRACKING && needNewKeyFrame()) insertKeyFrame();
    } else if (state_ == State::LOST) {
        if (tryRelocalize()) {
            state_ = State::TRACKING;
        } else if (camera_->hasPerFrameDepth()) {
            // 双目/RGB-D 兜底：重定位失败时直接用当前帧重新初始化——
            // 单帧即有绝对尺度深度，无需历史地图（ORB-SLAM stereo 的 Lost→Reset 语义），
            // 避免高速段长时间卡在 LOST 循环（每帧 × 30 KF 匹配 → 帧率暴跌）
            map_->clear();
            ref_frame_.reset();
            prev_frame_.reset();
            trajectory_.clear();
            last_kf_frame_id_ = 0;
            if (tryInitialize()) {
                LOG_WARN("Stereo re-init with current frame (map reset)");
                state_ = State::TRACKING;
            }
        }
    }

    prev_frame_ = curr_frame_;
    trajectory_.push_back(curr_frame_->pose_cw.t);
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

    for (size_t i = 0; i < status.size() && i < curr_frame_->keypoints.size(); i++) {
        if (!status[i]) continue;
        double disparity = curr_frame_->keypoints[i].pt.x - right_pts[i].x;
        if (disparity <= 0) continue;
        double depth = camera_->fx * camera_->baseline() / disparity;  // z = fx*b/d
        if (depth < cfg_.stereo_min_depth || depth > cfg_.stereo_max_depth) continue;
        curr_frame_->pts_c[i] = camera_->pixel2camera(
            Vec2(curr_frame_->keypoints[i].pt.x, curr_frame_->keypoints[i].pt.y), depth);
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
}

// ============================================================
// 初始化
// ============================================================
bool VisualOdometry::tryInitialize() {
    // 双目/RGB-D：首帧即有绝对尺度深度，直接建图进入 TRACKING
    // （无需对极初始化——这是双目相对单目的本质区别：尺度可观测）
    if (camera_->hasPerFrameDepth()) {
        if (!ref_frame_) {
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

    // 第一帧 = 世界原点（T_cw1 = I）
    // recoverPose 返回的相对变换 T_rel 满足 p_c2 = T_rel * p_c1，
    // 即 T_cw2 = T_rel（世界系 = 帧1相机系）
    ref_frame_->pose_cw = SE3();
    Eigen::Matrix3d R_eigen;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_eigen(i, j) = R.at<double>(i, j);
    SE3 T_cw2(Eigen::Quaterniond(R_eigen),
              Vec3(t.at<double>(0), t.at<double>(1), t.at<double>(2)));
    curr_frame_->pose_cw = T_cw2;

    triangulateNewPoints(ref_frame_, curr_frame_, matches);

    map_->insertKeyFrame(ref_frame_);
    map_->insertKeyFrame(curr_frame_);
    last_kf_frame_id_ = curr_frame_->id;   // 初始化插入的两个关键帧也参与冷却

    LOG_INFO("Init OK! parallax=" << parallax << " inliers=" << inliers
             << " mp=" << map_->mapPointCount());
    updateStatus((int)matches.size(), inliers, parallax);
    return true;
}

// ============================================================
// 跟踪
// ============================================================
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
        if (ok && inliers.size() >= 10) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            curr_frame_->pose_cw = matToSE3(R, tvec);
            inliers_cnt = (int)inliers.size();
            // 把内点对应的地图点关联到当前帧（这是关键帧共视统计的基础：
            // 若不关联，关键帧只与紧邻帧共视，Local BA 窗口永远只有 2 帧）
            for (int idx : inliers) {
                if (idx < 0 || idx >= (int)match_idx.size()) continue;
                auto& mp = ref_frame_->map_points[matches[match_idx[idx]].queryIdx];
                if (mp) {
                    mp->observed_count++;
                    curr_frame_->map_points[matches[match_idx[idx]].trainIdx] = mp;
                }
            }
            updateStatus((int)matches.size(), inliers_cnt, 0.0);

            // 位姿跳变保护：单帧位移异常说明数值发散（如旋转-平移歧义的假平移）。
            // 阈值按传感器分派：双目有绝对尺度（真实帧间位移小），用运动先验相对阈值；
            // 单目尺度不定（recoverPose 归一化），保留宽松的固定阈值。
            SE3 Twc_new  = curr_frame_->pose_cw.inverse();
            SE3 Twc_ref  = ref_frame_->pose_cw.inverse();
            double jump_thresh = camera_->hasPerFrameDepth()
                ? motionPrior() * 6.0 + 2.0 : 30.0;
            if ((Twc_new.t - Twc_ref.t).norm() > jump_thresh) {
                LOG_WARN("Pose jump detected (" << (Twc_new.t - Twc_ref.t).norm()
                         << "m), retry stereo 3D-3D");
                // 旋转-平移歧义时 PnP 的假平移不可信，改用双目真实深度重新估计
                if (tryTrack3D3D(matches)) return curr_frame_->pose_cw;
                state_ = State::LOST;
                updateStatus(0, 0, 0.0);
                return ref_frame_->pose_cw;
            }
            return curr_frame_->pose_cw;
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
            state_ = State::LOST;
            updateStatus((int)matches.size(), 0, 0.0);
            return ref_frame_->pose_cw;
        }
    } else {
        // 匹配太少 → LOST
        curr_frame_->pose_cw = ref_frame_->pose_cw;
        state_ = State::LOST;
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
    if (!camera_->hasPerFrameDepth() || matches.size() < 10) return false;

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
    if ((int)pts_w.size() < 6) return false;

    cv::Mat affine, inliers;
    // RANSAC 3D-3D：返回 3x4 [R|t] 满足 dst = R*src + t → 即 T_cw（世界→相机）
    bool ok = cv::estimateAffine3D(pts_w, pts_c, affine, inliers, 1.0, 0.99);
    if (!ok) return false;

    // affine 允许缩放/剪切，把 R 投影回 SO(3)（SVD 正交化）
    Eigen::Matrix3d Rm;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Rm(i, j) = affine.at<double>(i, j);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rm, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_orth = svd.matrixU() * svd.matrixV().transpose();
    if (R_orth.determinant() < 0) R_orth.col(2) *= -1;
    Vec3 t(affine.at<double>(0, 3), affine.at<double>(1, 3), affine.at<double>(2, 3));
    SE3 pose_cw(Eigen::Quaterniond(R_orth), t);

    // 跳变保护（运动先验）：3D-3D 用真实深度，异常解概率低，但兜底拒绝
    SE3 Twc_new = pose_cw.inverse();
    SE3 Twc_ref = ref_frame_->pose_cw.inverse();
    if ((Twc_new.t - Twc_ref.t).norm() > motionPrior() * 6.0 + 2.0) {
        LOG_WARN("3D-3D pose jump (" << (Twc_new.t - Twc_ref.t).norm() << "m), rejected");
        return false;
    }

    curr_frame_->pose_cw = pose_cw;
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
// 运动先验：ref→上一帧 位移（跳变保护阈值基准）
// ============================================================
double VisualOdometry::motionPrior() const {
    if (prev_frame_ && ref_frame_ && prev_frame_->id != ref_frame_->id) {
        SE3 Twc_prev = prev_frame_->pose_cw.inverse();
        SE3 Twc_ref  = ref_frame_->pose_cw.inverse();
        return std::max((Twc_prev.t - Twc_ref.t).norm(), 0.3);
    }
    return cfg_.keyframe_translation;
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
        if (ok && inliers.size() >= 10) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            curr_frame_->pose_cw = matToSE3(R, tvec);
            inliers_cnt = (int)inliers.size();
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

    // LK PnP 失败 → 重新提取 ORB 特征，回退到 ORB 匹配跟踪
    LOG_WARN("LK PnP failed (" << pts3d.size() << " 3D pts), fallback to ORB track");
    feature_matcher_.extract(curr_frame_);
    return trackFrame();
}

// ============================================================
// 重定位（LOST 状态下尝试匹配所有关键帧恢复跟踪）
// ============================================================
bool VisualOdometry::tryRelocalize() {
    auto all_kfs = map_->getAllKeyFrames();
    if (all_kfs.empty()) return false;

    // LK 模式：当前帧可能无描述子，重定位前先提取 ORB
    if (curr_frame_->descriptors.empty())
        feature_matcher_.extract(curr_frame_);

    int best_inliers = 0;
    SE3 best_pose;
    Frame::Ptr best_kf;

    // 对单个关键帧做 PnP 匹配，内点达标(20)即返回 true
    auto try_kf = [&](const Frame::Ptr& kf) -> bool {
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
        if (ok && (int)inliers.size() > best_inliers) {
            best_inliers = (int)inliers.size();
            best_pose = matToSE3(cv::Mat(rvec), tvec);
            best_kf = kf;
        }
        return best_inliers >= 20;
    };

    // 从最新关键帧向历史方向尝试，最多 kMaxRelocTries 帧：
    // 最近帧时间邻近成功率最高，兜底覆盖回环场景，同时限制每帧 LOST 的匹配开销
    constexpr int kMaxRelocTries = 30;
    int tried = 0;
    for (int i = (int)all_kfs.size() - 1; i >= 0 && tried < kMaxRelocTries; i--, tried++) {
        if (try_kf(all_kfs[i])) break;
    }

    if (best_inliers >= 20) {
        curr_frame_->pose_cw = best_pose;
        ref_frame_ = best_kf;
        LOG_INFO("Relocalized! inliers=" << best_inliers);
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
    updateStatus(status_.matches, status_.inliers, status_.parallax);
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
    return dtrans > kf_trans || drot > cfg_.keyframe_rotation || weak_match;
}

std::vector<Vec3> VisualOdometry::getTrajectory() const {
    return trajectory_;
}

} // namespace vslam
