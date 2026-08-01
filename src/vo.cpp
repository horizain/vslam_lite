#include "vslam/vo.h"
#include "vslam/optimizer.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

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
    unsigned long frame_id = frame_count_++;
    LOG_INFO("--- Frame " << frame_id << " ---");

    // 1. 创建当前帧 + CLAHE 增强
    curr_frame_ = std::make_shared<Frame>(frame_id, timestamp);
    if (image.channels() == 3)
        cv::cvtColor(image, curr_frame_->image_gray, cv::COLOR_BGR2GRAY);
    else
        curr_frame_->image_gray = image;
    curr_frame_->image = image;

    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(curr_frame_->image_gray, curr_frame_->image_gray);

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
        if (tryRelocalize()) { state_ = State::TRACKING; }
    }

    prev_frame_ = curr_frame_;
    trajectory_.push_back(curr_frame_->pose_cw.t);
    return curr_frame_->pose_cw;
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

    cv::Mat E = cv::findEssentialMat(pts1, pts2, camera_.K(),
                                     cv::RANSAC, 0.999, 1.0);
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts1, pts2, camera_.K(), R, t);
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
        bool ok = cv::solvePnPRansac(pts3d, pts2d, camera_.K(),
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

            // 位姿跳变保护：单帧位移异常说明数值发散（如对极尺度错误累积），
            // 拒绝该位姿并进入 LOST，等待重定位
            SE3 Twc_new  = curr_frame_->pose_cw.inverse();
            SE3 Twc_ref  = ref_frame_->pose_cw.inverse();
            if ((Twc_new.t - Twc_ref.t).norm() > 30.0) {
                LOG_WARN("Pose jump detected (" << (Twc_new.t - Twc_ref.t).norm()
                         << "m), tracking lost");
                state_ = State::LOST;
                updateStatus(0, 0, 0.0);
                return ref_frame_->pose_cw;
            }
            return curr_frame_->pose_cw;
        }
    }

    // 对极几何回退
    // recoverPose 返回 T_rel 满足 p_c2 = T_rel * p_c1 → T_cw2 = T_rel * T_cw1
    if (matches.size() >= (size_t)cfg_.min_matches_track) {
        std::vector<cv::Point2f> pts1, pts2;
        FeatureMatcher::getMatchedPoints(ref_frame_, curr_frame_, matches, pts1, pts2);
        cv::Mat E = cv::findEssentialMat(pts1, pts2, camera_.K(), cv::RANSAC, 0.999, 1.0);
        cv::Mat R, t;
        cv::recoverPose(E, pts1, pts2, camera_.K(), R, t);
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
        bool ok = cv::solvePnPRansac(pts3d, pts2d, camera_.K(),
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
        bool ok = cv::solvePnPRansac(pts3d, pts2d, camera_.K(),
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

    cv::Mat K = camera_.K();
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
    return dtrans > cfg_.keyframe_translation || drot > cfg_.keyframe_rotation || weak_match;
}

std::vector<Vec3> VisualOdometry::getTrajectory() const {
    return trajectory_;
}

} // namespace vslam
