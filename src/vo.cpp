#include "vslam/vo.h"
#include "vslam/optimizer.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace vslam {

VisualOdometry::VisualOdometry(const Camera& camera)
    : camera_(camera), map_(std::make_shared<Map>()) {
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

    // 2. 提取特征
    feature_matcher_.extract(curr_frame_);
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
        trackFrame();
        if (needNewKeyFrame()) insertKeyFrame();
    } else if (state_ == State::LOST) {
        if (tryRelocalize()) { state_ = State::TRACKING; }
    }

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

    auto matches = feature_matcher_.match(ref_frame_, curr_frame_, 0.7, true);
    if (matches.size() < 30) {
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

    auto matches = feature_matcher_.match(ref_frame_, curr_frame_, 0.7, true);

    // 收集 3D-2D 对应（保留 pts3d[i] 与 matches 的映射，供内点观测计数）
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<int> match_idx;
    for (size_t k = 0; k < matches.size(); k++) {
        const auto& m = matches[k];
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
                                     false, 100, 4.0, 0.99, inliers);
        if (ok && inliers.size() >= 10) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            curr_frame_->pose_cw = matToSE3(R, tvec);
            inliers_cnt = (int)inliers.size();
            // 内点对应的地图点被当前帧再次观测到，累计观测计数
            for (int idx : inliers) {
                if (idx >= 0 && idx < (int)match_idx.size()) {
                    auto& mp = ref_frame_->map_points[matches[match_idx[idx]].queryIdx];
                    if (mp) mp->observed_count++;
                }
            }
            updateStatus((int)matches.size(), inliers_cnt, 0.0);
            return curr_frame_->pose_cw;
        }
    }

    // 对极几何回退
    // recoverPose 返回 T_rel 满足 p_c2 = T_rel * p_c1 → T_cw2 = T_rel * T_cw1
    if (matches.size() >= 20) {
        std::vector<cv::Point2f> pts1, pts2;
        FeatureMatcher::getMatchedPoints(ref_frame_, curr_frame_, matches, pts1, pts2);
        cv::Mat E = cv::findEssentialMat(pts1, pts2, camera_.K(), cv::RANSAC, 0.999, 1.0);
        cv::Mat R, t;
        cv::recoverPose(E, pts1, pts2, camera_.K(), R, t);
        SE3 T_rel = matToSE3(R, t);
        curr_frame_->pose_cw = T_rel * ref_frame_->pose_cw;
        inliers_cnt = (int)matches.size();
    } else {
        // 匹配太少 → LOST
        curr_frame_->pose_cw = ref_frame_->pose_cw;
        state_ = State::LOST;
        LOG_WARN("Tracking lost! matches=" << matches.size());
    }

    updateStatus((int)matches.size(), inliers_cnt, 0.0);
    return curr_frame_->pose_cw;
}

// ============================================================
// 重定位（LOST 状态下尝试匹配所有关键帧恢复跟踪）
// ============================================================
bool VisualOdometry::tryRelocalize() {
    auto all_kfs = map_->getAllKeyFrames();
    if (all_kfs.empty()) return false;

    int best_inliers = 0;
    SE3 best_pose;
    Frame::Ptr best_kf;

    for (auto& kf : all_kfs) {
        auto matches = feature_matcher_.match(kf, curr_frame_, 0.8, true);
        if ((int)matches.size() < 30) continue;

        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        for (auto& m : matches) {
            auto& mp = kf->map_points[m.queryIdx];
            if (mp) {
                pts3d.emplace_back((float)mp->pos_w.x(), (float)mp->pos_w.y(), (float)mp->pos_w.z());
                pts2d.push_back(curr_frame_->keypoints[m.trainIdx].pt);
            }
        }
        if (pts3d.size() < 10) continue;

        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        bool ok = cv::solvePnPRansac(pts3d, pts2d, camera_.K(),
                                     cv::Mat(), rvec, tvec,
                                     false, 100, 4.0, 0.99, inliers);
        if (ok && (int)inliers.size() > best_inliers) {
            best_inliers = (int)inliers.size();
            best_pose = matToSE3(cv::Mat(rvec), tvec);
            best_kf = kf;
        }
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
    triangulateNewPoints(ref_frame_, curr_frame_,
        feature_matcher_.match(ref_frame_, curr_frame_, 0.7, true));
    ref_frame_ = curr_frame_;

    LOG_INFO("New KF. mp=" << map_->mapPointCount());

    auto all_kfs = map_->getAllKeyFrames();
    int n = 8;
    std::vector<Frame::Ptr> window;
    int start = std::max(0, (int)all_kfs.size() - n);
    for (int i = start; i < (int)all_kfs.size(); i++)
        window.push_back(all_kfs[i]);

    Optimizer::localBundleAdjustment(camera_, map_, window);
    updateStatus(status_.matches, status_.inliers, status_.parallax);
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
    return dtrans > 0.5 || drot > 0.3;
}

std::vector<Vec3> VisualOdometry::getTrajectory() const {
    return trajectory_;
}

} // namespace vslam
