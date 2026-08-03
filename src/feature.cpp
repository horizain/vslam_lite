#include "vslam/feature.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <cmath>

namespace vslam {

FeatureMatcher::FeatureMatcher() {
    // ORB 特征提取器：1000 个特征点，8 层金字塔，尺度因子 1.2
    orb_ = cv::ORB::create(1000, 1.2f, 8, 15, 0, 2,
                           cv::ORB::HARRIS_SCORE, 19, 10);
    // 暴力匹配器（汉明距离）
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void FeatureMatcher::setParams(int num_features, double scale_factor, int pyramid_levels) {
    orb_ = cv::ORB::create(num_features, (float)scale_factor, pyramid_levels,
                           15, 0, 2, cv::ORB::HARRIS_SCORE, 19, 10);
}

void FeatureMatcher::extract(Frame::Ptr frame) {
    if (frame->image_gray.empty()) {
        cv::cvtColor(frame->image, frame->image_gray, cv::COLOR_BGR2GRAY);
    }

    frame->keypoints.clear();
    orb_->detectAndCompute(frame->image_gray, cv::Mat(),
                           frame->keypoints, frame->descriptors);

    // 初始化 map_points 占位
    frame->map_points.resize(frame->keypoints.size(), nullptr);
}

std::vector<cv::DMatch> FeatureMatcher::match(
    const Frame::Ptr& f1,
    const Frame::Ptr& f2,
    double ratio_thresh,
    bool use_ransac) {

    if (f1->descriptors.empty() || f2->descriptors.empty()) {
        LOG_WARN("Empty descriptors, cannot match");
        return {};
    }

    // Step 1: 暴力匹配 + 最近邻 / 次近邻比率测试
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(f1->descriptors, f2->descriptors, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (auto& m : knn_matches) {
        if (m.size() < 2) continue;
        if (m[0].distance < ratio_thresh * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }
    LOG_INFO("Feature matching: " << good_matches.size() << " / " << knn_matches.size());

    // Step 2: RANSAC 基础矩阵剔除误匹配
    if (use_ransac && good_matches.size() >= 8) {
        std::vector<cv::Point2f> pts1, pts2;
        getMatchedPoints(f1, f2, good_matches, pts1, pts2);

        std::vector<unsigned char> inlier_mask;
        cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, inlier_mask);

        std::vector<cv::DMatch> inlier_matches;
        for (size_t i = 0; i < inlier_mask.size(); i++) {
            if (inlier_mask[i]) {
                inlier_matches.push_back(good_matches[i]);
            }
        }
        LOG_INFO("After RANSAC: " << inlier_matches.size() << " inliers");
        return inlier_matches;
    }

    return good_matches;
}

std::vector<unsigned char> FeatureMatcher::trackLK(
    const Frame::Ptr& prev_frame,
    Frame::Ptr& curr_frame) {

    if (prev_frame->keypoints.empty()) return {};

    std::vector<cv::Point2f> prev_pts;
    for (const auto& kp : prev_frame->keypoints) {
        prev_pts.push_back(kp.pt);
    }

    std::vector<cv::Point2f> curr_pts;
    std::vector<unsigned char> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        prev_frame->image_gray, curr_frame->image_gray,
        prev_pts, curr_pts, status, err,
        cv::Size(21, 21), 3);

    // 将跟踪成功的点写入当前帧
    curr_frame->keypoints.clear();
    curr_frame->map_points.clear();
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            cv::KeyPoint kp;
            kp.pt = curr_pts[i];
            kp.size = 31;   // ORB 默认 patch 尺寸（否则描述子退化）
            kp.angle = 0;
            curr_frame->keypoints.push_back(kp);
            // 保持地图点的关联
            if (i < prev_frame->map_points.size())
                curr_frame->map_points.push_back(prev_frame->map_points[i]);
            else
                curr_frame->map_points.push_back(nullptr);
        }
    }

    return status;
}

std::vector<unsigned char> FeatureMatcher::matchStereo(
    const cv::Mat& left_gray,
    const cv::Mat& right_gray,
    const std::vector<cv::KeyPoint>& left_keypoints,
    std::vector<cv::Point2f>& right_pts) {

    right_pts.clear();
    if (left_keypoints.empty() || left_gray.empty() || right_gray.empty()) return {};

    std::vector<cv::Point2f> left_pts;
    left_pts.reserve(left_keypoints.size());
    for (const auto& kp : left_keypoints) left_pts.push_back(kp.pt);

    // 校正后的双目图像行对齐：左目→右目 的 LK 光流退化为近一维搜索，
    // 稳定且给出亚像素视差（比 ORB 左右匹配精度更高）。
    // 金字塔 5 层 + 31x31 窗口：KITTI 近点视差可达 100px+，默认 3 层/21px 会追丢。
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        left_gray, right_gray,
        left_pts, right_pts, status, err,
        cv::Size(31, 31), 5);

    // 右目→左目反向跟踪：仅使用单向 LK 的 status 会把重复纹理上的错误点
    // 当成有效深度。前后向一致性是稀疏双目最便于理解、也最关键的质量门槛。
    std::vector<cv::Point2f> back_pts;
    std::vector<unsigned char> back_status;
    std::vector<float> back_err;
    cv::calcOpticalFlowPyrLK(
        right_gray, left_gray,
        right_pts, back_pts, back_status, back_err,
        cv::Size(31, 31), 5);

    constexpr float kMaxEpipolarError = 1.5f;
    constexpr float kMaxForwardBackwardError = 1.0f;
    constexpr float kMaxLkError = 25.0f;
    for (size_t i = 0; i < status.size(); i++) {
        if (!status[i] || i >= back_status.size() || !back_status[i]) {
            status[i] = 0;
            continue;
        }
        const auto& pl = left_pts[i];
        const auto& pr = right_pts[i];
        const bool in_image = pr.x >= 1.0f && pr.y >= 1.0f
            && pr.x < right_gray.cols - 1.0f && pr.y < right_gray.rows - 1.0f;
        const float fb_error = cv::norm(back_pts[i] - pl);
        const float stereo_error = i < err.size() ? err[i] : kMaxLkError + 1.0f;
        const float reverse_error = i < back_err.size() ? back_err[i] : kMaxLkError + 1.0f;
        if (!in_image || std::abs(pl.y - pr.y) > kMaxEpipolarError
            || fb_error > kMaxForwardBackwardError
            || stereo_error > kMaxLkError || reverse_error > kMaxLkError) {
            status[i] = 0;
        }
    }

    return status;
}

void FeatureMatcher::getMatchedPoints(
    const Frame::Ptr& f1,
    const Frame::Ptr& f2,
    const std::vector<cv::DMatch>& matches,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2) {

    pts1.clear(); pts2.clear();
    for (const auto& m : matches) {
        pts1.push_back(f1->keypoints[m.queryIdx].pt);
        pts2.push_back(f2->keypoints[m.trainIdx].pt);
    }
}

} // namespace vslam
