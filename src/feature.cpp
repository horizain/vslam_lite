#include "vslam/feature.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace vslam {

FeatureMatcher::FeatureMatcher() {
    // ORB 特征提取器：1000 个特征点，8 层金字塔，尺度因子 1.2
    orb_ = cv::ORB::create(1000, 1.2f, 8, 15, 0, 2,
                           cv::ORB::HARRIS_SCORE, 19, 10);
    // 暴力匹配器（汉明距离）
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void FeatureMatcher::setParams(int num_features, double scale_factor, int pyramid_levels,
                               int orb_max_bands, bool stereo_reverse_prune) {
    num_features_ = num_features;
    scale_factor_ = scale_factor;
    pyramid_levels_ = pyramid_levels;
    orb_max_bands_ = std::clamp(orb_max_bands, 1, 8);
    stereo_reverse_prune_ = stereo_reverse_prune;
    orb_ = cv::ORB::create(num_features, (float)scale_factor, pyramid_levels,
                           15, 0, 2, cv::ORB::HARRIS_SCORE, 19, 10);
}

void FeatureMatcher::extract(Frame::Ptr frame) {
    if (frame->image_gray.empty()) {
        cv::cvtColor(frame->image, frame->image_gray, cv::COLOR_BGR2GRAY);
    }
    const cv::Mat& img = frame->image_gray;
    const int rows = img.rows;

    // ---- 并行分带提取 ----
    // ORB 内部已按金字塔 octave 并行（约 8 个任务）。把图像按行分成 N 个带，
    // 各带独立 detectAndCompute，得到 N×octave 个可并行任务，进一步发挥多核。
    // 每个带在上下各扩 kBorder 像素提取、只保留核心区 [y0,y1) 内的特征：
    // 边界角点由相邻带的核心区自然保留（无重复、不丢点），且每个保留特征
    // 距提取 ROI 边界 ≥ kBorder（> ORB edgeThreshold=15），描述子不退化。
    // 小图退化为单带串行，行为与旧实现一致（保证单元测试确定性）。
    // 分带数取固定上限 4：TBB 内嵌 parallel_for_ 与外层并行度相乘会波动
    //（getNumThreads()/2 推导），固定后任务数稳定、行为可复现。
    constexpr int kBorder = 24;
    int nbands = 1;
    if (rows >= 2 * kBorder + 64 && num_features_ >= 500) {
        nbands = std::clamp(cv::getNumThreads() / 2, 1,
                            std::min(orb_max_bands_, 4));
        while (nbands > 1 && rows / nbands < 48) --nbands;
    }

    if (nbands <= 1) {
        orb_->detectAndCompute(img, cv::Mat(), frame->keypoints, frame->descriptors);
        frame->map_points.resize(frame->keypoints.size(), nullptr);
        return;
    }

    // 每带预算放大以补偿带间重叠区的重复提取，合并后统一截断到 num_features_
    const int per_band = std::max(32, num_features_ * 2 / nbands);
    std::vector<std::vector<cv::KeyPoint>> band_kps(nbands);
    std::vector<cv::Mat> band_desc(nbands);

    cv::parallel_for_(cv::Range(0, nbands), [&](const cv::Range& r) {
        for (int b = r.start; b < r.end; b++) {
            const int y0 = rows * b / nbands;
            const int y1 = rows * (b + 1) / nbands;
            const int top = std::max(0, y0 - kBorder);
            const int bot = std::min(rows, y1 + kBorder);
            const int roi_h = bot - top;
            // 带高受限时收敛金字塔层数，避免最深层退化到过小分辨率。
            // 注意按真实 scale_factor（1.2）计算，不能用 >>（那是按 2 倍）。
            int levels = pyramid_levels_;
            while (levels > 1 && roi_h / std::pow(scale_factor_, levels - 1) < 16.0)
                --levels;
            cv::Mat roi = img(cv::Rect(0, top, img.cols, roi_h));
            auto orb = cv::ORB::create(per_band, (float)scale_factor_, levels,
                                       15, 0, 2, cv::ORB::HARRIS_SCORE, 19, 10);
            std::vector<cv::KeyPoint> kps;
            cv::Mat desc;
            orb->detectAndCompute(roi, cv::Mat(), kps, desc);
            // 只保留核心区 [y0, y1) 内的特征
            for (size_t i = 0; i < kps.size(); i++) {
                const float gy = kps[i].pt.y + (float)top;  // 带内坐标 → 全局
                if (gy >= (float)y0 && gy < (float)y1) {
                    band_kps[b].push_back(kps[i]);
                    band_kps[b].back().pt.y = gy;
                    band_desc[b].push_back(desc.row((int)i));
                }
            }
        }
    });

    // 合并各带（各带至多 per_band 个；总量可能超 num_features_，按响应截断）
    std::vector<cv::KeyPoint> kps;
    std::vector<cv::Mat> descs;
    for (int b = 0; b < nbands; b++) {
        kps.insert(kps.end(), band_kps[b].begin(), band_kps[b].end());
        if (!band_desc[b].empty()) descs.push_back(band_desc[b]);
    }
    cv::Mat desc;
    if (descs.size() == 1) desc = descs[0];
    else if (!descs.empty()) cv::vconcat(descs, desc);

    if ((int)kps.size() > num_features_) {
        std::vector<int> idx(kps.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + num_features_, idx.end(),
                          [&](int a, int b) { return kps[a].response > kps[b].response; });
        std::vector<cv::KeyPoint> sel(num_features_);
        cv::Mat sel_desc(num_features_, desc.cols, desc.type());
        for (int i = 0; i < num_features_; i++) {
            sel[i] = kps[idx[i]];
            desc.row(idx[i]).copyTo(sel_desc.row(i));
        }
        kps.swap(sel);
        desc = sel_desc;
    }

    frame->keypoints = std::move(kps);
    frame->descriptors = desc;
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
    if (use_ransac)
        LOG_INFO("Feature matching: " << good_matches.size() << " / " << knn_matches.size());

    // Step 2: RANSAC 基础矩阵剔除误匹配
    if (use_ransac && good_matches.size() >= 8) {
        const auto inlier_matches = filterFundamental(
            f1, f2, good_matches, 3.0);
        LOG_INFO("After RANSAC: " << inlier_matches.size() << " inliers");
        return inlier_matches;
    }

    return good_matches;
}

std::vector<cv::DMatch> FeatureMatcher::filterFundamental(
    const Frame::Ptr& f1, const Frame::Ptr& f2,
    const std::vector<cv::DMatch>& matches,
    double ransac_threshold) const {
    if (!f1 || !f2 || matches.size() < 8) return matches;
    std::vector<cv::Point2f> pts1, pts2;
    getMatchedPoints(f1, f2, matches, pts1, pts2);
    std::vector<unsigned char> inlier_mask;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                           ransac_threshold, 0.99, inlier_mask);
    std::vector<cv::DMatch> inlier_matches;
    inlier_matches.reserve(matches.size());
    for (size_t i = 0; i < inlier_mask.size() && i < matches.size(); ++i)
        if (inlier_mask[i]) inlier_matches.push_back(matches[i]);
    return inlier_matches;
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
    // 曾实验 4 层/25px（每帧省 ~40ms），但近点视差匹配失败导致建点减少、
    // LOST 增 6 倍（219 vs 32），回滚——精度优先，速度靠别处优化。
    // 注：calcOpticalFlowPyrLK 内部已用 TBB 并行处理关键点；外部再分块会重复
    // 构建金字塔反而更慢（OpenCV 4.6 无接收预建金字塔的重载），故保持串行调用。
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        left_gray, right_gray,
        left_pts, right_pts, status, err,
        cv::Size(31, 31), 5);

    if (!stereo_reverse_prune_) {
        // 确定性回归兼容路径：保留父版本的全量反向 LK 调用顺序。
        std::vector<cv::Point2f> back_pts;
        std::vector<unsigned char> back_status;
        std::vector<float> back_err;
        cv::calcOpticalFlowPyrLK(
            right_gray, left_gray, right_pts, back_pts, back_status, back_err,
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
                || stereo_error > kMaxLkError || reverse_error > kMaxLkError)
                status[i] = 0;
        }
        return status;
    }

    // 右目→左目反向跟踪：只对正向 LK 已成功且仍在图像内的点做反向检查。
    // 失败点不可能通过后续深度门，重新把它们送进反向金字塔只增加 CPU 工作。
    std::vector<int> reverse_indices;
    std::vector<cv::Point2f> reverse_input;
    reverse_indices.reserve(status.size());
    reverse_input.reserve(status.size());
    for (size_t i = 0; i < status.size() && i < right_pts.size(); ++i) {
        if (!status[i]) continue;
        const auto& pr = right_pts[i];
        if (!std::isfinite(pr.x) || !std::isfinite(pr.y) ||
            pr.x < 1.0f || pr.y < 1.0f ||
            pr.x >= right_gray.cols - 1.0f ||
            pr.y >= right_gray.rows - 1.0f ||
            std::abs(left_pts[i].y - pr.y) > 1.5f) {
            status[i] = 0;
            continue;
        }
        reverse_indices.push_back(static_cast<int>(i));
        reverse_input.push_back(pr);
    }
    std::vector<cv::Point2f> reverse_output;
    std::vector<unsigned char> reverse_status;
    std::vector<float> reverse_err;
    if (!reverse_input.empty()) {
        cv::calcOpticalFlowPyrLK(
            right_gray, left_gray,
            reverse_input, reverse_output, reverse_status, reverse_err,
            cv::Size(31, 31), 5);
    }
    std::vector<cv::Point2f> back_pts(left_pts.size());
    std::vector<unsigned char> back_status(left_pts.size(), 0);
    std::vector<float> back_err(left_pts.size(), 0.0f);
    for (size_t j = 0; j < reverse_indices.size(); ++j) {
        const int i = reverse_indices[j];
        if (j < reverse_output.size()) back_pts[i] = reverse_output[j];
        if (j < reverse_status.size()) back_status[i] = reverse_status[j];
        if (j < reverse_err.size()) back_err[i] = reverse_err[j];
    }

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

int FeatureMatcher::quickMatchCount(const cv::Mat& desc1, const cv::Mat& desc2,
                                    int max_query, double dist_thresh) const {
    if (desc1.empty() || desc2.empty()) return 0;
    const int rows = std::min(max_query, desc2.rows);
    if (rows <= 0) return 0;
    // 子集 × 全量的 BF 单匹配：距离 < 阈值的数量即候选相似度粗分
    cv::Mat sub = desc2.rowRange(0, rows);
    std::vector<cv::DMatch> matches;
    matcher_->match(sub, desc1, matches);
    int cnt = 0;
    for (const auto& m : matches)
        if (m.distance < dist_thresh) cnt++;
    return cnt;
}

} // namespace vslam
