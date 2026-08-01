#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/mappoint.h"
#include <opencv2/features2d.hpp>

namespace vslam {

/// 特征提取与匹配
class FeatureMatcher {
public:
    FeatureMatcher();

    /// 设置 ORB 提取参数（重建提取器）
    void setParams(int num_features, double scale_factor, int pyramid_levels);

    /// 对一帧图像提取 ORB 特征，结果写入 frame.keypoints / frame.descriptors
    void extract(Frame::Ptr frame);

    /// 在两帧间做暴力匹配 + 比率测试 + RANSAC 基础矩阵滤除外点
    /// 返回匹配点对索引
    std::vector<cv::DMatch> match(
        const Frame::Ptr& f1,
        const Frame::Ptr& f2,
        double ratio_thresh = 0.7,
        bool use_ransac = true);

    /// 在上一帧特征点位置做光流跟踪（用于与 ORB 对比学习用）
    /// 返回追踪成功的索引
    std::vector<unsigned char> trackLK(
        const Frame::Ptr& prev_frame,
        Frame::Ptr& curr_frame);

    /// 从匹配中提取匹配点的像素坐标对
    static void getMatchedPoints(
        const Frame::Ptr& f1,
        const Frame::Ptr& f2,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::Point2f>& pts1,
        std::vector<cv::Point2f>& pts2);

private:
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

} // namespace vslam
