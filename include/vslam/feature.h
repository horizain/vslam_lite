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
    void setParams(int num_features, double scale_factor, int pyramid_levels,
                   int orb_max_bands = 8);

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

    /// 双目左右目匹配：LK 光流 左目→右目（校正图像行对齐，亚像素视差）。
    /// 输入左目特征点，输出每点对应的右目像素坐标；返回成功掩码。
    std::vector<unsigned char> matchStereo(
        const cv::Mat& left_gray,
        const cv::Mat& right_gray,
        const std::vector<cv::KeyPoint>& left_keypoints,
        std::vector<cv::Point2f>& right_pts);

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
    int num_features_ = 1000;      // ORB 特征预算（并行分带提取用）
    double scale_factor_ = 1.2;
    int pyramid_levels_ = 8;
    int orb_max_bands_ = 8;        // 分带上限；1 禁用外层分带并行
};

} // namespace vslam
