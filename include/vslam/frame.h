#pragma once

#include "vslam/common.h"
#include <opencv2/features2d.hpp>
#include <memory>

namespace vslam {

// 前向声明
struct MapPoint;
struct Feature;

/// 帧：VO/SLAM 的基本处理单元
struct Frame {
    using Ptr = std::shared_ptr<Frame>;

    unsigned long id        = 0;   // 帧序号
    double timestamp        = 0.0; // 时间戳（秒）
    SE3   pose_cw;                  // 世界到相机的变换（T_cw）：p_c = pose_cw * p_w
                                   // 注意：PnP/三角化/BA 全部统一使用此 T_cw 语义
    cv::Mat image;                  // 主目（左目）原始图像（可选，可仅保留灰度图）
    cv::Mat image_gray;             // 主目灰度图（所有特征操作都用这个）
    cv::Mat image_right;            // 右目原始图像（双目；单目为空）
    cv::Mat image_right_gray;       // 右目灰度图（双目；单目为空）

    // 特征
    std::vector<cv::KeyPoint> keypoints;   // 关键点
    cv::Mat                   descriptors; // 描述子

    // 与地图点的关联：keypoints[i] 对应 map_points[i]
    std::vector<std::shared_ptr<MapPoint>> map_points;

    // 每特征点的相机系 3D 观测（双目视差 / RGB-D 深度反投影；z>0 有效）。
    // 统一的"单帧绝对尺度深度"表达：双目/RGB-D 直接填，单目为空走多帧三角化。
    std::vector<Vec3> pts_c;

    bool is_keyframe = false;

    Frame() = default;
    Frame(unsigned long id_, double ts_) : id(id_), timestamp(ts_) {}
};

} // namespace vslam
