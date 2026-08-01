#pragma once

#include "vslam/common.h"
#include <opencv2/core.hpp>

namespace vslam {

/// 地图点：3D 空间中的一个点，被多帧观测到
struct MapPoint {
    using Ptr = std::shared_ptr<MapPoint>;

    unsigned long id   = 0;     // 全局唯一 ID
    Vec3 pos_w         = Vec3::Zero();  // 世界坐标系下的位置
    cv::Mat descriptor;          // 最具代表性的描述子（用于匹配）
    int observed_count = 0;     // 被观测到的次数
    int inlier_count   = 0;     // 内点计数（用于剔除不可靠点）

    MapPoint() = default;
    explicit MapPoint(unsigned long id_) : id(id_) {}

    /// 工厂方法：从两个观测创建地图点（三角化）
    static Ptr create(unsigned long id,
                      const Vec2& px1, const Vec2& px2,
                      const SE3& T1, const SE3& T2,
                      const cv::Mat& K);
};

} // namespace vslam
