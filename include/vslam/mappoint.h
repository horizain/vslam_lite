#pragma once

#include "vslam/common.h"
#include <opencv2/core.hpp>

namespace vslam {

/// 地图点：3D 空间中的一个点，被多帧观测到
/// M3：坐标存"子地图局部系"（pos_s）；全局坐标由 Submap::T_ws 派生：
/// p_w = T_ws · p_s。子地图内部（PnP/BA/三角化/回环）全部使用局部系。
struct MapPoint {
    using Ptr = std::shared_ptr<MapPoint>;

    unsigned long id   = 0;     // 全局唯一 ID
    Vec3 pos_s         = Vec3::Zero();  // 子地图局部系坐标（M3 前为 pos_s 世界坐标）
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
