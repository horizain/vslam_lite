#pragma once

#include "vslam/common.h"
#include "vslam/observation.h"
#include <opencv2/core.hpp>

#include <optional>
#include <set>

namespace vslam {

/// 地图点：3D 空间中的一个点，被多帧观测到
/// M3：坐标存"子地图局部系"（pos_s）；全局坐标由 Submap::T_ws 派生：
/// p_w = T_ws · p_s。子地图内部（PnP/BA/三角化/回环）全部使用局部系。
struct MapPoint {
    using Ptr = std::shared_ptr<MapPoint>;

    MapPointId id       = 0;     // 所属 Map 内唯一 ID；0 是合法地图点 id
    Vec3 pos_s         = Vec3::Zero();  // 子地图局部系坐标（M3 前为 pos_s 世界坐标）
    cv::Mat descriptor;          // 最具代表性的描述子（用于匹配）
    int inlier_count   = 0;     // 内点计数（用于剔除不可靠点）

    using ObservationSet = std::set<Observation>;

    MapPoint() = default;
    explicit MapPoint(unsigned long id_) : id(id_) {}

    /// 添加正式关键帧观测；同一地图点在同一关键帧最多绑定一个特征。
    /// 重复添加同一观测幂等，返回 false；冲突特征也返回 false。
    /// 通过 Map 使用时，调用方必须持有外部 map_mutex_ 写锁。
    bool addObservation(const Observation& observation);

    /// 删除正式关键帧观测；不存在时返回 false。
    /// 通过 Map 使用时，调用方必须持有外部 map_mutex_ 写锁。
    bool removeObservation(const Observation& observation);

    const ObservationSet& observations() const { return observations_; }
    size_t observationCount() const { return observations_.size(); }
    bool hasObservation(const Observation& observation) const {
        return observations_.find(observation) != observations_.end();
    }

    /// 查询该地图点在指定关键帧中的特征索引。
    std::optional<FeatureIndex> featureIndex(KeyframeId keyframe_id) const;

    /// 工厂方法：从两个观测创建地图点（三角化）
    static Ptr create(unsigned long id,
                      const Vec2& px1, const Vec2& px2,
                      const SE3& T1, const SE3& T2,
                      const cv::Mat& K);

private:
    ObservationSet observations_;
};

} // namespace vslam
