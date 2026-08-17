#pragma once

#include "vslam/atlas.h"
#include "vslam/camera.h"
#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/optimizer.h"

#include <opencv2/features2d.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace vslam {

/// Local BA 快照是否应保留某个地图点（M1.5：从 vo.h 迁移，公式不变）。
[[nodiscard]] bool includeLocalBALandmark(int observation_count,
                                          int min_observed);

/// 局部建图模块（M1.5，§5.5）：消费前端提议，通过 Map API 插入关键帧、
/// 地图点与正式 Observation；负责 Local BA 快照与共视窗口选择。
///
/// 只操作调用方传入的 Map/Frame（调用方须按旧约定持 map_mutex_ 相应锁），
/// 不持有 VO 状态、不切换子地图、不执行优化求解、不创建线程。
class LocalMapper {
public:
    explicit LocalMapper(const Camera& camera);

    /// 双目/RGB-D 单帧建点：pts_c 有效 → 子地图局部系 3D 地图点。
    /// 已注册 KF 通过 Map 维护双向正式观测；未注册帧先写 slot。
    /// 调用方须持 map_mutex_ 独占锁（旧 createMapPointsFromStereo）。
    void createMapPointsFromStereo(const Map::Ptr& map,
                                   const Frame::Ptr& frame,
                                   size_t max_map_points =
                                       std::numeric_limits<size_t>::max()) const;

    /// 两帧三角化建点（单目/初始化）：MapPoint::create + 正式观测绑定。
    /// 调用方须持 map_mutex_ 独占锁（旧 triangulateNewPoints）。
    void triangulateNewPoints(const Map::Ptr& map,
                              const Frame::Ptr& f1, const Frame::Ptr& f2,
                              const std::vector<cv::DMatch>& matches,
                              size_t max_map_points =
                                  std::numeric_limits<size_t>::max()) const;

    /// 共视图滑动窗口选择（Local BA 窗口）：当前帧 + 共视最多的 KF，
    /// 共视不足退化为最近 n 帧，最终按 id 升序。调用方锁约定同旧
    /// selectLocalWindow（同步路径无锁、异步路径持读锁）。
    [[nodiscard]] std::vector<Frame::Ptr> selectLocalWindow(
        const Map::Ptr& map, const Frame::Ptr& curr_frame, int n) const;

    /// Local BA 只读快照（§5.5：OptimizationSnapshot，锁外求解）。
    /// min_observed 门槛过滤弱观测点；只保留 anchor KF 所在观测连通分量。
    /// 调用方须持 map_mutex_ 读锁（旧 buildLocalBASnapshot）。
    [[nodiscard]] OptimizationSnapshot buildLocalBASnapshot(
        const Map::Ptr& map, const Atlas::Ptr& atlas,
        const std::vector<Frame::Ptr>& window,
        KeyframeId anchor_kf_id, int min_observed = 0,
        size_t max_landmarks = std::numeric_limits<size_t>::max()) const;

private:
    Camera camera_;
};

} // namespace vslam
