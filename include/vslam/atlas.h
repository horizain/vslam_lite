#pragma once

#include "vslam/common.h"
#include "vslam/map.h"
#include <deque>

namespace vslam {

/// 轻量 Atlas：保存多个局部地图，并让每个子地图共享同一个全局世界坐标系。
///
/// 这里不实现子地图融合或位姿图优化。子地图只在跟踪长期失败时创建，
/// origin_Twc 记录其第一帧在全局世界系中的锚定位姿。
struct Submap {
    unsigned long id = 0;
    Map::Ptr map;
    SE3 origin_Twc;
    bool frozen = false;
};

class Atlas {
public:
    using Ptr = std::shared_ptr<Atlas>;

    Atlas() = default;

    /// 创建并激活一个子地图。origin_Twc 是新子地图原点在全局系中的位姿。
    Submap& createSubmap(const SE3& origin_Twc);

    /// 激活已有子地图，用于后续重定位回旧地图。
    bool activate(unsigned long id);

    Submap* activeSubmap();
    const Submap* activeSubmap() const;
    Map::Ptr activeMap() const;

    const std::deque<Submap>& submaps() const { return submaps_; }
    size_t submapCount() const { return submaps_.size(); }

private:
    std::deque<Submap> submaps_;
    unsigned long active_id_ = 0;
    unsigned long next_id_ = 0;
};

} // namespace vslam
