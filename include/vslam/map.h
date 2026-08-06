#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/mappoint.h"
#include <atomic>
#include <map>
#include <memory>

namespace vslam {

/// 地图：管理所有关键帧和地图点
/// M1：版本模型——拓扑（集合增删）与几何（坐标变更）分离计数。
/// 后端任务快照绑定版本号，提交前必须 stale 检查；只有 BackendCommitter
/// （M2）能通过 bumpGeometry 发布几何新版本。
class Map {
public:
    using Ptr = std::shared_ptr<Map>;

    Map() = default;

    // ---- 版本（M1）----
    /// 拓扑版本：关键帧/地图点集合增删（insert/cull/clear）
    uint64_t topologyRevision() const { return topology_rev_.load(std::memory_order_relaxed); }
    /// 几何版本：位姿/点坐标变更（仅 BackendCommitter 发布）
    uint64_t geometryRevision() const { return geometry_rev_.load(std::memory_order_relaxed); }
    void bumpTopology() { topology_rev_.fetch_add(1, std::memory_order_relaxed); }
    void bumpGeometry() { geometry_rev_.fetch_add(1, std::memory_order_relaxed); }

    // ---- 地图点 ----
    /// 分配一个全局唯一的地图点 id
    unsigned long nextMapPointId() { return next_mp_id_++; }
    void insertMapPoint(MapPoint::Ptr mp);
    MapPoint::Ptr getMapPoint(unsigned long id) const;
    /// 原子计数：供状态栏/状态轮询在锁外读取（无需锁住整个 std::map）。
    /// 与 map_points_.size() 的唯一差别是"插入后计数立即生效"，对统计足够。
    size_t mapPointCount() const { return mp_count_.load(std::memory_order_relaxed); }

    /// 剔除观测次数不足的地图点
    void cullMapPoints(int min_observations = 2);

    // ---- 关键帧 ----
    void insertKeyFrame(Frame::Ptr kf);
    Frame::Ptr getKeyFrame(unsigned long id) const;
    size_t keyFrameCount() const { return kf_count_.load(std::memory_order_relaxed); }

    /// 获取所有关键帧（按 ID 排序）
    std::vector<Frame::Ptr> getAllKeyFrames() const;

    /// 获取所有地图点
    std::vector<MapPoint::Ptr> getAllMapPoints() const;

    /// 清空地图
    void clear();

private:
    std::map<unsigned long, MapPoint::Ptr> map_points_;
    std::map<unsigned long, Frame::Ptr>    keyframes_;

    std::atomic<size_t> mp_count_{0};  // 地图点数量（原子，锁外可读）
    std::atomic<size_t> kf_count_{0};  // 关键帧数量（原子，锁外可读）

    std::atomic<uint64_t> topology_rev_{0};  // 集合变化版本（M1）
    std::atomic<uint64_t> geometry_rev_{0};  // 坐标变化版本（M1）

    unsigned long next_mp_id_ = 0;
    unsigned long next_kf_id_ = 0;
};

} // namespace vslam
