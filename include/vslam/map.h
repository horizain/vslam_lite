#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/mappoint.h"
#include <atomic>
#include <map>
#include <memory>
#include <optional>

namespace vslam {

/// 地图：管理所有关键帧和地图点
/// M1：版本模型——拓扑（集合增删）与几何（坐标变更）分离计数。
/// 后端任务快照绑定版本号，提交前必须 stale 检查；只有 BackendCommitter
/// （M2）能通过 bumpGeometry 发布几何新版本。
class Map {
public:
    using Ptr = std::shared_ptr<Map>;

    struct Covisibility {
        KeyframeId keyframe_id = 0;
        size_t shared_points = 0;
    };

    Map() = default;

    // ---- 版本（M1）----
    /// 拓扑版本：关键帧/地图点集合增删（insert/cull/clear）
    uint64_t topologyRevision() const { return topology_rev_.load(std::memory_order_relaxed); }
    /// 几何版本：位姿/点坐标变更（仅 BackendCommitter 发布）
    uint64_t geometryRevision() const { return geometry_rev_.load(std::memory_order_relaxed); }
    void bumpTopology() { topology_rev_.fetch_add(1, std::memory_order_relaxed); }
    void bumpGeometry() { geometry_rev_.fetch_add(1, std::memory_order_relaxed); }

    // ---- 地图点 ----
    /// 分配当前 Map 内唯一的地图点 id
    unsigned long nextMapPointId() { return next_mp_id_++; }
    void insertMapPoint(MapPoint::Ptr mp);
    MapPoint::Ptr getMapPoint(unsigned long id) const;
    /// 原子删除地图点及其全部 observation、共视计数、KF slot。
    /// 调用方必须持有 map_mutex_ 独占锁；不存在时幂等返回 false。
    bool removeMapPoint(MapPointId id);
    /// 原子计数：供状态栏/状态轮询在锁外读取（无需锁住整个 std::map）。
    /// 与 map_points_.size() 的唯一差别是"插入后计数立即生效"，对统计足够。
    size_t mapPointCount() const { return mp_count_.load(std::memory_order_relaxed); }

    /// 剔除观测次数不足的地图点
    void cullMapPoints(int min_observations = 2);

    // ---- 资源预算（M2.2，§6.3）----
    /// 原子移除关键帧：清空该 KF 全部正式观测（反向集合 + 共视计数）与
    /// slot 后从地图集合删除。调用方必须持有 map_mutex_ 独占锁；
    /// 不存在时幂等返回 false。
    bool removeKeyFrame(KeyframeId id);

    /// 记录一次跟踪命中（旁路统计，§6.3 第 2 步的"最近命中 KF"）。
    /// 与 observed_count 语义无关：普通帧的临时关联也计入，用于弱点回收
    /// 判断。调用方须持 map_mutex_ 写锁；正式观测建立时自动记录。
    void recordTrackingHit(MapPointId map_point_id);

    /// 该点最后一次命中时的关键帧计数（0 = 从未命中）。
    /// 用于"超过 30 个 KF 未被跟踪命中"的陈旧判断。
    [[nodiscard]] size_t lastHitKeyframeCount(MapPointId map_point_id) const;

    // ---- 正式观测关系（调用方必须持有 map_mutex_ 独占锁）----
    /// 原子维护关键帧 feature slot 与 MapPoint 反向观测。
    /// 传入的关键帧和地图点必须已经注册到本 Map。
    bool setObservation(const Frame::Ptr& keyframe,
                        FeatureIndex feature_index,
                        const MapPoint::Ptr& map_point);

    /// 原子清除关键帧 feature slot 与 MapPoint 反向观测。
    bool clearObservation(KeyframeId keyframe_id,
                          FeatureIndex feature_index);

    /// 关键帧插入或历史代码直接写 slot 后，重建该 KF 的正式观测关系。
    void syncKeyframeObservations(const Frame::Ptr& keyframe);

    /// 统计两个关键帧共享的唯一地图点数量。调用方需持有外部
    /// map_mutex_ 读锁（或更强锁）。
    size_t sharedObservationCount(KeyframeId a, KeyframeId b) const;

    /// 返回与指定关键帧共视点数达标的关键帧，按数量降序、id 升序排列。
    /// 调用方需持有外部 map_mutex_ 读锁（或更强锁）。
    std::vector<Covisibility> covisibleKeyframes(
        KeyframeId id, size_t min_shared = 1) const;

    /// 检查 Frame slot、MapPoint 观测集合和共视计数是否相互一致。
    /// 调用方需持有外部 map_mutex_ 读锁（或更强锁）。
    bool verifyObservationConsistency() const;

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
    bool setObservationUnlocked(const Frame::Ptr& keyframe,
                                FeatureIndex feature_index,
                                const MapPoint::Ptr& map_point);
    bool addObservationUnlocked(const Observation& observation,
                                const MapPoint::Ptr& map_point);
    bool clearObservationUnlocked(KeyframeId keyframe_id,
                                  FeatureIndex feature_index);
    /// 仅把当前 slots 注册为正式观测；用于刚插入、按定义没有历史反向观测的 KF。
    bool registerKeyframeObservationsUnlocked(const Frame::Ptr& keyframe);
    void addCovisibilityUnlocked(const Observation& added,
                                 const MapPoint::Ptr& map_point);
    void removeCovisibilityUnlocked(const Observation& removed,
                                    const MapPoint::Ptr& map_point);
    bool clearDuplicateSlotsUnlocked(const Frame::Ptr& keyframe,
                                     const MapPoint::Ptr& map_point,
                                     std::optional<FeatureIndex> keep_feature);
    // 批量剔除时先处理所有正式观测，再对 KF slot 做一次全量扫描；
    // 调用方必须持有 map_mutex_ 独占锁。
    void removeMapPointsUnlocked(
        const std::vector<MapPoint::Ptr>& map_points);
    void removeMapPointUnlocked(const MapPoint::Ptr& map_point);

    std::map<unsigned long, MapPoint::Ptr> map_points_;
    std::map<unsigned long, Frame::Ptr>    keyframes_;

    // 以无向 KF 对为键的持久共视计数；唯一来源仍是 MapPoint 观测集合。
    std::map<std::pair<KeyframeId, KeyframeId>, size_t> covisibility_;

    // M2.2：地图点"最近命中 KF"旁路统计（§6.3 第 2 步）。记录命中时的
    // 关键帧计数，与 observed_count 无关；随点删除/清图一并清理。
    std::map<MapPointId, size_t> last_hit_kf_;

    std::atomic<size_t> mp_count_{0};  // 地图点数量（原子，锁外可读）
    std::atomic<size_t> kf_count_{0};  // 关键帧数量（原子，锁外可读）

    std::atomic<uint64_t> topology_rev_{0};  // 集合变化版本（M1）
    std::atomic<uint64_t> geometry_rev_{0};  // 坐标变化版本（M1）

    unsigned long next_mp_id_ = 0;
    unsigned long next_kf_id_ = 0;
};

} // namespace vslam
