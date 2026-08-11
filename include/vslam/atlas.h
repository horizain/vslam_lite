#pragma once

#include "vslam/common.h"
#include "vslam/map.h"
#include <deque>
#include <vector>

namespace vslam {

/// 轻量 Atlas：保存多个局部地图，每个子地图拥有独立局部坐标系。
///
/// M3：Submap::T_ws 是"子地图→世界"的唯一权威变换——地图内 KF 位姿
/// （Frame::pose_cs）与地图点（MapPoint::pos_s）全部存局部坐标：
///   p_w = T_ws · p_s；T_cw = T_cs · T_ws⁻¹
/// 更新子地图全局位置（子地图对齐/跨子地图约束）只修改 T_ws，不再遍历
/// 所有 KF 和地图点（§14.3）。
struct Submap {
    unsigned long id = 0;
    Map::Ptr map;
    SE3 T_ws;               // 子地图系 → 世界系（M3 前为 origin_Twc）
    bool frozen = false;
    bool connected = true;  // 是否已通过重定位/约束连接到全局世界系
};

/// M5：Atlas 子地图约束图边（§14.5）——只连坐标，不做地图融合。
enum class AtlasConstraintType {
    TrackingBridge,   // 跟丢后运动外推锚定（低权重，可被后续约束修正）
    Relocalization,   // 跨子地图重定位产生的 SE3 约束（中权重）
    LoopClosure       // 完整几何验证后的高置信约束（高权重）
};

struct AtlasConstraint {
    unsigned long a = 0;       // 起点子地图 id
    unsigned long b = 0;       // 终点子地图 id
    SE3 T_rel;                 // 满足 T_ws_b = T_ws_a · T_rel
    double weight = 1.0;
    AtlasConstraintType type = AtlasConstraintType::Relocalization;
};

/// 从同一相机的跨子地图回环测量构造 Atlas 约束。
/// verifyLoop 输出 Z = T_cs_loop_kf * P^-1，其中 P 是当前相机在回环
/// 子地图中的 T_cw；Atlas 边要求 T_ws_loop = T_ws_current * T_rel。
[[nodiscard]] AtlasConstraint makeCrossSubmapLoopConstraint(
    SubmapId current_submap_id, SubmapId loop_submap_id,
    const SE3& current_pose_cs, const SE3& loop_pose_cs,
    const SE3& T_loop_curr, double weight = 10.0);

/// 同一子地图局部相机位姿不变时，把世界系 T_cw 从旧 Atlas 锚重基到新锚。
/// T_cw_new = T_cw_old * T_ws_old * T_ws_new^-1。
[[nodiscard]] SE3 rebaseWorldPoseForSubmapAnchor(
    const SE3& T_cw_old, const SE3& T_ws_old, const SE3& T_ws_new);

class Atlas {
public:
    using Ptr = std::shared_ptr<Atlas>;

    Atlas() = default;

    /// 创建并激活一个子地图。T_ws 是新子地图原点在世界系中的位姿。
    Submap& createSubmap(const SE3& T_ws, bool connected = true);

    /// 激活已有子地图，用于后续重定位回旧地图。
    bool activate(unsigned long id);

    Submap* activeSubmap();
    const Submap* activeSubmap() const;
    /// 按 id 查找子地图（写 T_ws 等用）
    Submap* getSubmap(unsigned long id);
    const Submap* getSubmap(unsigned long id) const;
    Map::Ptr activeMap() const;

    const std::deque<Submap>& submaps() const { return submaps_; }
    size_t submapCount() const { return submaps_.size(); }

    /// M5：约束图（TrackingBridge / Relocalization / LoopClosure 边）
    void addConstraint(const AtlasConstraint& c) { constraints_.push_back(c); }
    /// 事务回滚：移除自 index 起的全部约束（失败的重定位必须逐项恢复）
    void removeConstraintsFrom(size_t index) {
        if (index < constraints_.size()) constraints_.resize(index);
    }
    const std::vector<AtlasConstraint>& constraints() const { return constraints_; }
    /// 按子地图过滤约束（a 或 b 命中）
    std::vector<AtlasConstraint> constraintsOf(unsigned long submap_id) const;
    /// 检查全部高置信 LoopClosure 边在当前 T_ws 解上的残差；低置信
    /// TrackingBridge/Relocalization 不参与此硬门。
    [[nodiscard]] bool loopConstraintsConsistent(
        double max_translation, double max_rotation,
        double* max_translation_seen = nullptr,
        double* max_rotation_seen = nullptr) const;

private:
    std::deque<Submap> submaps_;
    std::vector<AtlasConstraint> constraints_;  // M5
    unsigned long active_id_ = 0;
    unsigned long next_id_ = 0;
};

} // namespace vslam
