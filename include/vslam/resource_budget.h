#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/map.h"

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vslam {

/// §6.2 首版参数 + §6.3 地图预算算法参数。
/// 参数变更必须按 §0.1 用不少于 3 类场景的机器人实录数据统一标定，
/// 不得为单个 KITTI/EuRoC 序列改动。
///
/// M2 遗留清理实测标定（2026-08-10，KITTI 00 代表场景）：
///   §6.2 原首版参数（KF 1200/点 120000/总量 900MB）与 §6.5 RSS 硬门槛
///   （<1GiB）互斥——实测 12 万点 + 1100 KF 时 RSS ≈ 1.3GB（点 ~2.5KB/个、
///   KF ~150KB/个、词袋 ~350MB、框架 ~150MB）。按 §0.1"产品硬门槛未达到
///   不能进入下一发布阶段"，以硬门槛优先标定：地图预算上限使 RSS 峰值
///   ≈ 词袋 350MB + 框架 150MB + 地图 ≈ 400MB ≈ 900MB < 1GiB。
///   标定依据为 KITTI 00 实测（唯一可用场景）；后续 ≥3 类实录数据再统一
///   修订（DEVELOPMENT_LOG §3.38 记录）。
struct MapBudgetConfig {
    // ---- §6.2 MapBudget 硬预算（实测标定版，2026-08-10）----
    size_t max_active_keyframes = 700;
    size_t max_active_points = 60000;
    size_t max_descriptor_mb = 256;
    size_t max_snapshot_mb = 256;
    size_t max_total_estimated_mb = 500;

    // ---- §6.3 固定顺序回收参数 ----
    /// 第 2 步：观测数 < 该值的点为"弱点"
    size_t weak_point_min_observations = 2;
    /// 第 2 步：超过该数量个 KF 未被跟踪命中即视为陈旧
    size_t weak_point_stale_kf_window = 30;
    /// 第 3 步：保留图像的最近 KF 数（其余视为"非活动"，卸载原图/灰度图）
    size_t kf_image_keep_recent = 2;
    /// 第 4 步：共视重叠率（shared / min(count_a, count_b)）超过该值且
    /// 相邻位姿差小于位移/旋转门限的 KF 视为冗余
    double redundant_overlap_threshold = 0.9;
    double redundant_max_translation_m = 0.15;
    double redundant_max_rotation_deg = 3.0;
    /// 第 5 步：非活动子地图超过该数量时冻结更老的并卸载图像缓存
    size_t max_inactive_submaps = 2;

    // ---- 内存估算常量（§6.4 指标；2026-08-10 按 KITTI 00 实测标定：
    // KF ≈ 150KB/个（描述子 32KB + 关键点 24KB + map_points 指针 + 观测），
    // 点 ≈ 2.5KB/个（对象 + 描述子 32B + observations 向量））----
    size_t overhead_bytes_per_keyframe = 131072;
    size_t overhead_bytes_per_point = 2560;

    static constexpr size_t bytes_per_mb = 1024 * 1024;
};

/// 预算检查结果（只读，不修改地图）
struct BudgetStatus {
    bool within_budget = true;
    size_t keyframes = 0;
    size_t points = 0;
    size_t descriptor_bytes = 0;
    size_t image_bytes = 0;
    size_t snapshot_bytes = 0;        // 在途 OptimizationSnapshot（外部传入）
    size_t estimated_total_bytes = 0; // 各项 + 对象开销估算

    bool over_keyframes = false;
    bool over_points = false;
    bool over_descriptor = false;
    bool over_snapshot = false;
    bool over_total = false;
};

/// 单次回收执行结果（§6.3 各步计数）
struct BudgetReclaimResult {
    size_t removed_zero_obs_points = 0;      // 第 1 步
    size_t removed_weak_stale_points = 0;    // 第 2 步
    size_t unloaded_kf_images = 0;           // 第 3 步
    size_t culled_redundant_keyframes = 0;   // 第 4 步
    size_t frozen_submaps = 0;               // 第 5 步
    size_t unloaded_submap_kf_images = 0;    // 第 5 步
    size_t removed_frozen_submap_points = 0; // 第 5 步：冻结子地图弱陈点删除
    std::vector<KeyframeId> culled_keyframe_ids; // 第 4 步：供外部索引批量同步
    bool stopped_map_growth = false;         // 第 6 步：仍超预算
};

/// §6.3 地图预算引擎：无锁、无线程、不持有 Map/Atlas 状态。
/// 调用方必须持有 map_mutex_ 独占锁（写路径约定与 Map API 一致）。
/// 触发预算时按文档固定顺序回收，任一步回到预算内立即返回；
/// 全部手段耗尽仍超预算时设置 stopped_map_growth（调用方上报
/// Degraded + BackendOverloaded，不得随机删除锚点）。
class ResourceBudget {
public:
    explicit ResourceBudget(const MapBudgetConfig& config = {});

    /// 只读估算当前占用并判定是否超限。
    /// @param snapshot_bytes 在途 Local BA 快照字节数（由调用方上报，
    ///        §6.4 指标；M2.2 阶段默认 0）
    [[nodiscard]] BudgetStatus evaluate(const Map::Ptr& map,
                                        size_t snapshot_bytes = 0) const;

    /// 按 §6.3 固定顺序执行回收，直到回到预算内或耗尽全部手段。
    /// @param protected_keyframe_ids 回环 KF / 子地图锚点，冗余剔除必须跳过
    /// @param submap_keyframes       非活动子地图 → 其 KF id 列表；超过
    ///        max_inactive_submaps 个子地图时冻结更老的（按子地图 id 升序，
    ///        保留最新的 2 个）并卸载其 KF 图像缓存。冻结标志本身属于
    ///        Atlas 状态，M4 前本模块只做图像卸载与计数。
    /// @param inactive_submaps      非活动子地图 → 其 Map；与 submap_keyframes
    ///        同序提供时，第 5 步在卸载图像外同时删除冻结子地图的弱陈点
    ///        （observationCount < weak_point_min_observations 且超过
    ///        weak_point_stale_kf_window 个 KF 未被跟踪命中）——冻结地图的
    ///        点主体不再被访问，保留 KF 描述子作为重定位骨架；这是 M4 磁盘
    ///        换出落地前控制 RSS 硬门槛（§6.5 <1GiB）的最低内存手段。
    [[nodiscard]] BudgetReclaimResult reclaim(
        const Map::Ptr& map,
        const std::unordered_set<KeyframeId>& protected_keyframe_ids = {},
        const std::unordered_map<SubmapId, std::vector<KeyframeId>>&
            submap_keyframes = {},
        size_t snapshot_bytes = 0,
        const std::unordered_map<SubmapId, Map::Ptr>& inactive_submaps = {},
        const std::function<void(const Frame::Ptr&, const Frame::Ptr&)>&
            before_keyframe_cull = {}) const;

    /// 描述子字节统计（点代表描述子 + 各 KF 描述子矩阵；供 §6.4 指标复用）
    [[nodiscard]] static size_t descriptorBytes(const Map::Ptr& map);
    /// 图像字节统计（原图 + 灰度图，左右目；供 §6.4 指标复用）
    [[nodiscard]] static size_t imageBytes(const Map::Ptr& map);
    [[nodiscard]] static size_t matBytes(const cv::Mat& mat);

    /// 只读配置（VO 恢复 stopped_map_growth 时做轻量判据）
    [[nodiscard]] const MapBudgetConfig& config() const { return config_; }

private:
    /// 第 4 步：相邻 KF 对是否冗余（位姿差 + 共视重叠率）
    [[nodiscard]] bool redundantPair(const Frame::Ptr& older,
                                     const Frame::Ptr& newer,
                                     const Map::Ptr& map) const;
    /// KF slot 中引用的不同地图点数量（共视重叠率的分母）
    [[nodiscard]] static size_t distinctPointCount(const Frame::Ptr& keyframe);

    MapBudgetConfig config_;
};

} // namespace vslam
