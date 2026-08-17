#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/map.h"
#include "vslam/observation.h"

#include <opencv2/features2d.hpp>
#include <optional>
#include <unordered_set>
#include <vector>

namespace vslam {

/// 两个位姿均按 T_cw 表示时，相机光心实际发生的米制位移。
/// 该量不受姿态旋转参数化影响，供 BA 质量指标和回归测试使用。
[[nodiscard]] double cameraPositionDelta(const SE3& old_cw,
                                         const SE3& new_cw);

/// (Phase 2) 位姿图边：关键帧 a ↔ b 的相对位姿约束
struct LoopEdge {
    unsigned long a = 0;      // 起点关键帧 id
    unsigned long b = 0;      // 终点关键帧 id
    SE3 T_rel;                // X_b = X_a * T_rel（X 为 T_wc）
    double weight = 1.0;      // 信息权重（共视多 → 置信高）
};

// ============================================================
// M1：Optimizer 只读快照/Result 契约（§14.1/§14.2）
//
// 强制不变量：Optimizer 永远不接收可写的实时 Map::Ptr，只在不可变
// OptimizationSnapshot 上做纯计算，返回 OptimizationResult（候选增量 +
// 质量指标）。调用方（唯一 BackendCommitter，M2）负责 stale 检查与
// 原子提交；优化失败/结果过期/验收失败时实时状态逐项保持不变。
// ============================================================

/// 快照关键帧：仅保存位姿；正式观测统一位于 OptimizationSnapshot::observations。
struct KeyframeState {
    unsigned long id = 0;
    SE3 pose_cs;  // T_cw（与全局项目语义一致）
};

/// 优化快照中的正式观测；map_point_id=0 也是合法地图点。
/// camera_point 有值时表示该观测带有相机系深度，可用于深度权重。
struct ObservationState {
    KeyframeId keyframe_id = 0;
    FeatureIndex feature_index = 0;
    MapPointId map_point_id = 0;
    Vec2 pixel = Vec2::Zero();
    std::optional<Vec3> camera_point;
};

/// 快照地图点：坐标 + 观测数
struct LandmarkState {
    unsigned long id = 0;
    Vec3 pos_s;
    int observations = 0;
};

/// 位姿图约束（里程计边 / 回环边统一表达）
struct Constraint {
    unsigned long a = 0;
    unsigned long b = 0;
    SE3 T_rel;               // X_b = X_a * T_rel
    double weight = 1.0;
    bool is_loop = false;    // 回环边：Huber 核 + 残差预检（M0 防爆）
};

/// 优化输入：不可变快照（绑定子地图与版本）
struct OptimizationSnapshot {
    unsigned long submap_id = 0;
    uint64_t topology_revision = 0;
    uint64_t geometry_revision = 0;
    std::vector<KeyframeState> keyframes;
    std::vector<ObservationState> observations;
    std::vector<LandmarkState> landmarks;
    std::vector<Constraint> constraints;
    std::vector<unsigned long> fixed_kf_ids;  // BA/位姿图锚定（窗口最早帧、已发布 Atlas 子图等）
};

struct PoseUpdate {
    unsigned long id = 0;
    SE3 pose_cs;
};
struct PointUpdate {
    unsigned long id = 0;
    Vec3 pos_s;
};

/// 优化质量指标（供提交器验收）
struct OptimizationMetrics {
    int iterations = 0;
    double initial_chi2 = 0.0;
    double final_chi2 = 0.0;
    double max_correction = 0.0;   // 最大位姿校正（m）
    double max_neighbor_step = 0.0;  // 相邻关键帧最大步长（m）
    size_t vertices = 0;
    size_t edges = 0;
    bool converged = false;        // chi2 未发散（final ≤ initial*1.01）
};

/// 优化输出：候选增量 + 质量指标。valid=false 表示优化失败/保护拒绝，
/// 调用方不得提交任何内容。
struct OptimizationResult {
    unsigned long submap_id = 0;
    uint64_t base_topology_revision = 0;  // 快照基准版本（stale 检查用）
    uint64_t base_geometry_revision = 0;
    std::vector<PoseUpdate> poses;
    std::vector<PointUpdate> points;
    OptimizationMetrics metrics;
    bool valid = false;
};

/// 图优化后端（基于 g2o）——纯计算，只读快照，不修改任何实时地图对象
class Optimizer {
public:
    /// 局部 Bundle Adjustment（快照上）：优化窗口关键帧 + 地图点。
    /// 返回结果含全部位姿/点候选增量；fixed_kf_ids 内位姿固定（锚定）。
    /// @param max_points  参与优化的地图点数量上限（按观测数降序截断）
    static OptimizationResult solveLocalBA(
        const Camera& camera,
        const OptimizationSnapshot& snap,
        int max_iterations = 10,
        std::optional<bool> fix_points = std::nullopt,
        size_t max_points = 4000,
        int passes = 2);

    /// 位姿图优化（快照上）：约束 = 里程计边（无鲁棒核）+ 回环边（Huber）。
    /// 内置防护（M0）：回环边残差预检、chi2 收敛、有限值、最大校正/相邻步长
    /// 检查——任一失败 → valid=false，调用方不得提交。
    static OptimizationResult solvePoseGraph(
        const OptimizationSnapshot& snap,
        int max_iterations = 20);
};

} // namespace vslam
