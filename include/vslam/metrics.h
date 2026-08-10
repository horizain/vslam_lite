#pragma once

#include "vslam/backend_scheduler.h"
#include "vslam/localization_types.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vslam {

/// §6.4：结构化指标快照（JSON/CSV 输出，soak_test.py / nightly 消费）。
/// 字段命名即输出契约；未接数据源的字段保持 -1（见 DEVELOPMENT_LOG §3.37
/// 未接边界：grid occupancy、PnP condition number、loop queried/candidates/
/// verified 由 M3.1/M3.2/M5 数据源补齐）。
struct MetricsSnapshot {
    // ---- frame latency / deadline（§6.4）----
    long long frames_processed = 0;
    long long deadline_ms = 80;
    long long deadline_miss = 0;
    double deadline_miss_ratio = 0.0;
    double latency_p50_ms = 0.0;
    double latency_p95_ms = 0.0;
    double latency_p99_ms = 0.0;
    double latency_max_ms = 0.0;

    // ---- input（§6.4）----
    long long input_received = 0;
    long long input_processed = 0;
    long long input_dropped = 0;
    long long input_queue_hwm = 0;

    // ---- tracking（§6.4；features/stereo/inliers/ratio/RMSE 均值）----
    double features_avg = -1.0;
    double stereo_points_avg = -1.0;
    double pnp_inliers_avg = -1.0;
    double pnp_inlier_ratio_avg = -1.0;
    double pnp_rmse_avg = -1.0;

    // ---- pose（§6.4）----
    long long pose_accepted = 0;
    long long pose_rejected = 0;
    long long pose_prediction_only = 0;
    /// 每类 FailureReason 计数（按枚举序；0 = 无失败）
    std::array<long long, 11> failure_reasons{};

    // ---- backend（§6.4；stale/invalid 数据源在 BackendCommitter，M2.3 遗留清理后已接）----
    long long backend_submitted = 0;
    long long backend_executed = 0;
    long long backend_dropped = 0;
    long long backend_pending = 0;
    double backend_task_age_max_ms = -1.0;
    double backend_task_age_avg_ms = -1.0;
    /// 提交结果计数（§6.4 backend committed/stale/invalid；queued 即 submitted）
    long long backend_committed = 0;
    long long backend_stale = 0;
    long long backend_invalid = 0;
    long long backend_not_found = 0;

    // ---- loop（§6.4；committed 已接，queried/candidates/verified 未接）----
    long long loop_committed = 0;

    // ---- map / RSS（§6.4；snapshot_bytes 由 VO 在途 Local BA 快照上报）----
    long long map_keyframes = 0;
    long long map_points = 0;
    long long map_observations = 0;
    long long map_descriptor_bytes = 0;
    long long map_image_bytes = 0;
    long long map_snapshot_bytes = -1;
    long long map_estimated_total_bytes = 0;

    // ---- lost / relocalization（§6.4）----
    long long lost_count = 0;
    double lost_duration_s = 0.0;
    double relocalization_latency_p95_ms = -1.0;
};

/// FailureReason 的 JSON 名字（与枚举序一致）
constexpr const char* kFailureReasonNames[] = {
    "None", "InvalidInput", "TimestampRollback", "StereoUnsynchronized",
    "InsufficientFeatures", "GeometricRejection", "MotionDiscontinuity",
    "RelocalizationTimeout", "BackendOverloaded", "MapIncompatible",
    "InternalError"};

/// §6.4 线程安全的结构化指标采集器（M2.3）。
/// 只收值对象、不持有 VO/Map/Localizer 状态；由 Localizer 各路径喂数据，
/// 退出时输出 JSON/CSV 供 soak_test.py / nightly 断言（禁止正则解析日志）。
class MetricsCollector {
public:
    explicit MetricsCollector(long long deadline_ms = 80);

    /// 单帧跟踪延迟（ms）；超过 deadline 计为 deadline miss
    void recordFrameLatency(double latency_ms);
    /// 输入路径计数（队列 hwm/dropped 由 BoundedQueue 提供）
    void recordInput(long long received, long long dropped, long long queue_hwm);
    /// 每帧跟踪质量（均值；-1 表示无样本——调用方传入无效值时不更新）
    void recordTracking(double features, double stereo_points, double pnp_inliers,
                        double inlier_ratio, double rmse);
    /// 每帧位姿结果（accepted/rejected/prediction_only + FailureReason 计数）
    void recordPose(const PoseEstimate& pose);
    /// 后端调度统计（取最新值；§6.4 backend 指标）
    void recordBackend(const BackendSchedulerStats& stats);
    /// 已闭合回环总数（取最新值）
    void recordLoopCommitted(long long committed);
    /// 地图规模与字节统计（取最新值；snapshot_bytes < 0 表示未上报）
    void recordMap(long long keyframes, long long map_points, long long observations,
                   long long descriptor_bytes, long long image_bytes,
                   long long snapshot_bytes, long long estimated_total_bytes);
    /// 状态机转换（timestamp 秒）：用于 LOST 段计数/时长与
    /// relocalization latency（Relocalizing → Tracking/Degraded）
    void recordStateChange(TrackingState from, TrackingState to, double timestamp);

    void setDeadlineMs(long long deadline_ms);

    [[nodiscard]] MetricsSnapshot snapshot() const;
    [[nodiscard]] std::string toJson() const;
    [[nodiscard]] std::string toCsv() const;
    void writeJson(const std::string& path) const;
    void writeCsv(const std::string& path) const;

private:
    mutable std::mutex mutex_;
    long long deadline_ms_;

    // latency 历史（退出时一次性算分位；2h@10Hz ≈ 7 万样本，内存可控）
    std::vector<double> latencies_;
    long long frames_processed_ = 0;
    long long deadline_miss_ = 0;

    long long input_received_ = 0;
    long long input_processed_ = 0;
    long long input_dropped_ = 0;
    long long input_queue_hwm_ = 0;

    double features_sum_ = 0.0;
    double stereo_sum_ = 0.0;
    double inliers_sum_ = 0.0;
    double ratio_sum_ = 0.0;
    double rmse_sum_ = 0.0;
    long long tracking_samples_ = 0;

    long long pose_accepted_ = 0;
    long long pose_rejected_ = 0;
    long long pose_prediction_only_ = 0;
    std::array<long long, 11> failure_reasons_{};

    BackendSchedulerStats backend_stats_;
    long long loop_committed_ = 0;

    long long map_keyframes_ = 0;
    long long map_points_ = 0;
    long long map_observations_ = 0;
    long long map_descriptor_bytes_ = 0;
    long long map_image_bytes_ = 0;
    long long map_snapshot_bytes_ = -1;
    long long map_estimated_total_bytes_ = 0;

    long long lost_count_ = 0;
    double lost_duration_s_ = 0.0;
    bool in_lost_ = false;
    double lost_start_ = 0.0;
    bool in_reloc_ = false;
    double reloc_start_ = 0.0;
    std::vector<double> reloc_latencies_ms_;
};

} // namespace vslam
