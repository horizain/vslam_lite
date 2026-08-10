#include "vslam/metrics.h"

#include "metrics_json.h"  // vslam::percentile（utils/，库 PUBLIC include 路径）

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace vslam {

MetricsCollector::MetricsCollector(long long deadline_ms)
    : deadline_ms_(deadline_ms) {}

void MetricsCollector::setDeadlineMs(long long deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    deadline_ms_ = deadline_ms;
}

void MetricsCollector::recordFrameLatency(double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    latencies_.push_back(latency_ms);
    frames_processed_++;
    if (latency_ms > static_cast<double>(deadline_ms_)) deadline_miss_++;
}

void MetricsCollector::recordInput(long long received, long long dropped,
                                   long long queue_hwm) {
    std::lock_guard<std::mutex> lock(mutex_);
    input_received_ = received;
    input_dropped_ = dropped;
    input_queue_hwm_ = queue_hwm;
    input_processed_ = frames_processed_;
}

void MetricsCollector::recordTracking(double features, double stereo_points,
                                      double pnp_inliers, double inlier_ratio,
                                      double rmse) {
    std::lock_guard<std::mutex> lock(mutex_);
    // 无效值（-1/NaN）不污染均值；数据源未接时保持 -1 哨兵
    if (features >= 0.0) features_sum_ += features;
    if (stereo_points >= 0.0) stereo_sum_ += stereo_points;
    if (pnp_inliers >= 0.0) inliers_sum_ += pnp_inliers;
    if (inlier_ratio >= 0.0) ratio_sum_ += inlier_ratio;
    if (rmse >= 0.0) rmse_sum_ += rmse;
    tracking_samples_++;
}

void MetricsCollector::recordPose(const PoseEstimate& pose) {
    std::lock_guard<std::mutex> lock(mutex_);
    // 类别互斥（§6.4）：完整质量 accepted / prediction_only / rejected
    if (pose.pose_valid) {
        if (pose.prediction_only)
            pose_prediction_only_++;
        else
            pose_accepted_++;
    } else {
        pose_rejected_++;
    }
    if (pose.reason != FailureReason::None)
        failure_reasons_[static_cast<size_t>(pose.reason)]++;
}

void MetricsCollector::recordBackend(const BackendSchedulerStats& stats) {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_stats_ = stats;
}

void MetricsCollector::recordLoopCommitted(long long committed) {
    std::lock_guard<std::mutex> lock(mutex_);
    loop_committed_ = committed;
}

void MetricsCollector::recordMap(long long keyframes, long long map_points,
                                 long long observations, long long descriptor_bytes,
                                 long long image_bytes, long long snapshot_bytes,
                                 long long estimated_total_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    map_keyframes_ = keyframes;
    map_points_ = map_points;
    map_observations_ = observations;
    map_descriptor_bytes_ = descriptor_bytes;
    map_image_bytes_ = image_bytes;
    map_snapshot_bytes_ = snapshot_bytes;
    map_estimated_total_bytes_ = estimated_total_bytes;
}

void MetricsCollector::recordStateChange(TrackingState from, TrackingState to,
                                         double timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    // LOST 段：进入 Lost 开始，离开（任意非 Lost 状态）结束（§6.4）
    if (from != TrackingState::Lost && to == TrackingState::Lost) {
        in_lost_ = true;
        lost_start_ = timestamp;
    } else if (from == TrackingState::Lost && to != TrackingState::Lost) {
        if (in_lost_) {
            lost_count_++;
            lost_duration_s_ += std::max(0.0, timestamp - lost_start_);
            in_lost_ = false;
        }
    }
    // relocalization latency：Relocalizing → Tracking/Degraded 恢复
    if (from != TrackingState::Relocalizing && to == TrackingState::Relocalizing) {
        in_reloc_ = true;
        reloc_start_ = timestamp;
    } else if (from == TrackingState::Relocalizing &&
               (to == TrackingState::Tracking || to == TrackingState::Degraded)) {
        if (in_reloc_) {
            reloc_latencies_ms_.push_back(
                std::max(0.0, (timestamp - reloc_start_) * 1000.0));
            in_reloc_ = false;
        }
    }
}

MetricsSnapshot MetricsCollector::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MetricsSnapshot s;
    s.frames_processed = frames_processed_;
    s.deadline_ms = deadline_ms_;
    s.deadline_miss = deadline_miss_;
    s.deadline_miss_ratio = frames_processed_ > 0
        ? static_cast<double>(deadline_miss_) / static_cast<double>(frames_processed_)
        : 0.0;
    if (!latencies_.empty()) {
        std::vector<double> sorted = latencies_;
        std::ranges::sort(sorted);
        s.latency_p50_ms = percentile(sorted, 50.0);
        s.latency_p95_ms = percentile(sorted, 95.0);
        s.latency_p99_ms = percentile(sorted, 99.0);
        s.latency_max_ms = sorted.back();
    }
    s.input_received = input_received_;
    s.input_processed = input_processed_;
    s.input_dropped = input_dropped_;
    s.input_queue_hwm = input_queue_hwm_;
    if (tracking_samples_ > 0) {
        s.features_avg = features_sum_ / tracking_samples_;
        s.stereo_points_avg = stereo_sum_ / tracking_samples_;
        s.pnp_inliers_avg = inliers_sum_ / tracking_samples_;
        s.pnp_inlier_ratio_avg = ratio_sum_ / tracking_samples_;
        s.pnp_rmse_avg = rmse_sum_ / tracking_samples_;
    }
    s.pose_accepted = pose_accepted_;
    s.pose_rejected = pose_rejected_;
    s.pose_prediction_only = pose_prediction_only_;
    s.failure_reasons = failure_reasons_;
    s.backend_submitted = backend_stats_.submitted;
    s.backend_executed = backend_stats_.executed;
    s.backend_dropped = backend_stats_.dropped;
    s.backend_pending = backend_stats_.pending;
    s.backend_task_age_max_ms = backend_stats_.age_samples > 0
        ? backend_stats_.task_age_max_ms : -1.0;
    s.backend_task_age_avg_ms = backend_stats_.taskAgeAvgMs();
    s.backend_committed = backend_stats_.committed;
    s.backend_stale = backend_stats_.stale;
    s.backend_invalid = backend_stats_.invalid;
    s.backend_not_found = backend_stats_.not_found;
    s.loop_committed = loop_committed_;
    s.map_keyframes = map_keyframes_;
    s.map_points = map_points_;
    s.map_observations = map_observations_;
    s.map_descriptor_bytes = map_descriptor_bytes_;
    s.map_image_bytes = map_image_bytes_;
    s.map_snapshot_bytes = map_snapshot_bytes_;
    s.map_estimated_total_bytes = map_estimated_total_bytes_;
    s.lost_count = lost_count_;
    s.lost_duration_s = lost_duration_s_;
    if (!reloc_latencies_ms_.empty()) {
        std::vector<double> sorted = reloc_latencies_ms_;
        std::ranges::sort(sorted);
        s.relocalization_latency_p95_ms = percentile(sorted, 95.0);
    }
    return s;
}

std::string MetricsCollector::toJson() const {
    const MetricsSnapshot s = snapshot();
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "{\n";
    os << "  \"frames_processed\": " << s.frames_processed << ",\n";
    os << "  \"deadline_ms\": " << s.deadline_ms << ",\n";
    os << "  \"deadline_miss\": " << s.deadline_miss << ",\n";
    os << "  \"deadline_miss_ratio\": " << s.deadline_miss_ratio << ",\n";
    os << "  \"latency_p50_ms\": " << s.latency_p50_ms << ",\n";
    os << "  \"latency_p95_ms\": " << s.latency_p95_ms << ",\n";
    os << "  \"latency_p99_ms\": " << s.latency_p99_ms << ",\n";
    os << "  \"latency_max_ms\": " << s.latency_max_ms << ",\n";
    os << "  \"input_received\": " << s.input_received << ",\n";
    os << "  \"input_processed\": " << s.input_processed << ",\n";
    os << "  \"input_dropped\": " << s.input_dropped << ",\n";
    os << "  \"input_queue_hwm\": " << s.input_queue_hwm << ",\n";
    os << "  \"features_avg\": " << s.features_avg << ",\n";
    os << "  \"stereo_points_avg\": " << s.stereo_points_avg << ",\n";
    os << "  \"pnp_inliers_avg\": " << s.pnp_inliers_avg << ",\n";
    os << "  \"pnp_inlier_ratio_avg\": " << s.pnp_inlier_ratio_avg << ",\n";
    os << "  \"pnp_rmse_avg\": " << s.pnp_rmse_avg << ",\n";
    os << "  \"pose_accepted\": " << s.pose_accepted << ",\n";
    os << "  \"pose_rejected\": " << s.pose_rejected << ",\n";
    os << "  \"pose_prediction_only\": " << s.pose_prediction_only << ",\n";
    os << "  \"failure_reasons\": {\n";
    for (size_t i = 0; i < s.failure_reasons.size(); i++) {
        os << "    \"" << kFailureReasonNames[i] << "\": "
           << s.failure_reasons[i];
        if (i + 1 < s.failure_reasons.size()) os << ",";
        os << "\n";
    }
    os << "  },\n";
    os << "  \"backend_submitted\": " << s.backend_submitted << ",\n";
    os << "  \"backend_executed\": " << s.backend_executed << ",\n";
    os << "  \"backend_dropped\": " << s.backend_dropped << ",\n";
    os << "  \"backend_pending\": " << s.backend_pending << ",\n";
    os << "  \"backend_task_age_max_ms\": " << s.backend_task_age_max_ms << ",\n";
    os << "  \"backend_task_age_avg_ms\": " << s.backend_task_age_avg_ms << ",\n";
    os << "  \"backend_committed\": " << s.backend_committed << ",\n";
    os << "  \"backend_stale\": " << s.backend_stale << ",\n";
    os << "  \"backend_invalid\": " << s.backend_invalid << ",\n";
    os << "  \"backend_not_found\": " << s.backend_not_found << ",\n";
    os << "  \"loop_committed\": " << s.loop_committed << ",\n";
    os << "  \"map_keyframes\": " << s.map_keyframes << ",\n";
    os << "  \"map_points\": " << s.map_points << ",\n";
    os << "  \"map_observations\": " << s.map_observations << ",\n";
    os << "  \"map_descriptor_bytes\": " << s.map_descriptor_bytes << ",\n";
    os << "  \"map_image_bytes\": " << s.map_image_bytes << ",\n";
    os << "  \"map_snapshot_bytes\": " << s.map_snapshot_bytes << ",\n";
    os << "  \"map_estimated_total_bytes\": " << s.map_estimated_total_bytes << ",\n";
    os << "  \"lost_count\": " << s.lost_count << ",\n";
    os << "  \"lost_duration_s\": " << s.lost_duration_s << ",\n";
    os << "  \"relocalization_latency_p95_ms\": "
       << s.relocalization_latency_p95_ms << "\n";
    os << "}\n";
    return os.str();
}

std::string MetricsCollector::toCsv() const {
    const MetricsSnapshot s = snapshot();
    std::ostringstream os;
    os << "frames_processed,deadline_ms,deadline_miss,deadline_miss_ratio,"
          "latency_p50_ms,latency_p95_ms,latency_p99_ms,latency_max_ms,"
          "input_received,input_processed,input_dropped,input_queue_hwm,"
          "features_avg,stereo_points_avg,pnp_inliers_avg,pnp_inlier_ratio_avg,"
          "pnp_rmse_avg,pose_accepted,pose_rejected,pose_prediction_only,"
          "backend_submitted,backend_executed,backend_dropped,backend_pending,"
          "backend_task_age_max_ms,backend_task_age_avg_ms,"
          "backend_committed,backend_stale,backend_invalid,backend_not_found,"
          "loop_committed,"
          "map_keyframes,map_points,map_observations,map_descriptor_bytes,"
          "map_image_bytes,map_snapshot_bytes,map_estimated_total_bytes,"
          "lost_count,lost_duration_s,relocalization_latency_p95_ms\n";
    os << std::fixed << std::setprecision(6);
    os << s.frames_processed << "," << s.deadline_ms << "," << s.deadline_miss
       << "," << s.deadline_miss_ratio << "," << s.latency_p50_ms << ","
       << s.latency_p95_ms << "," << s.latency_p99_ms << "," << s.latency_max_ms
       << "," << s.input_received << "," << s.input_processed << ","
       << s.input_dropped << "," << s.input_queue_hwm << "," << s.features_avg
       << "," << s.stereo_points_avg << "," << s.pnp_inliers_avg << ","
       << s.pnp_inlier_ratio_avg << "," << s.pnp_rmse_avg << ","
       << s.pose_accepted << "," << s.pose_rejected << "," << s.pose_prediction_only
       << "," << s.backend_submitted << "," << s.backend_executed << ","
       << s.backend_dropped << "," << s.backend_pending << ","
        << s.backend_task_age_max_ms << "," << s.backend_task_age_avg_ms << ","
        << s.backend_committed << "," << s.backend_stale << ","
        << s.backend_invalid << "," << s.backend_not_found << ","
        << s.loop_committed << "," << s.map_keyframes << "," << s.map_points << ","
       << s.map_observations << "," << s.map_descriptor_bytes << ","
       << s.map_image_bytes << "," << s.map_snapshot_bytes << ","
       << s.map_estimated_total_bytes << "," << s.lost_count << ","
       << s.lost_duration_s << "," << s.relocalization_latency_p95_ms << "\n";
    return os.str();
}

void MetricsCollector::writeJson(const std::string& path) const {
    std::ofstream ofs(path);
    if (ofs.is_open()) ofs << toJson();
}

void MetricsCollector::writeCsv(const std::string& path) const {
    std::ofstream ofs(path);
    if (ofs.is_open()) ofs << toCsv();
}

} // namespace vslam
