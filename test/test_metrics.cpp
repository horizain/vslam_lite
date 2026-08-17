/**
 * test_metrics.cpp - M2.3 结构化指标（§6.4）单元测试
 *
 * 覆盖：
 *   1. MetricsCollector：latency 分位/deadline miss、input、tracking 均值、
 *      pose/reason 计数、LOST 段与 relocalization latency、backend 透传
 *   2. MetricsSnapshot JSON/CSV 序列化与落盘
 *   3. BackendScheduler 调度统计（§6.4 backend 指标数据源）
 *   4. 并发喂数（多线程 record 后快照一致）
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_metrics
 * 运行: ./build/test_metrics（独立 CTest）
 */

#include "vslam/backend_scheduler.h"
#include "vslam/metrics.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::BackendScheduler;
using vslam::BackendTask;
using vslam::FailureReason;
using vslam::MetricsCollector;
using vslam::MetricsSnapshot;
using vslam::PoseEstimate;
using vslam::TrackingState;

namespace {

PoseEstimate makePose(TrackingState state, FailureReason reason, bool valid,
                      bool prediction_only, double timestamp) {
    PoseEstimate p;
    p.timestamp = timestamp;
    p.state = state;
    p.reason = reason;
    p.pose_valid = valid;
    p.prediction_only = prediction_only;
    return p;
}

void test_empty_snapshot_defaults() {
    TEST("collector: 空快照默认值") {
        MetricsCollector collector(80);
        const MetricsSnapshot s = collector.snapshot();
        assert(s.frames_processed == 0);
        assert(s.pose_accepted == 0 && s.pose_rejected == 0);
        assert(s.lost_count == 0);
        assert(s.latency_p99_ms == 0.0 && s.latency_max_ms == 0.0);
    } TEST_PASS();
}

void test_latency_percentiles_and_deadline() {
    TEST("collector: latency 分位与 deadline miss") {
        MetricsCollector collector(80);
        // 样本 1..100 ms：p50=50.5, p95=95.05, p99=99.01, max=100；miss=20 个（81..100）
        for (int i = 1; i <= 100; i++) collector.recordFrameLatency((double)i);
        const MetricsSnapshot s = collector.snapshot();
        assert(s.frames_processed == 100);
        assert(std::abs(s.latency_p50_ms - 50.5) < 1e-9);
        assert(std::abs(s.latency_p95_ms - 95.05) < 1e-9);
        assert(std::abs(s.latency_p99_ms - 99.01) < 1e-9);
        assert(s.latency_max_ms == 100.0);
        assert(s.deadline_ms == 80);
        assert(s.deadline_miss == 20);
        assert(std::abs(s.deadline_miss_ratio - 0.20) < 1e-9);
    } TEST_PASS();
}

void test_input_counters() {
    TEST("collector: input 计数（received/dropped/hwm）") {
        MetricsCollector collector;
        collector.recordInput(50, 7, 3);
        const MetricsSnapshot s = collector.snapshot();
        assert(s.input_received == 50);
        assert(s.input_dropped == 7);
        assert(s.input_queue_hwm == 3);
    } TEST_PASS();
}

void test_tracking_averages() {
    TEST("collector: tracking 均值（features/inliers/ratio/RMSE）") {
        MetricsCollector collector;
        collector.recordTracking(200, 40, 30, 0.6, 2.0);
        collector.recordTracking(300, 60, 45, 0.5, 1.0);
        const MetricsSnapshot s = collector.snapshot();
        assert(s.features_avg == 250.0);
        assert(s.stereo_points_avg == 50.0);
        assert(s.pnp_inliers_avg == 37.5);
        assert(std::abs(s.pnp_inlier_ratio_avg - 0.55) < 1e-9);
        assert(s.pnp_rmse_avg == 1.5);
    } TEST_PASS();
}

void test_pose_and_reason_counters() {
    TEST("collector: pose accepted/rejected/prediction + 每类 FailureReason") {
        MetricsCollector collector;
        collector.recordPose(makePose(TrackingState::Tracking, FailureReason::None,
                                      true, false, 1.0));
        collector.recordPose(makePose(TrackingState::Tracking, FailureReason::None,
                                      true, false, 2.0));
        collector.recordPose(makePose(TrackingState::Degraded, FailureReason::None,
                                      true, true, 3.0));   // prediction_only 计入独立计数
        collector.recordPose(makePose(TrackingState::Lost, FailureReason::GeometricRejection,
                                      false, false, 4.0));
        collector.recordPose(makePose(TrackingState::Lost, FailureReason::MotionDiscontinuity,
                                      false, false, 5.0));
        collector.recordPose(makePose(TrackingState::Stopped, FailureReason::TimestampRollback,
                                      false, false, 6.0));
        const MetricsSnapshot s = collector.snapshot();
        assert(s.pose_accepted == 2);
        assert(s.pose_rejected == 3);
        assert(s.pose_prediction_only == 1);
        assert(s.failure_reasons[static_cast<size_t>(FailureReason::GeometricRejection)] == 1);
        assert(s.failure_reasons[static_cast<size_t>(FailureReason::MotionDiscontinuity)] == 1);
        assert(s.failure_reasons[static_cast<size_t>(FailureReason::TimestampRollback)] == 1);
        assert(s.failure_reasons[static_cast<size_t>(FailureReason::None)] == 0);
    } TEST_PASS();
}

void test_lost_segments_and_relocalization_latency() {
    TEST("collector: LOST 段计数/时长与 relocalization latency") {
        MetricsCollector collector;
        // ts: 0 Initializing → 1 Tracking → 2 Lost → 2.5 Relocalizing → 3.5 Tracking
        //     → 4 Lost → 5 Tracking
        collector.recordStateChange(TrackingState::Initializing, TrackingState::Tracking, 1.0);
        collector.recordStateChange(TrackingState::Tracking, TrackingState::Lost, 2.0);
        collector.recordStateChange(TrackingState::Lost, TrackingState::Relocalizing, 2.5);
        collector.recordStateChange(TrackingState::Relocalizing, TrackingState::Tracking, 3.5);
        collector.recordStateChange(TrackingState::Tracking, TrackingState::Lost, 4.0);
        collector.recordStateChange(TrackingState::Lost, TrackingState::Tracking, 5.0);
        const MetricsSnapshot s = collector.snapshot();
        assert(s.lost_count == 2);
        assert(std::abs(s.lost_duration_s - 1.5) < 1e-9);   // (2.5-2) + (5-4)
        assert(std::abs(s.relocalization_latency_p95_ms - 1000.0) < 1e-9);
    } TEST_PASS();
}

void test_backend_stats_transfer() {
    TEST("collector: backend 统计透传（真实 BackendScheduler 数据源）") {
        BackendScheduler scheduler([](BackendTask&) {});
        scheduler.start();
        BackendTask a;
        a.type = BackendTask::Type::LocalBA;
        BackendTask b;
        b.type = BackendTask::Type::LocalBA;
        scheduler.submit(a);
        scheduler.submit(b);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        scheduler.stop();  // 排空槽中任务

        const auto stats = scheduler.stats();
        assert(stats.submitted >= 2);
        assert(stats.executed >= 1 && "stop 排空后至少执行一个任务");
        assert(stats.pending == 0);
        assert(stats.age_samples >= 1);
        assert(stats.task_age_max_ms >= 0.0);

        MetricsCollector collector;
        collector.recordBackend(stats);
        const MetricsSnapshot s = collector.snapshot();
        assert(s.backend_submitted == stats.submitted);
        assert(s.backend_executed == stats.executed);
        assert(s.backend_pending == 0);
        assert(s.backend_task_age_max_ms == stats.task_age_max_ms);
    } TEST_PASS();
}

void test_backend_commit_outcomes() {
    TEST("collector: backend committed/stale/invalid/not_found 计数") {
        BackendScheduler scheduler([](BackendTask&) {});
        scheduler.recordTaskOutcome(vslam::TaskOutcome::Committed);
        scheduler.recordTaskOutcome(vslam::TaskOutcome::Committed);
        scheduler.recordTaskOutcome(vslam::TaskOutcome::Stale);
        scheduler.recordTaskOutcome(vslam::TaskOutcome::Invalid);
        scheduler.recordTaskOutcome(vslam::TaskOutcome::NotFound);
        const auto stats = scheduler.stats();
        assert(stats.committed == 2);
        assert(stats.stale == 1);
        assert(stats.invalid == 1);
        assert(stats.not_found == 1);

        MetricsCollector collector;
        collector.recordBackend(stats);
        const MetricsSnapshot s = collector.snapshot();
        assert(s.backend_committed == 2);
        assert(s.backend_stale == 1);
        assert(s.backend_invalid == 1);
        assert(s.backend_not_found == 1);
        assert(collector.toJson().find("backend_committed") != std::string::npos);
        assert(collector.toCsv().find("backend_committed") != std::string::npos);
    } TEST_PASS();
}

void test_map_snapshot_bytes_reported() {
    TEST("collector: map_snapshot_bytes 上报后非 -1") {
        MetricsCollector collector;
        const MetricsSnapshot empty = collector.snapshot();
        assert(empty.map_snapshot_bytes == -1 && "未上报保持 -1 哨兵");
        collector.recordMap(10, 100, 200, 1024, 2048, 4096, 8192);
        const MetricsSnapshot s = collector.snapshot();
        assert(s.map_keyframes == 10);
        assert(s.map_snapshot_bytes == 4096);
        assert(s.map_estimated_total_bytes == 8192);
    } TEST_PASS();
}

void test_json_output() {
    TEST("collector: JSON 输出包含全部 §6.4 键") {
        MetricsCollector collector(80);
        collector.recordFrameLatency(10.0);
        collector.recordPose(makePose(TrackingState::Tracking, FailureReason::None,
                                      true, false, 1.0));
        collector.recordPose(makePose(TrackingState::Lost, FailureReason::InternalError,
                                      false, false, 2.0));
        const std::string json = collector.toJson();
        for (const auto& key : {"frames_processed", "latency_p50_ms", "latency_p95_ms",
                                "latency_p99_ms", "latency_max_ms", "deadline_ms",
                                "deadline_miss", "deadline_miss_ratio",
                                "input_received", "input_processed", "input_dropped",
                                "input_queue_hwm", "features_avg", "stereo_points_avg",
                                "pnp_inliers_avg", "pnp_inlier_ratio_avg", "pnp_rmse_avg",
                                "pose_accepted", "pose_rejected", "pose_prediction_only",
                                "failure_reasons", "backend_submitted", "backend_executed",
                                 "backend_dropped", "backend_pending",
                                 "backend_task_age_max_ms", "backend_task_age_avg_ms",
                                 "backend_expired", "backend_service_max_ms",
                                 "backend_service_avg_ms",
                                "backend_committed", "backend_stale", "backend_invalid",
                                "backend_not_found",
                                "loop_committed", "map_keyframes", "map_points",
                                "map_observations", "map_descriptor_bytes",
                                 "map_image_bytes", "map_snapshot_bytes",
                                 "map_estimated_total_bytes", "process_rss_bytes",
                                 "process_thread_count", "allowed_cpu_count", "lost_count",
                                "lost_duration_s", "relocalization_latency_p95_ms"}) {
            assert(json.find(key) != std::string::npos);
        }
        assert(json.find("InternalError") != std::string::npos);
        assert(json.find("None") != std::string::npos);
        assert(json.find("GeometricRejection") != std::string::npos);  // 0 计数也输出
    } TEST_PASS();
}

void test_csv_output() {
    TEST("collector: CSV 输出表头与一行值") {
        MetricsCollector collector;
        collector.recordFrameLatency(5.0);
        const std::string csv = collector.toCsv();
        assert(csv.find("latency_p99_ms") != std::string::npos);
        const auto newline = csv.find('\n');
        assert(newline != std::string::npos);
        const auto second_newline = csv.find('\n', newline + 1);
        assert(second_newline != std::string::npos && "CSV 需要表头+一行值");
        assert(csv.find('\n', second_newline + 1) == std::string::npos &&
               "CSV 只有表头+一行");
    } TEST_PASS();
}

void test_concurrent_recording() {
    TEST("collector: 4 线程并发喂数后计数精确") {
        MetricsCollector collector;
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; t++) {
            threads.emplace_back([&collector, t] {
                for (int i = 0; i < 500; i++) {
                    collector.recordFrameLatency(
                        1.0 + static_cast<double>((t * 500 + i) % 10));
                    collector.recordPose(makePose(
                        TrackingState::Tracking, FailureReason::None, true, false, 1.0));
                }
            });
        }
        for (auto& th : threads) th.join();
        const MetricsSnapshot s = collector.snapshot();
        assert(s.frames_processed == 2000);
        assert(s.pose_accepted == 2000);
        assert(s.latency_max_ms >= 1.0 && s.latency_max_ms <= 10.0);
    } TEST_PASS();
}

void test_write_json_and_csv_files() {
    TEST("collector: writeJson/writeCsv 落盘") {
        MetricsCollector collector;
        collector.recordFrameLatency(3.0);

        // 使用系统临时目录和 mkstemp 取得唯一基名，测试不能依赖某个开发机
        // 的固定目录（旧测试写死 /tmp/opencode，干净环境会假绿/直接失败）。
        const auto tmp_dir = std::filesystem::temp_directory_path();
        std::string pattern = (tmp_dir / "vslam_metrics_XXXXXX").string();
        std::vector<char> native(pattern.begin(), pattern.end());
        native.push_back('\0');
        const int fd = mkstemp(native.data());
        assert(fd >= 0);
        close(fd);
        const std::filesystem::path base(native.data());
        std::filesystem::remove(base);
        const auto json_path = std::filesystem::path(base.string() + ".json");
        const auto csv_path = std::filesystem::path(base.string() + ".csv");

        collector.writeJson(json_path.string());
        collector.writeCsv(csv_path.string());
        std::ifstream ifs(json_path);
        assert(ifs.is_open());
        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        assert(content.find("\"latency_p50_ms\"") != std::string::npos);
        ifs.close();
        std::ifstream ifs2(csv_path);
        assert(ifs2.is_open());
        ifs2.close();

        // 文件不能作为父目录；写入失败时不得伪造输出文件。
        const auto invalid_json = base / "child.json";
        const auto invalid_csv = base / "child.csv";
        { std::ofstream blocker(base); assert(blocker.is_open()); }
        collector.writeJson(invalid_json.string());
        collector.writeCsv(invalid_csv.string());
        assert(!std::filesystem::exists(invalid_json));
        assert(!std::filesystem::exists(invalid_csv));

        std::filesystem::remove(base);
        std::filesystem::remove(json_path);
        std::filesystem::remove(csv_path);
    } TEST_PASS();
}

}  // namespace

int main() {
    cv::setNumThreads(1);
    cv::setRNGSeed(0x5A17);

    std::cout << "test_metrics (M2.3 MetricsCollector)" << std::endl;

    test_empty_snapshot_defaults();
    test_latency_percentiles_and_deadline();
    test_input_counters();
    test_tracking_averages();
    test_pose_and_reason_counters();
    test_lost_segments_and_relocalization_latency();
    test_backend_stats_transfer();
    test_backend_commit_outcomes();
    test_map_snapshot_bytes_reported();
    test_json_output();
    test_csv_output();
    test_concurrent_recording();
    test_write_json_and_csv_files();

    std::cout << "全部通过" << std::endl;
    return 0;
}
