#pragma once

// ============================================================
// metrics_json.h - 单次运行结构化指标（M1.6 / §6.4 方向）
//
// 用途：把 run_slam 的单次运行结果写成 JSON，供 benchmark.py 与提交门消费，
// 替代"正则解析自然语言日志"——后者易碎且无法给出延迟分位/状态时长等指标。
// 与 utils/perf_monitor.h 的关系：perf 关注阶段热点（vo.track 等），
// 这里关注整帧延迟分布、有效性、LOST、后端规模等端到端指标。
// ============================================================

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace vslam {

/// 单次运行的结构化指标
struct RunMetrics {
    long long frames_processed = 0;      // 处理的帧数
    long long valid_poses = 0;           // pose_valid=true 的帧数
    double    valid_ratio = 0.0;         // valid_poses / frames_processed
    double    fps = 0.0;                 // 全程平均 FPS
    double    seconds = 0.0;             // 全程耗时（s）
    long long deadline_ms = 100;         // 单帧延迟门限（默认 10Hz=100ms）
    long long deadline_miss = 0;         // 超过门限的帧数
    double    deadline_miss_ratio = 0.0; // deadline_miss / frames_processed
    double    latency_p50_ms = 0.0;      // 单帧 addFrame 延迟分位（ms）
    double    latency_p95_ms = 0.0;
    double    latency_p99_ms = 0.0;
    double    latency_max_ms = 0.0;
    long long lost_count = 0;            // 进入 LOST 的次数（状态迁移）
    double    lost_duration_s = 0.0;     // LOST 累计时长（s）
    long long submap_reinit = 0;         // 子地图重建次数（submap_id 递增）
    long long loops = 0;                 // 已闭合回环次数
    long long map_points = 0;            // 最终地图点
    long long keyframes = 0;             // 最终关键帧
};

/// 已排序向量的百分位（线性插值）
[[nodiscard]] inline double percentile(std::vector<double> sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (p <= 0.0) return sorted.front();
    if (p >= 100.0) return sorted.back();
    const double idx = (static_cast<double>(sorted.size()) - 1.0) * p / 100.0;
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

/// 写单次运行指标 JSON（benchmark.py 消费）
inline void writeRunMetricsJson(const std::string& path, const RunMetrics& m) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;
    ofs << std::fixed << std::setprecision(6);
    ofs << "{\n";
    ofs << "  \"frames_processed\": " << m.frames_processed << ",\n";
    ofs << "  \"valid_poses\": " << m.valid_poses << ",\n";
    ofs << "  \"valid_ratio\": " << m.valid_ratio << ",\n";
    ofs << "  \"fps\": " << m.fps << ",\n";
    ofs << "  \"seconds\": " << m.seconds << ",\n";
    ofs << "  \"deadline_ms\": " << m.deadline_ms << ",\n";
    ofs << "  \"deadline_miss\": " << m.deadline_miss << ",\n";
    ofs << "  \"deadline_miss_ratio\": " << m.deadline_miss_ratio << ",\n";
    ofs << "  \"latency_p50_ms\": " << m.latency_p50_ms << ",\n";
    ofs << "  \"latency_p95_ms\": " << m.latency_p95_ms << ",\n";
    ofs << "  \"latency_p99_ms\": " << m.latency_p99_ms << ",\n";
    ofs << "  \"latency_max_ms\": " << m.latency_max_ms << ",\n";
    ofs << "  \"lost_count\": " << m.lost_count << ",\n";
    ofs << "  \"lost_duration_s\": " << m.lost_duration_s << ",\n";
    ofs << "  \"submap_reinit\": " << m.submap_reinit << ",\n";
    ofs << "  \"loops\": " << m.loops << ",\n";
    ofs << "  \"map_points\": " << m.map_points << ",\n";
    ofs << "  \"keyframes\": " << m.keyframes << "\n";
    ofs << "}\n";
}

} // namespace vslam
