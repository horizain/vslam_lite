#pragma once

// ============================================================
// perf_monitor.h - VSLAM 轻量性能监测器（header-only）
//
// 用途：排查 vslam 各阶段的 CPU 耗时与卡顿尖峰（画面卡滞、FPS 掉、回环
//       校正慢等）。对代码侵入极小：在需要计时的代码段前放一个
//       PERF_SCOPE("name") 即可，作用域结束自动累计。
//
// 用法（配合 CMake 选项 VSLAM_ENABLE_PERF，默认 ON）：
//   void track() {
//       PERF_SCOPE("vo.track");        // 进入即计时，出作用域自动记录
//       ... 业务代码 ...
//   }
//
// 输出：
//   - 程序退出时调用 vslam::perf_dump("perf.csv")（run_vo/run_slam 已内置）：
//       ① 控制台打印汇总表：调用次数 / 总耗时 / 平均 / 最小 / 最大 / 占比
//       ② perf.csv        每阶段聚合统计（用于找热点阶段）
//       ③ perf.csv.history.csv  每阶段逐次耗时时间序列（用于找单次卡顿尖峰）
//
// 线程安全：内部用互斥锁保护，可在多线程流水线中使用（主线程 + 后端线程）。
// 性能：关闭 VSLAM_ENABLE_PERF 时 PERF_SCOPE 展开为空，零开销。
// ============================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vslam {

#ifdef VSLAM_ENABLE_PERF

// 线程安全的阶段计时器：按名字聚合统计 + 保留最近 N 次样本
class PerfMonitor {
public:
    struct Stats {
        size_t count = 0;
        double total_ms = 0.0;
        double min_ms = std::numeric_limits<double>::infinity();
        double max_ms = 0.0;
        double last_ms = 0.0;
    };

    static PerfMonitor& instance() {
        static PerfMonitor pm;
        return pm;
    }

    void add(const std::string& name, double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& s = stats_[name];
        s.count++;
        s.total_ms += ms;
        s.min_ms = std::min(s.min_ms, ms);
        s.max_ms = std::max(s.max_ms, ms);
        s.last_ms = ms;
        if ((int)history_[name].size() < kMaxHistory)
            history_[name].push_back(ms);
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
        history_.clear();
    }

    // 控制台报告：按总耗时降序
    void report(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stats_.empty()) {
            os << "[perf] 无采样数据（VSLAM_ENABLE_PERF 未开启或无 PERF_SCOPE 执行）\n";
            return;
        }
        std::vector<std::pair<std::string, Stats>> items(stats_.begin(), stats_.end());
        std::ranges::sort(items, {}, [](const auto& p) { return -p.second.total_ms; });
        double grand = 0.0;
        for (const auto& [n, s] : items) grand += s.total_ms;

        os << "===== VSLAM Perf Report =====" << std::endl;
        os << std::left << std::setw(36) << "name"
           << std::right << std::setw(10) << "count"
           << std::setw(12) << "total_ms"
           << std::setw(10) << "avg_ms"
           << std::setw(10) << "min_ms"
           << std::setw(10) << "max_ms"
           << std::setw(10) << "last_ms"
           << std::setw(8) << "pct%" << std::endl;
        os << std::string(96, '-') << std::endl;
        for (const auto& [n, s] : items) {
            const double pct = grand > 0.0 ? 100.0 * s.total_ms / grand : 0.0;
            os << std::left << std::setw(36) << n.substr(0, 35)
               << std::right << std::setw(10) << s.count
               << std::setw(12) << std::fixed << std::setprecision(2) << s.total_ms
               << std::setw(10) << std::setprecision(4) << (s.total_ms / s.count)
               << std::setw(10) << (std::isfinite(s.min_ms) ? s.min_ms : 0.0)
               << std::setw(10) << s.max_ms
               << std::setw(10) << s.last_ms
               << std::setw(8) << std::setprecision(2) << pct << std::endl;
        }
        os << std::setprecision(6) << std::defaultfloat;
        os << "总采样耗时: " << grand << " ms\n";
    }

    // 聚合统计 CSV: name,count,total_ms,avg_ms,min_ms,max_ms,last_ms,pct
    bool exportStatsCsv(const std::string& path) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ofstream ofs(path);
        if (!ofs.is_open()) return false;
        ofs << "name,count,total_ms,avg_ms,min_ms,max_ms,last_ms,pct\n";
        double grand = 0.0;
        for (const auto& [n, s] : stats_) grand += s.total_ms;
        std::vector<std::pair<std::string, Stats>> items(stats_.begin(), stats_.end());
        std::ranges::sort(items, {}, [](const auto& p) { return -p.second.total_ms; });
        for (const auto& [n, s] : items) {
            ofs << n << ',' << s.count << ',' << s.total_ms << ','
                << s.total_ms / s.count << ','
                << (std::isfinite(s.min_ms) ? s.min_ms : 0.0) << ','
                << s.max_ms << ',' << s.last_ms << ','
                << (grand > 0.0 ? 100.0 * s.total_ms / grand : 0.0) << '\n';
        }
        return true;
    }

    // 时间序列 CSV: name,index,ms —— 定位单次卡顿尖峰（图/脚本分析用）
    bool exportHistoryCsv(const std::string& path) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ofstream ofs(path);
        if (!ofs.is_open()) return false;
        ofs << "name,index,ms\n";
        for (const auto& [name, series] : history_) {
            for (size_t i = 0; i < series.size(); i++)
                ofs << name << ',' << i << ',' << series[i] << '\n';
        }
        return true;
    }

    // RAII 计时作用域
    class Scope {
    public:
        explicit Scope(std::string name) : name_(std::move(name)) {
            start_ = std::chrono::steady_clock::now();
        }
        ~Scope() {
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_).count();
            PerfMonitor::instance().add(name_, ms);
        }
    private:
        std::string name_;
        std::chrono::steady_clock::time_point start_;
    };

private:
    static constexpr int kMaxHistory = 200000;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Stats> stats_;
    std::unordered_map<std::string, std::vector<double>> history_;
};

#define PERF_MONITOR_CONCAT_IMPL(a, b) a##b
#define PERF_MONITOR_CONCAT(a, b) PERF_MONITOR_CONCAT_IMPL(a, b)
#define PERF_SCOPE(name) \
    ::vslam::PerfMonitor::Scope PERF_MONITOR_CONCAT(perf_scope_, __COUNTER__)(name)

// 汇总 dump：控制台报告 + 导出 stats.csv 与历史序列 csv
inline void perf_dump(const std::string& csv_path = "perf.csv") {
    PerfMonitor::instance().report();
    const bool ok = PerfMonitor::instance().exportStatsCsv(csv_path);
    const bool hok = PerfMonitor::instance().exportHistoryCsv(csv_path + ".history.csv");
    if (ok)
        std::cout << "[perf] stats -> " << csv_path << std::endl;
    if (hok)
        std::cout << "[perf] history -> " << csv_path << ".history.csv" << std::endl;
}

#else  // VSLAM_ENABLE_PERF 未定义：全部展开为空实现，零开销

#define PERF_SCOPE(name) /* no-op */

inline void perf_dump(const std::string& /*csv_path*/ = "perf.csv") {}

#endif

} // namespace vslam
