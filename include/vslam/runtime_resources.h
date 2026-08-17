#pragma once

#include <cstddef>

namespace vslam {

/// 进程级运行资源契约。配置值是硬上限，不是“建议使用”的提示。
struct RuntimeResourceConfig {
    size_t max_cpu_cores = 6;
    size_t max_rss_mb = 12288;
    size_t backend_reserved_cores = 1;
    bool enforce_cpu_affinity = false;
    bool pin_backend_worker = true;
};

struct RuntimeResourceSnapshot {
    size_t rss_bytes = 0;
    size_t thread_count = 0;
    size_t allowed_cpu_count = 0;
};

/// 进程资源控制与诊断。Linux 下收窄当前进程的 CPU affinity；其它平台只
/// 应用 OpenCV 线程上限并返回可观测的降级结果。
class RuntimeResources {
public:
    /// 配置 OpenCV 线程数，并在 enforce_cpu_affinity=true 时收窄当前线程的
    /// CPU 集合。backend_enabled=true 时预留 backend_reserved_cores 个核。
    static bool configure(const RuntimeResourceConfig& config,
                          int requested_opencv_threads,
                          bool backend_enabled);

    /// 把当前线程固定到当前允许 CPU 集合中的 ordinal 位置。
    [[nodiscard]] static bool pinCurrentThread(size_t ordinal);

    [[nodiscard]] static size_t processRssBytes();
    [[nodiscard]] static size_t processThreadCount();
    [[nodiscard]] static size_t allowedCpuCount();
    [[nodiscard]] static RuntimeResourceSnapshot snapshot();
    [[nodiscard]] static bool withinRssBudget(size_t max_rss_mb);
};

}  // namespace vslam
