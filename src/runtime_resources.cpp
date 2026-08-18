#include "vslam/runtime_resources.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if __has_include(<oneapi/tbb/global_control.h>)
#include <oneapi/tbb/global_control.h>
#define VSLAM_HAS_TBB_GLOBAL_CONTROL 1
#elif __has_include(<tbb/global_control.h>)
#include <tbb/global_control.h>
#define VSLAM_HAS_TBB_GLOBAL_CONTROL 1
#endif

namespace vslam {
namespace {

#ifdef VSLAM_HAS_TBB_GLOBAL_CONTROL
std::mutex tbb_mutex;
std::unique_ptr<tbb::global_control> tbb_control;
#endif

#ifdef __linux__
std::vector<int> allowedCpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return {};

    std::vector<int> cpus;
    cpus.reserve(CPU_SETSIZE);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
    return cpus;
}

bool restrictCurrentThread(const std::vector<int>& cpus, size_t max_cpu_cores) {
    if (cpus.empty()) return false;
    const size_t count = std::min(max_cpu_cores, cpus.size());
    cpu_set_t set;
    CPU_ZERO(&set);
    for (size_t i = 0; i < count; ++i) CPU_SET(cpus[i], &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

void restrictExistingThreads(const std::vector<int>& cpus, size_t max_cpu_cores) {
    if (cpus.empty()) return;
    const size_t count = std::min(max_cpu_cores, cpus.size());
    std::error_code ec;
    const auto task_dir = std::filesystem::path("/proc/self/task");
    const auto current_tid = static_cast<unsigned long>(syscall(SYS_gettid));
    size_t ordinal = 0;
    for (const auto& entry : std::filesystem::directory_iterator(task_dir, ec)) {
        if (ec) break;
        unsigned long tid = 0;
        try {
            tid = std::stoul(entry.path().filename().string());
        } catch (...) {
            continue;
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        if (tid == current_tid) {
            for (size_t i = 0; i < count; ++i) CPU_SET(cpus[i], &set);
        } else {
            const size_t cpu_index = count <= 1
                ? 0 : 1 + (ordinal++ % (count - 1));
            CPU_SET(cpus[cpu_index], &set);
        }
        (void)sched_setaffinity(static_cast<pid_t>(tid), sizeof(set), &set);
    }
}
#else
std::vector<int> allowedCpus() {
    const unsigned count = std::max(1u, std::thread::hardware_concurrency());
    std::vector<int> cpus(count);
    for (unsigned i = 0; i < count; ++i) cpus[i] = static_cast<int>(i);
    return cpus;
}

bool restrictCurrentThread(const std::vector<int>&, size_t) { return false; }
void restrictExistingThreads(const std::vector<int>&, size_t) {}
#endif

}  // namespace

bool RuntimeResources::configure(const RuntimeResourceConfig& config,
                                 int requested_opencv_threads,
                                 bool backend_enabled) {
    const auto cpus = allowedCpus();
    if (cpus.empty()) return false;

    const size_t max_cores = std::clamp<size_t>(
        config.max_cpu_cores, 1, cpus.size());
    bool affinity_ok = true;
    if (config.enforce_cpu_affinity)
        affinity_ok = restrictCurrentThread(cpus, max_cores);

    if (requested_opencv_threads > 0) {
        size_t threads = std::min<size_t>(
            static_cast<size_t>(requested_opencv_threads), max_cores);
        if (backend_enabled && config.backend_reserved_cores > 0 &&
            max_cores > config.backend_reserved_cores) {
            threads = std::min(
                threads, max_cores - config.backend_reserved_cores);
        }
        threads = std::max<size_t>(1, threads);
#ifdef __linux__
        const std::string thread_text = std::to_string(threads);
        (void)setenv("OPENCV_FOR_THREADS_NUM", thread_text.c_str(), 1);
#endif
#ifdef VSLAM_HAS_TBB_GLOBAL_CONTROL
        {
            std::lock_guard<std::mutex> lock(tbb_mutex);
            tbb_control = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism, threads);
        }
#endif
        cv::setNumThreads(static_cast<int>(threads));
        if (config.enforce_cpu_affinity) {
            // 先触发一次 OpenCV parallel backend，让它创建固定 worker 集合；
            // 随后把已存在的 worker 收窄并轮转固定到允许 CPU，避免 TBB
            // 线程继续带着启动时的全机 affinity 跨核迁移。
            cv::parallel_for_(cv::Range(0, 1), [](const cv::Range&) {});
            restrictExistingThreads(cpus, max_cores);
        }
    }
    return affinity_ok;
}

bool RuntimeResources::pinCurrentThread(size_t ordinal) {
    const auto cpus = allowedCpus();
    if (cpus.empty()) return false;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[std::min(ordinal, cpus.size() - 1)], &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)ordinal;
    return false;
#endif
}

size_t RuntimeResources::processRssBytes() {
#ifdef __linux__
    std::ifstream input("/proc/self/status");
    std::string line;
    while (std::getline(input, line)) {
        if (!line.starts_with("VmRSS:")) continue;
        const auto first = line.find_first_of("0123456789");
        if (first == std::string::npos) return 0;
        try {
            return std::stoull(line.substr(first)) * 1024ULL;
        } catch (...) {
            return 0;
        }
    }
#endif
    return 0;
}

size_t RuntimeResources::processThreadCount() {
#ifdef __linux__
    std::error_code ec;
    const auto task_dir = std::filesystem::path("/proc/self/task");
    size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(task_dir, ec)) {
        (void)entry;
        if (ec) break;
        ++count;
    }
    return count;
#else
    return 1;
#endif
}

size_t RuntimeResources::allowedCpuCount() { return allowedCpus().size(); }

RuntimeResourceSnapshot RuntimeResources::snapshot() {
    return {processRssBytes(), processThreadCount(), allowedCpuCount()};
}

bool RuntimeResources::withinRssBudget(size_t max_rss_mb) {
    const size_t rss = processRssBytes();
    return rss == 0 || rss <= max_rss_mb * 1024ULL * 1024ULL;
}

}  // namespace vslam
