#include "vslam/runtime_resources.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace vslam {
namespace {

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
#else
std::vector<int> allowedCpus() {
    const unsigned count = std::max(1u, std::thread::hardware_concurrency());
    std::vector<int> cpus(count);
    for (unsigned i = 0; i < count; ++i) cpus[i] = static_cast<int>(i);
    return cpus;
}

bool restrictCurrentThread(const std::vector<int>&, size_t) { return false; }
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
        cv::setNumThreads(static_cast<int>(std::max<size_t>(1, threads)));
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
    std::string key;
    size_t value = 0;
    std::string unit;
    while (input >> key >> value >> unit) {
        if (key == "VmRSS:") return value * 1024;
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
