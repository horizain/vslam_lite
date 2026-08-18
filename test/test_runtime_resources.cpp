#include "vslam/runtime_resources.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <iostream>

using vslam::RuntimeResourceConfig;
using vslam::RuntimeResources;

int main() {
    std::cout << "test_runtime_resources" << std::endl;

    RuntimeResourceConfig config;
    config.max_cpu_cores = 2;
    config.backend_reserved_cores = 1;
    config.enforce_cpu_affinity = true;
    assert(RuntimeResources::configure(config, 8, true));
    const auto snapshot = RuntimeResources::snapshot();
    assert(snapshot.allowed_cpu_count >= 1 && snapshot.allowed_cpu_count <= 2);
    assert(cv::getNumThreads() <= 1);
    assert(RuntimeResources::pinCurrentThread(0));
    assert(RuntimeResources::withinRssBudget(1024 * 1024));
    assert(snapshot.thread_count >= 1);

    std::cout << "全部通过" << std::endl;
    return 0;
}
