#include "vslam/backend_scheduler.h"

#include <utility>

namespace vslam {

BackendScheduler::BackendScheduler(TaskHandler handler)
    : handler_(std::move(handler)) {}

BackendScheduler::~BackendScheduler() {
    stop();
}

void BackendScheduler::start() {
    if (running_.exchange(true)) return;
    stop_.store(false);
    thread_ = std::thread(&BackendScheduler::loop, this);
}

void BackendScheduler::submit(BackendTask task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (task.type == BackendTask::Type::LoopClosure) {
            // LoopClosure 优先：覆盖任何等待任务（含 Local BA）
            slot_ = std::move(task);
        } else {
            // Local BA：只覆盖等待中的旧 Local BA；等待中的 LoopClosure 不让位
            if (!slot_ || slot_->type == BackendTask::Type::LocalBA)
                slot_ = std::move(task);
        }
    }
    cv_.notify_one();
}

void BackendScheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    running_.store(false);
}

bool BackendScheduler::hasPending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slot_.has_value();
}

void BackendScheduler::loop() {
    while (true) {
        BackendTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stop_.load() || slot_.has_value(); });
            if (stop_.load() && !slot_.has_value()) break;
            task = std::move(*slot_);
            slot_.reset();
        }
        if (handler_) handler_(task);
    }
}

} // namespace vslam
