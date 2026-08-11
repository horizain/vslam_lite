#include "vslam/backend_scheduler.h"

#include <chrono>
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
    const bool accepted = [&] {
        std::lock_guard<std::mutex> lock(mutex_);
        if (task.type == BackendTask::Type::LoopClosure) {
            // LoopClosure 优先：覆盖任何等待任务（含 Local BA）——被覆盖任务计丢弃
            if (slot_.has_value()) dropped_.fetch_add(1, std::memory_order_relaxed);
            slot_ = std::move(task);
        } else if (task.type == BackendTask::Type::LoopMaintenance) {
            // 索引维护优先于 Local BA；若 LoopClosure 已等待，它本身会先
            // drain 合并后的清理 id，因此无需覆盖高优先级任务。
            if (!slot_ || slot_->type == BackendTask::Type::LocalBA) {
                if (slot_.has_value())
                    dropped_.fetch_add(1, std::memory_order_relaxed);
                slot_ = std::move(task);
            } else {
                return false;
            }
        } else {
            // Local BA：只覆盖等待中的旧 Local BA；等待中的 LoopClosure 不让位
            if (!slot_ || slot_->type == BackendTask::Type::LocalBA) {
                if (slot_.has_value())
                    dropped_.fetch_add(1, std::memory_order_relaxed);
                slot_ = std::move(task);
            } else {
                return false;  // 槽被 LoopClosure 占用，新 Local BA 丢弃
            }
        }
        slot_enqueued_at_ = std::chrono::steady_clock::now();
        return true;
    }();
    submitted_.fetch_add(1, std::memory_order_relaxed);
    if (!accepted) dropped_.fetch_add(1, std::memory_order_relaxed);
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

BackendSchedulerStats BackendScheduler::stats() const {
    BackendSchedulerStats s;
    s.submitted = submitted_.load(std::memory_order_relaxed);
    s.executed = executed_.load(std::memory_order_relaxed);
    s.dropped = dropped_.load(std::memory_order_relaxed);
    s.pending = hasPending() ? 1 : 0;
    s.task_age_max_ms = task_age_max_ms_.load(std::memory_order_relaxed);
    s.task_age_total_ms = task_age_total_ms_.load(std::memory_order_relaxed);
    s.age_samples = age_samples_.load(std::memory_order_relaxed);
    s.committed = committed_.load(std::memory_order_relaxed);
    s.stale = stale_.load(std::memory_order_relaxed);
    s.invalid = invalid_.load(std::memory_order_relaxed);
    s.not_found = not_found_.load(std::memory_order_relaxed);
    return s;
}

void BackendScheduler::recordTaskOutcome(TaskOutcome outcome) {
    switch (outcome) {
        case TaskOutcome::Committed: committed_.fetch_add(1); break;
        case TaskOutcome::Stale:     stale_.fetch_add(1);     break;
        case TaskOutcome::Invalid:   invalid_.fetch_add(1);   break;
        case TaskOutcome::NotFound:  not_found_.fetch_add(1); break;
    }
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
            // §6.4（M2.3）：任务等待年龄（入队 → 开始执行）
            const double age_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - slot_enqueued_at_).count();
            double max_age = task_age_max_ms_.load(std::memory_order_relaxed);
            while (age_ms > max_age &&
                   !task_age_max_ms_.compare_exchange_weak(max_age, age_ms)) {}
            task_age_total_ms_.fetch_add(age_ms, std::memory_order_relaxed);
            age_samples_.fetch_add(1, std::memory_order_relaxed);
            executed_.fetch_add(1, std::memory_order_relaxed);
        }
        if (handler_) handler_(task);
    }
}

} // namespace vslam
