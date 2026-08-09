#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace vslam {

/// 有界输入队列（M2.1，§6.1）：固定容量 ring buffer + 覆盖式丢最旧。
///
/// - push 永不阻塞：满时丢最旧（队首）、保最新帧（§6.1 输入队列纪律，
///   禁止阻塞传感器线程和无限积压）。
/// - 消费者阻塞在 condition_variable，空闲不自旋（§6.1）。
/// - stop() 唤醒所有等待者；停止后仍排空剩余元素，随后 pop 返回 false
///   （与 BackendScheduler 的 stop 语义一致）。
/// - 记录 high water mark 与 dropped 计数（§6.4 结构化指标）。
///
/// 单生产者/单消费者 + 统计访问的锁模型；push 由传感器线程调用，
/// pop 由跟踪 worker 调用。
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity)
        : buf_(std::max<size_t>(1, capacity)) {}

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    /// 非阻塞入队；队列满时丢弃最旧元素（永不阻塞调用方/传感器线程）。
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) return;  // 停止后拒绝新元素
            if (count_ == buf_.size()) {
                head_ = (head_ + 1) % buf_.size();  // 丢最旧
                ++dropped_;
            } else {
                ++count_;
            }
            buf_[(head_ + count_ - 1) % buf_.size()] = std::move(item);
            high_water_ = std::max(high_water_, count_);
        }
        cv_.notify_one();
    }

    /// 阻塞出队：有元素立即返回 true；stop 且队列空返回 false；
    /// stop 后仍有元素则排空（与 BackendScheduler 一致）。
    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_ || count_ > 0; });
        if (count_ == 0) return false;
        out = std::move(buf_[head_]);
        head_ = (head_ + 1) % buf_.size();
        --count_;
        return true;
    }

    /// 非阻塞出队。
    bool tryPop(T& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) return false;
        out = std::move(buf_[head_]);
        head_ = (head_ + 1) % buf_.size();
        --count_;
        return true;
    }

    /// 停止：唤醒所有等待者；之后 pop 排空剩余元素再返回 false。
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    [[nodiscard]] bool stopped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stop_;
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    [[nodiscard]] size_t capacity() const { return buf_.size(); }

    /// §6.4：入队深度峰值（<= capacity）
    [[nodiscard]] size_t highWaterMark() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return high_water_;
    }

    /// §6.4：因满而丢弃的帧数
    [[nodiscard]] size_t droppedCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<T> buf_;    // 环形缓冲
    size_t head_ = 0;       // 队首索引
    size_t count_ = 0;      // 当前元素数
    size_t high_water_ = 0; // 深度峰值
    size_t dropped_ = 0;    // 丢弃计数
    bool stop_ = false;
};

} // namespace vslam
