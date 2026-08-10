#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/observation.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace vslam {

/// 异步后台任务（M1.3：调度单元，从 vo.h 迁移；§5.4）。
struct BackendTask {
    enum class Type { LocalBA, LoopClosure };
    Type type = Type::LocalBA;
    Map::Ptr map;                       // 入队时所属 Map；revision 不能替代实例身份
    unsigned long submap_id = 0;       // 入队时所属 Submap
    std::vector<Frame::Ptr> window;   // LocalBA：窗口关键帧（快照在后台锁内构造）
    KeyframeId anchor_kf_id = 0;      // LocalBA：提交时的当前/锚定 KF 身份
    Frame::Ptr curr_kf;               // LoopClosure：当前关键帧
};

/// §6.4 后端调度指标（M2.3）：排队/执行/丢弃计数 + 任务等待年龄。
/// 覆盖式单任务槽的"覆盖"对被覆盖任务计为 dropped。
struct BackendSchedulerStats {
    long long submitted = 0;      ///< 收到任务总数
    long long executed = 0;       ///< 实际执行完成数
    long long dropped = 0;        ///< 槽满/优先级被丢弃（含被 LoopClosure 覆盖）
    long long pending = 0;        ///< 当前等待槽占用（容量 1，0/1）
    double task_age_max_ms = 0.0; ///< 任务入队→开始执行的最大等待时长
    double task_age_total_ms = 0.0;
    long long age_samples = 0;

    [[nodiscard]] double taskAgeAvgMs() const {
        return age_samples > 0 ? task_age_total_ms / static_cast<double>(age_samples) : 0.0;
    }
};

/// 后台任务调度器（M1.3，§5.4）：单后台线程 + 覆盖式单任务槽。
///
/// 排队语义（§5.4，A/B 结论：LoopClosure 优先于 Local BA）：
///   - 同类 Local BA 新任务覆盖旧任务（保新鲜，等待槽恒为最新）。
///   - LoopClosure 优先于 Local BA：LoopClosure 覆盖任何等待任务；等待中的
///     LoopClosure 不被后续 Local BA 覆盖（槽满则丢弃新 Local BA）。
///   - 任务槽容量固定 1（等待中的任务内存由 M2 预算控制）。
///   - 正在执行的任务不强制取消，其结果由调用方 stale gate 丢弃。
///   - stop() 设置标志、notify、join；不得 detach。
///
/// 线程模型：一个 worker 线程；空闲时阻塞在 condition_variable，不自旋。
class BackendScheduler {
public:
    using TaskHandler = std::function<void(BackendTask&)>;

    explicit BackendScheduler(TaskHandler handler);
    ~BackendScheduler();  // 未 stop 时自动 stop + join

    BackendScheduler(const BackendScheduler&) = delete;
    BackendScheduler& operator=(const BackendScheduler&) = delete;

    /// 启动 worker 线程（幂等；重复调用无副作用）。
    void start();

    /// 提交任务到覆盖式单任务槽。永不阻塞（槽满或按优先级丢弃）。
    void submit(BackendTask task);

    /// 停止：设置标志 → notify → join 已启动的线程。可安全重复调用；
    /// 停止前仍在槽中的任务会被排空执行（与原 backendLoop 一致）。
    void stop();

    /// worker 线程是否在运行。
    bool running() const { return running_.load(); }

    /// 等待槽是否被占用（容量 1，故为 0/1；诊断/指标用）。
    bool hasPending() const;

    /// §6.4（M2.3）：调度统计快照（线程安全；stale/invalid 属提交器计数，
    /// 见 BackendCommitter——M2.3 未接入，见 DEVELOPMENT_LOG §3.37）。
    [[nodiscard]] BackendSchedulerStats stats() const;

private:
    void loop();

    TaskHandler handler_;
    mutable std::mutex mutex_;   // 保护槽/等待状态；const 访问器（hasPending）也加锁
    std::condition_variable cv_;
    std::optional<BackendTask> slot_;   // 覆盖式单任务槽（容量 1）
    std::chrono::steady_clock::time_point slot_enqueued_at_;  // 当前槽任务入队时刻
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};

    // ---- §6.4 调度统计（M2.3）----
    std::atomic<long long> submitted_{0};
    std::atomic<long long> executed_{0};
    std::atomic<long long> dropped_{0};
    std::atomic<double> task_age_max_ms_{0.0};
    std::atomic<double> task_age_total_ms_{0.0};
    std::atomic<long long> age_samples_{0};
};

} // namespace vslam
