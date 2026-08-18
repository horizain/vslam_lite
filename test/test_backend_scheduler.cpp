/**
 * test_backend_scheduler.cpp - BackendScheduler 单元测试
 *
 * 覆盖 M1.3（PRODUCTION_LOCALIZATION_PLAN §5.4）：
 *   1. 单后台线程 + 覆盖式单任务槽（容量 1）
 *   2. 同类 Local BA 新任务覆盖旧任务
 *   3. LoopClosure 优先于 Local BA（覆盖任何等待任务；不被后续 Local BA 覆盖）
 *   4. LoopMaintenance 优先于 Local BA、低于 LoopClosure
 *   5. stop() 设置标志、排空槽内任务、join；不得 detach；可重复调用
 *   6. start() 幂等、stop 后 pending 状态正确
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_backend_scheduler
 * 运行: ./build/test_backend_scheduler（独立 CTest）
 */

#include "vslam/backend_scheduler.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// 简单的测试辅助宏（与 test_vo.cpp 保持一致）
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::BackendScheduler;
using vslam::BackendTask;

namespace {

/// 记录执行过的任务（类型 + anchor_kf_id），带等待辅助。
struct Recorder {
    std::mutex m;
    std::condition_variable cv;
    std::vector<BackendTask::Type> types;
    std::vector<unsigned long> anchors;

    void record(const BackendTask& t) {
        std::lock_guard<std::mutex> lk(m);
        types.push_back(t.type);
        anchors.push_back(t.anchor_kf_id);
        cv.notify_all();
    }

    bool waitCount(int n, int ms = 3000) {
        std::unique_lock<std::mutex> lk(m);
        return cv.wait_for(lk, std::chrono::milliseconds(ms),
                           [&] { return (int)types.size() >= n; });
    }

    size_t count() {
        std::lock_guard<std::mutex> lk(m);
        return types.size();
    }
};

/// 可阻塞首任务、能释放的 handler 挂具：保证"worker 已开始执行首任务"这一
/// 确定时序点，随后提交的覆盖行为才是可测的。
struct Harness {
    Recorder rec;
    std::atomic<bool> hold{true};      // 首任务进入时等待 release()
    std::atomic<bool> in_handler{false};

    void handler(BackendTask& t) {
        in_handler.store(true);
        while (hold.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        rec.record(t);
    }

    bool waitInHandler(int ms = 3000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (!in_handler.load()) {
            if (std::chrono::steady_clock::now() > deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }

    void release() { hold.store(false); }
};

BackendTask makeLocalBA(unsigned long anchor) {
    BackendTask t;
    t.type = BackendTask::Type::LocalBA;
    t.anchor_kf_id = anchor;
    return t;
}

BackendTask makeLoop() {
    BackendTask t;
    t.type = BackendTask::Type::LoopClosure;
    return t;
}

BackendTask makeMaintenance() {
    BackendTask t;
    t.type = BackendTask::Type::LoopMaintenance;
    return t;
}

void test_basic_execution() {
    TEST("基础执行：提交 → worker 执行 → stop 正常退出") {
        Harness h;
        BackendScheduler sched([&](BackendTask& t) { h.handler(t); });
        sched.start();
        assert(sched.running());

        sched.submit(makeLocalBA(7));
        assert(h.waitInHandler());
        h.release();
        assert(h.rec.waitCount(1));

        sched.stop();
        assert(!sched.running());
        assert(h.rec.count() == 1 && h.rec.anchors[0] == 7);
    } TEST_PASS();
}

void test_same_type_overwrite() {
    TEST("同类 Local BA 覆盖：新任务覆盖等待中的旧任务") {
        Harness h;
        BackendScheduler sched([&](BackendTask& t) { h.handler(t); });
        sched.start();

        sched.submit(makeLocalBA(0));  // 阻塞任务，占住 worker
        assert(h.waitInHandler());

        sched.submit(makeLocalBA(1));  // 等待槽
        sched.submit(makeLocalBA(2));  // 覆盖槽中的 1
        assert(sched.hasPending());

        h.release();
        assert(h.rec.waitCount(2));
        assert(h.rec.count() == 2 && "anchor 1 必须被 anchor 2 覆盖");
        assert(h.rec.anchors[0] == 0 && h.rec.anchors[1] == 2);

        sched.stop();
    } TEST_PASS();
}

void test_loop_overwrites_localba() {
    TEST("LoopClosure 覆盖等待中的 Local BA（优先）") {
        Harness h;
        BackendScheduler sched([&](BackendTask& t) { h.handler(t); });
        sched.start();

        sched.submit(makeLocalBA(0));  // 阻塞任务
        assert(h.waitInHandler());

        sched.submit(makeLocalBA(1));  // 等待槽 = LocalBA
        sched.submit(makeLoop());       // 覆盖 → 槽 = LoopClosure

        h.release();
        assert(h.rec.waitCount(2));
        assert(h.rec.count() == 2 && "LocalBA(1) 必须被 LoopClosure 覆盖");
        assert(h.rec.types[0] == BackendTask::Type::LocalBA);
        assert(h.rec.types[1] == BackendTask::Type::LoopClosure);

        sched.stop();
    } TEST_PASS();
}

void test_localba_not_overwrite_pending_loop() {
    TEST("Local BA 不覆盖等待中的 LoopClosure（槽满则丢弃）") {
        Harness h;
        BackendScheduler sched([&](BackendTask& t) { h.handler(t); });
        sched.start();

        sched.submit(makeLocalBA(0));  // 阻塞任务
        assert(h.waitInHandler());

        sched.submit(makeLoop());       // 等待槽 = LoopClosure
        sched.submit(makeLocalBA(5));   // 槽被 LoopClosure 占住 → 丢弃
        assert(sched.hasPending());

        h.release();
        assert(h.rec.waitCount(2));
        assert(h.rec.count() == 2 && "LocalBA(5) 必须被丢弃");
        assert(h.rec.types[0] == BackendTask::Type::LocalBA);
        assert(h.rec.types[1] == BackendTask::Type::LoopClosure);

        sched.stop();
    } TEST_PASS();
}

void test_loop_maintenance_priority() {
    TEST("LoopMaintenance 覆盖 Local BA，但不覆盖 LoopClosure") {
        Harness h;
        BackendScheduler sched([&](BackendTask& t) { h.handler(t); });
        sched.start();

        sched.submit(makeLocalBA(0));
        assert(h.waitInHandler());
        sched.submit(makeLocalBA(1));
        sched.submit(makeMaintenance());
        sched.submit(makeLocalBA(2));
        sched.submit(makeLoop());
        sched.submit(makeMaintenance());

        h.release();
        assert(h.rec.waitCount(2));
        assert(h.rec.count() == 2);
        assert(h.rec.types[0] == BackendTask::Type::LocalBA);
        assert(h.rec.types[1] == BackendTask::Type::LoopClosure &&
               "最终等待槽必须保留最高优先级 LoopClosure");
        sched.stop();
    } TEST_PASS();
}

void test_stop_drains_pending() {
    TEST("stop() 排空槽内任务后 join") {
        Harness h;
        h.hold.store(false);  // 本测试不需要阻塞门
        BackendScheduler sched([&](BackendTask& t) { h.handler(t); });
        sched.start();

        sched.submit(makeLocalBA(3));
        sched.stop();  // 槽内有任务：stop 语义为排空后退出

        assert(!sched.running());
        assert(h.rec.count() == 1 && h.rec.anchors[0] == 3);
    } TEST_PASS();
}

void test_stop_without_start_and_double_stop() {
    TEST("未 start 的 stop / 重复 stop：无崩溃") {
        BackendScheduler sched([](BackendTask&) {});
        sched.stop();  // 从未 start
        sched.start();
        sched.start();  // 幂等
        assert(sched.running());
        sched.stop();
        sched.stop();  // 重复 stop
        assert(!sched.running());
    } TEST_PASS();
}

void test_submit_after_stop_is_rejected() {
    TEST("stop 后提交：不得重新产生 pending 任务") {
        BackendScheduler sched([](BackendTask&) {});
        sched.start();
        sched.stop();
        sched.submit(makeLocalBA(11));
        assert(!sched.hasPending() && "已停止的 scheduler 不得接收新任务");
        assert(sched.stats().dropped >= 1 && "stop 后任务应计为 dropped");
    } TEST_PASS();
}

void test_expired_task_is_dropped_before_handler() {
    TEST("后台任务年龄超限：过期任务不进入 handler") {
        std::atomic<int> handled{0};
        BackendScheduler sched([&](BackendTask&) { handled.fetch_add(1); });
        sched.setMaxTaskAgeMs(1.0);
        sched.submit(makeLocalBA(12));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        sched.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        sched.stop();
        const auto stats = sched.stats();
        assert(handled.load() == 0);
        assert(stats.expired == 1);
        assert(stats.executed == 0);
        assert(stats.dropped >= 1);
    } TEST_PASS();
}

void test_destructor_joins() {
    TEST("析构自动 stop + join（不 detach）") {
        {
            Harness h;
            h.hold.store(false);  // 不阻塞，确保析构 join 能完成
            BackendScheduler sched([&](BackendTask& t) { h.handler(t); });
            sched.start();
            sched.submit(makeLocalBA(9));
            // 作用域结束触发 ~BackendScheduler → stop + join，槽内任务被排空
        }
        std::cout << "(sched 已析构，无崩溃)" << std::endl;
    } TEST_PASS();
}

}  // namespace

int main() {
    std::cout << "test_backend_scheduler (M1.3 BackendScheduler)" << std::endl;

    test_basic_execution();
    test_same_type_overwrite();
    test_loop_overwrites_localba();
    test_localba_not_overwrite_pending_loop();
    test_loop_maintenance_priority();
    test_stop_drains_pending();
    test_stop_without_start_and_double_stop();
    test_submit_after_stop_is_rejected();
    test_expired_task_is_dropped_before_handler();
    test_destructor_joins();

    std::cout << "全部通过" << std::endl;
    return 0;
}
