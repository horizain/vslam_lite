/**
 * test_tracking_state_machine.cpp - 定位状态机单元测试
 *
 * 覆盖 M0.2 状态机（PRODUCTION_LOCALIZATION_PLAN §4.2）：
 *   1. Initializing → Tracking（连续 3 帧完整验收）
 *   2. Tracking ⇄ Degraded（连续 2 帧弱质量 / 连续 3 帧完整恢复）
 *   3. Tracking/Degraded → Relocalizing（连续 5 帧失败）
 *   4. Relocalizing → Tracking / Lost（20 帧或 2.0s 超时）
 *   5. Lost → Tracking（有效全局重定位）
 *   6. stop() 幂等；重复 stop、stop 后输入均不崩溃
 *   7. 发布语义：弱质量 pose_valid=true、失败帧预测窗口 0.5s
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_tracking_state_machine
 * 运行: ./build/test_tracking_state_machine（独立 CTest）
 */

#include "vslam/tracking_state_machine.h"

#include <cassert>
#include <iostream>

// 简单的测试辅助宏（与 test_vo.cpp 保持一致）
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::FailureReason;
using vslam::FrameQuality;
using vslam::StateMachineOutput;
using vslam::TrackingState;
using vslam::TrackingStateMachine;

namespace {

// 初始化到 Tracking 状态的公共入口
void reach_tracking(TrackingStateMachine& sm) {
    for (int i = 0; i < 3; i++) sm.on_tracking(FrameQuality::Full, 0.1);
    assert(sm.state() == TrackingState::Tracking);
}

// 到达 Relocalizing 状态的公共入口
void reach_relocalizing(TrackingStateMachine& sm) {
    reach_tracking(sm);
    for (int i = 0; i < 5; i++) sm.on_tracking(FrameQuality::Failed, 0.05);
    assert(sm.state() == TrackingState::Relocalizing);
}

void test_initial_state() {
    TEST("初始状态为 Initializing") {
        TrackingStateMachine sm;
        assert(sm.state() == TrackingState::Initializing);
    } TEST_PASS();
}

void test_initializing() {
    TEST("Initializing: 连续 3 帧 Full → Tracking 且位姿有效") {
        TrackingStateMachine sm;
        auto o1 = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o1.state == TrackingState::Initializing);
        assert(!o1.pose_valid);
        auto o2 = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o2.state == TrackingState::Initializing);
        auto o3 = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o3.state == TrackingState::Tracking);
        assert(o3.pose_valid && !o3.prediction_only);
    } TEST_PASS();

    TEST("Initializing: Weak/Failed 中断连续计数，重新累计 3 帧") {
        TrackingStateMachine sm;
        sm.on_tracking(FrameQuality::Full, 0.1);
        auto o = sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(o.state == TrackingState::Initializing);
        sm.on_tracking(FrameQuality::Full, 0.1);
        sm.on_tracking(FrameQuality::Full, 0.1);
        auto o2 = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o2.state == TrackingState::Tracking);
    } TEST_PASS();

    TEST("Initializing: 失败帧不发布有效位姿") {
        TrackingStateMachine sm;
        auto o = sm.on_tracking(FrameQuality::Failed, 0.1);
        assert(o.state == TrackingState::Initializing);
        assert(!o.pose_valid && !o.prediction_only);
    } TEST_PASS();
}

void test_tracking_weak() {
    TEST("Tracking: 连续 2 帧 Weak → Degraded（弱质量帧仍有效位姿）") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        auto o1 = sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(o1.state == TrackingState::Tracking);
        assert(o1.pose_valid && !o1.prediction_only);
        auto o2 = sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(o2.state == TrackingState::Degraded);
        assert(o2.pose_valid && !o2.prediction_only);
    } TEST_PASS();

    TEST("Tracking: Weak 被 Full 打断后不进入 Degraded") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        sm.on_tracking(FrameQuality::Weak, 0.1);
        auto o = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o.state == TrackingState::Tracking);
        sm.on_tracking(FrameQuality::Weak, 0.1);
        auto o2 = sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(o2.state == TrackingState::Degraded);  // 重新累计 2 帧 Weak
    } TEST_PASS();
}

void test_degraded_recovery() {
    TEST("Degraded: 连续 3 帧 Full → Tracking") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        sm.on_tracking(FrameQuality::Weak, 0.1);
        sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(sm.state() == TrackingState::Degraded);
        auto o1 = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o1.state == TrackingState::Degraded);
        auto o2 = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o2.state == TrackingState::Degraded);
        auto o3 = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o3.state == TrackingState::Tracking);
    } TEST_PASS();

    TEST("Degraded: 恢复被 Weak 打断，重新累计 3 帧 Full") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        sm.on_tracking(FrameQuality::Weak, 0.1);
        sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(sm.state() == TrackingState::Degraded);
        sm.on_tracking(FrameQuality::Full, 0.1);
        auto o = sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(o.state == TrackingState::Degraded);
        for (int i = 0; i < 3; i++) sm.on_tracking(FrameQuality::Full, 0.1);
        assert(sm.state() == TrackingState::Tracking);
    } TEST_PASS();
}

void test_failures_to_relocalizing() {
    TEST("Tracking: 连续 5 帧 Failed → Relocalizing") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        for (int i = 0; i < 4; i++) sm.on_tracking(FrameQuality::Failed, 0.05);
        auto o = sm.on_tracking(FrameQuality::Failed, 0.05);
        assert(o.state == TrackingState::Relocalizing);
    } TEST_PASS();

    TEST("Degraded: 失败帧同样累计 → Relocalizing") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        sm.on_tracking(FrameQuality::Weak, 0.1);
        sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(sm.state() == TrackingState::Degraded);
        for (int i = 0; i < 5; i++) sm.on_tracking(FrameQuality::Failed, 0.05);
        assert(sm.state() == TrackingState::Relocalizing);
    } TEST_PASS();

    TEST("Tracking: Full 帧清零失败计数，不提前进入 Relocalizing") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        for (int i = 0; i < 4; i++) sm.on_tracking(FrameQuality::Failed, 0.05);
        auto o = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o.state == TrackingState::Tracking);
        for (int i = 0; i < 4; i++) sm.on_tracking(FrameQuality::Failed, 0.05);
        assert(sm.state() == TrackingState::Tracking);
        sm.on_tracking(FrameQuality::Failed, 0.05);
        assert(sm.state() == TrackingState::Relocalizing);
    } TEST_PASS();
}

void test_prediction_window() {
    TEST("Tracking: 失败帧在 0.5s 窗口内发布 prediction_only") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        auto o = sm.on_tracking(FrameQuality::Failed, 0.05);
        assert(o.pose_valid && o.prediction_only);
        o = sm.on_tracking(FrameQuality::Failed, 0.1);
        assert(o.pose_valid && o.prediction_only);
    } TEST_PASS();

    TEST("Tracking: 预测累计超过 0.5s 后 pose_valid=false") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        sm.on_tracking(FrameQuality::Failed, 0.3);
        auto o = sm.on_tracking(FrameQuality::Failed, 0.3);  // 0.6 > 0.5
        assert(!o.pose_valid);
        assert(!o.prediction_only);
    } TEST_PASS();

    TEST("预测窗口在 Weak 帧后重置") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        sm.on_tracking(FrameQuality::Failed, 0.4);
        auto o = sm.on_tracking(FrameQuality::Weak, 0.1);
        assert(o.pose_valid);
        o = sm.on_tracking(FrameQuality::Failed, 0.4);  // 重新从 0.4 累计
        assert(o.pose_valid && o.prediction_only);
        o = sm.on_tracking(FrameQuality::Failed, 0.2);  // 0.6 > 0.5
        assert(!o.pose_valid);
    } TEST_PASS();
}

void test_relocalizing() {
    TEST("Relocalizing: 重定位成功 → Tracking 且位姿有效") {
        TrackingStateMachine sm;
        reach_relocalizing(sm);
        auto o = sm.on_relocalization(true, 0.1);
        assert(o.state == TrackingState::Tracking);
        assert(o.pose_valid && !o.prediction_only);
    } TEST_PASS();

    TEST("Relocalizing: 重定位失败保持状态，帧数 20 上限触发 Lost") {
        TrackingStateMachine sm;
        reach_relocalizing(sm);
        for (int i = 0; i < 19; i++) {
            auto o = sm.on_relocalization(false, 0.05);
            assert(o.state == TrackingState::Relocalizing);
        }
        auto o = sm.on_relocalization(false, 0.05);  // 第 20 帧
        assert(o.state == TrackingState::Lost);
        assert(o.reason == FailureReason::RelocalizationTimeout);
    } TEST_PASS();

    TEST("Relocalizing: 时长 2.0s 上限触发 Lost（帧数未到）") {
        TrackingStateMachine sm;
        reach_relocalizing(sm);
        for (int i = 0; i < 4; i++) sm.on_relocalization(false, 0.5);  // 第 4 次累计 2.0s
        assert(sm.state() == TrackingState::Lost);
    } TEST_PASS();
}

void test_lost() {
    TEST("Lost: 有效全局重定位 → Tracking") {
        TrackingStateMachine sm;
        reach_relocalizing(sm);
        for (int i = 0; i < 20; i++) sm.on_relocalization(false, 0.05);
        assert(sm.state() == TrackingState::Lost);
        auto o = sm.on_relocalization(true, 0.1);
        assert(o.state == TrackingState::Tracking);
        assert(o.pose_valid && !o.prediction_only);
    } TEST_PASS();

    TEST("Lost: 重定位失败保持 Lost 且不发布有效位姿") {
        TrackingStateMachine sm;
        reach_relocalizing(sm);
        for (int i = 0; i < 20; i++) sm.on_relocalization(false, 0.05);
        assert(sm.state() == TrackingState::Lost);
        auto o = sm.on_relocalization(false, 0.1);
        assert(o.state == TrackingState::Lost);
        assert(!o.pose_valid);
    } TEST_PASS();

    TEST("Lost: on_tracking 保持 Lost（只能经全局重定位离开）") {
        TrackingStateMachine sm;
        reach_relocalizing(sm);
        for (int i = 0; i < 20; i++) sm.on_relocalization(false, 0.05);
        auto o = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o.state == TrackingState::Lost);
        assert(!o.pose_valid);
    } TEST_PASS();
}

void test_stop() {
    TEST("任意状态 stop() → Stopped，重复 stop 不崩溃（§4.4）") {
        TrackingStateMachine sm;
        reach_tracking(sm);
        auto o = sm.stop();
        assert(o.state == TrackingState::Stopped);
        o = sm.stop();  // 重复 stop
        assert(o.state == TrackingState::Stopped);
        // stop 后输入不再改变状态
        o = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o.state == TrackingState::Stopped);
        o = sm.on_relocalization(true, 0.1);
        assert(o.state == TrackingState::Stopped);
    } TEST_PASS();

    TEST("Initializing 中 stop() → Stopped") {
        TrackingStateMachine sm;
        auto o = sm.stop();
        assert(o.state == TrackingState::Stopped);
    } TEST_PASS();

    TEST("Lost 中 stop() → Stopped") {
        TrackingStateMachine sm;
        reach_relocalizing(sm);
        for (int i = 0; i < 20; i++) sm.on_relocalization(false, 0.05);
        assert(sm.state() == TrackingState::Lost);
        sm.stop();
        assert(sm.state() == TrackingState::Stopped);
    } TEST_PASS();
}

void test_custom_params() {
    TEST("自定义 Params 生效（init=1、fail=2、reloc 上限 3 帧）") {
        TrackingStateMachine::Params p;
        p.init_required_frames = 1;
        p.max_tracking_failures = 2;
        p.relocalization_max_frames = 3;
        TrackingStateMachine sm(p);
        auto o = sm.on_tracking(FrameQuality::Full, 0.1);
        assert(o.state == TrackingState::Tracking);
        sm.on_tracking(FrameQuality::Failed, 0.05);
        o = sm.on_tracking(FrameQuality::Failed, 0.05);
        assert(o.state == TrackingState::Relocalizing);
        sm.on_relocalization(false, 0.1);
        sm.on_relocalization(false, 0.1);
        o = sm.on_relocalization(false, 0.1);  // 第 3 帧
        assert(o.state == TrackingState::Lost);
    } TEST_PASS();
}

} // namespace

int main() {
    std::cout << "[Tracking State Machine (M0.2)]\n";
    test_initial_state();
    test_initializing();
    test_tracking_weak();
    test_degraded_recovery();
    test_failures_to_relocalizing();
    test_prediction_window();
    test_relocalizing();
    test_lost();
    test_stop();
    test_custom_params();

    std::cout << "\n===== All Tests Completed =====\n";
    return 0;
}
