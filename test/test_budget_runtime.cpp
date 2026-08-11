/**
 * test_budget_runtime.cpp - M2.2/M2.3 遗留清理：§6.3 预算触发点运行时接线测试
 *
 * 覆盖（先失败后实现的集成级验证）：
 *   1. VOConfig 从 default.yaml 解析 MapBudget 段（§6.2 首版参数）
 *   2. KF 插入触发预算：极小预算下地图 KF 数被控制在门限内（第 4 步冗余剔除）
 *   3. 点配额耗尽后 mapGrowthStopped() 生效，但仍允许 KF 继续插入
 *   4. mapSnapshotBytes() 在异步 Local BA 提交后有非零在途快照字节
 *   5. 预算内配置（默认 §6.2 参数）100 帧全程零回收，地图行为不变
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_budget_runtime
 * 运行: ./build/test_budget_runtime（独立 CTest）
 */

#include "vslam/camera.h"
#include "vslam/dataset.h"
#include "vslam/vo.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <iostream>
#include <string>

#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::Camera;
using vslam::Dataset;
using vslam::VisualOdometry;
using vslam::VOConfig;

namespace {

// KITTI 00 前 80 帧（静态路径，随构建目录无关）
Dataset loadKitti() {
    return Dataset(std::string(VSLAM_SOURCE_DIR) + "/datasets/kitti/sequences/00",
                   Dataset::Type::KITTI);
}

Camera makeCamera() {
    auto cam = std::make_shared<vslam::StereoCamera>();
    cam->fx = 718.856;
    cam->fy = 718.856;
    cam->cx = 607.193;
    cam->cy = 185.216;
    cam->img_width = 1241;
    cam->img_height = 376;
    cam->baseline_m = 0.54;
    return cam;
}

bool runFrames(VisualOdometry& vo, Dataset& ds, int frames) {
    for (int i = 0; i < frames; i++) {
        cv::Mat left, right;
        double ts = 0.0, rt = 0.0;
        if (!ds.nextFrame(left, right, ts)) return false;
        vo.addFrame(left, right, ts);
    }
    return true;
}

void test_yaml_map_budget_parsing() {
    TEST("VOConfig: default.yaml 解析 MapBudget 段") {
        const VOConfig cfg = VOConfig::fromYaml(
            std::string(VSLAM_SOURCE_DIR) + "/config/default.yaml");
        // 2026-08-10 实测标定版（§6.5 RSS <1GiB 硬门槛，见 resource_budget.h）
        assert(cfg.map_budget.max_active_keyframes == 700);
        assert(cfg.map_budget.max_active_points == 60000);
        assert(cfg.map_budget.max_descriptor_mb == 256);
        assert(cfg.map_budget.max_snapshot_mb == 256);
        assert(cfg.map_budget.max_total_estimated_mb == 500);
    } TEST_PASS();
}

void test_default_budget_no_reclaim() {
    TEST("runtime: 默认预算 40 帧零回收，地图正常增长") {
        VOConfig cfg;
        cfg.rng_seed = 0x5A17;   // 确定性
        cfg.async_backend = false;
        cfg.enable_local_ba = false;
        VisualOdometry vo(makeCamera(), cfg);
        auto ds = loadKitti();
        assert(runFrames(vo, ds, 40));
        vo.finishPendingBackendWork();
        assert(vo.getMap()->keyFrameCount() >= 5 && "默认预算下 KF 正常插入");
        assert(!vo.mapGrowthStopped() && "默认预算远未耗尽");
    } TEST_PASS();
}

void test_tiny_budget_controls_keyframes() {
    TEST("runtime: 极小预算触发连续滚动且活动图受硬上限控制") {
        VOConfig cfg;
        cfg.rng_seed = 0x5A17;
        cfg.async_backend = false;
        cfg.enable_local_ba = false;
        cfg.map_budget.max_active_keyframes = 2;
        cfg.map_budget.max_active_points = 100000;
        cfg.map_budget.max_descriptor_mb = 256;
        cfg.map_budget.max_total_estimated_mb = 4096;
        VisualOdometry vo(makeCamera(), cfg);
        auto ds = loadKitti();
        assert(runFrames(vo, ds, 40));
        vo.finishPendingBackendWork();
        const size_t kfs = vo.getMap()->keyFrameCount();
        assert(kfs >= 2 && "仍应有最低限度的关键帧（锚点/最近窗口保护）");
        const bool stopped = vo.mapGrowthStopped();
        assert(stopped && "极小预算下当前活动图应报告硬预算耗尽");
        assert(vo.getAtlas()->submapCount() > 1 &&
               "硬预算不能通过永久拒绝 KF 维持，应主动滚动连续子图");

        // stopped 后继续少量帧：地图不得出现预算外增长。
        assert(runFrames(vo, ds, 10));
        vo.finishPendingBackendWork();
        const size_t kfs_after = vo.getMap()->keyFrameCount();
        assert(kfs_after <= kfs + 2 &&
               "每个滚动后的活动子图仍必须受同一 KF 硬上限约束");

    } TEST_PASS();
}

void test_point_budget_does_not_block_keyframes() {
    TEST("runtime: 点预算耗尽仍允许 KF 插入且点数不越界") {
        VOConfig cfg;
        cfg.rng_seed = 0x5A17;
        cfg.async_backend = false;
        cfg.enable_local_ba = false;
        cfg.map_budget.max_active_keyframes = 1000;
        cfg.map_budget.max_active_points = 200;
        cfg.map_budget.max_descriptor_mb = 256;
        cfg.map_budget.max_total_estimated_mb = 4096;
        VisualOdometry vo(makeCamera(), cfg);
        auto ds = loadKitti();
        assert(runFrames(vo, ds, 30));
        vo.finishPendingBackendWork();
        const size_t kfs = vo.getMap()->keyFrameCount();
        assert(vo.getMap()->mapPointCount() <= cfg.map_budget.max_active_points);
        assert(kfs >= 2);

        assert(runFrames(vo, ds, 20));
        vo.finishPendingBackendWork();
        assert(vo.getMap()->keyFrameCount() > kfs &&
               "点预算 stopped 不得阻断后续关键帧");
        assert(vo.getMap()->mapPointCount() <= cfg.map_budget.max_active_points);
    } TEST_PASS();
}

void test_hard_keyframe_budget_rollover_keeps_pose_live() {
    TEST("runtime: KF 硬预算触顶主动滚动子图且不进入 LOST") {
        VOConfig cfg;
        cfg.rng_seed = 0x5A17;
        cfg.async_backend = false;
        cfg.map_budget.max_active_keyframes = 1;
        cfg.map_budget.max_active_points = 10000;
        cfg.map_budget.max_descriptor_mb = 256;
        cfg.map_budget.max_total_estimated_mb = 4096;
        VisualOdometry vo(makeCamera(), cfg);
        auto ds = loadKitti();
        for (int i = 0; i < 40; i++) {
            cv::Mat left, right;
            double ts = 0.0, rt = 0.0;
            assert(ds.nextFrame(left, right, ts));
            vo.addFrame(left, right, ts);
            const auto status = vo.getStatus();
            assert(status.state != VisualOdometry::State::LOST &&
                   "KF 硬预算触顶不得把连续跟踪推入 LOST");
            if (i > 1)
                assert(status.pose_valid && "预算滚动后的当前帧仍应有有效世界位姿");
            (void)rt;
        }
        vo.finishPendingBackendWork();
        assert(vo.getAtlas()->submapCount() > 1 &&
               "KF 硬预算触顶后应创建连续的新子图");
    } TEST_PASS();
}

void test_map_snapshot_bytes_reported() {
    TEST("runtime: 异步 Local BA 提交后 mapSnapshotBytes 非零") {
        VOConfig cfg;
        cfg.rng_seed = 0x5A17;
        cfg.async_backend = true;          // 后台 Local BA（真实快照路径）
        cfg.enable_local_ba = true;
        VisualOdometry vo(makeCamera(), cfg);
        auto ds = loadKitti();
        assert(runFrames(vo, ds, 30));
        vo.finishPendingBackendWork();
        assert(vo.mapSnapshotBytes() > 0 && "在途快照字节应已上报（>0）");
    } TEST_PASS();
}

}  // namespace

int main() {
    cv::setNumThreads(1);
    cv::setRNGSeed(0x5A17);

    std::cout << "test_budget_runtime (M2.2/M2.3 遗留清理：预算触发接线)" << std::endl;

    test_yaml_map_budget_parsing();
    test_default_budget_no_reclaim();
    test_tiny_budget_controls_keyframes();
    test_point_budget_does_not_block_keyframes();
    test_hard_keyframe_budget_rollover_keeps_pose_live();
    test_map_snapshot_bytes_reported();

    std::cout << "全部通过" << std::endl;
    return 0;
}
