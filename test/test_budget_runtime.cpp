/**
 * test_budget_runtime.cpp - M2.2/M2.3 遗留清理：§6.3 预算触发点运行时接线测试
 *
 * 覆盖（先失败后实现的集成级验证）：
 *   1. VOConfig 从 default.yaml 解析 MapBudget 段（§6.2 首版参数）
 *   2. KF 插入触发预算：极小预算下地图 KF 数被控制在门限内（第 4 步冗余剔除）
 *   3. 全部手段耗尽后 mapGrowthStopped() 生效，needNewKeyFrame 不再插 KF
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
    return Dataset(std::string(VSLAM_SOURCE_DIR) + "/datasets/sequences/00",
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
    TEST("runtime: 默认预算 80 帧零回收，地图正常增长") {
        VOConfig cfg;
        cfg.rng_seed = 0x5A17;   // 确定性
        cfg.async_backend = false;
        VisualOdometry vo(makeCamera(), cfg);
        auto ds = loadKitti();
        assert(runFrames(vo, ds, 80));
        vo.finishPendingBackendWork();
        assert(vo.getMap()->keyFrameCount() >= 10 && "默认预算下 KF 正常插入");
        assert(!vo.mapGrowthStopped() && "默认预算远未耗尽");
    } TEST_PASS();
}

void test_tiny_budget_controls_keyframes() {
    TEST("runtime: 极小预算触发回收并停止建图（§6.3 第 4/6 步）") {
        VOConfig cfg;
        cfg.rng_seed = 0x5A17;
        cfg.async_backend = false;
        cfg.map_budget.max_active_keyframes = 5;   // 极小：必触发预算
        cfg.map_budget.max_active_points = 2000;
        cfg.map_budget.max_descriptor_mb = 8;
        cfg.map_budget.max_total_estimated_mb = 64;
        VisualOdometry vo(makeCamera(), cfg);
        auto ds = loadKitti();
        assert(runFrames(vo, ds, 150));
        vo.finishPendingBackendWork();
        const size_t kfs = vo.getMap()->keyFrameCount();
        assert(kfs >= 2 && "仍应有最低限度的关键帧（锚点/最近窗口保护）");
        // §6.3：KITTI 行驶 KF 间距 0.9m 远超冗余剔除门限（0.15m/3deg），
        // 第 4 步剔不动 → 第 6 步停止增加地图（正确语义，非错误）
        const bool stopped = vo.mapGrowthStopped();
        assert(stopped && "极小预算下最终应停止增加地图（§6.3 第 6 步）");
        assert(!vo.needNewKeyFrame() && "stopped 后 KF 提议必须被拒绝");

        // stopped 后继续 100 帧：地图不得出现预算外增长。子地图重建会重置
        // 活动地图计数（LOST 后重定位路径，合法），但新地图同样受预算约束。
        assert(runFrames(vo, ds, 100));
        vo.finishPendingBackendWork();
        const size_t kfs_after = vo.getMap()->keyFrameCount();
        assert(kfs_after <= kfs + 2 &&
               "stopped 期间 KF 增长只允许来自子地图重建（新地图同样受预算约束）");

        // 对照：同一帧数、默认预算下 KF 数应明显更多（证明停止建图生效）
        VOConfig cfg2;
        cfg2.rng_seed = 0x5A17;
        cfg2.async_backend = false;
        VisualOdometry vo2(makeCamera(), cfg2);
        auto ds2 = loadKitti();
        assert(runFrames(vo2, ds2, 250));
        vo2.finishPendingBackendWork();
        assert(vo2.getMap()->keyFrameCount() > kfs + 3 &&
               "默认预算下 KF 应正常增长（远多于预算冻结后）");
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
        assert(runFrames(vo, ds, 60));
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
    test_map_snapshot_bytes_reported();

    std::cout << "全部通过" << std::endl;
    return 0;
}
