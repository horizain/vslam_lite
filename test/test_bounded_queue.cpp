/**
 * test_bounded_queue.cpp - M2.1 输入队列与异步 Localizer 单元测试
 *
 * 覆盖 PRODUCTION_LOCALIZATION_PLAN §6.1/§6.2：
 *   A. BoundedQueue：
 *      1. FIFO 顺序
 *      2. 满时丢最旧、保最新（drop-oldest）
 *      3. high water mark / dropped 计数（§6.4）
 *      4. 阻塞 pop 唤醒、stop 唤醒、stop 后排空、stop 后拒绝 push
 *   B. Localizer 异步模式：
 *      5. submitFrame 只入队不阻塞；满丢最旧 → 最新帧存活（时间戳断言）
 *      6. 输入校验（§4.3）在提交侧生效
 *      7. 同步模式下 submitFrame 返回 false；stop 后提交拒绝
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_bounded_queue
 * 运行: ./build/test_bounded_queue（独立 CTest）
 */

#include "vslam/bounded_queue.h"
#include "vslam/localizer.h"
#include "vslam/localization_types.h"
#include "vslam/sensor_packet.h"
#include "vslam/camera.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

// 简单的测试辅助宏（与 test_vo.cpp 保持一致）
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::BoundedQueue;
using vslam::FailureReason;
using vslam::LocalizationMode;
using vslam::Localizer;
using vslam::LocalizerConfig;
using vslam::PoseEstimate;
using vslam::SE3;
using vslam::TrackingState;
using vslam::VOConfig;
using vslam::Vec3;

namespace {

// ============================================================
// A. BoundedQueue 单元测试
// ============================================================

void test_fifo_order() {
    TEST("BoundedQueue: FIFO 顺序") {
        BoundedQueue<int> q(4);
        for (int i = 1; i <= 3; i++) q.push(i);
        assert(q.size() == 3);
        int v = 0;
        assert(q.pop(v) && v == 1);
        assert(q.pop(v) && v == 2);
        assert(q.pop(v) && v == 3);
        assert(!q.tryPop(v) && "空队列 tryPop 返回 false");
        assert(q.size() == 0);
    } TEST_PASS();
}

void test_drop_oldest() {
    TEST("BoundedQueue: 满时丢最旧、保最新") {
        BoundedQueue<int> q(3);
        for (int i = 1; i <= 6; i++) q.push(i);
        assert(q.size() == 3);
        assert(q.droppedCount() == 3);
        int v = 0;
        assert(q.pop(v) && v == 4);
        assert(q.pop(v) && v == 5);
        assert(q.pop(v) && v == 6);
        assert(q.highWaterMark() == 3);
    } TEST_PASS();
}

void test_high_water_mark() {
    TEST("BoundedQueue: high water mark 记录峰值") {
        BoundedQueue<int> q(3);
        q.push(1);
        q.push(2);
        int v = 0;
        assert(q.pop(v) && v == 1);  // 深度 1
        q.push(3);
        q.push(4);                   // 深度 3（峰值）
        assert(q.highWaterMark() == 3);
        assert(q.pop(v) && q.pop(v) && q.pop(v));
        assert(q.highWaterMark() == 3 && "峰值不被 pop 重置");
    } TEST_PASS();
}

void test_try_pop_empty() {
    TEST("BoundedQueue: tryPop 空队列返回 false") {
        BoundedQueue<int> q(2);
        int v = 0;
        assert(!q.tryPop(v));
        q.push(7);
        assert(q.tryPop(v) && v == 7);
        assert(!q.tryPop(v));
    } TEST_PASS();
}

void test_pop_wakes_on_push() {
    TEST("BoundedQueue: 阻塞 pop 在 push 后唤醒") {
        BoundedQueue<int> q(2);
        std::atomic<bool> got{false};
        int result = 0;
        std::thread consumer([&] {
            int v = 0;
            if (q.pop(v)) { result = v; got.store(true); }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        q.push(42);
        for (int i = 0; i < 500 && !got.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        assert(got.load() && result == 42);
        consumer.join();
    } TEST_PASS();
}

void test_stop_wakes_blocker() {
    TEST("BoundedQueue: stop 唤醒阻塞消费者并返回 false") {
        BoundedQueue<int> q(2);
        std::atomic<bool> done{false};
        std::thread consumer([&] {
            int v = 0;
            bool ok = q.pop(v);  // 应被 stop 唤醒并返回 false
            done.store(!ok);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        q.stop();
        for (int i = 0; i < 500 && !done.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        assert(done.load());
        consumer.join();
    } TEST_PASS();
}

void test_stop_drains_then_rejects() {
    TEST("BoundedQueue: stop 后排空剩余元素，随后拒绝 push") {
        BoundedQueue<int> q(3);
        q.push(1);
        q.push(2);
        q.stop();
        int v = 0;
        assert(q.pop(v) && v == 1 && "stop 后仍排空剩余元素");
        assert(q.pop(v) && v == 2);
        assert(!q.pop(v));
        const size_t size_before = q.size();
        q.push(99);  // stop 后拒绝
        assert(q.size() == size_before);
        assert(q.droppedCount() == 0 && "stop 后提交不算丢弃");
    } TEST_PASS();
}

// ============================================================
// B. Localizer 异步模式
// ============================================================

struct Blk { cv::Point3f c; float sx, sy; int gray; };

void makeScene(std::vector<Blk>& blks, unsigned seed = 7) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dx(-4, 4), dy(-3, 3), dz(3, 6),
                                           ds(0.5, 1.5), dg(80, 255);
    for (int i = 0; i < 60; i++)
        blks.push_back({cv::Point3f(dx(gen), dy(gen), dz(gen)),
                        (float)ds(gen), (float)ds(gen), (int)dg(gen)});
}

auto makeStereoCamera() {
    auto cam = std::make_shared<vslam::StereoCamera>();
    cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
    cam->img_width = 640; cam->img_height = 480;
    cam->fx_r = 500; cam->fy_r = 500; cam->cx_r = 320; cam->cy_r = 240;
    cam->baseline_m = 0.5;
    return cam;
}

void renderStereoFrame(const vslam::Camera& cam, const std::vector<Blk>& blks,
                       const SE3& T_wc, cv::Mat& left, cv::Mat& right) {
    const SE3 T_cw = T_wc.inverse();
    left  = cv::Mat(480, 640, CV_8UC1, cv::Scalar(64));
    right = cv::Mat(480, 640, CV_8UC1, cv::Scalar(64));
    for (auto& b : blks) {
        std::vector<cv::Point3f> corners = {
            cv::Point3f(b.c.x - b.sx/2, b.c.y - b.sy/2, b.c.z),
            cv::Point3f(b.c.x + b.sx/2, b.c.y - b.sy/2, b.c.z),
            cv::Point3f(b.c.x + b.sx/2, b.c.y + b.sy/2, b.c.z),
            cv::Point3f(b.c.x - b.sx/2, b.c.y + b.sy/2, b.c.z)};
        std::vector<cv::Point> pi_l, pi_r;
        for (auto& q : corners) {
            vslam::Vec3 p(q.x, q.y, q.z);
            vslam::Vec2 pl = cam->world2pixel(p, T_cw);
            vslam::Vec2 pr = cam->camera2pixelRight(T_cw * p);
            pi_l.emplace_back(cvRound(pl.x()), cvRound(pl.y()));
            pi_r.emplace_back(cvRound(pr.x()), cvRound(pr.y()));
        }
        cv::fillConvexPoly(left,  pi_l, cv::Scalar(b.gray));
        cv::fillConvexPoly(right, pi_r, cv::Scalar(b.gray));
    }
}

VOConfig deterministicConfig() {
    VOConfig cfg;
    cfg.async_backend = false;
    cfg.enable_loop_closure = false;
    cfg.min_matches_track = 10;
    cfg.stereo_min_points = 40;
    cfg.opencv_threads = 1;
    cfg.orb_max_bands = 1;
    cfg.rng_seed = 0x5A17;  // worker 线程 RNG 播种（与主线程同规则，确定性）
    return cfg;
}

// 8 帧确定平移路径（双目第一帧即建图）；步长 0.15m 保证路径末端
// （队列满丢最旧后存活的帧 5~7，z<=1.05）仍在场景深度范围内可跟踪
void makePath(std::vector<SE3>& path) {
    for (int i = 0; i < 8; i++) {
        const double z = 0.15 * i;
        Eigen::Quaterniond q(Eigen::AngleAxisd(0.02 * i, Vec3::UnitY()));
        path.push_back(SE3(q, Vec3(0.0, 0.0, z)));
    }
}

void test_async_localizer_drop_oldest() {
    TEST("异步 Localizer: 满丢最旧、最新帧存活且进入 Tracking") {
        const auto cam = makeStereoCamera();
        LocalizerConfig cfg;
        cfg.mode = LocalizationMode::Mapping;
        cfg.enable_async_input = true;
        cfg.input_queue_capacity = 3;
        Localizer l(cam, deterministicConfig(), cfg);

        std::vector<Blk> blks;
        makeScene(blks);
        std::vector<SE3> path;
        makePath(path);

        // 先全部渲染（渲染较慢，若边渲边提交 worker 可能追平队列），
        // 再快速连续提交 8 帧（提交只有 Mat 头拷贝，~µs/帧 → 队列必然打满）
        std::vector<std::pair<cv::Mat, cv::Mat>> frames;
        frames.reserve(8);
        for (int i = 0; i < 8; i++) {
            cv::Mat left, right;
            renderStereoFrame(cam, blks, path[i], left, right);
            frames.emplace_back(left, right);
        }
        for (int i = 0; i < 8; i++)
            assert(l.submitFrame(frames[i].first, frames[i].second, 0.1 * i));

        // 轮询等待 worker 消费（最长 10s）
        bool tracking = false;
        for (int i = 0; i < 2000; i++) {
            if (l.state() == TrackingState::Tracking) { tracking = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        assert(tracking && "异步消费后应进入 Tracking");

        // 丢最旧保最新：最后消费的必须是时间戳最大（0.7）的帧
        const PoseEstimate latest = l.latestPose();
        assert(latest.pose_valid);
        assert(std::abs(latest.timestamp - 0.7) < 1e-9
               && "队列满丢最旧 → 最新帧（ts=0.7）必须存活");
        assert(l.inputQueueHighWaterMark() <= 3);
        assert(l.inputQueueDroppedCount() >= 3 && "8 帧进容量 3 队列必丢至少 3 帧");
        l.stop();
        assert(l.inputQueueDroppedCount() >= 3);
    } TEST_PASS();
}

void test_async_submit_validation() {
    TEST("异步 Localizer: §4.3 输入校验在提交侧生效") {
        const auto cam = makeStereoCamera();
        LocalizerConfig cfg;
        cfg.enable_async_input = true;
        cfg.input_queue_capacity = 3;
        Localizer l(cam, deterministicConfig(), cfg);

        std::vector<Blk> blks;
        makeScene(blks);
        cv::Mat left, right;
        renderStereoFrame(cam, blks, SE3(), left, right);

        assert(!l.submitFrame(cv::Mat(), cv::Mat(), 0.1) && "空图拒绝");
        assert(l.submitFrame(left, right, 0.2) && "有效帧入队");
        assert(!l.submitFrame(left, right, 0.2) && "时间戳相等拒绝（严格递增）");
        assert(!l.submitFrame(left, right, 0.15) && "时间倒退拒绝");
        // 异步模式下 processFrame 不可用（避免与 worker 并发驱动 VO）
        const PoseEstimate out = l.processFrame(left, right, 0.3);
        assert(!out.pose_valid && "异步模式下 processFrame 返回无效输出");
        l.stop();
    } TEST_PASS();
}

void test_sync_mode_submit_rejected() {
    TEST("同步 Localizer: submitFrame 返回 false；processFrame 不可用") {
        const auto cam = makeStereoCamera();
        Localizer l(cam, deterministicConfig());  // 默认同步
        std::vector<Blk> blks;
        makeScene(blks);
        cv::Mat left, right;
        renderStereoFrame(cam, blks, SE3(), left, right);
        assert(!l.submitFrame(left, right, 0.1) && "同步模式 submitFrame 拒绝");
        l.stop();
    } TEST_PASS();
}

void test_async_stop_rejects_submit() {
    TEST("异步 Localizer: stop 后提交拒绝、重复 stop 幂等") {
        const auto cam = makeStereoCamera();
        LocalizerConfig cfg;
        cfg.enable_async_input = true;
        Localizer l(cam, deterministicConfig(), cfg);
        std::vector<Blk> blks;
        makeScene(blks);
        cv::Mat left, right;
        renderStereoFrame(cam, blks, SE3(), left, right);
        l.stop();
        l.stop();  // 幂等
        assert(!l.submitFrame(left, right, 0.1) && "stop 后提交拒绝");
    } TEST_PASS();
}

}  // namespace

int main() {
    cv::setNumThreads(1);
    cv::setRNGSeed(0x5A17);

    std::cout << "test_bounded_queue (M2.1 输入队列 + 异步 Localizer)" << std::endl;

    test_fifo_order();
    test_drop_oldest();
    test_high_water_mark();
    test_try_pop_empty();
    test_pop_wakes_on_push();
    test_stop_wakes_blocker();
    test_stop_drains_then_rejects();
    test_async_localizer_drop_oldest();
    test_async_submit_validation();
    test_sync_mode_submit_rejected();
    test_async_stop_rejects_submit();

    std::cout << "全部通过" << std::endl;
    return 0;
}
