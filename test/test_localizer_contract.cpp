/**
 * test_localizer_contract.cpp - Localizer Facade 契约测试
 *
 * 覆盖 M0.3（PRODUCTION_LOCALIZATION_PLAN §4）：
 *   1. 输入硬检查（§4.3）：空图、时间倒退/相等、左右尺寸/类型不一致、
 *      图像尺寸与标定不一致、双目时间差 >1ms、T_bc 非单位四元数/非有限平移
 *   2. 非法输入不调用 VO、Map revision 逐项不变（§3-3）
 *   3. 有效输入与旧 run_slam 输出一致（平移差 <1e-12 m、旋转差 <1e-12 rad）
 *   4. 重复 stop、stop 后输入、析构均不崩溃（§4.4）
 *   5. robot.yaml 配置解析
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_localizer_contract
 * 运行: ./build/test_localizer_contract（独立 CTest）
 */

#include "vslam/localizer.h"
#include "vslam/camera.h"
#include "vslam/vo.h"

#include <opencv2/imgproc.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

// 简单的测试辅助宏（与 test_vo.cpp 保持一致）
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::FailureReason;
using vslam::Localizer;
using vslam::LocalizerConfig;
using vslam::LocalizationMode;
using vslam::PoseEstimate;
using vslam::SE3;
using vslam::StereoCamera;
using vslam::TrackingState;
using vslam::Vec3;
using vslam::VisualOdometry;
using vslam::VOConfig;

namespace {

// 与 test_vo.cpp test_stereo_vo 一致的合成双目方块场景
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
    auto cam = std::make_shared<StereoCamera>();
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

// 确定性的 VO 配置：异步关闭 + 回环关闭 + 单线程/单带提取
// （与 M1 §5.6 deterministic.yaml 同思路：OpenCV 单线程、固定 RNG seed）
VOConfig deterministicConfig() {
    VOConfig cfg;
    cfg.async_backend = false;
    cfg.enable_loop_closure = false;
    cfg.min_matches_track = 10;
    cfg.stereo_min_points = 40;
    cfg.opencv_threads = 1;     // OpenCV 单线程
    cfg.orb_max_bands = 1;      // 单带串行提取（特征提取确定性）
    return cfg;
}

// 一个由 8 帧组成的确定平移+轻微旋转路径（双目第一帧即建图）
void makePath(std::vector<SE3>& path) {
    for (int i = 0; i < 8; i++) {
        const double z = 0.5 * i;
        Eigen::Quaterniond q(Eigen::AngleAxisd(0.02 * i, Vec3::UnitY()));
        path.push_back(SE3(q, Vec3(0.0, 0.0, z)));
    }
}

} // namespace

// ============================================================
// 构造校验（§4.3：T_bc 归一化误差 <1e-6、平移有限）
// ============================================================

void test_constructor_validation() {
    TEST("构造: 非单位 T_bc 四元数被拒绝") {
        auto cam = makeStereoCamera();
        LocalizerConfig cfg;
        cfg.T_bc = SE3(Eigen::Quaterniond(0.5, 0, 0, 0), Vec3::Zero());  // norm=0.5
        bool threw = false;
        try {
            Localizer l(cam, deterministicConfig(), cfg);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    } TEST_PASS();

    TEST("构造: 非有限 T_bc 平移被拒绝") {
        auto cam = makeStereoCamera();
        LocalizerConfig cfg;
        cfg.T_bc = SE3(Eigen::Quaterniond::Identity(),
                       Vec3(std::numeric_limits<double>::quiet_NaN(), 0, 0));
        bool threw = false;
        try {
            Localizer l(cam, deterministicConfig(), cfg);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    } TEST_PASS();

    TEST("构造: 单位 T_bc 正常创建") {
        auto cam = makeStereoCamera();
        Localizer l(cam, deterministicConfig());
        assert(l.state() == TrackingState::Initializing);
    } TEST_PASS();
}

// ============================================================
// 输入硬检查（§4.3）与 Map revision 不变（§3-3）
// ============================================================

void test_input_validation() {
    auto cam = makeStereoCamera();
    Localizer l(cam, deterministicConfig());
    std::vector<Blk> blks;
    makeScene(blks);
    cv::Mat left, right;
    renderStereoFrame(cam, blks, SE3(), left, right);

    const uint64_t topo0 = l.mapTopologyRevision();
    const uint64_t geo0 = l.mapGeometryRevision();

    TEST("空图: InvalidInput 拒绝且 Map revision 不变") {
        cv::Mat empty;
        auto out = l.processFrame(empty, empty, 0.1);
        assert(out.reason == FailureReason::InvalidInput);
        assert(!out.pose_valid);
        assert(l.mapTopologyRevision() == topo0);
        assert(l.mapGeometryRevision() == geo0);
    } TEST_PASS();

    TEST("时间倒退: TimestampRollback 拒绝且 Map revision 不变") {
        renderStereoFrame(cam, blks, SE3(), left, right);
        auto out = l.processFrame(left, right, 1.0);
        assert(out.reason == FailureReason::None);  // 有效帧
        const uint64_t topo1 = l.mapTopologyRevision();
        out = l.processFrame(left, right, 0.5);      // 倒退
        assert(out.reason == FailureReason::TimestampRollback);
        assert(!out.pose_valid);
        assert(l.mapTopologyRevision() == topo1);
    } TEST_PASS();

    TEST("时间戳相等: TimestampRollback 拒绝（严格递增，§4.3）") {
        renderStereoFrame(cam, blks, SE3(), left, right);
        auto out = l.processFrame(left, right, 2.0);
        assert(out.reason == FailureReason::None);
        out = l.processFrame(left, right, 2.0);
        assert(out.reason == FailureReason::TimestampRollback);
    } TEST_PASS();

    TEST("左右尺寸不一致: InvalidInput 拒绝且 Map revision 不变") {
        const uint64_t topo1 = l.mapTopologyRevision();
        cv::Mat small = left(cv::Rect(0, 0, 320, 240));
        auto out = l.processFrame(left, small, 3.0);
        assert(out.reason == FailureReason::InvalidInput);
        assert(l.mapTopologyRevision() == topo1);
    } TEST_PASS();

    TEST("左右类型不一致: InvalidInput 拒绝且 Map revision 不变") {
        const uint64_t topo1 = l.mapTopologyRevision();
        cv::Mat right_c3;
        cv::cvtColor(right, right_c3, cv::COLOR_GRAY2BGR);
        auto out = l.processFrame(left, right_c3, 3.0);
        assert(out.reason == FailureReason::InvalidInput);
        assert(l.mapTopologyRevision() == topo1);
    } TEST_PASS();

    TEST("图像尺寸与标定不一致: InvalidInput 拒绝且 Map revision 不变") {
        const uint64_t topo1 = l.mapTopologyRevision();
        cv::Mat bad_size(240, 320, CV_8UC1, cv::Scalar(0));
        auto out = l.processFrame(bad_size, cv::Mat(), 3.0);
        assert(out.reason == FailureReason::InvalidInput);
        assert(l.mapTopologyRevision() == topo1);
    } TEST_PASS();

    TEST("双目时间差 >1ms: StereoUnsynchronized 拒绝且 Map revision 不变") {
        const uint64_t topo1 = l.mapTopologyRevision();
        renderStereoFrame(cam, blks, SE3(), left, right);
        auto out = l.processFrame(left, right, 4.0, 4.0 + 0.01);  // 10ms 偏差
        assert(out.reason == FailureReason::StereoUnsynchronized);
        assert(l.mapTopologyRevision() == topo1);
    } TEST_PASS();

    TEST("双目时间差 <=1ms: 接受") {
        renderStereoFrame(cam, blks, SE3(), left, right);
        auto out = l.processFrame(left, right, 5.0, 5.0 + 0.0005);  // 0.5ms
        assert(out.reason == FailureReason::None);
    } TEST_PASS();
}

// ============================================================
// 有效输入输出（§4.1/§4.2）与 run_slam 等价（§4.4）
// ============================================================

void test_valid_frames_and_equivalence() {
    auto cam = makeStereoCamera();
    std::vector<Blk> blks;
    makeScene(blks);
    std::vector<SE3> path;
    makePath(path);

    // 旧 run_slam 路径：直接驱动 VisualOdometry（异步/回环关闭，确定性）
    VisualOdometry raw_vo(cam, deterministicConfig());
    Localizer localizer(cam, deterministicConfig());

    // 有效输入会建图 → Map revision 应推进（证明帧确实被接受处理）
    const uint64_t topo0 = localizer.mapTopologyRevision();

    TEST("有效双目序列: 3 帧完整验收后进入 Tracking") {
        for (size_t i = 0; i < path.size(); i++) {
            cv::Mat l, r;
            renderStereoFrame(cam, blks, path[i], l, r);
            const double t = 0.1 * i;
            auto est = localizer.processFrame(l, r, t);
            if (i >= 3) assert(est.state == TrackingState::Tracking);
        }
        assert(localizer.state() == TrackingState::Tracking);
        assert(localizer.mapTopologyRevision() > topo0);  // 建图推进
    } TEST_PASS();

    TEST("run_slam 等价: 每帧 T_wb 与裸 VO 输出平移差 <1e-12 m、旋转差 <1e-12 rad") {
        // 固定 RNG seed（§5.6 deterministic.yaml：solvePnPRansac 内部 RNG 确定性）
        cv::setRNGSeed(0x5A17);
        // 用全新 Localizer 与裸 VO 平行驱动同一路径。
        // 注意：VO 会在 CLAHE 增强时原地修改传入的 cv::Mat 缓冲（已知问题 §3.27），
        // 因此两路必须各渲染一份独立拷贝（与 run_slam 每帧新矩阵的用法一致）。
        Localizer loc2(cam, deterministicConfig());
        VisualOdometry raw2(cam, deterministicConfig());
        for (size_t i = 0; i < path.size(); i++) {
            cv::Mat l_a, r_a, l_b, r_b;
            renderStereoFrame(cam, blks, path[i], l_a, r_a);
            renderStereoFrame(cam, blks, path[i], l_b, r_b);
            const double t = 0.1 * i;
            const SE3 raw_pose = raw2.addFrame(l_a, r_a, t);       // T_cw
            const SE3 raw_Twc = raw_pose.inverse();                // T_wc
            const auto est = loc2.processFrame(l_b, r_b, t);       // T_bc 单位阵 → T_wb = T_wc
            const double trans_diff = (est.T_wb.t - raw_Twc.t).norm();
            const SE3 recomposed = est.T_wo * est.T_ob;
            if (est.pose_valid) {
                assert((recomposed.t - est.T_wb.t).norm() < 1e-9);
                assert(recomposed.q.angularDistance(est.T_wb.q) < 1e-9);
            }
            // 相对旋转角：T_wb · T_cw 应接近单位
            const Eigen::Quaterniond rel = est.T_wb.q * raw_pose.q;
            const double rot_rad = 2.0 * std::acos(std::min(1.0, std::abs(rel.w())));
            assert(trans_diff < 1e-12);
            assert(rot_rad < 1e-12);
        }
    } TEST_PASS();

    TEST("有效帧输出字段完整（timestamp/pose_valid/状态机注释）") {
        Localizer loc3(cam, deterministicConfig());
        cv::Mat l, r;
        renderStereoFrame(cam, blks, path[0], l, r);
        auto est = loc3.processFrame(l, r, 0.0);
        assert(est.sequence == 1);
        assert(est.timestamp == 0.0);
        assert(est.state == TrackingState::Initializing);
        assert(!est.pose_valid);       // 初始化完成前不发布有效位姿（§4.2）
        // 第 2/3 帧：重新渲染新缓冲（VO 的 CLAHE 会原地改写输入，§3.27）
        renderStereoFrame(cam, blks, path[1], l, r);
        est = loc3.processFrame(l, r, 0.1);
        assert(est.state == TrackingState::Initializing);
        renderStereoFrame(cam, blks, path[2], l, r);
        est = loc3.processFrame(l, r, 0.2);
        assert(est.state == TrackingState::Tracking);   // 连续 3 帧完整验收（§4.2）
        assert(est.pose_valid);
        assert(!est.prediction_only);
        assert(est.covariance == vslam::Mat6::Identity());  // M0 占位协方差
    } TEST_PASS();
}

// ============================================================
// 停止与析构（§4.4）
// ============================================================

void test_stop_and_destructor() {
    auto cam = makeStereoCamera();
    std::vector<Blk> blks;
    makeScene(blks);
    std::vector<SE3> path;
    makePath(path);

    TEST("重复 stop() 不崩溃（幂等）") {
        Localizer l(cam, deterministicConfig());
        l.stop();
        l.stop();  // 重复 stop
        assert(l.state() == TrackingState::Stopped);
    } TEST_PASS();

    TEST("stop() 后输入返回 Stopped 且不调用 VO（Map revision 不变）") {
        Localizer l(cam, deterministicConfig());
        cv::Mat lf, r;
        renderStereoFrame(cam, blks, path[0], lf, r);
        l.processFrame(lf, r, 0.0);
        const uint64_t topo1 = l.mapTopologyRevision();
        l.stop();
        auto est = l.processFrame(lf, r, 0.1);
        assert(est.state == TrackingState::Stopped);
        assert(!est.pose_valid);
        assert(l.mapTopologyRevision() == topo1);
    } TEST_PASS();

    TEST("析构不崩溃（作用域结束自动 stop）") {
        {
            Localizer l(cam, deterministicConfig());
            cv::Mat lf, r;
            renderStereoFrame(cam, blks, path[0], lf, r);
            l.processFrame(lf, r, 0.0);
        }
        assert(true);
    } TEST_PASS();
}

// ============================================================
// robot.yaml 配置解析
// ============================================================

void test_robot_yaml() {
    TEST("robot.yaml 解析: 默认 LocalizationOnly + 单位 T_bc + 1ms 双目容差") {
        // 使用默认构造值验证默认值，再验证 fromYaml（路径由 VSLAM_SOURCE_DIR 提供）
        const std::string path =
            std::string(VSLAM_SOURCE_DIR) + "/config/robot.yaml";
        LocalizerConfig cfg = LocalizerConfig::fromYaml(path);
        assert(cfg.mode == LocalizationMode::LocalizationOnly);
        assert(vslam::isUnitQuaternion(cfg.T_bc.q));
        assert(cfg.T_bc.t.norm() < 1e-12);
        assert(std::abs(cfg.stereo_max_time_diff_s - 0.001) < 1e-12);
    } TEST_PASS();

    TEST("缺失 Robot 段返回默认配置") {
        const std::string path =
            std::string(VSLAM_SOURCE_DIR) + "/config/default.yaml";
        LocalizerConfig cfg = LocalizerConfig::fromYaml(path);
        assert(cfg.mode == LocalizationMode::LocalizationOnly);
        assert(cfg.T_bc.q.w() == 1.0);
    } TEST_PASS();
}

int main() {
    std::cout << "[Localizer Contract (M0.3)]\n";
    test_constructor_validation();
    test_input_validation();
    test_valid_frames_and_equivalence();
    test_stop_and_destructor();
    test_robot_yaml();

    std::cout << "\n===== All Tests Completed =====\n";
    return 0;
}
