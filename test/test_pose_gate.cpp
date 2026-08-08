/**
 * test_pose_gate.cpp - PoseGate 纯几何质量与运动连续性单元测试
 *
 * 覆盖 M1.1（PRODUCTION_LOCALIZATION_PLAN §5.2）：
 *   1. pnpReprojectionRmse：内点 RMSE 计算（含空/越界内点）
 *   2. checkMotionContinuity：平移/旋转门限
 *   3. acceptPoseCandidate：几何（内点/比例/RMSE）+ 连续性组合
 *   4. evaluate：统一入口（§5.2 API，值对象），dt 不参与
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_pose_gate
 * 运行: ./build/test_pose_gate（独立 CTest）
 */

#include "vslam/pose_gate.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

// 简单的测试辅助宏（与 test_vo.cpp 保持一致）
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::MotionPrediction;
using vslam::PoseCandidate;
using vslam::PoseGate;
using vslam::PoseQuality;
using vslam::SE3;
using vslam::TrackingQuality;
using vslam::Vec3;

namespace {

constexpr int    kMinInliers = 15;
constexpr double kMinRatio   = 0.3;
constexpr double kMaxRmse    = 2.5;

void test_pnp_reprojection_rmse() {
    TEST("pnpReprojectionRmse: 内点全部匹配时 RMSE≈0") {
        cv::Mat K = (cv::Mat_<double>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);
        std::vector<cv::Point3f> pts3d = {{1, 0, 3}, {0, 1, 4}, {-1, 0, 5}, {0, -1, 3}};
        std::vector<cv::Point2f> pts2d;
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
        cv::projectPoints(pts3d, rvec, tvec, K, cv::Mat(), pts2d);
        std::vector<int> inliers = {0, 1, 2, 3};
        double rmse = PoseGate::pnpReprojectionRmse(pts3d, pts2d, rvec, tvec, inliers, K);
        assert(rmse < 1e-6);
    } TEST_PASS();

    TEST("pnpReprojectionRmse: 内点带噪声时 RMSE 为噪声水平") {
        cv::Mat K = (cv::Mat_<double>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);
        std::vector<cv::Point3f> pts3d = {{1, 0, 3}, {0, 1, 4}, {-1, 0, 5}, {0, -1, 3}};
        std::vector<cv::Point2f> pts2d;
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
        cv::projectPoints(pts3d, rvec, tvec, K, cv::Mat(), pts2d);
        pts2d[0].x += 1.0;  // 1px 噪声
        std::vector<int> inliers = {0, 1, 2, 3};
        double rmse = PoseGate::pnpReprojectionRmse(pts3d, pts2d, rvec, tvec, inliers, K);
        assert(std::abs(rmse - 0.5) < 1e-9);  // sqrt(1/4) = 0.5
    } TEST_PASS();

    TEST("pnpReprojectionRmse: 空内点/越界内点返回 +inf") {
        cv::Mat K = (cv::Mat_<double>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);
        std::vector<cv::Point3f> pts3d = {{1, 0, 3}};
        std::vector<cv::Point2f> pts2d = {{320, 240}};
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
        assert(std::isinf(PoseGate::pnpReprojectionRmse(
            pts3d, pts2d, rvec, tvec, std::vector<int>{}, K)));
        assert(std::isinf(PoseGate::pnpReprojectionRmse(
            pts3d, pts2d, rvec, tvec, std::vector<int>{7}, K)));  // 越界
    } TEST_PASS();
}

void test_check_motion_continuity() {
    TEST("checkMotionContinuity: 平移 2m 通过、4m 拒绝（3m 门限）") {
        const SE3 baseline_twc;  // 相机在原点
        double trans = 0.0, rot = 0.0;
        const SE3 ok_pose(Eigen::Quaterniond::Identity(), Vec3(-2, 0, 0));
        assert(PoseGate::checkMotionContinuity(ok_pose, baseline_twc, 3.0, 0.35, trans, rot));
        assert(std::abs(trans - 2.0) < 1e-9);
        const SE3 bad_pose(Eigen::Quaterniond::Identity(), Vec3(-4, 0, 0));
        assert(!PoseGate::checkMotionContinuity(bad_pose, baseline_twc, 3.0, 0.35, trans, rot));
        assert(std::abs(trans - 4.0) < 1e-9);
    } TEST_PASS();

    TEST("checkMotionContinuity: 旋转 0.2rad 通过、0.5rad 拒绝（0.35rad 门限）") {
        const SE3 baseline_twc;
        double trans = 0.0, rot = 0.0;
        const SE3 ok_pose(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Vec3::UnitZ())), Vec3::Zero());
        assert(PoseGate::checkMotionContinuity(ok_pose, baseline_twc, 3.0, 0.35, trans, rot));
        assert(std::abs(rot - 0.2) < 1e-9);
        const SE3 bad_pose(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Vec3::UnitZ())), Vec3::Zero());
        assert(!PoseGate::checkMotionContinuity(bad_pose, baseline_twc, 3.0, 0.35, trans, rot));
        assert(std::abs(rot - 0.5) < 1e-9);
    } TEST_PASS();
}

void test_accept_pose_candidate() {
    TEST("acceptPoseCandidate: 无基线时几何验收即最终验收") {
        PoseQuality q;
        assert(PoseGate::acceptPoseCandidate(
            SE3(Eigen::Quaterniond::Identity(), Vec3(-2, 0, 0)),
            20, 40, 1.0, kMinInliers, kMinRatio, kMaxRmse,
            std::nullopt, 0.0, 0.0, q));
        assert(q.geometric_ok && q.motion_ok);
        assert(std::abs(q.inlier_ratio - 0.5) < 1e-9);
    } TEST_PASS();

    TEST("acceptPoseCandidate: 内点不足/比例不足/RMSE 超标 → 几何拒绝") {
        const SE3 baseline_twc;
        PoseQuality q;
        // 内点不足（10 < 15）
        assert(!PoseGate::acceptPoseCandidate(
            SE3(), 10, 40, 1.0, kMinInliers, kMinRatio, kMaxRmse,
            baseline_twc, 3.0, 0.35, q));
        assert(!q.geometric_ok && !q.motion_ok);
        // 比例不足（0.2 < 0.3）
        assert(!PoseGate::acceptPoseCandidate(
            SE3(), 10, 50, 1.0, kMinInliers, kMinRatio, kMaxRmse,
            baseline_twc, 3.0, 0.35, q));
        // RMSE 超标（3.0 > 2.5）
        assert(!PoseGate::acceptPoseCandidate(
            SE3(), 20, 40, 3.0, kMinInliers, kMinRatio, kMaxRmse,
            baseline_twc, 3.0, 0.35, q));
    } TEST_PASS();

    TEST("acceptPoseCandidate: 几何通过但连续性失败 → 拒绝（§3.19 回归）") {
        // 几何全优（30/40、rmse 0.5），但距基线 258m（门限 50m）→ 拒绝
        const SE3 baseline_twc(Eigen::Quaterniond::Identity(), Vec3(10, 0, 0));
        PoseQuality q;
        const SE3 far_pose(Eigen::Quaterniond::Identity(), Vec3(-268, 0, 0));
        assert(!PoseGate::acceptPoseCandidate(
            far_pose, 30, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 50.0, 60.0 * M_PI / 180.0, q));
        assert(q.geometric_ok && !q.motion_ok);
        assert(std::abs(q.translation - 258.0) < 1e-6);
    } TEST_PASS();
}

void test_evaluate() {
    TEST("evaluate: 值对象统一入口，几何+连续性通过") {
        PoseGate gate;
        PoseCandidate candidate;
        candidate.pose_cs = SE3(Eigen::Quaterniond::Identity(), Vec3(-2, 0, 0));
        candidate.min_inliers = kMinInliers;
        candidate.min_ratio = kMinRatio;
        candidate.max_rmse = kMaxRmse;
        MotionPrediction prediction;
        prediction.baseline_twc = SE3();
        prediction.max_translation = 3.0;
        prediction.max_rotation = 0.35;
        TrackingQuality quality;
        quality.inliers = 20;
        quality.total = 40;
        quality.rmse = 1.0;

        auto decision = gate.evaluate(candidate, prediction, quality, 0.1);
        assert(decision.accepted);
        assert(decision.quality.geometric_ok && decision.quality.motion_ok);
        assert(std::abs(decision.quality.translation - 2.0) < 1e-9);
    } TEST_PASS();

    TEST("evaluate: 无运动基线时仅几何验收") {
        PoseGate gate;
        PoseCandidate candidate;
        candidate.pose_cs = SE3();
        candidate.min_inliers = kMinInliers;
        candidate.min_ratio = kMinRatio;
        candidate.max_rmse = kMaxRmse;
        TrackingQuality quality;
        quality.inliers = 8;   // < 15 → 几何拒绝
        quality.total = 40;
        quality.rmse = 1.0;
        auto decision = gate.evaluate(candidate, MotionPrediction{}, quality, 0.1);
        assert(!decision.accepted);
        assert(!decision.quality.geometric_ok);
    } TEST_PASS();

    TEST("evaluate: dt 不影响 M1.1 结果（相同输入两种 dt 决策一致）") {
        PoseGate gate;
        PoseCandidate candidate;
        candidate.pose_cs = SE3(Eigen::Quaterniond::Identity(), Vec3(-1, 0, 0));
        candidate.min_inliers = kMinInliers;
        candidate.min_ratio = kMinRatio;
        candidate.max_rmse = kMaxRmse;
        MotionPrediction prediction;
        prediction.baseline_twc = SE3();
        prediction.max_translation = 3.0;
        prediction.max_rotation = 0.35;
        TrackingQuality quality;
        quality.inliers = 20;
        quality.total = 40;
        quality.rmse = 1.0;
        auto d1 = gate.evaluate(candidate, prediction, quality, 0.01);
        auto d2 = gate.evaluate(candidate, prediction, quality, 0.5);
        assert(d1.accepted == d2.accepted);
        assert(d1.quality.translation == d2.quality.translation);
    } TEST_PASS();
}

} // namespace

int main() {
    std::cout << "[PoseGate (M1.1)]\n";
    test_pnp_reprojection_rmse();
    test_check_motion_continuity();
    test_accept_pose_candidate();
    test_evaluate();

    std::cout << "\n===== All Tests Completed =====\n";
    return 0;
}
