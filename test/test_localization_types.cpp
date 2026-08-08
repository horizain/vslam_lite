/**
 * test_localization_types.cpp - 定位契约类型单元测试
 *
 * 覆盖 M0.1 类型契约：
 *   1. LocalizationMode / TrackingState / FailureReason 枚举
 *   2. PoseEstimate 默认值与字段语义
 *   3. 输出质量契约谓词（§3 硬不变量静态部分）
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_localization_types
 * 运行: ./build/test_localization_types（独立 CTest，不并入 test_vo.cpp）
 */

#include "vslam/localization_types.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

// 简单的测试辅助宏（与 test_vo.cpp 保持一致）
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::Mat6;
using vslam::PoseEstimate;
using vslam::SE3;
using vslam::Vec3;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// ============================================================
// 枚举契约
// ============================================================

void test_enums() {
    TEST("LocalizationMode 四个模式可寻址可比较") {
        vslam::LocalizationMode m = vslam::LocalizationMode::OdometryOnly;
        m = vslam::LocalizationMode::Mapping;
        m = vslam::LocalizationMode::LocalizationOnly;
        m = vslam::LocalizationMode::MapMaintenance;
        assert(m == vslam::LocalizationMode::MapMaintenance);
        assert(m != vslam::LocalizationMode::OdometryOnly);
    } TEST_PASS();

    TEST("TrackingState 六个状态可寻址可比较") {
        vslam::TrackingState s = vslam::TrackingState::Initializing;
        s = vslam::TrackingState::Tracking;
        s = vslam::TrackingState::Degraded;
        s = vslam::TrackingState::Relocalizing;
        s = vslam::TrackingState::Lost;
        s = vslam::TrackingState::Stopped;
        assert(s == vslam::TrackingState::Stopped);
        assert(s != vslam::TrackingState::Initializing);
    } TEST_PASS();

    TEST("FailureReason 十一种原因码可寻址") {
        vslam::FailureReason r = vslam::FailureReason::None;
        r = vslam::FailureReason::InvalidInput;
        r = vslam::FailureReason::TimestampRollback;
        r = vslam::FailureReason::StereoUnsynchronized;
        r = vslam::FailureReason::InsufficientFeatures;
        r = vslam::FailureReason::GeometricRejection;
        r = vslam::FailureReason::MotionDiscontinuity;
        r = vslam::FailureReason::RelocalizationTimeout;
        r = vslam::FailureReason::BackendOverloaded;
        r = vslam::FailureReason::MapIncompatible;
        r = vslam::FailureReason::InternalError;
        assert(r == vslam::FailureReason::InternalError);
    } TEST_PASS();
}

// ============================================================
// PoseEstimate 字段契约
// ============================================================

void test_pose_estimate_default() {
    TEST("PoseEstimate 默认值（§4.1 初值）") {
        PoseEstimate p;
        assert(p.sequence == 0);
        assert(p.timestamp == 0.0);
        assert(p.state == vslam::TrackingState::Initializing);
        assert(p.reason == vslam::FailureReason::None);
        assert(!p.pose_valid);
        assert(!p.prediction_only);
        assert(p.map_generation == 0);
        assert(p.covariance == Mat6::Zero());
        assert(p.T_ob.q.w() == 1.0 && p.T_ob.t.norm() == 0.0);
        assert(p.T_wb.q.w() == 1.0 && p.T_wb.t.norm() == 0.0);
    } TEST_PASS();

    TEST("PoseEstimate 全字段赋值与回读") {
        PoseEstimate p;
        p.sequence = 42;
        p.timestamp = 123.456;
        p.T_ob = SE3(Eigen::Quaterniond::Identity(), Vec3(1, 2, 3));
        p.T_wb = SE3(Eigen::Quaterniond(
                         Eigen::AngleAxisd(0.5, Vec3::UnitZ())),
                     Vec3(0.5, 0.25, 0.125));
        p.covariance = Mat6::Identity() * 2.0;
        p.state = vslam::TrackingState::Degraded;
        p.reason = vslam::FailureReason::MotionDiscontinuity;
        p.pose_valid = true;
        p.prediction_only = false;
        p.map_generation = 7;

        assert(p.sequence == 42);
        assert(p.timestamp == 123.456);
        assert(p.T_ob.t == Vec3(1, 2, 3));
        assert(std::abs(p.T_wb.t.z() - 0.125) < 1e-12);
        assert(p.covariance(0, 0) == 2.0 && p.covariance(5, 5) == 2.0);
        assert(p.state == vslam::TrackingState::Degraded);
        assert(p.reason == vslam::FailureReason::MotionDiscontinuity);
        assert(p.pose_valid && !p.prediction_only);
        assert(p.map_generation == 7);
    } TEST_PASS();
}

// ============================================================
// 输出质量契约谓词（§3 硬不变量静态部分）
// ============================================================

void test_is_unit_quaternion() {
    TEST("isUnitQuaternion: 单位模长通过") {
        assert(vslam::isUnitQuaternion(Eigen::Quaterniond::Identity()));
        Eigen::Quaterniond q(Eigen::AngleAxisd(0.7, Vec3(1, 2, 3).normalized()));
        q.normalize();
        assert(vslam::isUnitQuaternion(q));
    } TEST_PASS();

    TEST("isUnitQuaternion: 非单位模长拒绝") {
        Eigen::Quaterniond q(Eigen::Quaterniond::Identity());
        q.coeffs() *= 0.5;
        assert(!vslam::isUnitQuaternion(q));
        q.coeffs() *= 3.0;  // 重新放大回非单位
        assert(!vslam::isUnitQuaternion(q));
    } TEST_PASS();

    TEST("isUnitQuaternion: NaN/Inf 拒绝") {
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        q.x() = kNaN;
        assert(!vslam::isUnitQuaternion(q));
        q = Eigen::Quaterniond::Identity();
        q.w() = kInf;
        assert(!vslam::isUnitQuaternion(q));
    } TEST_PASS();
}

void test_is_finite_se3() {
    TEST("isFinite(SE3): 正常位姿通过") {
        assert(vslam::isFinite(SE3()));
        SE3 T(Eigen::Quaterniond::Identity(), Vec3(1.5, -2.0, 3.25));
        assert(vslam::isFinite(T));
    } TEST_PASS();

    TEST("isFinite(SE3): NaN/Inf 拒绝") {
        SE3 T;
        T.t.x() = kNaN;
        assert(!vslam::isFinite(T));
        T = SE3();
        T.q.z() = kInf;
        assert(!vslam::isFinite(T));
    } TEST_PASS();
}

void test_is_valid_timestamp() {
    TEST("isValidTimestamp: 有限时间戳通过") {
        assert(vslam::isValidTimestamp(0.0));
        assert(vslam::isValidTimestamp(1e9));
        assert(vslam::isValidTimestamp(-0.5));
    } TEST_PASS();

    TEST("isValidTimestamp: NaN/Inf 拒绝") {
        assert(!vslam::isValidTimestamp(kNaN));
        assert(!vslam::isValidTimestamp(kInf));
        assert(!vslam::isValidTimestamp(-kInf));
    } TEST_PASS();
}

void test_positive_definite_covariance() {
    TEST("isPositiveDefiniteCovariance: 正定通过") {
        assert(vslam::isPositiveDefiniteCovariance(Mat6::Identity()));
        Mat6 cov = Mat6::Identity();
        cov(0, 0) = 2.0;
        cov(3, 3) = 0.5;
        assert(vslam::isPositiveDefiniteCovariance(cov));
    } TEST_PASS();

    TEST("isPositiveDefiniteCovariance: 非正定拒绝") {
        assert(!vslam::isPositiveDefiniteCovariance(Mat6::Zero()));
        Mat6 cov = Mat6::Identity();
        cov(0, 0) = -1.0;
        assert(!vslam::isPositiveDefiniteCovariance(cov));
        cov = Mat6::Zero();
        cov(1, 1) = 1.0;  // 特征值含 0
        assert(!vslam::isPositiveDefiniteCovariance(cov));
    } TEST_PASS();

    TEST("isPositiveDefiniteCovariance: 非对称/非有限拒绝") {
        Mat6 cov = Mat6::Identity();
        cov(0, 1) = 0.5;  // 破坏对称
        assert(!vslam::isPositiveDefiniteCovariance(cov));
        cov = Mat6::Identity();
        cov(4, 4) = kNaN;
        assert(!vslam::isPositiveDefiniteCovariance(cov));
    } TEST_PASS();
}

void test_is_publishable() {
    TEST("isPublishable: 默认输出通过（pose_valid=false）") {
        assert(vslam::isPublishable(PoseEstimate()));
    } TEST_PASS();

    TEST("isPublishable: 有效位姿 + 正定协方差通过") {
        PoseEstimate p;
        p.pose_valid = true;
        p.covariance = Mat6::Identity();
        assert(vslam::isPublishable(p));
        p.prediction_only = true;  // 预测标记不影响静态校验
        assert(vslam::isPublishable(p));
    } TEST_PASS();

    TEST("isPublishable: pose_valid 但协方差非正定拒绝（§3-1）") {
        PoseEstimate p;
        p.pose_valid = true;
        p.covariance = Mat6::Zero();
        assert(!vslam::isPublishable(p));
    } TEST_PASS();

    TEST("isPublishable: NaN/非单位四元数/非法时间戳拒绝") {
        PoseEstimate p;
        p.T_ob.t.x() = kNaN;
        assert(!vslam::isPublishable(p));

        p = PoseEstimate();
        p.T_wb.q.coeffs() *= 2.0;
        assert(!vslam::isPublishable(p));

        p = PoseEstimate();
        p.timestamp = kInf;
        assert(!vslam::isPublishable(p));
    } TEST_PASS();
}

} // namespace

int main() {
    std::cout << "[Localization Types (M0.1)]\n";
    test_enums();
    test_pose_estimate_default();
    test_is_unit_quaternion();
    test_is_finite_se3();
    test_is_valid_timestamp();
    test_positive_definite_covariance();
    test_is_publishable();

    std::cout << "\n===== All Tests Completed =====\n";
    return 0;
}
