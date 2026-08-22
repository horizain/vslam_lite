/**
 * test_pose_covariance.cpp - M3.2 位姿数值协方差单元测试
 *
 * 覆盖 PRODUCTION_LOCALIZATION_PLAN §7.5：
 *   1. se3Exp / se3Adjoint：SE(3) 指数映射与左伴随（切空间 [tx,ty,tz,rx,ry,rz]）
 *   2. pnpPoseCovariance：中心有限差分 Jacobian → σ²H⁻¹（σ² 下限、特征截断、
 *      cond(H)>1e8 判退化），噪声二次缩放 / 点数缩放 / 确定性
 *   3. FrontendTracker 集成：接受的 PnP 位姿携带有效正定协方差
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_pose_covariance
 * 运行: ./build/test_pose_covariance（独立 CTest）
 */

#include "vslam/camera.h"
#include "vslam/frontend_tracker.h"
#include "vslam/localization_types.h"
#include "vslam/pose_covariance.h"
#include "vslam/relocalizer.h"
#include "vslam/vo.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using namespace vslam;

namespace {

constexpr double kFx = 500.0;
constexpr double kFy = 500.0;
constexpr double kCx = 320.0;
constexpr double kCy = 240.0;

Mat33 testK() {
    Mat33 K = Mat33::Identity();
    K(0, 0) = kFx; K(0, 2) = kCx;
    K(1, 1) = kFy; K(1, 2) = kCy;
    return K;
}

cv::Mat eigenToCv(const Mat33& m) {
    cv::Mat out(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out.at<double>(r, c) = m(r, c);
    return out;
}

/// 合成场景：GT 位姿（T_cw 语义）+ 前方 3D 点 + 投影像素（可选噪声缩放）
struct Scene {
    SE3 gt_pose_cs;
    std::vector<cv::Point3f> points;
    std::vector<cv::Point2f> pixels;
};

Scene makeScene(int n, unsigned seed) {
    Scene s;
    // GT 位姿与投影使用同一 rvec/tvec（Rodrigues 精确派生），保证无噪场景
    // 残差只受像素 float 存储量化影响
    const cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0.03, -0.05, 0.02);
    const cv::Mat tvec = (cv::Mat_<double>(3, 1) << -0.2, 0.15, -0.25);
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Mat33 R = Mat33::Identity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = R_cv.at<double>(r, c);
    s.gt_pose_cs = SE3(Eigen::Quaterniond(R),
                       Vec3(tvec.at<double>(0), tvec.at<double>(1),
                            tvec.at<double>(2)));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(-2.0, 2.0);
    std::uniform_real_distribution<double> uy(-1.5, 1.5);
    std::uniform_real_distribution<double> uz(4.0, 9.0);
    for (int i = 0; i < n; ++i)
        s.points.emplace_back(static_cast<float>(ux(rng)),
                              static_cast<float>(uy(rng)),
                              static_cast<float>(uz(rng)));
    // projectPoints 的 p_c = R·p_w + t 与 T_cs 语义一致
    cv::projectPoints(s.points, rvec, tvec, eigenToCv(testK()), cv::Mat(),
                      s.pixels);
    return s;
}

/// 给像素加确定性高斯噪声：返回标准化样本，供多组实验复用同一几何扰动
std::vector<std::pair<double, double>> standardizedDraws(
    const std::vector<cv::Point2f>& pixels, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<std::pair<double, double>> draws;
    draws.reserve(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i)
        draws.emplace_back(normal(rng), normal(rng));
    return draws;
}

std::vector<cv::Point2f> applyNoise(
    const std::vector<cv::Point2f>& pixels,
    const std::vector<std::pair<double, double>>& draws, double sigma_px) {
    assert(pixels.size() == draws.size());
    std::vector<cv::Point2f> noisy;
    noisy.reserve(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i)
        noisy.emplace_back(static_cast<float>(pixels[i].x + sigma_px * draws[i].first),
                           static_cast<float>(pixels[i].y + sigma_px * draws[i].second));
    return noisy;
}

} // namespace

void test_se3_exp_adjoint() {
    TEST("se3Exp: 零向量 → 单位位姿") {
        Eigen::Matrix<double, 6, 1> xi = Eigen::Matrix<double, 6, 1>::Zero();
        const SE3 T = se3Exp(xi);
        assert(std::abs(T.t.x()) < 1e-12 && std::abs(T.t.y()) < 1e-12 &&
               std::abs(T.t.z()) < 1e-12);
        const Mat44 M = T.matrix();
        assert((M - Mat44::Identity()).norm() < 1e-12);
    } TEST_PASS();

    TEST("se3Exp: 纯平移 → t 直通、旋转为单位") {
        Eigen::Matrix<double, 6, 1> xi;
        xi << 1.0, -2.0, 3.0, 0.0, 0.0, 0.0;
        const SE3 T = se3Exp(xi);
        assert((T.t - Vec3(1, -2, 3)).norm() < 1e-12);
        assert((T.q.toRotationMatrix() - Mat33::Identity()).norm() < 1e-12);
    } TEST_PASS();

    TEST("se3Exp: 绕 z 转 π/2 → x 轴映到 y 轴") {
        Eigen::Matrix<double, 6, 1> xi;
        xi << 0, 0, 0, 0.0, 0.0, M_PI / 2;
        const SE3 T = se3Exp(xi);
        const Vec3 px = T * Vec3(1, 0, 0);
        assert((px - Vec3(0, 1, 0)).norm() < 1e-12);
    } TEST_PASS();

    TEST("se3Adjoint: 单位变换 → 单位矩阵；块结构 [[R,[t]^R],[0,R]]") {
        assert((se3Adjoint(SE3()) - Mat6::Identity()).norm() < 1e-12);
        const SE3 T(Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Vec3::UnitY())),
                    Vec3(0.4, -0.5, 0.6));
        const Mat33 R = T.q.toRotationMatrix();
        Mat6 expect = Mat6::Zero();
        expect.block<3, 3>(0, 0) = R;
        Mat33 tx_r;
        tx_r << 0, -T.t.z(), T.t.y(),
            T.t.z(), 0, -T.t.x(),
            -T.t.y(), T.t.x(), 0;
        expect.block<3, 3>(0, 3) = tx_r * R;
        expect.block<3, 3>(3, 3) = R;
        assert((se3Adjoint(T) - expect).norm() < 1e-12);
    } TEST_PASS();

    TEST("se3Adjoint: 同态性 Ad_AB = A·B；共轭 Exp(δ) 共轭 ≈ Exp(Ad·δ)") {
        const SE3 A(Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Vec3::UnitX())),
                    Vec3(0.3, -0.2, 0.5));
        const SE3 B(Eigen::Quaterniond(Eigen::AngleAxisd(-0.4, Vec3::UnitZ())),
                    Vec3(-0.6, 0.1, 0.25));
        assert((se3Adjoint(A * B) - se3Adjoint(A) * se3Adjoint(B)).norm() < 1e-12);

        Eigen::Matrix<double, 6, 1> dxi;
        dxi << 0.01, -0.02, 0.015, 0.002, 0.001, -0.0015;
        const SE3 lhs = A * se3Exp(dxi) * A.inverse();
        const SE3 rhs = se3Exp(se3Adjoint(A) * dxi);
        assert((lhs.t - rhs.t).norm() < 1e-9 &&
               lhs.q.angularDistance(rhs.q) < 1e-9);
    } TEST_PASS();
}

void test_pnp_covariance_basics() {
    TEST("pnpPoseCovariance: 无噪合成场景 → 有效/非退化/对称正定/σ² 取下限") {
        const Scene s = makeScene(80, 42);
        std::vector<int> inliers(s.points.size());
        for (size_t i = 0; i < s.points.size(); ++i) inliers[i] = static_cast<int>(i);
        const PoseCovarianceResult r = pnpPoseCovariance(
            s.gt_pose_cs, s.points, s.pixels, inliers, testK());
        assert(r.valid);
        assert(!r.degenerate);
        assert(std::isfinite(r.condition) && r.condition < 1e8);
        assert(std::abs(r.sigma2 - 0.25) < 1e-12);   // SSE=0 → σ² 下限
        assert(r.covariance_cs.allFinite());
        assert((r.covariance_cs - r.covariance_cs.transpose()).norm() <
               1e-12 * std::max(1.0, r.covariance_cs.norm()));
        assert(isPositiveDefiniteCovariance(r.covariance_cs));
        assert(r.covariance_cs.diagonal().minCoeff() > 0.0);
    } TEST_PASS();

    TEST("pnpPoseCovariance: 噪声二次缩放（同一标准化样本 σ=1 vs σ=2）") {
        const Scene s = makeScene(120, 7);
        std::vector<int> inliers(s.points.size());
        for (size_t i = 0; i < s.points.size(); ++i) inliers[i] = static_cast<int>(i);
        const auto draws = standardizedDraws(s.pixels, 99);
        const PoseCovarianceResult r1 = pnpPoseCovariance(
            s.gt_pose_cs, s.points, applyNoise(s.pixels, draws, 1.0), inliers, testK());
        const PoseCovarianceResult r2 = pnpPoseCovariance(
            s.gt_pose_cs, s.points, applyNoise(s.pixels, draws, 2.0), inliers, testK());
        assert(r1.valid && r2.valid);
        // J 只依赖几何，σ² 随像素噪声线性进入 → 协方差整体 ×4
        const double sig_ratio = r2.sigma2 / r1.sigma2;
        assert(sig_ratio > 3.0 && sig_ratio < 5.0);
        const double trace_ratio = r2.covariance_cs.trace() / r1.covariance_cs.trace();
        assert(trace_ratio > 2.5 && trace_ratio < 5.5);
    } TEST_PASS();

    TEST("pnpPoseCovariance: 点数增加 → 信息增加 → 迹变小") {
        const Scene small = makeScene(40, 11);
        const Scene large = makeScene(160, 11);
        const auto draws_small = standardizedDraws(small.pixels, 13);
        const auto draws_large = standardizedDraws(large.pixels, 13);
        std::vector<int> si(small.points.size()), li(large.points.size());
        for (int i = 0; i < (int)small.points.size(); ++i) si[i] = i;
        for (int i = 0; i < (int)large.points.size(); ++i) li[i] = i;
        const PoseCovarianceResult r_small = pnpPoseCovariance(
            small.gt_pose_cs, small.points,
            applyNoise(small.pixels, draws_small, 1.0), si, testK());
        const PoseCovarianceResult r_large = pnpPoseCovariance(
            large.gt_pose_cs, large.points,
            applyNoise(large.pixels, draws_large, 1.0), li, testK());
        assert(r_small.valid && r_large.valid);
        assert(r_large.covariance_cs.trace() < r_small.covariance_cs.trace());
    } TEST_PASS();

    TEST("pnpPoseCovariance: 视轴共线点 → 深度不可观 → 判退化拒绝发布") {
        Scene s = makeScene(30, 21);
        s.points.clear();
        for (int i = 0; i < 30; ++i)
            s.points.emplace_back(0.f, 0.f, 4.0f + 0.15f * i);  // 全在光轴上
        const cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0.03, -0.05, 0.02);
        const cv::Mat tvec = (cv::Mat_<double>(3, 1) << -0.2, 0.15, -0.25);
        std::vector<cv::Point2f> pix;
        cv::projectPoints(s.points, rvec, tvec,
                          eigenToCv(testK()), cv::Mat(), pix);
        std::vector<int> inliers(s.points.size());
        for (size_t i = 0; i < s.points.size(); ++i) inliers[i] = static_cast<int>(i);
        const PoseCovarianceResult r = pnpPoseCovariance(
            s.gt_pose_cs, s.points, pix, inliers, testK());
        // 深度/横向耦合方向病态：必须判退化且不得给出"可用"协方差
        assert(!r.valid);
        assert(r.degenerate || !isPositiveDefiniteCovariance(r.covariance_cs));
    } TEST_PASS();

    TEST("pnpPoseCovariance: 内点不足 6 → 直接退化") {
        const Scene s = makeScene(10, 5);
        const std::vector<int> few = {0, 1, 2};
        const PoseCovarianceResult r = pnpPoseCovariance(
            s.gt_pose_cs, s.points, s.pixels, few, testK());
        assert(!r.valid && r.degenerate);
    } TEST_PASS();

    TEST("pnpPoseCovariance: 相同输入逐位确定") {
        const Scene s = makeScene(60, 31);
        const auto draws = standardizedDraws(s.pixels, 17);
        const auto noisy = applyNoise(s.pixels, draws, 1.0);
        std::vector<int> inliers(s.points.size());
        for (size_t i = 0; i < s.points.size(); ++i) inliers[i] = static_cast<int>(i);
        const PoseCovarianceResult a = pnpPoseCovariance(
            s.gt_pose_cs, s.points, noisy, inliers, testK());
        const PoseCovarianceResult b = pnpPoseCovariance(
            s.gt_pose_cs, s.points, noisy, inliers, testK());
        assert(a.valid && b.valid);
        assert(a.sigma2 == b.sigma2 && a.condition == b.condition);
        assert(a.covariance_cs == b.covariance_cs);
    } TEST_PASS();
}

void test_frontend_integration() {
    TEST("trackPnP: 接受位姿携带有效正定协方差") {
        auto cam = std::make_shared<MonocularCamera>();
        cam->fx = kFx; cam->fy = kFy; cam->cx = kCx; cam->cy = kCy;
        cam->img_width = 640; cam->img_height = 480;
        FrontendTracker tracker(cam);

        const Scene s = makeScene(100, 23);
        const auto draws = standardizedDraws(s.pixels, 19);
        const auto noisy = applyNoise(s.pixels, draws, 0.6);

        // 用带噪像素做一次 PnP，再对收敛位姿求协方差
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        const bool ok = cv::solvePnPRansac(
            s.points, noisy, eigenToCv(testK()),
            cv::Mat(), rvec, tvec, false, 200, 3.0, 0.99, inliers);
        assert(ok && (int)inliers.size() >= 15);
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        const SE3 pose_cs = Relocalizer::matToSE3(R, tvec);
        const PoseCovarianceResult cov =
            tracker.estimatePnPCovariance(pose_cs, s.points, noisy, inliers);
        assert(cov.valid && !cov.degenerate);
        assert(isPositiveDefiniteCovariance(cov.covariance_cs));
        assert(cov.covariance_cs.trace() > 0.0);
    } TEST_PASS();
}

/// 端到端传播：真实 VO 跟踪建立后 Status 必须携带有效数值协方差
///（trackPnP/refinePnP → TrackingResult → Status → Localizer 链路的 VO 侧）。
void test_vo_status_propagation() {
    TEST("VO Status: 跟踪建立后 pose_covariance_valid=true 且非占位单位阵") {
        auto cam = std::make_shared<MonocularCamera>();
        cam->fx = 350; cam->fy = 350;
        cam->cx = 320; cam->cy = 188;
        cam->img_width = 640; cam->img_height = 376;
        const cv::Mat K = cam->K();

        // 带深度变化的 3D 块场景（避免平面+平移的本质矩阵退化）
        struct Blk { cv::Point3f c; float sx, sy; int gray; };
        std::vector<Blk> blks;
        std::mt19937 rng(29);
        std::uniform_real_distribution<double> dx(-2.0, 2.0), dy(-1.0, 1.0),
            dz(3.0, 7.0), ds(0.25, 0.7);
        for (int i = 0; i < 90; ++i)
            blks.push_back({cv::Point3f(dx(rng), dy(rng), dz(rng)),
                            (float)ds(rng) * 1.5f, (float)ds(rng), 64 + i * 2});
        const auto render = [&](double tx) {
            cv::Mat img(376, 640, CV_8UC1, cv::Scalar(64));
            const cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            const cv::Mat tvec = (cv::Mat_<double>(3, 1) << -tx, 0.0, 0.0);
            for (const auto& b : blks) {
                std::vector<cv::Point3f> corners = {
                    {b.c.x - b.sx / 2, b.c.y - b.sy / 2, b.c.z},
                    {b.c.x + b.sx / 2, b.c.y - b.sy / 2, b.c.z},
                    {b.c.x + b.sx / 2, b.c.y + b.sy / 2, b.c.z},
                    {b.c.x - b.sx / 2, b.c.y + b.sy / 2, b.c.z}};
                std::vector<cv::Point2f> px;
                cv::projectPoints(corners, rvec, tvec, K, cv::Mat(), px);
                std::vector<cv::Point> poly;
                for (const auto& q : px) poly.emplace_back(cvRound(q.x), cvRound(q.y));
                cv::fillConvexPoly(img, poly, cv::Scalar(b.gray));
            }
            return img;
        };

        VOConfig cfg;
        cfg.feature_method = 0;
        VisualOdometry vo(cam, cfg);

        bool tracked = false;
        for (int i = 0; i < 14; ++i) {
            vo.addFrame(render(0.25 * i), i * 0.1);
            tracked = tracked ||
                      (vo.state() == VisualOdometry::State::TRACKING &&
                       vo.getStatus().pose_valid);
        }
        assert(tracked);  // 非空工作证明

        // 连续多帧都必须携带有效、正定、非占位的协方差。
        // 例外：EPIPOLAR 回退帧（recoverPose 无尺度、无内点集）按设计无数值
        // 协方差——Localizer 对这些帧回退保守占位单位阵。
        int pnp_frames = 0;
        for (int k = 14; k < 20 && pnp_frames < 3; ++k) {
            vo.addFrame(render(0.25 * k), k * 0.1);
            const VisualOdometry::Status st = vo.getStatus();
            if (!st.pose_valid) continue;
            if (st.pose_method != "PNP" && st.pose_method != "LK_PNP") continue;
            ++pnp_frames;
            assert(st.pose_covariance_valid);
            assert(st.pose_covariance.allFinite());
            assert((st.pose_covariance - st.pose_covariance.transpose()).norm() <
                   1e-9 * std::max(1.0, st.pose_covariance.norm()));
            assert(isPositiveDefiniteCovariance(st.pose_covariance));
            assert((st.pose_covariance - Mat6::Identity()).norm() > 1e-3);  // 非占位
        }
        assert(pnp_frames >= 1);  // 必须至少覆盖一个 PnP 跟踪帧
    } TEST_PASS();
}

int main() {
    std::cout << "== test_pose_covariance ==" << std::endl;
    test_se3_exp_adjoint();
    test_pnp_covariance_basics();
    test_frontend_integration();
    test_vo_status_propagation();
    std::cout << "ALL PASSED" << std::endl;
    return 0;
}
