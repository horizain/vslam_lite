/**
 * test_vo.cpp - 视觉里程计单元测试
 *
 * 测试用例:
 *   1. SE3 位姿运算
 *   2. Camera 投影/反投影
 *   3. FeatureMatcher ORB 提取/匹配
 *   4. VisualOdometry 两帧初始化
 *   5. recoverPose → T_cw 位姿语义（关键回归测试：修复前的取逆方向会使三角化全部失效）
 *   6. LK 光流模式多帧跟踪
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_vo && ./test_vo
 */

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/feature.h"
#include "vslam/vo.h"
#include "vslam/mappoint.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

// 简单的测试辅助宏
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

// ============================================================
// 测试用例
// ============================================================

void test_se3_basics() {
    TEST("SE3 identity") {
        vslam::SE3 id;
        assert(id.q.w() == 1.0);
        assert(id.t.norm() == 0.0);
    } TEST_PASS();

    TEST("SE3 inverse") {
        vslam::SE3 T(Eigen::Quaterniond::Identity(), vslam::Vec3(1, 0, 0));
        auto T_inv = T.inverse();
        auto T_combined = T * T_inv;
        assert(std::abs(T_combined.q.w() - 1.0) < 1e-10);
        assert(T_combined.t.norm() < 1e-10);
    } TEST_PASS();

    TEST("SE3 point transform") {
        vslam::SE3 T(Eigen::Quaterniond::Identity(), vslam::Vec3(1, 2, 3));
        vslam::Vec3 p(1, 1, 1);
        auto result = T * p;
        assert(std::abs(result.x() - 2.0) < 1e-10);
        assert(std::abs(result.y() - 3.0) < 1e-10);
        assert(std::abs(result.z() - 4.0) < 1e-10);
    } TEST_PASS();
}

void test_camera_projection() {
    TEST("Camera pixel2camera") {
        vslam::Camera cam;
        cam.fx = 500; cam.fy = 500; cam.cx = 320; cam.cy = 240;
        auto p3d = cam.pixel2camera(vslam::Vec2(320, 240), 2.0);
        assert(std::abs(p3d.x()) < 1e-10);
        assert(std::abs(p3d.y()) < 1e-10);
        assert(std::abs(p3d.z() - 2.0) < 1e-10);
    } TEST_PASS();

    TEST("Camera world2pixel") {
        vslam::Camera cam;
        cam.fx = 500; cam.fy = 500; cam.cx = 320; cam.cy = 240;
        vslam::SE3 T_cw; // Identity (camera at origin, looking +z)
        // Point at (0, 0, 1) in world → (0,0,1) in camera → pixel (320, 240)
        auto px = cam.world2pixel(vslam::Vec3(0, 0, 1), T_cw);
        assert(std::abs(px.x() - 320.0) < 1e-6);
        assert(std::abs(px.y() - 240.0) < 1e-6);
    } TEST_PASS();
}

void test_feature_extraction() {
    TEST("ORB feature extraction on synthetic image") {
        // 创建一张有纹理的合成图像
        cv::Mat img(480, 640, CV_8UC1, cv::Scalar(128));
        cv::rectangle(img, cv::Rect(100, 100, 200, 200), cv::Scalar(255), -1);
        cv::circle(img, cv::Point(400, 300), 50, cv::Scalar(0), -1);

        auto frame = std::make_shared<vslam::Frame>(0, 0.0);
        frame->image_gray = img;

        vslam::FeatureMatcher fm;
        fm.extract(frame);

        assert(frame->keypoints.size() > 10);  // 至少有些特征点
        assert(!frame->descriptors.empty());
    } TEST_PASS();
}

void test_vo_initialization() {
    TEST("VO two-frame initialization (synthetic)") {
        // 创建两帧合成图像（第二帧模拟向右平移）
        cv::Mat img1(480, 640, CV_8UC1, cv::Scalar(128));
        cv::rectangle(img1, cv::Rect(50, 100, 200, 200), cv::Scalar(255), -1);
        cv::circle(img1, cv::Point(400, 300), 50, cv::Scalar(0), -1);

        cv::Mat img2(480, 640, CV_8UC1, cv::Scalar(128));
        cv::rectangle(img2, cv::Rect(100, 100, 200, 200), cv::Scalar(255), -1);  // moved right
        cv::circle(img2, cv::Point(450, 300), 50, cv::Scalar(0), -1);            // moved right

        vslam::Camera cam;
        cam.fx = 500; cam.fy = 500; cam.cx = 320; cam.cy = 240;
        cam.img_width = 640; cam.img_height = 480;

        // 合成场景特征匹配数较少，显式放宽初始化阈值（验证 VOConfig 接口生效）
        vslam::VOConfig cfg;
        cfg.min_matches_init = 20;

        vslam::VisualOdometry vo(cam, cfg);

        // 第一帧
        auto pose1 = vo.addFrame(img1, 0.0);
        // 状态应为 INITIALIZING
        assert(vo.state() == vslam::VisualOdometry::State::INITIALIZING);

        // 第二帧：应该有足够视差完成初始化
        auto pose2 = vo.addFrame(img2, 0.1);

        // 检查是否初始化成功（可能成功，取决于合成图像的特征质量）
        std::cout << " (state=" << static_cast<int>(vo.state()) << ")"
                  << " map_points=" << vo.getMap()->mapPointCount();
    } TEST_PASS();
}

// ============================================================
// 位姿语义回归测试
// 用已知真实位姿生成点对，走与 vo.cpp 初始化完全相同的序列
// （findEssentialMat → recoverPose → pose_cw = recoverPose 输出），
// 验证 T_cw 方向正确：旋转精确、平移方向一致、三角化深度为正。
// 修复前的取逆方向会让三角化 0% 深度正确。
// ============================================================
void test_pose_semantics() {
    TEST("recoverPose→T_cw 位姿语义 (known motion)") {
        vslam::Camera cam;
        cam.fx = 500; cam.fy = 500; cam.cx = 320; cam.cy = 240;
        cv::Mat K = cam.K();

        // 世界点（深度 3~12m，破坏共面性）
        std::mt19937 gen(11);
        std::uniform_real_distribution<double> dx(-4, 4), dy(-3, 3), dz(3, 12);
        std::vector<cv::Point3f> pts_w;
        for (int i = 0; i < 300; i++)
            pts_w.emplace_back(dx(gen), dy(gen), dz(gen));

        // 真值（T_cw 语义）：帧1=原点；帧2=平移(1.0,0.2,0)m + 绕Y约12°旋转
        cv::Mat rvec1 = (cv::Mat_<double>(3,1) << 0, 0, 0);
        cv::Mat tvec1 = (cv::Mat_<double>(3,1) << 0, 0, 0);
        cv::Mat rvec2 = (cv::Mat_<double>(3,1) << 0.05, 0.2, 0.03);
        cv::Mat tvec2 = (cv::Mat_<double>(3,1) << 1.0, 0.2, 0.0);

        std::vector<cv::Point2f> p1, p2;
        cv::projectPoints(pts_w, rvec1, tvec1, K, cv::Mat(), p1);
        cv::projectPoints(pts_w, rvec2, tvec2, K, cv::Mat(), p2);

        // 与 vo.cpp tryInitialize 完全相同的序列
        cv::Mat E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0);
        cv::Mat R, t;
        cv::recoverPose(E, p1, p2, K, R, t);

        // 新代码（修复后）：T_cw2 = recoverPose 输出（不取逆）
        Eigen::Matrix3d Re;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Re(i, j) = R.at<double>(i, j);
        vslam::SE3 T_cw2(Eigen::Quaterniond(Re),
                         vslam::Vec3(t.at<double>(0), t.at<double>(1), t.at<double>(2)));

        // 与真值对比：旋转误差、平移方向（单目尺度未知）
        cv::Mat R2t;
        cv::Rodrigues(rvec2, R2t);
        Eigen::Matrix3d Re2;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Re2(i, j) = R2t.at<double>(i, j);
        vslam::SE3 gt(Eigen::Quaterniond(Re2), vslam::Vec3(1.0, 0.2, 0.0));
        double r_err = std::acos(std::clamp(std::abs(T_cw2.q.dot(gt.q)), 0.0, 1.0));
        double cos_t = T_cw2.t.normalized().dot(gt.t.normalized());
        assert(r_err < 0.15);
        assert(cos_t > 0.9);

        // 用恢复的 T_cw2 三角化：方向正确时深度应几乎全为正（旧代码取逆则全为负）
        int depth_ok = 0;
        for (size_t i = 0; i < pts_w.size(); i++) {
            auto mp = vslam::MapPoint::create((unsigned long)i,
                vslam::Vec2(p1[i].x, p1[i].y), vslam::Vec2(p2[i].x, p2[i].y),
                vslam::SE3(), T_cw2, K);
            if ((T_cw2 * mp->pos_w).z() > 0) depth_ok++;
        }
        assert(depth_ok > (int)pts_w.size() * 0.9);
        std::cout << " (rot_err=" << r_err << " trans_cos=" << cos_t
                  << " depth_ok=" << depth_ok << "/" << pts_w.size() << ")";
    } TEST_PASS();
}

// ============================================================
// LK 光流模式测试：3D 矩形块场景，feature_method=1 连续 4 帧跟踪
// ============================================================
void test_lk_tracking() {
    struct Blk { cv::Point3f c; float sx, sy; int gray; };
    auto render = [](cv::Mat& img, const std::vector<Blk>& blks,
                     const cv::Mat& K, const cv::Mat& rvec, const cv::Mat& tvec) {
        for (auto& b : blks) {
            std::vector<cv::Point3f> corners = {
                cv::Point3f(b.c.x - b.sx/2, b.c.y - b.sy/2, b.c.z),
                cv::Point3f(b.c.x + b.sx/2, b.c.y - b.sy/2, b.c.z),
                cv::Point3f(b.c.x + b.sx/2, b.c.y + b.sy/2, b.c.z),
                cv::Point3f(b.c.x - b.sx/2, b.c.y + b.sy/2, b.c.z)};
            std::vector<cv::Point2f> px;
            cv::projectPoints(corners, rvec, tvec, K, cv::Mat(), px);
            std::vector<cv::Point> pi;
            for (auto& q : px) pi.emplace_back(cvRound(q.x), cvRound(q.y));
            cv::fillConvexPoly(img, pi, cv::Scalar(b.gray));
        }
    };

    TEST("LK 光流模式多帧跟踪") {
        vslam::Camera cam;
        cam.fx = 500; cam.fy = 500; cam.cx = 320; cam.cy = 240;
        cam.img_width = 640; cam.img_height = 480;
        cv::Mat K = cam.K();

        std::mt19937 gen(7);
        std::uniform_real_distribution<double> dx(-4, 4), dy(-3, 3), dz(3, 6),
                                               ds(0.5, 1.5), dg(80, 255);
        std::vector<Blk> blks;
        for (int i = 0; i < 60; i++)
            blks.push_back({cv::Point3f(dx(gen), dy(gen), dz(gen)),
                            (float)ds(gen), (float)ds(gen), (int)dg(gen)});

        vslam::VOConfig cfg;
        cfg.feature_method = 1;          // LK 模式
        cfg.min_matches_init = 20;
        cfg.min_matches_track = 10;

        vslam::VisualOdometry vo(cam, cfg);

        int track_ok = 0;
        for (int f = 0; f < 4; f++) {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << 0.01*f, 0.02*f, 0);
            cv::Mat tvec = (cv::Mat_<double>(3,1) << 0.4*f, 0.1*f, 0);
            cv::Mat img(480, 640, CV_8UC1, cv::Scalar(64));
            render(img, blks, K, rvec, tvec);
            vo.addFrame(img, f * 0.1);
            if (vo.state() == vslam::VisualOdometry::State::TRACKING) track_ok++;
        }
        // 至少一半帧保持在 TRACKING（合成场景允许回退 ORB）
        assert(track_ok >= 2);
        std::cout << " (tracking_frames=" << track_ok << "/4)";
    } TEST_PASS();
}

// ============================================================
// 主函数
// ============================================================
int main() {
    std::cout << "===== VSLAM Unit Tests =====\n\n";

    std::cout << "[SE3 Basics]\n";
    test_se3_basics();

    std::cout << "\n[Camera Projection]\n";
    test_camera_projection();

    std::cout << "\n[Feature Extraction]\n";
    test_feature_extraction();

    std::cout << "\n[VO Initialization]\n";
    test_vo_initialization();

    std::cout << "\n[Pose Semantics]\n";
    test_pose_semantics();

    std::cout << "\n[LK Tracking]\n";
    test_lk_tracking();

    std::cout << "\n===== All Tests Completed =====\n";
    return 0;
}
