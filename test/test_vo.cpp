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
 *   7. T_cw 平移与世界系相机轨迹的语义
 *   8. MiniAtlas 子地图锚定与切换
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_vo && ./test_vo
 */

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/dataset.h"
#include "vslam/feature.h"
#include "vslam/vo.h"
#include "vslam/pose_gate.h"
#include "vslam/atlas.h"
#include "vslam/mappoint.h"
#include "vslam/optimizer.h"
#include "vslam/backend_committer.h"
#include "vslam/loop_closure.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <map>
#include <thread>

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

    TEST("T_cw 原地旋转时相机世界位置不绕圈") {
        const vslam::Vec3 fixed_camera_position(3.0, 0.5, 8.0);
        for (int degree = 0; degree <= 180; degree += 15) {
            Eigen::Quaterniond q_wc(
                Eigen::AngleAxisd(degree * M_PI / 180.0, vslam::Vec3::UnitY()));
            vslam::SE3 T_wc(q_wc, fixed_camera_position);
            vslam::SE3 T_cw = T_wc.inverse();

            // T_cw.t 会随旋转变化，camera_position() 必须始终返回固定光心 C_w。
            assert((T_cw.camera_position() - fixed_camera_position).norm() < 1e-10);
        }
    } TEST_PASS();
}

void test_camera_projection() {
    TEST("Camera pixel2camera") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        auto p3d = cam->pixel2camera(vslam::Vec2(320, 240), 2.0);
        assert(std::abs(p3d.x()) < 1e-10);
        assert(std::abs(p3d.y()) < 1e-10);
        assert(std::abs(p3d.z() - 2.0) < 1e-10);
    } TEST_PASS();

    TEST("Camera world2pixel") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        vslam::SE3 T_cw; // Identity (camera at origin, looking +z)
        // Point at (0, 0, 1) in world → (0,0,1) in camera → pixel (320, 240)
        auto px = cam->world2pixel(vslam::Vec3(0, 0, 1), T_cw);
        assert(std::abs(px.x() - 320.0) < 1e-6);
        assert(std::abs(px.y() - 240.0) < 1e-6);
    } TEST_PASS();

    TEST("EuRoC sensor.yaml 内参/畸变与图像去畸变") {
        const auto unique_id = std::chrono::steady_clock::now()
                                   .time_since_epoch().count();
        const auto root = std::filesystem::temp_directory_path()
                         / ("vslam_euroc_sensor_test_"
                            + std::to_string(unique_id));
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root / "mav0/cam0/data");
        {
            std::ofstream sensor(root / "mav0/cam0/sensor.yaml");
            sensor << "camera_model: pinhole\n"
                   << "resolution: [32, 24]\n"
                   << "intrinsics: [20.0, 21.0, 16.0, 12.0]\n"
                   << "distortion_model: radial-tangential\n"
                   << "distortion_coefficients: [-0.2, 0.03, 0.01, -0.02]\n";
        }
        {
            std::ofstream csv(root / "mav0/cam0/data.csv");
            csv << "#timestamp [ns],filename\n"
                << "1000000000,000000.png\n";
        }
        cv::Mat source(24, 32, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::rectangle(source, cv::Rect(12, 8, 8, 8),
                      cv::Scalar(255, 255, 255), -1);
        cv::imwrite((root / "mav0/cam0/data/000000.png").string(), source);

        vslam::Dataset dataset(root.string(), vslam::Dataset::Type::EUROC);
        assert(dataset.hasCalibration());
        assert(dataset.totalFrames() == 1);
        auto camera = dataset.getCamera();
        assert(camera && camera->type() == vslam::CameraType::MONOCULAR);
        assert(std::abs(camera->fx - 20.0) < 1e-12);
        assert(std::abs(camera->fy - 21.0) < 1e-12);
        assert(camera->img_width == 32 && camera->img_height == 24);
        assert(std::abs(camera->k1 + 0.2) < 1e-12);
        assert(std::abs(camera->p2 + 0.02) < 1e-12);

        cv::Mat left, right;
        double timestamp = 0.0;
        assert(dataset.nextFrame(left, right, timestamp));
        assert(std::abs(timestamp - 1.0) < 1e-12);
        assert(right.empty());
        assert(left.size() == source.size());
        assert(cv::norm(left, source, cv::NORM_INF) > 0.0);
        std::filesystem::remove_all(root);
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

void test_frame_image_lifecycle() {
    TEST("Frame 像素缓冲分阶段释放且关键帧数据保留") {
        auto frame = std::make_shared<vslam::Frame>(7, 0.7);
        frame->image = cv::Mat(48, 64, CV_8UC3, cv::Scalar(1, 2, 3));
        frame->image_gray = cv::Mat(48, 64, CV_8UC1, cv::Scalar(4));
        frame->image_right = cv::Mat(48, 64, CV_8UC3, cv::Scalar(5, 6, 7));
        frame->image_right_gray = cv::Mat(48, 64, CV_8UC1, cv::Scalar(8));
        frame->keypoints.emplace_back(cv::Point2f(10, 12), 31.0f);
        frame->descriptors = cv::Mat::ones(1, 32, CV_8UC1);
        frame->map_points.push_back(std::make_shared<vslam::MapPoint>(3));
        frame->pts_c.emplace_back(1.0, 2.0, 3.0);

        frame->releaseImages(true);
        assert(frame->image.empty());
        assert(!frame->image_gray.empty());
        assert(frame->image_right.empty());
        assert(frame->image_right_gray.empty());

        frame->releaseImages();
        assert(frame->image_gray.empty());
        assert(frame->keypoints.size() == 1);
        assert(frame->descriptors.rows == 1);
        assert(frame->map_points.size() == 1 && frame->map_points[0]);
        assert(frame->pts_c.size() == 1 && frame->pts_c[0].z() == 3.0);
    } TEST_PASS();
}

void test_mobile_config() {
    TEST("mobile.yaml 资源预算参数可加载") {
        const auto config_path = std::filesystem::path(__FILE__).parent_path()
            .parent_path() / "config/mobile.yaml";
        const auto cfg = vslam::VOConfig::fromYaml(config_path.string());
        assert(cfg.num_features == 600);
        assert(cfg.pyramid_levels == 6);
        assert(cfg.orb_max_bands == 1);
        assert(cfg.opencv_threads == 4);
        assert(cfg.local_window_size == 6);
        assert(cfg.local_ba_iterations == 3);
        assert(std::abs(cfg.local_ba_max_correction - 1.0) < 1e-12);
        assert(cfg.enable_loop_closure);
        assert(std::abs(cfg.min_parallax - 0.025) < 1e-12);
        assert(cfg.min_init_inliers == 18);
    } TEST_PASS();
}

void test_monocular_initialization_quality() {
    TEST("单目初始化几何判定拒绝纯旋转/近零基线") {
        const vslam::Mat33 R = Eigen::AngleAxisd(
            0.35, vslam::Vec3::UnitY()).toRotationMatrix();
        std::vector<vslam::Vec2> ref, curr;
        for (int i = 0; i < 20; ++i) {
            const vslam::Vec3 ray(0.1 * (i - 10), 0.03 * (i - 5), 1.0);
            const vslam::Vec3 rotated = R * ray;
            ref.emplace_back(ray.x() / ray.z(), ray.y() / ray.z());
            curr.emplace_back(rotated.x() / rotated.z(),
                              rotated.y() / rotated.z());
        }
        // recoverPose 的 t 即使在纯旋转退化时也可能是任意单位方向；
        // 必须依靠光线夹角而不是 ||t|| 拒绝它。
        const auto pure_rotation = vslam::assessMonocularInitialization(
            ref, curr, R, vslam::Vec3(1.0, 0.0, 0.0), 0.1);
        assert(!pure_rotation.accepted);
        assert(pure_rotation.parallax_rad < 1e-8);

        const auto near_zero_baseline = vslam::assessMonocularInitialization(
            ref, curr, R, vslam::Vec3(1e-12, 0.0, 0.0), 0.1);
        assert(!near_zero_baseline.accepted);
    } TEST_PASS();

    TEST("单目初始化几何判定要求足够视差和双正深度") {
        const vslam::Mat33 R = vslam::Mat33::Identity();
        const vslam::Vec3 t(1.0, 0.0, 0.0);
        std::vector<vslam::Vec2> ref, curr;
        for (int i = 0; i < 20; ++i) {
            const double x = -0.8 + 0.08 * i;
            const double y = -0.3 + 0.03 * (i % 10);
            const double z = 4.0 + 0.2 * (i % 7);
            ref.emplace_back(x / z, y / z);
            curr.emplace_back((x + t.x()) / z, y / z);
        }
        const auto good = vslam::assessMonocularInitialization(
            ref, curr, R, t, 0.1);
        assert(good.accepted);
        assert(good.parallax_rad > 0.1);
        assert(good.positive_depth_ratio > 0.99);

        std::vector<vslam::Vec2> mostly_invalid_ref, mostly_invalid_curr;
        for (int i = 0; i < 20; ++i) {
            const double z = 4.0 + 0.1 * i;
            mostly_invalid_ref.emplace_back(0.0, 0.02 * (i % 5));
            const double disparity_sign = i < 5 ? 1.0 : -1.0;
            mostly_invalid_curr.emplace_back(disparity_sign / z,
                                             0.02 * (i % 5));
        }
        const auto low_positive = vslam::assessMonocularInitialization(
            mostly_invalid_ref, mostly_invalid_curr, R, t, 0.1);
        assert(!low_positive.accepted);
        assert(low_positive.positive_depth_ratio < 0.7);
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

        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;

        // 合成场景特征匹配数较少，显式放宽初始化阈值（验证 VOConfig 接口生效）
        vslam::VOConfig cfg;
        cfg.min_matches_init = 20;
        cfg.min_init_inliers = 10;

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
        if (vo.getMap()->keyFrameCount() > 0)
            assert(vo.getMap()->verifyObservationConsistency());

        // 几何内点门槛必须在原始匹配数足够时仍能阻止退化初始化。
        vslam::VOConfig strict_cfg;
        strict_cfg.min_matches_init = 20;
        strict_cfg.min_init_inliers = 1000;
        vslam::VisualOdometry strict_vo(cam, strict_cfg);
        strict_vo.addFrame(img1, 0.0);
        strict_vo.addFrame(img2, 0.1);
        assert(strict_vo.state() == vslam::VisualOdometry::State::INITIALIZING);
        assert(strict_vo.getMap()->mapPointCount() == 0);
    } TEST_PASS();
}

// ============================================================
// 位姿语义回归测试
// 用已知真实位姿生成点对，走与 vo.cpp 初始化完全相同的序列
// （findEssentialMat → recoverPose → pose_cs = recoverPose 输出），
// 验证 T_cw 方向正确：旋转精确、平移方向一致、三角化深度为正。
// 修复前的取逆方向会让三角化 0% 深度正确。
// ============================================================
void test_pose_semantics() {
    TEST("recoverPose→T_cw 位姿语义 (known motion)") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        cv::Mat K = cam->K();

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
            if ((T_cw2 * mp->pos_s).z() > 0) depth_ok++;
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
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;
        cv::Mat K = cam->K();

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
        cfg.min_init_inliers = 20;
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
        assert(vo.getMap()->mapPointCount() > 0);
        std::cout << " (tracking_frames=" << track_ok << "/4)";
    } TEST_PASS();
}

void test_monocular_3d_initialization() {
    TEST("单目确定性 3D 场景初始化进入 TRACKING") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;
        const cv::Mat K = cam->K();

        struct Block { cv::Point3f center; float sx, sy; int gray; };
        std::vector<Block> blocks;
        for (int iy = -2; iy <= 2; ++iy) {
            for (int ix = -4; ix <= 4; ++ix) {
                const float z = 4.0f + ((ix + iy) & 1) * 0.8f;
                const int gray = 80 + ((ix + 5) * 31 + (iy + 3) * 17) % 150;
                blocks.push_back({cv::Point3f(0.48f * ix, 0.48f * iy, z),
                                  0.22f, 0.22f, gray});
            }
        }

        auto render = [&](double tx) {
            cv::Mat image(480, 640, CV_8UC1, cv::Scalar(32));
            const cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            const cv::Mat tvec = (cv::Mat_<double>(3, 1) << tx, 0.0, 0.0);
            for (const auto& block : blocks) {
                const auto& c = block.center;
                std::vector<cv::Point3f> corners = {
                    {c.x - block.sx, c.y - block.sy, c.z},
                    {c.x + block.sx, c.y - block.sy, c.z},
                    {c.x + block.sx, c.y + block.sy, c.z},
                    {c.x - block.sx, c.y + block.sy, c.z}};
                std::vector<cv::Point2f> pixels;
                cv::projectPoints(corners, rvec, tvec, K, cv::Mat(), pixels);
                std::vector<cv::Point> polygon;
                for (const auto& pixel : pixels)
                    polygon.emplace_back(cvRound(pixel.x), cvRound(pixel.y));
                cv::fillConvexPoly(image, polygon, cv::Scalar(block.gray));
            }
            return image;
        };

        vslam::VOConfig cfg;
        cfg.min_matches_init = 20;
        cfg.min_init_inliers = 10;
        vslam::VisualOdometry vo(cam, cfg);
        vo.addFrame(render(0.0), 0.0);
        vo.addFrame(render(0.4), 0.1);

        assert(vo.state() == vslam::VisualOdometry::State::TRACKING);
        assert(vo.getMap()->mapPointCount() > 0);
        assert(vo.getMap()->verifyObservationConsistency());
    } TEST_PASS();
}

// ============================================================
// 长时间运行稳定性测试
// 守护性能优化：关键帧/地图点必须有界（防"跑一段时间后卡死"），
// 帧耗时应稳定（不随运行时间增长）。
// ============================================================
void test_long_run_stability() {
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

    TEST("长时间运行：关键帧/地图点有界 + 帧耗时稳定") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;
        cv::Mat K = cam->K();

        std::mt19937 gen(7);
        std::uniform_real_distribution<double> dx(-4, 4), dy(-3, 3), dz(3, 6),
                                               ds(0.5, 1.5), dg(80, 255);
        std::vector<Blk> blks;
        for (int i = 0; i < 60; i++)
            blks.push_back({cv::Point3f(dx(gen), dy(gen), dz(gen)),
                            (float)ds(gen), (float)ds(gen), (int)dg(gen)});

        vslam::VOConfig cfg;
        cfg.min_matches_init = 20;
        cfg.min_init_inliers = 10;

        vslam::VisualOdometry vo(cam, cfg);

        // 螺旋路径 150 帧：缓慢前进 + 小幅摆动
        constexpr int kFrames = 150;
        std::vector<double> frame_ms;
        bool initialized = false;
        for (int f = 0; f < kFrames; f++) {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << 0.003*f, 0.005*f, 0);
            cv::Mat tvec = (cv::Mat_<double>(3,1) << 0.1*f, 0.02*std::sin(f*0.05), 0.1*std::sin(f*0.03));
            cv::Mat img(480, 640, CV_8UC1, cv::Scalar(64));
            render(img, blks, K, rvec, tvec);

            auto t0 = std::chrono::steady_clock::now();
            vo.addFrame(img, f * 0.1);
            auto t1 = std::chrono::steady_clock::now();
            frame_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            initialized = initialized ||
                (vo.state() == vslam::VisualOdometry::State::TRACKING &&
                 vo.getMap()->mapPointCount() > 0);
        }

        auto avg = [&](int from, int to) {
            double s = 0;
            for (int i = from; i < to; i++) s += frame_ms[i];
            return s / (to - from);
        };

        size_t kf = vo.getMap()->keyFrameCount();
        size_t mp = vo.getMap()->mapPointCount();
        double avg_first = avg(0, 30), avg_last = avg(kFrames - 30, kFrames);

        std::cout << " (kf=" << kf << " mp=" << mp
                  << " avg_first=" << avg_first << "ms avg_last=" << avg_last << "ms)";

        // 有界性：关键帧 ≈ 30（每 0.5m 一个，共 15m），地图点几千
        assert(initialized);
        assert(kf < 80);
        assert(mp < 15000);
        // 稳定性：末尾帧耗时不能比开头差 3 倍以上（防增长型卡死回归）
        assert(avg_last < avg_first * 3.0 + 3.0);
    } TEST_PASS();
}

// ============================================================
// Local BA 语义回归测试
// 3 帧已知位姿（1m/帧）+ 精确投影 → BA 后尺度必须保持。
// 守护：本机 apt 版 g2o 的 EdgeProjectXYZ2UV 用 T_cw 语义，
// 若喂 T_wc 会被反向优化（实测帧位姿被拉飞、轨迹尺度膨胀）。
// ============================================================
void test_camera_position_delta() {
    TEST("T_cw 光心位移指标不混入旋转") {
        const vslam::Vec3 old_center(2.0, -1.0, 4.0);
        const vslam::Vec3 new_center(2.12, -1.07, 4.20);
        const Eigen::Quaterniond old_rotation(
            Eigen::AngleAxisd(0.4, vslam::Vec3::UnitY()));
        const Eigen::Quaterniond new_rotation(
            Eigen::AngleAxisd(-0.7, vslam::Vec3::UnitZ()) *
            Eigen::AngleAxisd(0.2, vslam::Vec3::UnitX()));
        const vslam::SE3 old_cw(old_rotation, -(old_rotation * old_center));
        const vslam::SE3 new_cw(new_rotation, -(new_rotation * new_center));
        const double expected = (new_center - old_center).norm();
        assert(std::abs(vslam::cameraPositionDelta(old_cw, new_cw) - expected)
               < 1e-12);
        std::cout << " (delta=" << expected << "m)";
    } TEST_PASS();
}

void test_local_ba_filter_helper() {
    TEST("Local BA 弱观测过滤") {
        assert(vslam::includeLocalBALandmark(3, 3));
        assert(!vslam::includeLocalBALandmark(2, 3));
        assert(vslam::includeLocalBALandmark(0, 0));
    } TEST_PASS();

    TEST("帧内回环后按参考 KF 重基并保持 T_ca") {
        const vslam::SE3 old_ref(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.2, vslam::Vec3::UnitY())),
            vslam::Vec3(2, -1, 4));
        const vslam::SE3 frame(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.24, vslam::Vec3::UnitY())),
            vslam::Vec3(2.7, -0.8, 4.4));
        const vslam::SE3 new_ref(
            Eigen::Quaterniond(Eigen::AngleAxisd(-0.35, vslam::Vec3::UnitZ())),
            vslam::Vec3(-30, 7, 12));
        const auto rebased = vslam::rebaseAnchoredFramePose(
            frame, old_ref, new_ref);
        const auto old_T_ca = frame * old_ref.inverse();
        const auto new_T_ca = rebased * new_ref.inverse();
        assert((old_T_ca.matrix() - new_T_ca.matrix()).norm() < 1e-12);
        assert((rebased.matrix() - frame.matrix()).norm() > 1.0);
    } TEST_PASS();

    TEST("Local BA 改写锚点时反向重基 T_ca 并保持历史世界位姿") {
        const vslam::SE3 T_ca_old(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.08, vslam::Vec3::UnitX())),
            vslam::Vec3(0.7, -0.1, 0.2));
        const vslam::SE3 old_anchor(
            Eigen::Quaterniond(Eigen::AngleAxisd(-0.12, vslam::Vec3::UnitY())),
            vslam::Vec3(8.0, 1.0, -3.0));
        const vslam::SE3 new_anchor(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.16, vslam::Vec3::UnitZ())),
            vslam::Vec3(8.6, 0.7, -2.8));
        const auto T_ca_new = vslam::rebaseTrajectoryAnchor(
            T_ca_old, old_anchor, new_anchor);
        assert((T_ca_new * new_anchor).matrix().isApprox(
            (T_ca_old * old_anchor).matrix(), 1e-12));
        assert((T_ca_new.matrix() - T_ca_old.matrix()).norm() > 0.1);
    } TEST_PASS();

    TEST("跨子地图锚校正从连续边界平滑过渡到完整闭环校正") {
        const vslam::SE3 T_ca(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.04, vslam::Vec3::UnitX())),
            vslam::Vec3(0.3, 0.1, -0.2));
        const vslam::SE3 anchor(
            Eigen::Quaterniond(Eigen::AngleAxisd(-0.1, vslam::Vec3::UnitY())),
            vslam::Vec3(4.0, -2.0, 1.0));
        const vslam::SE3 old_T_ws(Eigen::Quaterniond::Identity(),
                                  vslam::Vec3(20.0, 0.0, 5.0));
        const vslam::SE3 new_T_ws(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.2, vslam::Vec3::UnitZ())),
            vslam::Vec3(27.0, -1.0, 4.0));
        const auto at_start = vslam::rebaseTrajectoryForSubmapAnchor(
            T_ca, anchor, old_T_ws, new_T_ws, 0.0);
        const auto at_end = vslam::rebaseTrajectoryForSubmapAnchor(
            T_ca, anchor, old_T_ws, new_T_ws, 1.0);
        assert((at_start * anchor * new_T_ws.inverse()).matrix().isApprox(
            (T_ca * anchor * old_T_ws.inverse()).matrix(), 1e-12));
        assert((at_end.matrix() - T_ca.matrix()).norm() < 1e-12);
        assert(vslam::submapTrajectoryCorrectionAlpha(80, 100, 100, 200) == 0.0);
        assert(vslam::submapTrajectoryCorrectionAlpha(100, 100, 100, 200) == 0.0);
        assert(std::abs(vslam::submapTrajectoryCorrectionAlpha(
            150, 100, 100, 200) - 0.5) < 1e-12);
        assert(vslam::submapTrajectoryCorrectionAlpha(220, 100, 100, 200) == 1.0);
        const auto fallback_world = vslam::composeAnchoredWorldPose(
            T_ca, anchor, old_T_ws);
        assert(fallback_world.matrix().isApprox(
            (T_ca * anchor * old_T_ws.inverse()).matrix(), 1e-12));
        assert(fallback_world.t.norm() > 1.0);  // 锚被剔除时不得退化为 SE3()

        const auto remapped = vslam::rebaseTrajectoryAnchor(
            T_ca, anchor, new_T_ws);
        assert((remapped * new_T_ws).matrix().isApprox(
            (T_ca * anchor).matrix(), 1e-12));
    } TEST_PASS();
}

void test_local_ba() {
    TEST("Local BA 保持尺度 (3 帧已知位姿)") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 620; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;

        auto map = std::make_shared<vslam::Map>();

        // 3 帧 T_wc：原点、前进 1m、前进 2m（真实尺度）
        std::vector<vslam::SE3> Twc = {
            vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(0, 0, 0)),
            vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(1, 0, 0)),
            vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(2, 0, 0)),
        };
        std::vector<vslam::Frame::Ptr> kfs;
        for (int i = 0; i < 3; i++) {
            auto kf = std::make_shared<vslam::Frame>((unsigned long)i, i * 0.1);
            kf->pose_cs = Twc[i].inverse();
            kfs.push_back(kf);
            map->insertKeyFrame(kf);
        }
        // 20 个世界点（前方 4~8m），投影到 3 帧
        std::mt19937 gen(5);
        std::uniform_real_distribution<double> dx(-2, 2), dy(-2, 2), dz(4, 8);
        for (int i = 0; i < 20; i++) {
            auto mp = std::make_shared<vslam::MapPoint>((unsigned long)i);
            mp->pos_s = vslam::Vec3(dx(gen), dy(gen), dz(gen));
            map->insertMapPoint(mp);
            for (int f = 0; f < 3; f++) {
                vslam::Vec2 px = cam->world2pixel(mp->pos_s, kfs[f]->pose_cs);
                cv::KeyPoint kp;
                kp.pt = cv::Point2f((float)px.x(), (float)px.y());
                kfs[f]->keypoints.push_back(kp);
                kfs[f]->map_points.push_back(mp);
            }
        }
        for (const auto& kf : kfs) map->syncKeyframeObservations(kf);
        assert(map->verifyObservationConsistency());

        // M1：构建只读快照 → 纯计算 → 结果携带候选增量。
        // ObservationState 使用非连续 feature_index，且 id=0 是合法地图点。
        vslam::OptimizationSnapshot snap;
        for (auto& kf : kfs) {
            vslam::KeyframeState ks;
            ks.id = kf->id;
            ks.pose_cs = kf->pose_cs;
            snap.keyframes.push_back(std::move(ks));
        }
        for (const auto& mp : map->getAllMapPoints()) {
            snap.landmarks.push_back({mp->id, mp->pos_s,
                                      static_cast<int>(mp->observationCount())});
        }
        for (size_t f = 0; f < kfs.size(); ++f) {
            for (size_t i = 0; i < kfs[f]->map_points.size(); ++i) {
                const auto& mp = kfs[f]->map_points[i];
                if (!mp) continue;
                const vslam::Vec2 pixel = cam->world2pixel(
                    mp->pos_s, kfs[f]->pose_cs);
                snap.observations.push_back({
                    kfs[f]->id, static_cast<vslam::FeatureIndex>(7 + 2 * i),
                    mp->id, pixel, std::nullopt});
            }
        }
        snap.fixed_kf_ids = {0, 1};  // 帧 0/1 固定（基线锚定）

        // 观测仍来自真值，但让非固定帧从一个带旋转/平移的初值开始；
        // 这样会验证 BA 的投影模型和 Jacobian 真正驱动了收敛，而不是
        // 仅凭“初始残差为 0”通过。
        const vslam::SE3 perturb(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.08, vslam::Vec3::UnitY())),
            vslam::Vec3(0.18, -0.12, 0.10));
        const vslam::SE3 perturbed_twc = Twc[2] * perturb;
        snap.keyframes[2].pose_cs = perturbed_twc.inverse();
        const double initial_pose_error =
            (perturbed_twc.t - Twc[2].t).norm();

        auto result = vslam::Optimizer::solveLocalBA(cam, snap, 10);
        assert(result.valid);
        assert(result.metrics.edges == 60);  // 20 点×3 KF，包含合法 id=0
        // 非方形像素测试自定义投影边确实使用 fy；精确输入不应被
        // 错误的单一 focal_length 模型拉动位姿。
        std::map<unsigned long, vslam::SE3> opt_pose;
        for (auto& u : result.poses) opt_pose[u.id] = u.pose_cs;
        const double final_pose_error =
            (opt_pose[2].inverse().t - Twc[2].t).norm();
        assert(result.metrics.max_correction > initial_pose_error * 0.5);
        assert(final_pose_error < initial_pose_error * 0.35);
        std::cout << " (pose_error=" << initial_pose_error << "m -> "
                  << final_pose_error << "m)";
        // 点固定（motion-only）+ 帧 0/1 固定：尺度必须保持 1m/帧
        auto disp = [&](int a, int b) {
            return (opt_pose[(unsigned long)b].inverse().t
                    - opt_pose[(unsigned long)a].inverse().t).norm();
        };
        assert(std::abs(disp(0, 1) - 1.0) < 0.2);
        assert(std::abs(disp(1, 2) - 1.0) < 0.2);
        std::cout << " (disp01=" << disp(0, 1) << " disp12=" << disp(1, 2) << ")";
        // 快照 API 不修改任何实时对象
        for (auto& kf : kfs)
            assert((kf->pose_cs.matrix() - Twc[kf->id].inverse().matrix()).norm() < 1e-12);
    } TEST_PASS();
}

// ============================================================
// Local BA gauge 回归：窗口含两个不相交的观测分量时，调用方只提供
// 分量 A 的固定帧，Optimizer 必须为分量 B 选择局部锚，不能让其自由漂移。
// ============================================================
void test_local_ba_disconnected_components() {
    TEST("Local BA 为断开观测分量选择局部锚") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 620; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;

        vslam::OptimizationSnapshot snap;
        std::vector<vslam::SE3> true_twc;
        for (int component = 0; component < 2; component++) {
            const double x_base = component == 0 ? 0.0 : 10.0;
            for (int frame = 0; frame < 3; frame++) {
                true_twc.emplace_back(
                    Eigen::Quaterniond::Identity(),
                    vslam::Vec3(x_base + frame, 0, 0));
            }
        }

        constexpr int kPointsPerComponent = 12;
        for (int component = 0; component < 2; component++) {
            const unsigned long id_base = component == 0 ? 100 : 200;
            const double x_base = component == 0 ? 0.0 : 10.0;
            for (int frame = 0; frame < 3; frame++) {
                vslam::KeyframeState kf;
                kf.id = id_base + static_cast<unsigned long>(frame);
                kf.pose_cs = true_twc[component * 3 + frame].inverse();
                for (int point = 0; point < kPointsPerComponent; point++) {
                    const unsigned long mp_id =
                        static_cast<unsigned long>(component * 1000 + point);
                    const vslam::Vec3 p(
                        x_base - 1.0 + 0.17 * point,
                        -0.7 + 0.11 * (point % 7),
                        5.0 + 0.18 * point);
                    if (frame == 0)
                        snap.landmarks.push_back({mp_id, p, 3});
                    const vslam::Vec2 pixel = cam->world2pixel(p, kf.pose_cs);
                    snap.observations.push_back({
                        kf.id, static_cast<vslam::FeatureIndex>(11 + point * 3),
                        mp_id, pixel, std::nullopt});
                }
                snap.keyframes.push_back(std::move(kf));
            }
        }

        // B 的初值整体平移，且只锚定 A；Optimizer 必须自动固定 B 的
        // 前两帧，否则带自由点的 BA 存在未约束的刚体/尺度 gauge。
        const vslam::SE3 shift(
            Eigen::Quaterniond::Identity(), vslam::Vec3(2.0, 0, 0));
        for (size_t i = 3; i < 6; i++)
            snap.keyframes[i].pose_cs = (true_twc[i] * shift).inverse();
        snap.fixed_kf_ids = {100, 101};

        auto result = vslam::Optimizer::solveLocalBA(cam, snap, 20, false);
        assert(result.valid);
        std::map<unsigned long, vslam::SE3> poses;
        for (const auto& update : result.poses) poses[update.id] = update.pose_cs;
        assert(poses.size() == 6);
        // B 的局部锚必须保持其快照初值；若沿用全局固定集，这两个
        // 顶点会被整体拉回真值，回归即可区分两种行为。
        for (size_t i = 3; i < 5; i++) {
            const unsigned long id = 200 + static_cast<unsigned long>(i - 3);
            assert((poses[id].matrix() - snap.keyframes[i].pose_cs.matrix()).norm()
                   < 1e-10);
        }
        for (const auto& [id, pose] : poses) {
            (void)id;
            assert(pose.t.allFinite() && pose.q.coeffs().allFinite());
        }
        assert(std::isfinite(result.metrics.max_correction));
    } TEST_PASS();
}

// ============================================================
// 纯旋转检测判据测试
// 对极几何在纯旋转时退化（E≈0，旋转/平移不可分），trackFrame 用
// "匹配点像素位移方向一致性"判断：平移主导方向一致，旋转主导方向分散。
// ============================================================
void test_rotation_detection() {
    // 与 vo.cpp 对极回退分支相同的判据逻辑（单位位移方向向量的平均模长）
    auto rot_dominant = [](const std::vector<cv::Point2f>& ds) {
        double sx = 0, sy = 0;
        int cnt = 0;
        for (auto& d : ds) {
            double l = std::hypot(d.x, d.y);
            if (l > 0.5) { sx += d.x / l; sy += d.y / l; cnt++; }
        }
        double consistency = (cnt > 0) ? std::hypot(sx, sy) / cnt : 0.0;
        return consistency < 0.5;   // 方向分散 → 旋转主导
    };

    TEST("纯旋转检测判据 (平移≠旋转)") {
        // 平移主导：所有像素位移方向一致 (5,3)
        std::vector<cv::Point2f> trans;
        for (int i = 0; i < 10; i++) trans.emplace_back(5.f, 3.f);
        // 旋转主导：绕图像中心 (320,240) 旋转 10°（位移方向绕中心辐射，方向分散）
        std::vector<cv::Point2f> rot;
        for (int i = 0; i < 12; i++) {
            double a = i * 2 * M_PI / 12;
            cv::Point2f p(320 + 200 * std::cos(a), 240 + 200 * std::sin(a));
            cv::Point2f q(320 + 200 * std::cos(a + 0.17), 240 + 200 * std::sin(a + 0.17));
            rot.push_back(q - p);
        }
        assert(!rot_dominant(trans));   // 平移 → 判为平移
        assert(rot_dominant(rot));      // 旋转 → 判为旋转
        std::cout << " (trans=" << rot_dominant(trans) << " rot=" << rot_dominant(rot) << ")";
    } TEST_PASS();
}

// ============================================================
// 旋转-平移歧义现象记录测试（单目本质限制，不设严格断言）
// 纯远点场景（15~35m，平移不可观测）+ 原地 yaw 旋转：
// PnP 假平移会振荡漂移 ~4.5m——这是单目"旋转-平移歧义"的本质
// （yaw 转头流场≈横向平移流场，数学上不可区分），任何单目启发式
// 只能部分缓解。精确解决需 VIO（IMU）或双目/RGB-D 提供尺度。
// 本测试仅记录现象（打印漂移值），不因漂移大而失败。
// ============================================================
void test_rotation_ambiguity() {
    TEST("旋转-平移歧义抑制 (远点原地转头)") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;

        // 纯远点 3D 点云（无近点 → 平移不可观测）
        std::mt19937 gen(11);
        std::uniform_real_distribution<double> dz(15, 35), dxy(-10, 10);
        std::vector<vslam::Vec3> pts;
        for (int i = 0; i < 500; i++)
            pts.emplace_back(dxy(gen), dxy(gen), dz(gen));

        auto render = [&](const vslam::SE3& T_wc, const Eigen::Matrix3d& R_scene) {
            cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC1);
            for (auto& p : pts) {
                vslam::Vec3 pr = R_scene * p;
                vslam::Vec2 px = cam->world2pixel(pr, T_wc.inverse());
                if (px.x() > 5 && px.x() < 635 && px.y() > 5 && px.y() < 475)
                    cv::rectangle(img, cv::Rect((int)px.x() - 4, (int)px.y() - 4, 8, 8), 200, -1);
            }
            cv::GaussianBlur(img, img, {3, 3}, 0);
            return img;
        };

        vslam::VOConfig cfg;
        cfg.keyframe_translation = 0.5;
        cfg.num_features = 800;
        vslam::VisualOdometry vo(cam, cfg);
        std::vector<vslam::SE3> poses;
        auto wpos = [&](int i) { return poses[i].inverse().t; };

        // 段 A：前进 20 帧（建图）
        vslam::SE3 T_wc = vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(0, 0, 0));
        Eigen::Matrix3d R_scene = Eigen::Matrix3d::Identity();
        for (int i = 0; i < 20; i++) {
            T_wc = vslam::SE3(Eigen::Quaterniond::Identity(), T_wc.t + vslam::Vec3(0, 0, 0.4));
            poses.push_back(vo.addFrame(render(T_wc, R_scene), i * 0.1));
        }
        vslam::Vec3 fixed_pos = T_wc.t;

        // 段 B：场景绕 y 轴旋转（等效相机原地 yaw），3°/帧 × 45 帧 = 135°
        // （>2.3°/帧，触发 rotation_shrink 的旋转角判据）
        double total = 0;
        for (int i = 0; i < 45; i++) {
            total += 3 * M_PI / 180;
            R_scene = Eigen::AngleAxisd(total, vslam::Vec3(0, 1, 0)).toRotationMatrix();
            T_wc = vslam::SE3(Eigen::Quaterniond::Identity(), fixed_pos);
            poses.push_back(vo.addFrame(render(T_wc, R_scene), (20 + i) * 0.1));
        }

        vslam::Vec3 ref_pos = wpos(19);
        double max_drift = 0;
        for (int i = 20; i < (int)poses.size(); i++)
            max_drift = std::max(max_drift, (wpos(i) - ref_pos).norm());
        std::cout << " (远点 yaw 旋转 45 帧最大漂移 " << max_drift
                  << "m —— 单目歧义本质，记录现象不作断言)";
    } TEST_PASS();
}

// ============================================================
// M0 统一位姿验收测试：正常跟踪与重定位共用几何 + 连续性门限。
// 回归 §3.19：重定位曾只查内点/RMSE，把跟踪刚拒绝的 258m 坏解重新接受。
// ============================================================
void test_pose_acceptance() {
    using vslam::SE3;
    using vslam::Vec3;
    using PoseQuality = vslam::PoseQuality;
    constexpr int    kMinInliers = 15;
    constexpr double kMinRatio   = 0.3;
    constexpr double kMaxRmse    = 2.5;
    constexpr double kRelocRot   = 60.0 * M_PI / 180.0;

    TEST("正常跟踪：2m 位移通过、4m 位移被拒绝（3m 门限）") {
        const SE3 baseline_twc;  // 相机在原点
        PoseQuality q;
        // 相机前进 2m（T_cw.t = -2）
        const SE3 ok_pose(Eigen::Quaterniond::Identity(), Vec3(-2, 0, 0));
        assert(vslam::PoseGate::acceptPoseCandidate(
            ok_pose, 20, 40, 1.0, kMinInliers, kMinRatio, kMaxRmse,
            baseline_twc, 3.0, 0.35, q));
        assert(q.geometric_ok && q.motion_ok);
        assert(std::abs(q.translation - 2.0) < 1e-9);
        // 相机前进 4m → 超 3m 门限
        const SE3 bad_pose(Eigen::Quaterniond::Identity(), Vec3(-4, 0, 0));
        assert(!vslam::PoseGate::acceptPoseCandidate(
            bad_pose, 20, 40, 1.0, kMinInliers, kMinRatio, kMaxRmse,
            baseline_twc, 3.0, 0.35, q));
        assert(q.geometric_ok && !q.motion_ok);
        assert(std::abs(q.translation - 4.0) < 1e-9);
    } TEST_PASS();

    TEST("正常跟踪：0.2rad 旋转通过、0.5rad 被拒绝（0.35rad 门限）") {
        const SE3 baseline_twc;
        PoseQuality q;
        const SE3 ok_pose(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Vec3::UnitZ())), Vec3::Zero());
        assert(vslam::PoseGate::acceptPoseCandidate(
            ok_pose, 20, 40, 1.0, kMinInliers, kMinRatio, kMaxRmse,
            baseline_twc, 3.0, 0.35, q));
        const SE3 bad_pose(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Vec3::UnitZ())), Vec3::Zero());
        assert(!vslam::PoseGate::acceptPoseCandidate(
            bad_pose, 20, 40, 1.0, kMinInliers, kMinRatio, kMaxRmse,
            baseline_twc, 3.0, 0.35, q));
        assert(q.geometric_ok && !q.motion_ok);
    } TEST_PASS();

    TEST("重定位：远离外推基线的 258m 假位姿被拒绝（§3.19 回归）") {
        // 丢失 4 帧、每步 2.5m → 外推基线 T_wc.t=(10,0,0)，expected=10m，
        // 平移门限 = max(50, 3×10) = 50m。几何全优（30/40 内点、rmse 0.5）
        // 但相机实际在 268m 处（距基线 258m）→ 必须拒绝。
        const SE3 baseline_twc(Eigen::Quaterniond::Identity(), Vec3(10, 0, 0));
        PoseQuality q;
        const SE3 far_pose(Eigen::Quaterniond::Identity(), Vec3(-268, 0, 0));
        assert(!vslam::PoseGate::acceptPoseCandidate(
            far_pose, 30, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 50.0, kRelocRot, q));
        assert(q.geometric_ok && !q.motion_ok);
        assert(std::abs(q.translation - 258.0) < 1e-6);
    } TEST_PASS();

    TEST("重定位：贴近外推基线的好位姿通过") {
        const SE3 baseline_twc(Eigen::Quaterniond::Identity(), Vec3(10, 0, 0));
        PoseQuality q;
        // 相机在 12m 处（距基线 2m）
        const SE3 near_pose(Eigen::Quaterniond::Identity(), Vec3(-12, 0, 0));
        assert(vslam::PoseGate::acceptPoseCandidate(
            near_pose, 30, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 50.0, kRelocRot, q));
        assert(q.geometric_ok && q.motion_ok);
    } TEST_PASS();

    TEST("重定位：长丢失后门限随外推位移放宽（20 帧×2.5m → 150m）") {
        // expected = 50m → limit = max(50, 3×50) = 150m
        const SE3 baseline_twc(Eigen::Quaterniond::Identity(), Vec3(50, 0, 0));
        PoseQuality q;
        // 相机在 130m（距基线 80m）→ 通过
        const SE3 ok_pose(Eigen::Quaterniond::Identity(), Vec3(-130, 0, 0));
        assert(vslam::PoseGate::acceptPoseCandidate(
            ok_pose, 30, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 150.0, kRelocRot, q));
        // 相机在 250m（距基线 200m）→ 拒绝
        const SE3 bad_pose(Eigen::Quaterniond::Identity(), Vec3(-250, 0, 0));
        assert(!vslam::PoseGate::acceptPoseCandidate(
            bad_pose, 30, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 150.0, kRelocRot, q));
    } TEST_PASS();

    TEST("重定位：旋转超 60° 被拒绝、45° 通过") {
        const SE3 baseline_twc;
        PoseQuality q;
        const SE3 ok_pose(
            Eigen::Quaterniond(Eigen::AngleAxisd(45.0 * M_PI / 180.0, Vec3::UnitY())),
            Vec3(0, 0, 0));
        assert(vslam::PoseGate::acceptPoseCandidate(
            ok_pose, 30, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 50.0, kRelocRot, q));
        const SE3 bad_pose(
            Eigen::Quaterniond(Eigen::AngleAxisd(90.0 * M_PI / 180.0, Vec3::UnitY())),
            Vec3(0, 0, 0));
        assert(!vslam::PoseGate::acceptPoseCandidate(
            bad_pose, 30, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 50.0, kRelocRot, q));
    } TEST_PASS();

    TEST("几何不达标直接拒绝（重定位不得绕过内点/RMSE）") {
        const SE3 baseline_twc(Eigen::Quaterniond::Identity(), Vec3(10, 0, 0));
        PoseQuality q;
        // rmse 5.0 > 2.5：即使运动贴近基线也拒绝
        const SE3 near_pose(Eigen::Quaterniond::Identity(), Vec3(-12, 0, 0));
        assert(!vslam::PoseGate::acceptPoseCandidate(
            near_pose, 30, 40, 5.0, 20, 0.4, kMaxRmse,
            baseline_twc, 50.0, kRelocRot, q));
        // 内点 5 < 20
        assert(!vslam::PoseGate::acceptPoseCandidate(
            near_pose, 5, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 50.0, kRelocRot, q));
        // 比例 0.1 < 0.4
        assert(!vslam::PoseGate::acceptPoseCandidate(
            near_pose, 4, 40, 0.5, 20, 0.4, kMaxRmse,
            baseline_twc, 50.0, kRelocRot, q));
        assert(!q.geometric_ok);
    } TEST_PASS();

    TEST("无运动基线时几何达标即通过（初始化首帧/单目）") {
        const std::optional<SE3> no_baseline;
        PoseQuality q;
        const SE3 pose(Eigen::Quaterniond::Identity(), Vec3(-300, 0, 0));
        assert(vslam::PoseGate::acceptPoseCandidate(
            pose, 30, 40, 0.5, 20, 0.4, kMaxRmse, no_baseline, 0.0, 0.0, q));
        assert(q.geometric_ok && q.motion_ok);
        assert(!vslam::PoseGate::acceptPoseCandidate(
            pose, 3, 40, 0.5, 20, 0.4, kMaxRmse, no_baseline, 0.0, 0.0, q));
    } TEST_PASS();
}

// ============================================================
// M1 版本模型测试：Map topology/geometry revision 语义
// ============================================================
void test_map_revision() {
    TEST("Map revision：集合变更 bump topology、坐标变更 bump geometry") {
        auto map = std::make_shared<vslam::Map>();
        assert(map->topologyRevision() == 0);
        assert(map->geometryRevision() == 0);

        auto kf = std::make_shared<vslam::Frame>(0, 0.0);
        map->insertKeyFrame(kf);
        assert(map->topologyRevision() == 1);
        assert(map->geometryRevision() == 0);

        auto mp = std::make_shared<vslam::MapPoint>(0);
        map->insertMapPoint(mp);
        assert(map->topologyRevision() == 2);

        // 坐标变更（仅 BackendCommitter 发布，测试直接调用）
        map->bumpGeometry();
        assert(map->geometryRevision() == 1);
        assert(map->topologyRevision() == 2);

        // 剔除触发 topology
        auto mp2 = std::make_shared<vslam::MapPoint>(1);
        map->insertMapPoint(mp2);
        map->cullMapPoints(2);  // 正式观测数为 0 < 2 → 全部剔除
        assert(map->topologyRevision() == 4);

        // 快照版本绑定 + stale 检测（M2 提交器的前置）
        vslam::OptimizationSnapshot snap;
        snap.topology_revision = map->topologyRevision();
        snap.geometry_revision = map->geometryRevision();
        assert(snap.topology_revision == map->topologyRevision());
        // 快照后地图又插入 KF → 快照过期（stale）
        map->insertKeyFrame(std::make_shared<vslam::Frame>(1, 0.1));
        assert(snap.topology_revision != map->topologyRevision());
    } TEST_PASS();
}

void test_map_observation_model() {
    TEST("MapPoint 正式观测：幂等、重绑、共视、剔除与 clear") {
        auto map = std::make_shared<vslam::Map>();
        auto kf0 = std::make_shared<vslam::Frame>(0, 0.0);
        auto kf1 = std::make_shared<vslam::Frame>(1, 0.1);
        kf0->keypoints.resize(2);
        kf1->keypoints.resize(2);
        kf0->map_points.resize(2);
        kf1->map_points.resize(2);
        map->insertKeyFrame(kf0);
        map->insertKeyFrame(kf1);

        auto mp0 = std::make_shared<vslam::MapPoint>(0);  // id=0 合法
        auto mp1 = std::make_shared<vslam::MapPoint>(1);
        map->insertMapPoint(mp0);
        map->insertMapPoint(mp1);
        assert(map->getMapPoint(0) == mp0);

        assert(map->setObservation(kf0, 0, mp0));
        assert(!map->setObservation(kf0, 0, mp0));  // 幂等
        assert(!map->setObservation(kf0, 1, mp0));  // 同 KF 冲突
        assert(mp0->observationCount() == 1);
        assert(map->sharedObservationCount(0, 1) == 0);
        assert(map->setObservation(kf1, 0, mp0));
        assert(!map->setObservation(kf1, 0, mp0));  // 重复添加不得重复计数
        assert(!mp0->addObservation({0, 1}));  // MapPoint 冲突添加也不得改状态
        assert(mp0->observationCount() == 2);
        assert(map->sharedObservationCount(0, 1) == 1);
        assert(map->verifyObservationConsistency());

        // 普通跟踪帧只持有临时指针，不应把一次 PnP/LK 关联计入正式观测。
        auto tracking = std::make_shared<vslam::Frame>(2, 0.2);
        tracking->map_points.resize(1);
        tracking->map_points[0] = mp0;
        assert(mp0->observationCount() == 2);

        // 直接构造历史重复 slot：清除 formal slot 必须同时清掉 stale slot。
        kf0->map_points[1] = mp0;
        assert(!map->verifyObservationConsistency());
        assert(map->clearObservation(0, 0));
        assert(!kf0->map_points[0] && !kf0->map_points[1]);
        assert(mp0->observationCount() == 1);  // kf1 的正式观测仍在
        assert(map->verifyObservationConsistency());

        // 重绑必须同步删除旧点的反向观测和共视计数。
        assert(map->setObservation(kf0, 0, mp1));
        assert(mp0->observationCount() == 1);
        assert(mp1->observationCount() == 1);
        assert(map->sharedObservationCount(0, 1) == 0);

        // 重绑 duplicate slot 不得误删另一个 KF slot 的合法 formal 观测。
        kf1->map_points[1] = mp0;
        assert(!map->verifyObservationConsistency());
        assert(map->setObservation(kf1, 1, mp1));
        assert(kf1->map_points[0] == mp0);
        assert(kf1->map_points[1] == mp1);
        assert(mp0->observationCount() == 1);
        assert(map->sharedObservationCount(0, 1) == 1);
        assert(map->verifyObservationConsistency());

        // feature_index 必须同时落在 keypoints/map_points 范围内；
        // 直接构造越界历史 slot 后 sync 应将其自愈清空。
        kf0->map_points.resize(3);
        assert(!map->setObservation(kf0, 2, mp1));
        kf0->map_points[2] = mp1;
        assert(!map->verifyObservationConsistency());
        map->syncKeyframeObservations(kf0);
        assert(!kf0->map_points[2]);
        assert(map->verifyObservationConsistency());

        // mp0 只有一个正式观测，应被剔除；slot 与反向集合一并清空。
        map->cullMapPoints(2);
        assert(!map->getMapPoint(0));
        assert(!kf1->map_points[0]);
        assert(map->getMapPoint(1) == mp1);
        assert(map->verifyObservationConsistency());

        assert(map->clearObservation(1, 1));
        map->cullMapPoints(2);
        assert(!map->getMapPoint(1));
        assert(!kf0->map_points[0]);
        assert(!kf1->map_points[1]);
        assert(map->verifyObservationConsistency());

        // sync 覆盖历史代码直接写 slot 的迁移路径。
        auto mp2 = std::make_shared<vslam::MapPoint>(0);
        map->insertMapPoint(mp2);
        kf0->map_points[1] = mp2;
        map->syncKeyframeObservations(kf0);
        assert(mp2->observationCount() == 1);
        assert(mp2->featureIndex(0).value() == 1);
        assert(map->verifyObservationConsistency());

        // removeMapPoint 必须同时清掉 id=0 点的双向观测、KF slot 和共视；
        // 重复删除不应再次减少计数或改变拓扑。
        const auto topology_before_remove = map->topologyRevision();
        assert(map->removeMapPoint(0));
        assert(map->topologyRevision() == topology_before_remove + 1);
        assert(map->mapPointCount() == 0);
        assert(!map->getMapPoint(0));
        assert(mp2->observationCount() == 0);
        assert(!kf0->map_points[1]);
        assert(map->sharedObservationCount(0, 1) == 0);
        assert(map->verifyObservationConsistency());
        assert(!map->removeMapPoint(0));
        assert(map->mapPointCount() == 0);

        // clear 覆盖空/非空关系，不留下任意反向观测。
        map->clear();
        assert(!map->getMapPoint(0));
        assert(!kf0->map_points[0]);
    } TEST_PASS();
}

void test_map_batch_cull() {
    TEST("Map 批量剔除：stale slot、共视和计数一次性保持一致") {
        auto map = std::make_shared<vslam::Map>();
        std::vector<vslam::Frame::Ptr> kfs;
        for (unsigned long id = 0; id < 3; id++) {
            auto kf = std::make_shared<vslam::Frame>(id, id * 0.1);
            kf->keypoints.resize(8);
            kf->map_points.resize(8);
            map->insertKeyFrame(kf);
            kfs.push_back(kf);
        }

        auto keep = std::make_shared<vslam::MapPoint>(10);
        auto weak = std::make_shared<vslam::MapPoint>(11);
        auto stale_only = std::make_shared<vslam::MapPoint>(12);
        map->insertMapPoint(keep);
        map->insertMapPoint(weak);
        map->insertMapPoint(stale_only);

        // keep 出现在三个 KF，weak 出现在两个 KF；剔除阈值为 3，
        // 因而 kf0/kf1 的共视由 2 降为 keep 的 1。
        assert(map->setObservation(kfs[0], 0, keep));
        assert(map->setObservation(kfs[1], 0, keep));
        assert(map->setObservation(kfs[2], 0, keep));
        assert(map->setObservation(kfs[0], 1, weak));
        assert(map->setObservation(kfs[1], 1, weak));
        assert(map->setObservation(kfs[2], 2, stale_only));
        assert(map->sharedObservationCount(0, 1) == 2);

        // 直接构造迁移期间可能遗留的 duplicate/stale slot。批量剔除
        // 必须在正式 observation 删除后仍能一次扫描把这些引用清干净。
        kfs[0]->map_points[4] = weak;
        kfs[1]->map_points[4] = stale_only;
        assert(!map->verifyObservationConsistency());

        const auto topology_before = map->topologyRevision();
        map->cullMapPoints(3);

        assert(map->mapPointCount() == 1);
        assert(map->getMapPoint(10) == keep);
        assert(!map->getMapPoint(11));
        assert(!map->getMapPoint(12));
        assert(keep->observationCount() == 3);
        assert(weak->observationCount() == 0);
        assert(stale_only->observationCount() == 0);
        assert(!kfs[0]->map_points[1] && !kfs[0]->map_points[4]);
        assert(!kfs[1]->map_points[1] && !kfs[1]->map_points[4]);
        assert(!kfs[2]->map_points[2]);
        assert(map->sharedObservationCount(0, 1) == 1);
        assert(map->sharedObservationCount(0, 2) == 1);
        assert(map->sharedObservationCount(1, 2) == 1);
        assert(map->topologyRevision() == topology_before + 1);
        assert(map->verifyObservationConsistency());

        // 再次剔除是幂等的，不会重复减少计数或破坏共视。
        map->cullMapPoints(3);
        assert(map->mapPointCount() == 1);
        assert(map->sharedObservationCount(0, 1) == 1);
        assert(map->verifyObservationConsistency());

        // 滚动地图会一次删除数万点；显式批量 API 必须只发布一次拓扑
        // revision，并保持正式观测/slot 一致。
        auto p13 = std::make_shared<vslam::MapPoint>(13);
        auto p14 = std::make_shared<vslam::MapPoint>(14);
        map->insertMapPoint(p13);
        map->insertMapPoint(p14);
        assert(map->setObservation(kfs[0], 3, p13));
        assert(map->setObservation(kfs[1], 3, p13));
        assert(map->setObservation(kfs[1], 5, p14));
        assert(map->setObservation(kfs[2], 5, p14));
        const auto before_bulk_remove = map->topologyRevision();
        assert(map->removeMapPoints({10, 13, 14}) == 3);
        assert(map->topologyRevision() == before_bulk_remove + 1);
        assert(map->mapPointCount() == 0);
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

// ============================================================
// M2 BackendCommitter 测试：stale/质量验收/原子提交/skip 保护
// ============================================================
void test_backend_committer() {
    using vslam::BackendCommitter;
    using vslam::CommitStatus;

    TEST("M3 子地图坐标组合不变量：T_cw = T_cs·T_ws⁻¹，p_w = T_ws·p_s") {
        const vslam::SE3 T_ws(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.3, vslam::Vec3::UnitY())),
            vslam::Vec3(10, -2, 5));
        const vslam::SE3 T_cs(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.1, vslam::Vec3::UnitZ())),
            vslam::Vec3(0.5, 0.2, -1.0));
        const vslam::Vec3 p_s(2, 3, 8);

        // p_c = T_cs·p_s；p_w = T_ws·p_s；T_cw = T_cs·T_ws⁻¹ → p_c = T_cw·p_w
        const vslam::Vec3 p_w = T_ws * p_s;
        const vslam::Vec3 p_c_direct = T_cs * p_s;
        const vslam::SE3 T_cw = T_cs * T_ws.inverse();
        assert((T_cw * p_w - p_c_direct).norm() < 1e-9);

        // 相机光心世界位置：C_w = T_ws · (T_cs⁻¹ 平移) = T_cw.camera_position()
        const vslam::Vec3 c_w = T_ws * T_cs.inverse().t;
        assert((T_cw.camera_position() - c_w).norm() < 1e-9);

        // 同一局部运动在世界/局部系下增量一致（连续性验收跨系不变）
        const vslam::SE3 T_cs2(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.12, vslam::Vec3::UnitZ())),
            vslam::Vec3(0.6, 0.2, -1.2));
        const vslam::SE3 T_cw2 = T_cs2 * T_ws.inverse();
        const double d_local = (T_cs2.inverse().t - T_cs.inverse().t).norm();
        const double d_world = (T_cw2.inverse().t - T_cw.inverse().t).norm();
        assert(std::abs(d_local - d_world) < 1e-9);
    } TEST_PASS();

    TEST("M5 Atlas 约束图：Relocalization 边对齐子地图锚点") {
        // 子地图 A 固定（T_ws=(10,0,0)），B 锚点差 30m（T_ws=(40,0,0)）。
        // 相机在两子地图局部系均为原点（T_cs_a=T_cs_b=I）→ 一致性要求
        // T_ws_a == T_ws_b；约束 T_rel = T_cs_a⁻¹ ∘ T_cs_b = I。
        vslam::OptimizationSnapshot snap;
        vslam::KeyframeState ka;
        ka.id = 0;
        const vslam::SE3 expected_a(
            Eigen::Quaterniond::Identity(), vslam::Vec3(10, 0, 0));
        ka.pose_cs = expected_a.inverse();  // 位姿图接口输入 T_cw
        vslam::KeyframeState kb;
        kb.id = 1;
        kb.pose_cs = vslam::SE3(
            Eigen::Quaterniond::Identity(), vslam::Vec3(40, 0, 0)).inverse();
        snap.keyframes = {ka, kb};
        // 旧 TrackingBridge（锚点差 30m，低权重）+ 新 Relocalization（0m，高权重）
        snap.constraints.push_back({
            0, 1, vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(30, 0, 0)),
            0.3, true});
        snap.constraints.push_back({
            0, 1, vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3::Zero()),
            1.0, true});  // T_ws_b = T_ws_a · T_rel

        auto result = vslam::Optimizer::solvePoseGraph(snap);
        assert(result.valid);
        vslam::SE3 T_ws_a, T_ws_b;
        for (const auto& u : result.poses) {
            if (u.id == 0) T_ws_a = u.pose_cs.inverse();
            if (u.id == 1) T_ws_b = u.pose_cs.inverse();
        }
        // A 固定不动；B 被高权重 reloc 约束拉回 A（Huber 下桥边被降权，
        // 30m 锚点差收敛到几米内——跨子地图连接不跳变）
        assert((T_ws_a.t - vslam::Vec3(10, 0, 0)).norm() < 1e-9);
        assert(std::abs(T_ws_b.t.x() - 10.0) < 8.0);
        std::cout << " (B.x=" << T_ws_b.t.x() << ")";
    } TEST_PASS();

    TEST("BackendCommitter：正常提交 COMMITTED + geometry++ + 写回生效") {
        auto map = std::make_shared<vslam::Map>();
        auto kf = std::make_shared<vslam::Frame>(0, 0.0);
        map->insertKeyFrame(kf);
        auto mp = std::make_shared<vslam::MapPoint>(0);
        map->insertMapPoint(mp);

        vslam::OptimizationResult r;
        r.base_topology_revision = map->topologyRevision();
        r.base_geometry_revision = map->geometryRevision();
        r.valid = true;
        r.metrics.max_correction = 0.5;
        const vslam::SE3 new_pose(Eigen::Quaterniond::Identity(), vslam::Vec3(1, 2, 3));
        r.poses.push_back({0, new_pose});
        r.points.push_back({0, vslam::Vec3(4, 5, 6)});

        const uint64_t geo_before = map->geometryRevision();
        assert(BackendCommitter::commit(map, r) == CommitStatus::COMMITTED);
        assert(map->geometryRevision() == geo_before + 1);
        assert((map->getKeyFrame(0)->pose_cs.matrix() - new_pose.matrix()).norm() < 1e-12);
        assert((map->getMapPoint(0)->pos_s - vslam::Vec3(4, 5, 6)).norm() < 1e-12);
    } TEST_PASS();

    TEST("BackendCommitter：相同 revision/id 的另一 Map 不能接收旧结果") {
        auto old_map = std::make_shared<vslam::Map>();
        auto new_map = std::make_shared<vslam::Map>();
        auto old_kf = std::make_shared<vslam::Frame>(0, 0.0);
        auto new_kf = std::make_shared<vslam::Frame>(0, 0.0);
        old_map->insertKeyFrame(old_kf);
        new_map->insertKeyFrame(new_kf);

        vslam::OptimizationResult r;
        r.valid = true;
        r.base_geometry_revision = old_map->geometryRevision();
        r.metrics.max_correction = 1.0;
        r.poses.push_back({0, vslam::SE3(Eigen::Quaterniond::Identity(),
                                         vslam::Vec3(1, 0, 0))});

        const auto new_geo = new_map->geometryRevision();
        assert(BackendCommitter::commit(new_map, r, {}, 10.0, old_map)
               == CommitStatus::STALE);
        assert(new_kf->pose_cs.t.norm() < 1e-12);
        assert(new_map->geometryRevision() == new_geo);
    } TEST_PASS();

    TEST("BackendCommitter：几何过期 STALE 且地图逐项不变；拓扑过期允许追加 rebase") {
        auto map = std::make_shared<vslam::Map>();
        auto kf = std::make_shared<vslam::Frame>(0, 0.0);
        map->insertKeyFrame(kf);
        const auto pose_before = kf->pose_cs;

        vslam::OptimizationResult r;
        r.base_topology_revision = map->topologyRevision() - 1;  // 拓扑过期
        r.base_geometry_revision = map->geometryRevision();
        r.valid = true;
        r.poses.push_back({0, vslam::SE3(Eigen::Quaterniond::Identity(),
                                         vslam::Vec3(9, 9, 9))});
        // M6：几何未变 → 追加 rebase 允许提交（异步后端核心）
        assert(BackendCommitter::commit(map, r) == CommitStatus::COMMITTED);
        // 几何过期 → 整笔丢弃，地图逐项不变
        r.base_topology_revision = map->topologyRevision();
        r.base_geometry_revision = map->geometryRevision() - 1;
        const auto pose_after = kf->pose_cs;
        assert(BackendCommitter::commit(map, r) == CommitStatus::STALE);
        assert((kf->pose_cs.matrix() - pose_after.matrix()).norm() < 1e-12);
        // isStale 纯函数
        assert(BackendCommitter::isStale(r, map));
        r.base_geometry_revision = map->geometryRevision();
        assert(!BackendCommitter::isStale(r, map));
    } TEST_PASS();

    TEST("BackendCommitter：快照对象被预算剔除时整笔 STALE，不得部分提交") {
        auto map = std::make_shared<vslam::Map>();
        auto kf = std::make_shared<vslam::Frame>(7, 0.7);
        auto mp = std::make_shared<vslam::MapPoint>(9);
        map->insertKeyFrame(kf);
        map->insertMapPoint(mp);
        vslam::OptimizationResult r;
        r.base_geometry_revision = map->geometryRevision();
        r.valid = true;
        r.metrics.max_correction = 0.2;
        r.poses.push_back({7, vslam::SE3(
            Eigen::Quaterniond::Identity(), vslam::Vec3(1, 0, 0))});
        r.points.push_back({9, vslam::Vec3(2, 3, 4)});
        assert(map->removeMapPoints({9}) == 1);
        const auto geo_before = map->geometryRevision();
        assert(BackendCommitter::commit(map, r) == CommitStatus::STALE);
        assert(map->geometryRevision() == geo_before);
        assert(kf->pose_cs.t.norm() < 1e-12);
    } TEST_PASS();

    TEST("BackendCommitter：无效结果/超大校正被拒绝 INVALID") {
        auto map = std::make_shared<vslam::Map>();
        vslam::OptimizationResult r;
        r.base_topology_revision = map->topologyRevision();
        r.base_geometry_revision = map->geometryRevision();
        r.valid = false;
        assert(BackendCommitter::commit(map, r) == CommitStatus::INVALID);
        r.valid = true;
        r.metrics.max_correction = 50.0;  // > 10m 上限
        assert(BackendCommitter::commit(map, r) == CommitStatus::INVALID);
        assert(!BackendCommitter::passesQuality(r, 10.0));
    } TEST_PASS();

    TEST("BackendCommitter：skip_pose 保护活动参考帧") {
        auto map = std::make_shared<vslam::Map>();
        map->insertKeyFrame(std::make_shared<vslam::Frame>(0, 0.0));
        map->insertKeyFrame(std::make_shared<vslam::Frame>(1, 0.1));
        const auto pose0 = map->getKeyFrame(0)->pose_cs;

        vslam::OptimizationResult r;
        r.base_topology_revision = map->topologyRevision();
        r.base_geometry_revision = map->geometryRevision();
        r.valid = true;
        r.poses.push_back({0, vslam::SE3(Eigen::Quaterniond::Identity(),
                                         vslam::Vec3(9, 9, 9))});
        r.poses.push_back({1, vslam::SE3(Eigen::Quaterniond::Identity(),
                                         vslam::Vec3(3, 3, 3))});
        assert(BackendCommitter::commit(map, r, {0}) == CommitStatus::COMMITTED);
        assert((map->getKeyFrame(0)->pose_cs.matrix() - pose0.matrix()).norm() < 1e-12);
        assert((map->getKeyFrame(1)->pose_cs.t - vslam::Vec3(3, 3, 3)).norm() < 1e-12);
    } TEST_PASS();
}

// ============================================================
// 双目相机模型测试：右目投影 / 视差-深度互逆 / 离轴点投影
// ============================================================
void test_stereo_camera() {
    auto cam = std::make_shared<vslam::StereoCamera>();
    cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
    cam->img_width = 640; cam->img_height = 480;
    cam->fx_r = 500; cam->fy_r = 500; cam->cx_r = 320; cam->cy_r = 240;
    cam->baseline_m = 0.5;

    TEST("StereoCamera 右目投影/视差/深度一致性") {
        // 主轴上 5m 的点：右目 x 应左移 fx*b/z = 500*0.5/5 = 50px
        vslam::Vec3 p_c(0, 0, 5.0);
        vslam::Vec2 pr = cam->camera2pixelRight(p_c);
        assert(std::abs(pr.x() - 270.0) < 1e-6);
        assert(std::abs(pr.y() - 240.0) < 1e-6);

        // 视差→深度 与 深度→视差 互逆：z = fx*b/d
        double disparity = 320.0 - pr.x();
        double depth = cam->disparityToDepth(disparity);
        assert(std::abs(depth - 5.0) < 1e-9);

        // 左目像素 + 视差深度反投影 → 相机系 3D 坐标
        vslam::Vec3 p3 = cam->pixel2camera(vslam::Vec2(320, 240), depth);
        assert(std::abs(p3.z() - 5.0) < 1e-9);
        assert(std::abs(p3.x()) < 1e-9);
    } TEST_PASS();

    TEST("StereoCamera 离轴点右目投影") {
        // 点 (1, 0.5, 4)：右目 x = fx_r*(x-b)/z + cx_r = 500*(1-0.5)/4+320 = 382.5
        //                   右目 y = fy_r*y/z + cy_r = 500*0.5/4+240 = 302.5
        vslam::Vec3 p_c(1.0, 0.5, 4.0);
        vslam::Vec2 pr = cam->camera2pixelRight(p_c);
        assert(std::abs(pr.x() - 382.5) < 1e-6);
        assert(std::abs(pr.y() - 302.5) < 1e-6);
    } TEST_PASS();
}

void test_stereo_match_quality() {
    TEST("双目 LK：水平视差通过，纵向错位被拒绝") {
        cv::Mat left(240, 320, CV_8UC1);
        cv::RNG rng(42);
        rng.fill(left, cv::RNG::UNIFORM, 0, 255);

        vslam::FeatureMatcher matcher;
        auto frame = std::make_shared<vslam::Frame>(0, 0.0);
        frame->image_gray = left;
        matcher.extract(frame);

        auto shifted = [&](double dy) {
            cv::Mat right;
            cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, -8, 0, 1, dy);
            cv::warpAffine(left, right, affine, left.size(), cv::INTER_LINEAR,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
            return right;
        };

        std::vector<cv::Point2f> right_pts;
        auto horizontal_status = matcher.matchStereo(
            left, shifted(0.0), frame->keypoints, right_pts);
        std::vector<double> disparities;
        for (size_t i = 0; i < horizontal_status.size(); i++) {
            if (horizontal_status[i])
                disparities.push_back(frame->keypoints[i].pt.x - right_pts[i].x);
        }
        assert(disparities.size() > 100);
        std::ranges::sort(disparities);
        const double median_disparity = disparities[disparities.size() / 2];
        assert(std::abs(median_disparity - 8.0) < 0.5);

        auto vertical_status = matcher.matchStereo(
            left, shifted(3.0), frame->keypoints, right_pts);
        const int vertical_valid = std::count(vertical_status.begin(), vertical_status.end(), 1);
        assert(vertical_valid < (int)disparities.size() / 10);
    } TEST_PASS();
}

void test_invalid_stereo_initialization() {
    TEST("双目初始化：错误左右目不能创建空地图") {
        auto cam = std::make_shared<vslam::StereoCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 160; cam->cy = 120;
        cam->img_width = 320; cam->img_height = 240;
        cam->fx_r = 500; cam->fy_r = 500; cam->cx_r = 160; cam->cy_r = 120;
        cam->baseline_m = 0.5;

        cv::Mat left(240, 320, CV_8UC1);
        cv::RNG rng(7);
        rng.fill(left, cv::RNG::UNIFORM, 0, 255);
        cv::Mat right;
        cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, -8, 0, 1, 3);
        cv::warpAffine(left, right, affine, left.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(0));

        vslam::VOConfig cfg;
        cfg.stereo_min_points = 40;
        vslam::VisualOdometry vo(cam, cfg);
        vo.addFrame(left, right, 0.0);
        assert(vo.state() == vslam::VisualOdometry::State::INITIALIZING);
        assert(!vo.getStatus().pose_valid);
        assert(vo.getMap()->mapPointCount() == 0);
    } TEST_PASS();
}

// ============================================================
// 双目 VO 测试：合成方块场景精确渲染左右目
// 验证：1) 首帧即 TRACKING（绝对尺度，无需对极初始化）
//       2) 前进 1m 的位移尺度正确（双目核心优势 vs 单目尺度不可观测）
// ============================================================
void test_stereo_vo() {
    auto cam = std::make_shared<vslam::StereoCamera>();
    cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
    cam->img_width = 640; cam->img_height = 480;
    cam->fx_r = 500; cam->fy_r = 500; cam->cx_r = 320; cam->cy_r = 240;
    cam->baseline_m = 0.5;

    struct Blk { cv::Point3f c; float sx, sy; int gray; };
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> dx(-4, 4), dy(-3, 3), dz(3, 6),
                                           ds(0.5, 1.5), dg(80, 255);
    std::vector<Blk> blks;
    for (int i = 0; i < 60; i++)
        blks.push_back({cv::Point3f(dx(gen), dy(gen), dz(gen)),
                        (float)ds(gen), (float)ds(gen), (int)dg(gen)});

    // 渲染左右目：左目用 world2pixel，右目用 camera2pixelRight
    auto render = [&](const vslam::SE3& T_wc, cv::Mat& left, cv::Mat& right) {
        vslam::SE3 T_cw = T_wc.inverse();
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
            cv::fillConvexPoly(left, pi_l, cv::Scalar(b.gray));
            cv::fillConvexPoly(right, pi_r, cv::Scalar(b.gray));
        }
    };

    TEST("双目 VO：首帧即建图 + 绝对尺度跟踪") {
        vslam::VOConfig cfg;
        cfg.min_matches_track = 10;
        vslam::VisualOdometry vo(cam, cfg);

        // 帧 1（原点）：双目首帧直接建图，状态应为 TRACKING
        cv::Mat l1, r1;
        render(vslam::SE3(), l1, r1);
        vo.addFrame(l1, r1, 0.0);
        assert(vo.state() == vslam::VisualOdometry::State::TRACKING);
        assert(vo.getStatus().pose_valid);
        assert(vo.getStatus().stereo_points >= cfg.stereo_min_points);
        assert(vo.getMap()->mapPointCount() > 50);
        auto first_kf = vo.getMap()->getKeyFrame(0);
        assert(first_kf);
        assert(!first_kf->image.empty());       // 当前帧必须仍可供 Viewer 使用
        assert(!first_kf->image_gray.empty());

        // 帧 2：相机沿 +z 前进 1m，位移尺度应 ≈1m（绝对尺度）
        cv::Mat l2, r2;
        vslam::SE3 T_wc2(Eigen::Quaterniond::Identity(), vslam::Vec3(0, 0, 1.0));
        render(T_wc2, l2, r2);
        auto pose2 = vo.addFrame(l2, r2, 0.1);
        double disp = pose2.inverse().t.norm();   // T_cw → T_wc 位移
        std::cout << " (disp=" << disp << "m mp=" << vo.getMap()->mapPointCount() << ")";
        assert(vo.getMap()->verifyObservationConsistency());
        assert(std::abs(disp - 1.0) < 0.3);
        assert(vo.getStatus().pose_valid);
        assert(first_kf->image.empty());        // 历史关键帧不再持有像素缓冲
        assert(first_kf->image_gray.empty());
        assert(first_kf->image_right.empty());
        assert(first_kf->image_right_gray.empty());
        assert(!first_kf->descriptors.empty());
        assert(!first_kf->keypoints.empty());
        assert(!first_kf->map_points.empty());
        assert(!first_kf->pts_c.empty());
        assert(!vo.currentFrame()->image.empty());
        assert(!vo.currentFrame()->image_right.empty());

        // LK 普通帧在插入关键帧时必须先重提 ORB、再重算深度；否则
        // extract 会清掉刚建立的 slot，留下反向 stale Observation。
        vslam::VOConfig lk_cfg = cfg;
        lk_cfg.feature_method = 1;
        vslam::VisualOdometry lk_vo(cam, lk_cfg);
        lk_vo.addFrame(l1, r1, 0.0);
        assert(lk_vo.getMap()->verifyObservationConsistency());
        lk_vo.addFrame(l2, r2, 0.1);
        assert(lk_vo.getMap()->keyFrameCount() >= 2);
        assert(lk_vo.getMap()->verifyObservationConsistency());
    } TEST_PASS();
}

void test_mini_atlas() {
    TEST("MiniAtlas 子地图锚定与切换") {
        vslam::Atlas atlas;
        auto& first = atlas.createSubmap(vslam::SE3());
        const auto first_id = first.id;
        auto first_map = first.map;
        assert(first_id == 0);
        assert(first.map != nullptr);
        assert(atlas.activeMap() == first.map);

        vslam::SE3 anchor(Eigen::Quaterniond::Identity(), vslam::Vec3(10, 2, -3));
        auto& second = atlas.createSubmap(anchor);
        assert(second.id == 1);
        assert(second.T_ws.t.isApprox(anchor.t));
        assert(first.frozen);
        assert(atlas.activeMap() == second.map);
        assert(atlas.submapCount() == 2);

        auto& disconnected = atlas.createSubmap(anchor, false);
        assert(!disconnected.connected);
        assert(atlas.submapCount() == 3);

        assert(atlas.activate(first_id));
        assert(atlas.activeMap() == first_map);
    } TEST_PASS();
}

// ============================================================
// Phase 2: Sim3 相似变换（回环校正核心）
// ============================================================
void test_sim3() {
    TEST("Sim3 正变换/逆变换/matrix 互转/toSE3") {
        const double s = 1.37;
        const vslam::Sim3 sim(s,
            Eigen::Quaterniond(Eigen::AngleAxisd(0.5, vslam::Vec3::UnitY())),
            vslam::Vec3(2.0, -1.5, 3.0));
        const vslam::Vec3 p(1.0, 2.0, 3.0);

        // 正变换：p' = s·R·p + t
        vslam::Vec3 p1 = sim * p;
        vslam::Vec3 expected = s * (sim.q * p) + sim.t;
        assert(p1.isApprox(expected, 1e-9));

        // 逆变换恢复
        vslam::Vec3 p2 = sim.inverse() * p1;
        assert(p2.isApprox(p, 1e-9));

        // matrix()/fromMatrix 互转
        vslam::Mat44 M = sim.matrix();
        vslam::Sim3 sim2 = vslam::Sim3::fromMatrix(M);
        assert(std::abs(sim2.s - sim.s) < 1e-9);
        assert(sim2.q.toRotationMatrix().isApprox(sim.q.toRotationMatrix(), 1e-9));
        assert(sim2.t.isApprox(sim.t, 1e-9));
        // 左上 3x3 = s·R，列范数 = s
        assert(std::abs(M.block<3, 3>(0, 0).col(0).norm() - s) < 1e-9);

        // toSE3 丢尺度：平移保持，旋转保持，尺度丢失
        vslam::SE3 se3 = sim.toSE3();
        assert(se3.t.isApprox(sim.t, 1e-9));
        assert(se3.q.toRotationMatrix().isApprox(sim.q.toRotationMatrix(), 1e-9));

        // Sim3 组合一致性：sim1 ∘ sim2 作用于点 = 依次作用
        const vslam::Sim3 sim_a(0.9,
            Eigen::Quaterniond(Eigen::AngleAxisd(-0.3, vslam::Vec3::UnitX())),
            vslam::Vec3(1, 1, 1));
        vslam::Vec3 pa = sim * (sim_a * p);
        vslam::Mat44 combined = sim.matrix() * sim_a.matrix();
        assert(pa.isApprox((vslam::Sim3::fromMatrix(combined) * p), 1e-9));
    } TEST_PASS();

    TEST("Sim3::estimate Umeyama 恢复精度") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-5.0, 5.0);
        const vslam::Sim3 sim(1.37,
            Eigen::Quaterniond(Eigen::AngleAxisd(0.5, vslam::Vec3::UnitY())),
            vslam::Vec3(2.0, -1.5, 3.0));

        std::vector<vslam::Vec3> src, dst;
        for (int i = 0; i < 30; i++) {
            vslam::Vec3 p(dist(rng), dist(rng), dist(rng));
            src.push_back(p);
            dst.push_back(sim * p);
        }
        vslam::Sim3 out;
        assert(vslam::Sim3::estimate(src, dst, out));
        assert(std::abs(out.s - sim.s) < 1e-6);
        assert(out.matrix().isApprox(sim.matrix(), 1e-6));

        // 无噪声下逐点恢复
        for (size_t i = 0; i < src.size(); i++)
            assert((out * src[i]).isApprox(dst[i], 1e-6));
    } TEST_PASS();

    TEST("Sim3::estimate 退化输入返回 false") {
        vslam::Sim3 out;
        std::vector<vslam::Vec3> src, dst;

        // 点数 < 3
        src = {vslam::Vec3(0, 0, 0), vslam::Vec3(1, 0, 0)};
        dst = {vslam::Vec3(0, 0, 0), vslam::Vec3(2, 0, 0)};
        assert(!vslam::Sim3::estimate(src, dst, out));

        // 共线点（秩 1 退化）
        src.clear(); dst.clear();
        for (int i = 0; i < 5; i++) {
            vslam::Vec3 p(i, 2 * i, -3 * i);
            src.push_back(p);
            dst.push_back(1.5 * p + vslam::Vec3(1, 1, 1));
        }
        assert(!vslam::Sim3::estimate(src, dst, out));

        // 共面点（z=0，秩 2 退化：深度方向不可观）
        src.clear(); dst.clear();
        for (int i = 0; i < 5; i++) {
            vslam::Vec3 p(i, i * i, 0);
            src.push_back(p);
            dst.push_back(1.2 * p + vslam::Vec3(0, 0, 1));
        }
        assert(!vslam::Sim3::estimate(src, dst, out));
    } TEST_PASS();
}

// ============================================================
// Phase 2: 位姿图优化（回环约束校正累积漂移）
// ============================================================
void test_pose_graph() {
#ifdef HAS_G2O
    TEST("位姿图优化：回环约束校正累积漂移") {
        const int N = 10;
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> noise(-0.01, 0.01);

        // 真实位姿（T_wc）：沿 X 轴每帧 1m
        std::vector<vslam::SE3> gt;
        for (int i = 0; i < N; i++)
            gt.push_back(vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(i, 0, 0)));

        // 估计位姿：每帧系统偏航漂移 0.05 rad + 小噪声 → 累计漂移
        std::vector<vslam::SE3> est;  // T_wc（含漂移）
        vslam::SE3 cur = gt[0];
        est.push_back(cur);
        for (int i = 1; i < N; i++) {
            vslam::SE3 rel_true(Eigen::Quaterniond::Identity(), vslam::Vec3(1, 0, 0));
            vslam::SE3 drift(
                Eigen::Quaterniond(Eigen::AngleAxisd(0.05 + noise(rng), vslam::Vec3::UnitZ())),
                vslam::Vec3(noise(rng), noise(rng), 0));
            cur = cur * rel_true * drift;
            est.push_back(cur);
        }

        // 建图：关键帧 pose_cs = T_wc⁻¹
        auto map = std::make_shared<vslam::Map>();
        for (int i = 0; i < N; i++) {
            auto kf = std::make_shared<vslam::Frame>((unsigned long)i, (double)i);
            kf->pose_cs = est[i].inverse();
            map->insertKeyFrame(kf);
        }

        // 确认存在显著漂移（否则测试无意义）
        double err_before = (est[N - 1].t - gt[N - 1].t).norm();
        assert(err_before > 1.0);

        // M1：构建只读快照（位姿 + 约束），纯计算
        vslam::OptimizationSnapshot snap;
        snap.topology_revision = map->topologyRevision();
        snap.geometry_revision = map->geometryRevision();
        for (int i = 0; i < N; i++) {
            vslam::KeyframeState ks;
            ks.id = (unsigned long)i;
            ks.pose_cs = map->getKeyFrame((unsigned long)i)->pose_cs;
            snap.keyframes.push_back(std::move(ks));
        }
        // 冻结每个 KF 插入时得到的里程计相对测量。
        for (int i = 1; i < N; i++) {
            snap.constraints.push_back({
                (unsigned long)(i - 1), (unsigned long)i,
                est[i - 1].inverse() * est[i], 1.0, false});
        }
        // 回环边：末帧 → 首帧（已知真实回环约束，高置信）
        snap.constraints.push_back({
            0, (unsigned long)(N - 1),
            gt[0].inverse() * gt[N - 1], 10.0, true});

        auto result = vslam::Optimizer::solvePoseGraph(snap);
        assert(result.valid);

        // 应用候选增量后，末帧误差应显著下降
        vslam::SE3 Twc_after;
        for (auto& u : result.poses) {
            if (u.id == (unsigned long)(N - 1)) {
                Twc_after = u.pose_cs.inverse();
                break;
            }
        }
        double err_after = (Twc_after.t - gt[N - 1].t).norm();
        std::cout << " (drift_before=" << err_before << "m after=" << err_after << "m)";
        assert(err_after < err_before * 0.2);
        assert(err_after < 0.5);
    } TEST_PASS();

    TEST("位姿图拒绝百万米错误回环且不修改地图") {
        constexpr int N = 10;
        auto map = std::make_shared<vslam::Map>();
        vslam::OptimizationSnapshot snap;
        for (int i = 0; i < N; i++) {
            auto kf = std::make_shared<vslam::Frame>((unsigned long)i, (double)i);
            const vslam::SE3 Twc(Eigen::Quaterniond::Identity(), vslam::Vec3(i, 0, 0));
            kf->pose_cs = Twc.inverse();
            map->insertKeyFrame(kf);
            vslam::KeyframeState ks;
            ks.id = (unsigned long)i;
            ks.pose_cs = kf->pose_cs;
            snap.keyframes.push_back(std::move(ks));
            if (i > 0) {
                snap.constraints.push_back({
                    (unsigned long)(i - 1), (unsigned long)i,
                    vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(1, 0, 0)),
                    1.0, false});
            }
        }
        snap.constraints.push_back({
            0, (unsigned long)(N - 1),
            vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(1e6, 0, 0)),
            10.0, true});

        auto result = vslam::Optimizer::solvePoseGraph(snap);
        assert(!result.valid);   // 恶性边被防护拒绝，无候选增量
        assert(result.poses.empty());
        for (int i = 0; i < N; i++) {
            const auto Twc = map->getKeyFrame((unsigned long)i)->pose_cs.inverse();
            assert((Twc.t - vslam::Vec3(i, 0, 0)).norm() < 1e-12);
        }
    } TEST_PASS();

    TEST("位姿图显式固定已发布历史节点，只校正活动尾节点") {
        vslam::OptimizationSnapshot snap;
        for (unsigned long i = 0; i < 3; ++i) {
            const double x = i == 2 ? 30.0 : 10.0 * static_cast<double>(i);
            const vslam::SE3 Twc(Eigen::Quaterniond::Identity(),
                                  vslam::Vec3(x, 0, 0));
            snap.keyframes.push_back({i, Twc.inverse()});
        }
        snap.fixed_kf_ids = {0, 1};
        snap.constraints.push_back({
            0, 1, vslam::SE3(Eigen::Quaterniond::Identity(),
                             vslam::Vec3(10, 0, 0)), 1.0, false});
        snap.constraints.push_back({
            1, 2, vslam::SE3(Eigen::Quaterniond::Identity(),
                             vslam::Vec3(10, 0, 0)), 10.0, true});
        const auto result = vslam::Optimizer::solvePoseGraph(snap);
        assert(result.valid);
        for (const auto& update : result.poses) {
            const double x = update.pose_cs.inverse().t.x();
            if (update.id == 1) assert(std::abs(x - 10.0) < 1e-10);
            if (update.id == 2) assert(std::abs(x - 20.0) < 0.1);
        }
    } TEST_PASS();
#else
    TEST("位姿图优化：回环约束校正累积漂移") {
        vslam::OptimizationSnapshot snap;
        assert(!vslam::Optimizer::solvePoseGraph(snap).valid);
        std::cout << "SKIPPED (built without g2o)";
    } TEST_PASS();
#endif
}

// ============================================================
// Phase 2: 合成回环（先走远再回起点，验证词袋检测 + 几何验证 + Sim3）
// ============================================================
void test_loop_closure() {
#ifdef HAS_DBOW3
    TEST("LoopClosure 合成回环：词袋命中 + PnP 验证 + SE3 回环边") {
        // 相机（合成针孔）
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = cam->fy = 500; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;

        // 测试资源固定相对源码根定位，避免独立构建目录下依赖 cwd。
        const std::string vocab = VSLAM_SOURCE_DIR "/config/ORBvoc.dbow3";
        assert(std::filesystem::exists(vocab));

        vslam::LoopClosure lc;
        // min_score 取 0.05：合成圆点特征的 ORB 描述子对微小视差很敏感，
        // 末帧偏离原点 0.3m 后 BoW 分数会跌破 0.3。时间窗（30）才是防误报的
        // 主闸，分数阈值只兜底，PnP 验证（min_loop_inliers=30, ratio=0.7）
        // 仍会拒绝弱候选。
        // 合成场景是随机圆点，ORB 描述子误匹配率高：PnP 内点 22/77（ratio 0.29）。
        // 本测试的目的是覆盖 T_cw 求逆语义回归（line 1075 后的平移断言），
        // 不是调阈值——生产参数（min_loop_inliers=50 / ratio=0.85）对合成图过严，
        // 0.7/30 也会在部分 OpenCV 版本下 PnP 内点率不足而误失败（2026-08-06 复现）。
        // 放宽到 0.4/15，仍要求 PnP 真实支撑 + 平移断言（<0.1m）把关语义。
        lc.setParams(0.05, 30, 15, 0.4, 3.0, cam);
        assert(lc.loadVocabulary(vocab));

        // 世界点云：相机前方（世界系 +Z 方向 10~30m，相机光轴默认朝 +Z）
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> d(-12.0, 12.0);
        std::uniform_real_distribution<double> dz(10.0, 30.0);
        std::vector<vslam::Vec3> pts;
        for (int i = 0; i < 800; i++)
            pts.push_back(vslam::Vec3(d(rng), d(rng), dz(rng)));

        // 渲染：把点云投影到图像（圆点 = 角点），返回像素与对应世界坐标
        auto render = [&](const vslam::SE3& T_wc,
                          std::vector<cv::Point2f>& px_out,
                          std::vector<vslam::Vec3>& pw_out) {
            cv::Mat img(cam->img_height, cam->img_width, CV_8UC1, cv::Scalar(0));
            const vslam::SE3 T_cw = T_wc.inverse();
            px_out.clear(); pw_out.clear();
            for (size_t i = 0; i < pts.size(); i++) {
                if ((T_cw * pts[i]).z() < 0.5) continue;
                const vslam::Vec2 uv = cam->world2pixel(pts[i], T_cw);
                if (uv.x() < 5 || uv.y() < 5 ||
                    uv.x() > cam->img_width - 5 || uv.y() > cam->img_height - 5) continue;
                // 每个点画成半径 4 的实心圆盘 + 唯一灰度值：
                // 旧实现 2px 圆点 + 黑色背景 → ORB 31x31 patch 几乎全黑，
                // 描述子高度相似 → 帧间大量误匹配 → PnP 内点率低 → verifyLoop
                // 的 solvePnPRansac 在 100 次迭代内找不到好假设（本机稳定 ok=0，
                // 见 §3.16）。实心圆盘让每个点有独特局部纹理，匹配可重复。
                cv::circle(img, cv::Point((int)uv.x(), (int)uv.y()), 4,
                           cv::Scalar((i * 37) % 256), -1);
                px_out.push_back(cv::Point2f((float)uv.x(), (float)uv.y()));
                pw_out.push_back(pts[i]);
            }
            return img;
        };

        // 相机位姿序列（T_wc，朝向恒为 -Z）：沿矩形走一圈回到起点
        const double step = 1.0;
        std::vector<vslam::SE3> poses;
        for (int i = 0; i <= 10; i++)  // 段1：+X
            poses.push_back(vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(i * step, 0, 0)));
        for (int i = 1; i <= 10; i++)  // 段2：+Y
            poses.push_back(vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(10 * step, i * step, 0)));
        for (int i = 1; i <= 10; i++)  // 段3：-X
            poses.push_back(vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3((10 - i) * step, 10 * step, 0)));
        for (int i = 1; i <= 10; i++)  // 段4：-Y（回到起点附近）
            poses.push_back(vslam::SE3(Eigen::Quaterniond::Identity(), vslam::Vec3(0, (10 - i) * step, 0)));
        // 末帧故意偏离原点（非单位位姿，T_cw ≠ T_wc）：若 verifyLoop 把
        // solvePnP 输出的 T_cw 误当 T_wc 而漏掉求逆，回环边测量会反号，
        // 下面的期望断言（cand->pose_cs * T_wc_curr）必然失败——覆盖 6c311d7
        // 回归（当时末帧恰为单位位姿，正反求逆结果相同，测试无法区分）。
        // 偏移取 0.15m 级：足够区分（反号误差 2×||t||≈0.36m >> 0.1m 容差），
        // 又尽量保持合成圆点特征的 ORB 匹配内点比例（0.3m 时跌到 0.67）。
        poses.back() = vslam::SE3(Eigen::Quaterniond::Identity(),
                                  vslam::Vec3(0.15, 0.10, 0.0));

        // 逐帧：渲染 → 提取 ORB → 关联地图点 → 入词袋数据库
        vslam::FeatureMatcher matcher;
        matcher.setParams(1000, 1.2, 8);
        vslam::Frame::Ptr last_kf, mid_kf;
        unsigned long kf_id = 0;
        for (auto& T_wc : poses) {
            std::vector<cv::Point2f> px;
            std::vector<vslam::Vec3> pw;
            cv::Mat img = render(T_wc, px, pw);

            auto kf = std::make_shared<vslam::Frame>(kf_id, (double)kf_id);
            kf->image = img;
            kf->image_gray = img;
            kf->pose_cs = T_wc.inverse();
            matcher.extract(kf);

            // 关键点 → 最近圆点 → 世界坐标（3px 内才关联）
            kf->map_points.assign(kf->keypoints.size(), nullptr);
            for (size_t j = 0; j < kf->keypoints.size(); j++) {
                double best_d2 = 9.0;
                int best_i = -1;
                for (size_t k = 0; k < px.size(); k++) {
                    const double dx = kf->keypoints[j].pt.x - px[k].x;
                    const double dy = kf->keypoints[j].pt.y - px[k].y;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 < best_d2) { best_d2 = d2; best_i = (int)k; }
                }
                if (best_i >= 0) {
                    auto mp = std::make_shared<vslam::MapPoint>((unsigned long)j);
                    mp->pos_s = pw[best_i];
                    kf->map_points[j] = mp;
                }
            }
            lc.addKeyFrame(kf, 0, vslam::SE3());
            last_kf = kf;
            if (kf_id == 20) mid_kf = kf;  // 中段帧（负例用）
            kf_id++;
        }

        // 回环检测：最后一帧（回到原点）应命中早期帧（id 差 > 30，时间窗过滤通过）
        auto cands = lc.detectLoop(last_kf, 0, vslam::SE3(),
                                   {{0, vslam::SE3()}});
        std::cout << " (cands=" << cands.size()
                  << " top=kf#" << (cands.empty() ? 0 : cands.front().frame->id)
                  << " last=kf#" << last_kf->id << ")";
        assert(!cands.empty());
        assert(cands.front().frame->id < last_kf->id);

        // 几何验证：PnP 直接输出 loop→curr 的位姿图 SE3 测量。
        // 它应与无漂移合成真值一致，不依赖 last_kf 中保存的 VO 漂移位姿。
        last_kf->pose_cs.t += vslam::Vec3(3.0, -2.0, 1.0);
        vslam::SE3 T_loop_curr;
        assert(lc.verifyLoop(last_kf, cands.front().frame, T_loop_curr));
        const vslam::SE3 expected = cands.front().frame->pose_cs * poses.back();
        // isApprox 是相对精度（|a-b|² ≤ prec²·min(|a|²,|b|²)），期望平移为零时
        // 分母恒为 0，任何非零误差都会失败，故用绝对范数判据。
        assert((T_loop_curr.t - expected.t).norm() < 0.1);
        assert(T_loop_curr.q.toRotationMatrix().isApprox(
            expected.q.toRotationMatrix(), 0.02));

        // 负例：中段帧（id 20，所有高分候选都在时间窗内）应返回空候选
        assert(mid_kf != nullptr);
        auto none = lc.detectLoop(mid_kf, 0, vslam::SE3(),
                                  {{0, vslam::SE3()}});
        assert(none.empty());
    } TEST_PASS();
#else
    TEST("LoopClosure 合成回环") {
        std::cout << "SKIPPED (built without DBoW3)";
    } TEST_PASS();
#endif
}

void test_loop_closure_concurrency_and_position_cache() {
#ifdef HAS_DBOW3
    TEST("LoopClosure 并发 add/detect 与光心位置缓存") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = cam->fy = 500;
        cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;

        const std::string vocab = VSLAM_SOURCE_DIR "/config/ORBvoc.dbow3";
        assert(std::filesystem::exists(vocab));

        vslam::LoopClosure lc;
        lc.setParams(0.0, 1, 8, 0.1, 3.0, cam, 20, 5.0, 1);
        assert(lc.loadVocabulary(vocab));

        constexpr int kFrameCount = 32;
        std::vector<vslam::Frame::Ptr> frames;
        frames.reserve(kFrameCount);
        for (int i = 0; i < kFrameCount; ++i) {
            auto frame = std::make_shared<vslam::Frame>(
                static_cast<unsigned long>(i), static_cast<double>(i));
            frame->pose_cs = vslam::SE3(
                Eigen::Quaterniond::Identity(), vslam::Vec3(-i * 0.1, 0, 0));
            frame->descriptors = cv::Mat(48, 32, CV_8U);
            for (int r = 0; r < frame->descriptors.rows; ++r)
                for (int c = 0; c < frame->descriptors.cols; ++c)
                    frame->descriptors.at<unsigned char>(r, c) =
                        static_cast<unsigned char>((r * 17 + c * 31 + i * 13) & 255);
            frames.push_back(std::move(frame));
        }
        auto query = std::make_shared<vslam::Frame>(1000, 1000.0);
        query->pose_cs = vslam::SE3();
        query->descriptors = frames.front()->descriptors.clone();

        // addKeyFrame 缓存的是加入瞬间的 T_cw 光心；之后修改 live Frame
        // 位姿不应改变位置先验结果。
        auto cached_position = std::make_shared<vslam::Frame>(200, 200.0);
        cached_position->pose_cs = vslam::SE3();
        cached_position->descriptors = frames.front()->descriptors.clone();
        lc.addKeyFrame(cached_position, 10, vslam::SE3());
        cached_position->pose_cs.t = vslam::Vec3(-1000, 0, 0);
        auto position_candidates = lc.detectLoop(query, 10, vslam::SE3(),
                                                 {{10, vslam::SE3()}});
        assert(std::ranges::any_of(position_candidates,
                                   [](const auto& candidate) {
                                       return candidate.frame &&
                                              candidate.frame->id == 200 &&
                                              candidate.submap_id == 10;
                                   }));

        // 并发回归只验证状态安全；关闭位置先验和候选日志，避免测试输出
        // 被重复检测结果淹没。
        lc.setParams(1.1, 1, 8, 0.1, 3.0, cam, 20, 0.0, 1);

        std::vector<std::thread> workers;
        for (int worker = 0; worker < 4; ++worker) {
            workers.emplace_back([&, worker] {
                for (int i = worker; i < kFrameCount; i += 4)
                    lc.addKeyFrame(frames[i], 10, vslam::SE3());
            });
        }
        for (int worker = 0; worker < 4; ++worker) {
            workers.emplace_back([&] {
                for (int i = 0; i < 10; ++i)
                    (void)lc.detectLoop(query, 10, vslam::SE3(),
                                        {{10, vslam::SE3()}});
            });
        }
        for (auto& worker : workers) worker.join();

    } TEST_PASS();
#else
    TEST("LoopClosure 并发 add/detect 与光心位置缓存") {
        std::cout << "SKIPPED (built without DBoW3)";
    } TEST_PASS();
#endif
}

void test_loop_closure_cross_submap_metadata() {
#ifdef HAS_DBOW3
    TEST("LoopClosure 跨子地图候选身份与世界位置先验") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = cam->fy = 500;
        cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;
        const std::string vocab = VSLAM_SOURCE_DIR "/config/ORBvoc.dbow3";
        assert(std::filesystem::exists(vocab));

        vslam::LoopClosure lc;
        // 关闭 BoW 分数候选，仅验证位置先验；跨子图不应使用全局 id 时间窗。
        lc.setParams(1.1, 30, 8, 0.1, 3.0, cam, 20, 2.0, 30);
        assert(lc.loadVocabulary(vocab));

        cv::Mat descriptors(48, 32, CV_8U);
        for (int r = 0; r < descriptors.rows; ++r)
            for (int c = 0; c < descriptors.cols; ++c)
                descriptors.at<unsigned char>(r, c) =
                    static_cast<unsigned char>((r * 17 + c * 31) & 255);

        // 两个候选的局部坐标都接近子地图原点，但最新 Atlas 锚点相反：
        // A 的局部光心为 +10，最新 T_ws=-10 后位于世界原点；
        // B 的局部光心为 0，最新 T_ws=+100 后远离查询帧。
        auto candidate_a = std::make_shared<vslam::Frame>(10, 10.0);
        candidate_a->pose_cs = vslam::SE3(
            Eigen::Quaterniond::Identity(), vslam::Vec3(0, 0, 0));
        candidate_a->pose_cs.t = vslam::Vec3(-10, 0, 0);
        candidate_a->descriptors = descriptors.clone();
        auto candidate_b = std::make_shared<vslam::Frame>(11, 11.0);
        candidate_b->pose_cs = vslam::SE3();
        candidate_b->descriptors = descriptors.clone();
        lc.addKeyFrame(candidate_a, 101, vslam::SE3());
        lc.addKeyFrame(candidate_b, 202, vslam::SE3());

        auto query = std::make_shared<vslam::Frame>(12, 12.0);
        query->pose_cs = vslam::SE3();
        query->descriptors = descriptors.clone();
        const vslam::LoopClosure::SubmapPoses latest = {
            {101, vslam::SE3(Eigen::Quaterniond::Identity(),
                             vslam::Vec3(-10, 0, 0))},
            {202, vslam::SE3(Eigen::Quaterniond::Identity(),
                             vslam::Vec3(100, 0, 0))},
            {303, vslam::SE3()}};
        auto candidates = lc.detectLoop(query, 303, vslam::SE3(), latest);
        assert(!candidates.empty());
        assert(candidates.front().frame == candidate_a);
        assert(candidates.front().submap_id == 101);
    } TEST_PASS();
#else
    TEST("LoopClosure 跨子地图候选身份与世界位置先验") {
        std::cout << "SKIPPED (built without DBoW3)";
    } TEST_PASS();
#endif
}

void test_loop_closure_keeps_all_place_clusters() {
#ifdef HAS_DBOW3
    TEST("LoopClosure 回归：不把词袋候选硬截为 Top-3") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = cam->fy = 500;
        cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;
        const std::string vocab = VSLAM_SOURCE_DIR "/config/ORBvoc.dbow3";
        assert(std::filesystem::exists(vocab));

        vslam::LoopClosure lc;
        lc.setParams(0.0, 1, 8, 0.1, 3.0, cam, 20, 0.0, 1);
        assert(lc.loadVocabulary(vocab));

        cv::Mat descriptors(48, 32, CV_8U);
        for (int r = 0; r < descriptors.rows; ++r)
            for (int c = 0; c < descriptors.cols; ++c)
                descriptors.at<unsigned char>(r, c) =
                    static_cast<unsigned char>((r * 17 + c * 31) & 255);

        // 5 个相距较远的历史地点都与查询帧有相同词袋外观。
        // 旧实现只返回分数最高的 3 个，导致排名靠后的真实地点没有机会
        // 进入 PnP；相距较远也确保不会被聚合成同一历史时间邻域。
        for (unsigned long id : {10UL, 100UL, 200UL, 300UL, 400UL}) {
            auto frame = std::make_shared<vslam::Frame>(id, static_cast<double>(id));
            frame->pose_cs = vslam::SE3();
            frame->descriptors = descriptors.clone();
            lc.addKeyFrame(frame, 7, vslam::SE3());
        }
        auto query = std::make_shared<vslam::Frame>(1000, 1000.0);
        query->pose_cs = vslam::SE3();
        query->descriptors = descriptors.clone();
        auto candidates = lc.detectLoop(query, 7, vslam::SE3(), {{7, vslam::SE3()}});
        assert(candidates.size() >= 5);
        assert(std::ranges::all_of(candidates, [](const auto& candidate) {
            return candidate.frame && !candidate.cluster_members.empty() &&
                   candidate.cluster_members.front() &&
                   candidate.temporal_support == 1 && !candidate.mature;
        }));

        // 连续查询才成熟；中间出现多个无效/缺测查询后，支持度必须重置。
        auto query_next = std::make_shared<vslam::Frame>(1001, 1001.0);
        query_next->pose_cs = vslam::SE3();
        query_next->descriptors = descriptors.clone();
        auto mature = lc.detectLoop(query_next, 7, vslam::SE3(),
                                    {{7, vslam::SE3()}});
        assert(std::ranges::any_of(mature, [](const auto& candidate) {
            return candidate.mature && candidate.temporal_support >= 2;
        }));
        auto missing = std::make_shared<vslam::Frame>(1002, 1002.0);
        for (int i = 0; i < 3; ++i) {
            missing->id = static_cast<unsigned long>(1002 + i);
            missing->descriptors.release();
            (void)lc.detectLoop(missing, 7, vslam::SE3(),
                                 {{7, vslam::SE3()}});
        }
        auto after_gap = std::make_shared<vslam::Frame>(1006, 1006.0);
        after_gap->pose_cs = vslam::SE3();
        after_gap->descriptors = descriptors.clone();
        auto reset = lc.detectLoop(after_gap, 7, vslam::SE3(),
                                   {{7, vslam::SE3()}});
        assert(std::ranges::all_of(reset, [](const auto& candidate) {
            return !candidate.mature && candidate.temporal_support == 1;
        }));
    } TEST_PASS();
#else
    TEST("LoopClosure 回归：不把词袋候选硬截为 Top-3") {
        std::cout << "SKIPPED (built without DBoW3)";
    } TEST_PASS();
#endif
}

void test_loop_closure_removes_culled_keyframes() {
#ifdef HAS_DBOW3
    TEST("LoopClosure 回归：预算剔除同步释放缓存并重建 DBoW 索引") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = cam->fy = 500;
        cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;
        const std::string vocab = VSLAM_SOURCE_DIR "/config/ORBvoc.dbow3";

        vslam::LoopClosure lc;
        lc.setParams(0.0, 1, 8, 0.1, 3.0, cam, 20, 0.0, 1);
        assert(lc.loadVocabulary(vocab));

        cv::Mat descriptors(48, 32, CV_8U);
        for (int r = 0; r < descriptors.rows; ++r)
            for (int c = 0; c < descriptors.cols; ++c)
                descriptors.at<unsigned char>(r, c) =
                    static_cast<unsigned char>((r * 17 + c * 31) & 255);

        auto removed_a = std::make_shared<vslam::Frame>(10, 10.0);
        auto removed_b = std::make_shared<vslam::Frame>(100, 100.0);
        auto survivor = std::make_shared<vslam::Frame>(200, 200.0);
        for (const auto& frame : {removed_a, removed_b, survivor}) {
            frame->pose_cs = vslam::SE3();
            frame->descriptors = descriptors.clone();
            lc.addKeyFrame(frame, 7, vslam::SE3());
        }
        assert(lc.indexedKeyFrameCount() == 3);

        auto query = std::make_shared<vslam::Frame>(1000, 1000.0);
        query->pose_cs = vslam::SE3();
        query->descriptors = descriptors.clone();
        {
            auto before = lc.detectLoop(query, 7, vslam::SE3(),
                                        {{7, vslam::SE3()}});
            assert(!before.empty());  // 同时建立 place_hypothesis 强引用
        }
        std::weak_ptr<vslam::Frame> removed_weak = removed_a;
        lc.removeKeyFrames({removed_a->id, removed_b->id});
        removed_a.reset();
        removed_b.reset();
        assert(removed_weak.expired() && "被剔除 KF 不得被缓存/地点假设强持有");
        assert(lc.indexedKeyFrameCount() == 1);

        auto after = lc.detectLoop(query, 7, vslam::SE3(),
                                   {{7, vslam::SE3()}});
        assert(!after.empty());
        assert(std::ranges::all_of(after, [&](const auto& candidate) {
            return candidate.frame && candidate.frame->id == survivor->id;
        }));

        // 重复通知必须幂等，不能误删存活条目。
        lc.removeKeyFrames({10, 100});
        assert(lc.indexedKeyFrameCount() == 1);
    } TEST_PASS();
#else
    TEST("LoopClosure 回归：预算剔除同步释放缓存并重建 DBoW 索引") {
        std::cout << "SKIPPED (built without DBoW3)";
    } TEST_PASS();
#endif
}

void test_loop_closure_reserves_multiple_position_priors() {
#ifdef HAS_DBOW3
    TEST("LoopClosure 回归：BoW 满槽时保留 4 个独立位置先验簇") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = cam->fy = 500;
        cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;
        const std::string vocab = VSLAM_SOURCE_DIR "/config/ORBvoc.dbow3";

        vslam::LoopClosure lc;
        lc.setParams(0.95, 30, 8, 0.1, 3.0, cam, 20, 10.0, 30);
        assert(lc.loadVocabulary(vocab));
        cv::Mat query_descriptors(48, 32, CV_8U);
        for (int r = 0; r < query_descriptors.rows; ++r)
            for (int c = 0; c < query_descriptors.cols; ++c)
                query_descriptors.at<unsigned char>(r, c) =
                    static_cast<unsigned char>((r * 17 + c * 31) & 255);

        // 12 个高分 BoW 历史地点占满总槽，但世界位置超出 prior 门。
        for (unsigned long id = 1000; id < 2200; id += 100) {
            auto frame = std::make_shared<vslam::Frame>(id, id * 0.1);
            frame->pose_cs = vslam::SE3(
                Eigen::Quaterniond::Identity(), vslam::Vec3(-100, 0, 0));
            frame->descriptors = query_descriptors.clone();
            lc.addKeyFrame(frame, 7, vslam::SE3());
        }
        // BoW kf#10 与下方 prior 35/60 通过 25/25 的链式间隔属同一地点；
        // 即使 BoW cluster_members 只有 10，也必须跨模态去重。
        auto overlapping_bow = std::make_shared<vslam::Frame>(10, 1.0);
        overlapping_bow->pose_cs = vslam::SE3(
            Eigen::Quaterniond::Identity(), vslam::Vec3(-100, 0, 0));
        overlapping_bow->descriptors = query_descriptors.clone();
        lc.addKeyFrame(overlapping_bow, 7, vslam::SE3());
        // 4 个低 BoW 分、但位置近且历史时间簇互异的地点。
        // 35/60 与 BoW#10 是同一跨模态链式簇，应整簇跳过；
        // 100/200/300/400 四个独立簇应恰好保留 4 个 prior。
        const std::vector<std::pair<unsigned long, double>> prior_specs{
            {35, 1.0}, {60, 2.0}, {100, 4.0}, {200, 5.0}, {300, 6.0}, {400, 7.0}};
        for (size_t i = 0; i < prior_specs.size(); ++i) {
            const auto [id, distance] = prior_specs[i];
            auto frame = std::make_shared<vslam::Frame>(id, id * 0.1);
            frame->pose_cs = vslam::SE3(
                Eigen::Quaterniond::Identity(),
                vslam::Vec3(-distance, 0, 0));
            frame->descriptors = cv::Mat(
                48, 32, CV_8U, cv::Scalar::all(static_cast<int>(31 + i * 47)));
            lc.addKeyFrame(frame, 7, vslam::SE3());
        }

        auto query = std::make_shared<vslam::Frame>(10000, 1000.0);
        query->pose_cs = vslam::SE3();
        query->descriptors = query_descriptors.clone();
        const auto candidates = lc.detectLoop(
            query, 7, vslam::SE3(), {{7, vslam::SE3()}});
        assert(candidates.size() == 12);
        for (const auto id : {100UL, 200UL, 300UL, 400UL}) {
            assert(std::ranges::any_of(candidates, [&](const auto& candidate) {
                return candidate.frame && candidate.frame->id == id &&
                       candidate.score == 0.0;
            }) && "每个独立位置先验簇都必须占有槽位");
        }
        assert(std::ranges::none_of(candidates, [](const auto& candidate) {
            return candidate.score == 0.0 && candidate.frame &&
                   (candidate.frame->id == 35 || candidate.frame->id == 60);
        }) && "与 BoW 链式连通的 prior 簇不得重复占槽");
    } TEST_PASS();
#else
    TEST("LoopClosure 回归：BoW 满槽时保留 4 个独立位置先验簇") {
        std::cout << "SKIPPED (built without DBoW3)";
    } TEST_PASS();
#endif
}

void test_cross_submap_loop_constraint_direction() {
    TEST("Atlas 跨子地图回环约束方向（非单位旋转/平移）") {
        const auto qx = Eigen::Quaterniond(
            Eigen::AngleAxisd(0.31, vslam::Vec3::UnitX()));
        const auto qy = Eigen::Quaterniond(
            Eigen::AngleAxisd(-0.27, vslam::Vec3::UnitY()));
        const auto qz = Eigen::Quaterniond(
            Eigen::AngleAxisd(0.43, vslam::Vec3::UnitZ()));
        const vslam::SE3 current_pose_cs(qx, vslam::Vec3(1.2, -0.4, 0.7));
        const vslam::SE3 loop_pose_cs(qy, vslam::Vec3(-2.1, 0.8, 1.3));
        const vslam::SE3 P(qz, vslam::Vec3(3.4, -1.7, 0.2));
        const vslam::SE3 Z = loop_pose_cs * P.inverse();

        const auto constraint = vslam::makeCrossSubmapLoopConstraint(
            7, 3, current_pose_cs, loop_pose_cs, Z);
        const vslam::SE3 expected = current_pose_cs.inverse() * P;
        assert(constraint.a == 7 && constraint.b == 3);
        assert(constraint.type == vslam::AtlasConstraintType::LoopClosure);
        assert((constraint.T_rel.t - expected.t).norm() < 1e-10);
        assert(constraint.T_rel.q.toRotationMatrix().isApprox(
            expected.q.toRotationMatrix(), 1e-10));

        const vslam::SE3 T_ws_current(qy, vslam::Vec3(12.0, -4.0, 2.0));
        const vslam::SE3 T_ws_loop = T_ws_current * constraint.T_rel;
        const vslam::SE3 world_from_current =
            current_pose_cs * T_ws_current.inverse();
        const vslam::SE3 world_from_loop = P * T_ws_loop.inverse();
        assert((world_from_current.t - world_from_loop.t).norm() < 1e-10);
        assert(world_from_current.q.toRotationMatrix().isApprox(
            world_from_loop.q.toRotationMatrix(), 1e-10));
    } TEST_PASS();
}

void test_frontend_world_pose_rebase_after_atlas_correction() {
    TEST("Atlas 锚点校正后前端世界基线保持同一局部位姿") {
        const vslam::SE3 T_ws_old(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.37, vslam::Vec3::UnitY())),
            vslam::Vec3(18.0, -3.0, 42.0));
        const vslam::SE3 T_ws_new(
            Eigen::Quaterniond(Eigen::AngleAxisd(-0.51, vslam::Vec3::UnitZ())),
            vslam::Vec3(-7.0, 5.0, 11.0));
        const vslam::SE3 T_cs(
            Eigen::Quaterniond(Eigen::AngleAxisd(0.23, vslam::Vec3::UnitX())),
            vslam::Vec3(2.0, -1.0, 6.0));
        const vslam::SE3 old_world = T_cs * T_ws_old.inverse();
        const vslam::SE3 expected = T_cs * T_ws_new.inverse();
        const vslam::SE3 rebased = vslam::rebaseWorldPoseForSubmapAnchor(
            old_world, T_ws_old, T_ws_new);
        assert((rebased.t - expected.t).norm() < 1e-10);
        assert(rebased.q.toRotationMatrix().isApprox(
            expected.q.toRotationMatrix(), 1e-10));
    } TEST_PASS();
}

void test_atlas_rejects_solution_that_reopens_old_loop() {
    TEST("Atlas 回环验收：新边不得把旧闭环重新拉开") {
        vslam::Atlas atlas;
        const auto a_id = atlas.createSubmap(vslam::SE3(), true).id;
        const auto b_id = atlas.createSubmap(vslam::SE3(), true).id;
        vslam::AtlasConstraint loop;
        loop.a = a_id;
        loop.b = b_id;
        loop.T_rel = vslam::SE3();
        loop.weight = 10.0;
        loop.type = vslam::AtlasConstraintType::LoopClosure;
        atlas.addConstraint(loop);
        assert(atlas.loopConstraintsConsistent(5.0, 10.0 * M_PI / 180.0));

        atlas.getSubmap(b_id)->T_ws = vslam::SE3(
            Eigen::Quaterniond(Eigen::AngleAxisd(
                20.0 * M_PI / 180.0, vslam::Vec3::UnitY())),
            vslam::Vec3(6.0, 0.0, 0.0));
        double trans = 0.0, rot = 0.0;
        assert(!atlas.loopConstraintsConsistent(
            5.0, 10.0 * M_PI / 180.0, &trans, &rot));
        assert(trans > 5.0 || rot > 10.0 * M_PI / 180.0);
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

    std::cout << "\n[Pose Acceptance (M0)]\n";
    test_pose_acceptance();

    std::cout << "\n[Map Revision (M1)]\n";
    test_map_revision();
    test_map_observation_model();
    test_map_batch_cull();

    std::cout << "\n[BackendCommitter (M2)]\n";
    test_backend_committer();

    std::cout << "\n[Stereo Camera]\n";
    test_stereo_camera();
    test_stereo_match_quality();
    test_invalid_stereo_initialization();

    std::cout << "\n[Stereo VO]\n";
    test_stereo_vo();

    std::cout << "\n[MiniAtlas]\n";
    test_mini_atlas();

    std::cout << "\n[Sim3 (Phase 2)]\n";
    test_sim3();

    std::cout << "\n[Pose Graph (Phase 2)]\n";
    test_pose_graph();

    std::cout << "\n[Loop Closure (Phase 2)]\n";
    test_loop_closure();
    test_loop_closure_concurrency_and_position_cache();
    test_loop_closure_cross_submap_metadata();
    test_loop_closure_keeps_all_place_clusters();
    test_loop_closure_removes_culled_keyframes();
    test_loop_closure_reserves_multiple_position_priors();
    test_cross_submap_loop_constraint_direction();
    test_frontend_world_pose_rebase_after_atlas_correction();
    test_atlas_rejects_solution_that_reopens_old_loop();

    std::cout << "\n[Feature Extraction]\n";
    test_feature_extraction();
    test_frame_image_lifecycle();
    test_mobile_config();
    test_monocular_initialization_quality();

    std::cout << "\n[VO Initialization]\n";
    test_vo_initialization();
    test_monocular_3d_initialization();

    std::cout << "\n[Pose Semantics]\n";
    test_pose_semantics();

    std::cout << "\n[LK Tracking]\n";
    test_lk_tracking();

    std::cout << "\n[Rotation Detection]\n";
    test_rotation_detection();
    test_rotation_ambiguity();

    std::cout << "\n[Local BA]\n";
    test_camera_position_delta();
    test_local_ba_filter_helper();
    test_local_ba();
    test_local_ba_disconnected_components();

    std::cout << "\n[Long-Run Stability]\n";
    test_long_run_stability();

    std::cout << "\n===== All Tests Completed =====\n";
    return 0;
}
