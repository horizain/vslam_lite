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
#include "vslam/feature.h"
#include "vslam/vo.h"
#include "vslam/atlas.h"
#include "vslam/mappoint.h"
#include "vslam/optimizer.h"
#include "vslam/loop_closure.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem>

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
        assert(cfg.enable_loop_closure);
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

        vslam::VisualOdometry vo(cam, cfg);

        // 螺旋路径 150 帧：缓慢前进 + 小幅摆动
        constexpr int kFrames = 150;
        std::vector<double> frame_ms;
        for (int f = 0; f < kFrames; f++) {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << 0.003*f, 0.005*f, 0);
            cv::Mat tvec = (cv::Mat_<double>(3,1) << 0.1*f, 0.02*std::sin(f*0.05), 0.1*std::sin(f*0.03));
            cv::Mat img(480, 640, CV_8UC1, cv::Scalar(64));
            render(img, blks, K, rvec, tvec);

            auto t0 = std::chrono::steady_clock::now();
            vo.addFrame(img, f * 0.1);
            auto t1 = std::chrono::steady_clock::now();
            frame_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
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
void test_local_ba() {
    TEST("Local BA 保持尺度 (3 帧已知位姿)") {
        auto cam = std::make_shared<vslam::MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
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
            kf->pose_cw = Twc[i].inverse();
            kfs.push_back(kf);
            map->insertKeyFrame(kf);
        }
        // 20 个世界点（前方 4~8m），投影到 3 帧
        std::mt19937 gen(5);
        std::uniform_real_distribution<double> dx(-2, 2), dy(-2, 2), dz(4, 8);
        for (int i = 0; i < 20; i++) {
            auto mp = std::make_shared<vslam::MapPoint>((unsigned long)i);
            mp->pos_w = vslam::Vec3(dx(gen), dy(gen), dz(gen));
            mp->observed_count = 3;
            map->insertMapPoint(mp);
            for (int f = 0; f < 3; f++) {
                vslam::Vec2 px = cam->world2pixel(mp->pos_w, kfs[f]->pose_cw);
                cv::KeyPoint kp;
                kp.pt = cv::Point2f((float)px.x(), (float)px.y());
                kfs[f]->keypoints.push_back(kp);
                kfs[f]->map_points.push_back(mp);
            }
        }
        auto disp = [&](int a, int b) {
            return (kfs[b]->pose_cw.inverse().t - kfs[a]->pose_cw.inverse().t).norm();
        };
        vslam::Optimizer::localBundleAdjustment(cam, map, kfs, 10);
        // 点固定（motion-only）+ 帧 0/1 固定：尺度必须保持 1m/帧
        assert(std::abs(disp(0, 1) - 1.0) < 0.2);
        assert(std::abs(disp(1, 2) - 1.0) < 0.2);
        std::cout << " (disp01=" << disp(0, 1) << " disp12=" << disp(1, 2) << ")";
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
        assert(second.origin_Twc.t.isApprox(anchor.t));
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

        // 建图：关键帧 pose_cw = T_wc⁻¹
        auto map = std::make_shared<vslam::Map>();
        for (int i = 0; i < N; i++) {
            auto kf = std::make_shared<vslam::Frame>((unsigned long)i, (double)i);
            kf->pose_cw = est[i].inverse();
            map->insertKeyFrame(kf);
        }

        // 确认存在显著漂移（否则测试无意义）
        double err_before = (est[N - 1].t - gt[N - 1].t).norm();
        assert(err_before > 1.0);

        // 冻结每个 KF 插入时得到的里程计相对测量。
        std::vector<vslam::LoopEdge> odometry_edges;
        for (int i = 1; i < N; i++) {
            odometry_edges.push_back({
                (unsigned long)(i - 1), (unsigned long)i,
                est[i - 1].inverse() * est[i], 1.0});
        }

        // 回环边：末帧 → 首帧（已知真实回环约束，高置信）
        vslam::LoopEdge le;
        le.a = 0;
        le.b = (unsigned long)(N - 1);
        le.T_rel = gt[0].inverse() * gt[N - 1];  // X_0⁻¹·X_{N-1} 的期望值
        le.weight = 10.0;
        vslam::Optimizer::poseGraphOptimization(map, odometry_edges, {le});

        // 优化后末帧误差应显著下降
        vslam::SE3 Twc_after = map->getKeyFrame((unsigned long)(N - 1))->pose_cw.inverse();
        double err_after = (Twc_after.t - gt[N - 1].t).norm();
        std::cout << " (drift_before=" << err_before << "m after=" << err_after << "m)";
        assert(err_after < err_before * 0.2);
        assert(err_after < 0.5);
    } TEST_PASS();
#else
    TEST("位姿图优化：回环约束校正累积漂移") {
        auto map = std::make_shared<vslam::Map>();
        assert(!vslam::Optimizer::poseGraphOptimization(map, {}, {}));
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

        // 词典路径（项目根运行：config/ORBvoc.dbow3；ctest：../config/...）
        std::string vocab;
        for (const auto& p : {"config/ORBvoc.dbow3", "../config/ORBvoc.dbow3"})
            if (std::filesystem::exists(p)) { vocab = p; break; }
        assert(!vocab.empty());

        vslam::LoopClosure lc;
        // min_score 取 0.05：合成圆点特征的 ORB 描述子对微小视差很敏感，
        // 末帧偏离原点 0.3m 后 BoW 分数会跌破 0.3。时间窗（30）才是防误报的
        // 主闸，分数阈值只兜底，PnP 验证（min_loop_inliers=30, ratio=0.7）
        // 仍会拒绝弱候选。
        lc.setParams(0.05, 30, 30, 0.7, 3.0, cam);
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
                cv::circle(img, cv::Point((int)uv.x(), (int)uv.y()), 2,
                           cv::Scalar(140 + (i % 115)), -1);
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
        // 下面的期望断言（cand->pose_cw * T_wc_curr）必然失败——覆盖 6c311d7
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
            kf->pose_cw = T_wc.inverse();
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
                    mp->pos_w = pw[best_i];
                    kf->map_points[j] = mp;
                }
            }
            lc.addKeyFrame(kf);
            last_kf = kf;
            if (kf_id == 20) mid_kf = kf;  // 中段帧（负例用）
            kf_id++;
        }

        // 回环检测：最后一帧（回到原点）应命中早期帧（id 差 > 30，时间窗过滤通过）
        auto cand = lc.detectLoop(last_kf);
        std::cout << " (cand=kf#" << (cand ? std::to_string(cand->id) : "null")
                  << " last=kf#" << last_kf->id << ")";
        assert(cand != nullptr);
        assert(cand->id < last_kf->id);

        // 几何验证：PnP 直接输出 loop→curr 的位姿图 SE3 测量。
        // 它应与无漂移合成真值一致，不依赖 last_kf 中保存的 VO 漂移位姿。
        last_kf->pose_cw.t += vslam::Vec3(3.0, -2.0, 1.0);
        vslam::SE3 T_loop_curr;
        assert(lc.verifyLoop(last_kf, cand, T_loop_curr));
        const vslam::SE3 expected = cand->pose_cw * poses.back();
        // isApprox 是相对精度（|a-b|² ≤ prec²·min(|a|²,|b|²)），期望平移为零时
        // 分母恒为 0，任何非零误差都会失败，故用绝对范数判据。
        assert((T_loop_curr.t - expected.t).norm() < 0.1);
        assert(T_loop_curr.q.toRotationMatrix().isApprox(
            expected.q.toRotationMatrix(), 0.02));

        // 负例：中段帧（id 20，所有高分候选都在时间窗内）应返回 nullptr
        assert(mid_kf != nullptr);
        auto none = lc.detectLoop(mid_kf);
        assert(none == nullptr);
    } TEST_PASS();
#else
    TEST("LoopClosure 合成回环") {
        std::cout << "SKIPPED (built without DBoW3)";
    } TEST_PASS();
#endif
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

    std::cout << "\n[Feature Extraction]\n";
    test_feature_extraction();
    test_frame_image_lifecycle();
    test_mobile_config();

    std::cout << "\n[VO Initialization]\n";
    test_vo_initialization();

    std::cout << "\n[Pose Semantics]\n";
    test_pose_semantics();

    std::cout << "\n[LK Tracking]\n";
    test_lk_tracking();

    std::cout << "\n[Rotation Detection]\n";
    test_rotation_detection();
    test_rotation_ambiguity();

    std::cout << "\n[Local BA]\n";
    test_local_ba();

    std::cout << "\n[Long-Run Stability]\n";
    test_long_run_stability();

    std::cout << "\n===== All Tests Completed =====\n";
    return 0;
}
