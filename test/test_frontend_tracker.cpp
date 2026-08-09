/**
 * test_frontend_tracker.cpp - FrontendTracker 单元测试
 *
 * 覆盖 M1.4（PRODUCTION_LOCALIZATION_PLAN §5.5）：
 *   1. estimateRigid3D3D：已知刚体变换恢复（Kabsch）、点数不足拒绝
 *   2. trackPnP：合成 3D-2D 恢复 identity、运动连续性拒绝
 *   3. computeStereoDepths：单目（无深度）全零 pts_c
 *   4. proposeKeyFrame：平移/旋转/弱匹配/最远间隔四种触发
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_frontend_tracker
 * 运行: ./build/test_frontend_tracker（独立 CTest）
 */

#include "vslam/frontend_tracker.h"
#include "vslam/camera.h"
#include "vslam/frame.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

// 简单的测试辅助宏（与 test_vo.cpp 保持一致）
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::Frame;
using vslam::FrontendTracker;
using vslam::KeyframeInput;
using vslam::KeyframeProposal;
using vslam::MotionBaseline;
using vslam::RigidResult;
using vslam::SE3;
using vslam::StereoStats;
using vslam::TrackerConfig;
using vslam::TrackingResult;
using vslam::Vec2;
using vslam::Vec3;

namespace {

vslam::Camera makeMonocular() {
    auto cam = std::make_shared<vslam::MonocularCamera>();
    cam->fx = 500.0;
    cam->fy = 500.0;
    cam->cx = 320.0;
    cam->cy = 240.0;
    cam->img_width = 640;
    cam->img_height = 480;
    return cam;
}

vslam::Camera makeStereo() {
    auto cam = std::make_shared<vslam::StereoCamera>();
    cam->fx = 500.0;
    cam->fy = 500.0;
    cam->cx = 320.0;
    cam->cy = 240.0;
    cam->img_width = 640;
    cam->img_height = 480;
    cam->fx_r = 500.0;
    cam->fy_r = 500.0;
    cam->cx_r = 320.0;
    cam->cy_r = 240.0;
    cam->baseline_m = 0.54;
    return cam;
}

void test_estimate_rigid_known_transform() {
    TEST("estimateRigid3D3D: 已知刚体变换恢复") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);

        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        const Vec3 t(1.0, 2.0, 0.5);

        cv::RNG rng(0x5A17);
        std::vector<cv::Point3f> pts_w, pts_c;
        for (int i = 0; i < 80; i++) {
            const Vec3 p(rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0),
                         rng.uniform(3.0, 12.0));
            const Vec3 q = R * p + t;
            pts_w.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
            pts_c.emplace_back((float)q.x(), (float)q.y(), (float)q.z());
        }

        const RigidResult res = tracker.estimateRigid3D3D(pts_w, pts_c, 15, 0.5);
        assert(res.ok);
        assert(res.inliers >= 15);
        assert(res.ratio >= 0.5);
        assert((res.pose_cs.q.toRotationMatrix() - R).norm() < 1e-3);
        assert((res.pose_cs.t - t).norm() < 1e-3);
        assert(res.rmse < 1e-6);
    } TEST_PASS();
}

void test_estimate_rigid_too_few_points() {
    TEST("estimateRigid3D3D: 点数不足（< max(20, min_inliers)）→ 拒绝") {
        FrontendTracker tracker(makeMonocular());
        std::vector<cv::Point3f> pts_w = {{1, 1, 3}, {2, 2, 4}, {3, 3, 5}};
        std::vector<cv::Point3f> pts_c = {{1, 1, 3}, {2, 2, 4}, {3, 3, 5}};
        const RigidResult res = tracker.estimateRigid3D3D(pts_w, pts_c, 15, 0.5);
        assert(!res.ok);
    } TEST_PASS();
}

void test_track_pnp_recovers_identity() {
    TEST("trackPnP: 合成 3D-2D 恢复 identity 位姿") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);

        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        cv::RNG rng(0x5A17);
        for (int i = 0; i < 100; i++) {
            const double u = rng.uniform(60.0, 580.0);
            const double v = rng.uniform(60.0, 420.0);
            const double depth = rng.uniform(3.0, 12.0);
            const Vec3 p = cam->pixel2camera(Vec2(u, v), depth);
            pts3d.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
            pts2d.emplace_back((float)u, (float)v);
        }

        const MotionBaseline motion;  // 空基线 → 跳过连续性
        const TrackingResult r = tracker.trackPnP(pts3d, pts2d, SE3(), motion,
                                                  15, 0.3, 2.5);
        assert(r.valid);
        assert(r.inliers >= 15);
        assert(r.pose_cs.t.norm() < 1e-3);
        assert(r.pose_cs.q.angularDistance(Eigen::Quaterniond::Identity()) < 1e-3);
    } TEST_PASS();
}

void test_track_pnp_continuity_rejects() {
    TEST("trackPnP: 运动连续性拒绝远离基线的候选") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);

        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        cv::RNG rng(0x5A17);
        for (int i = 0; i < 100; i++) {
            const double u = rng.uniform(60.0, 580.0);
            const double v = rng.uniform(60.0, 420.0);
            const double depth = rng.uniform(3.0, 12.0);
            const Vec3 p = cam->pixel2camera(Vec2(u, v), depth);
            pts3d.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
            pts2d.emplace_back((float)u, (float)v);
        }

        MotionBaseline far;
        far.baseline_twc = SE3(Eigen::Quaterniond::Identity(), Vec3(100.0, 0.0, 0.0));
        far.max_translation = 3.0;
        far.max_rotation = 0.35;
        const TrackingResult r = tracker.trackPnP(pts3d, pts2d, SE3(), far,
                                                  15, 0.3, 2.5);
        assert(!r.valid && "100m 外基线必须被连续性拒绝");
    } TEST_PASS();
}

void test_compute_stereo_depths_monocular() {
    TEST("computeStereoDepths: 单目 → pts_c 全零、stereo_points=0") {
        FrontendTracker tracker(makeMonocular());
        auto frame = std::make_shared<Frame>(0, 0.0);
        frame->keypoints = {cv::KeyPoint(100, 100, 20), cv::KeyPoint(300, 200, 20)};
        const StereoStats stats = tracker.computeStereoDepths(frame);
        assert(stats.stereo_points == 0);
        assert(frame->pts_c.size() == frame->keypoints.size());
        for (const auto& p : frame->pts_c) assert(p.z() == 0.0);
    } TEST_PASS();
}

void test_propose_keyframe() {
    TEST("proposeKeyFrame: 平移/弱匹配/最远间隔触发") {
        FrontendTracker tracker(makeStereo());
        const auto q_id = Eigen::Quaterniond::Identity();

        KeyframeInput in;
        in.curr_frame = std::make_shared<Frame>(100, 0.0);
        in.ref_pose_cs = SE3();
        in.inliers = 50;
        in.last_kf_frame_id = 95;
        in.map_keyframe_count = 100;

        // 1m 平移（双目阈值 0.9m）→ 需要 KF
        in.curr_frame->pose_cs = SE3(q_id, Vec3(1.0, 0.0, 0.0));
        const KeyframeProposal p1 = tracker.proposeKeyFrame(in);
        assert(p1.need && p1.translation > 0.9);

        // 0.2m 平移、内点充足、间隔 5 → 不需要
        in.curr_frame->pose_cs = SE3(q_id, Vec3(0.2, 0.0, 0.0));
        const KeyframeProposal p2 = tracker.proposeKeyFrame(in);
        assert(!p2.need);

        // 弱匹配：内点 5 < 15，但间隔 5 < min_keyframe_interval → 冷却抑制
        in.inliers = 5;
        const KeyframeProposal p3 = tracker.proposeKeyFrame(in);
        assert(!p3.need && !p3.weak_match);

        // 弱匹配 + 间隔足够（20 >= 10）→ 触发
        in.last_kf_frame_id = 80;
        const KeyframeProposal p4 = tracker.proposeKeyFrame(in);
        assert(p4.need && p4.weak_match);

        // 最远间隔：内点充足、小运动、间隔 20 >= 15 → 触发
        in.inliers = 50;
        in.last_kf_frame_id = 80;
        const KeyframeProposal p5 = tracker.proposeKeyFrame(in);
        assert(p5.need && p5.max_interval);
    } TEST_PASS();
}

}  // namespace

int main() {
    cv::setNumThreads(1);
    cv::setRNGSeed(0x5A17);

    std::cout << "test_frontend_tracker (M1.4 FrontendTracker)" << std::endl;

    test_estimate_rigid_known_transform();
    test_estimate_rigid_too_few_points();
    test_track_pnp_recovers_identity();
    test_track_pnp_continuity_rejects();
    test_compute_stereo_depths_monocular();
    test_propose_keyframe();

    std::cout << "全部通过" << std::endl;
    return 0;
}
