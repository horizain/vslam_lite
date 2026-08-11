/**
 * test_frontend_tracker.cpp - FrontendTracker 单元测试
 *
 * 覆盖 M1.4（PRODUCTION_LOCALIZATION_PLAN §5.5）：
 *   1. estimateRigid3D3D：已知刚体变换恢复（Kabsch）、点数不足拒绝
 *   2. trackPnP：合成 3D-2D 恢复 identity、运动连续性拒绝
 *   3. computeStereoDepths：单目（无深度）全零 pts_c
 *   4. proposeKeyFrame：平移/旋转/弱匹配/最远间隔四种触发
 *   5. 方案 A matchGuided：预测位姿正确时窗口匹配召回、预测偏离时召回下降
 *   6. 方案 B trackLocalMap：投影补匹配、occupied 防重复
 *   7. 方案 B refinePnP：确定性精修不消耗 RNG（与 trackPnP 交替调用一致）
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
        cv::setRNGSeed(0x5A17);
        const TrackingResult primary_only = tracker.trackPnP(
            pts3d, pts2d, SE3(), far, 15, 0.3, 2.5);
        const uint64_t rng_after_primary = cv::theRNG().state;
        assert(!primary_only.valid);

        far.predicted_pose_cs = SE3();  // 覆盖预测初值二次 RANSAC 分支
        cv::setRNGSeed(0x5A17);
        const TrackingResult r = tracker.trackPnP(pts3d, pts2d, SE3(), far,
                                                  15, 0.3, 2.5);
        assert(!r.valid && "100m 外基线必须被连续性拒绝");
        assert(cv::theRNG().state == rng_after_primary &&
               "容错重试不得扰动后续帧 RANSAC 随机序列");
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

// ============================================================
// 方案 A：运动模型引导匹配
// 合成两帧：ref 帧 3D 点 → 真位姿投影生成 curr 帧特征，描述子相同
// （投影噪声 < 搜索半径内）。预测位姿正确时窗口匹配应召回绝大多数；
// 预测位姿大幅偏离时（真值 + 大平移），投影点越出窗口 → 召回骤降。
// ============================================================
void test_match_guided_recall() {
    TEST("matchGuided: 预测位姿正确时窗口匹配召回（>90%）") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);

        // ref 帧：100 个 3D 点（相机前方），描述子 = 确定性伪随机 32B
        auto ref_frame = std::make_shared<Frame>(0, 0.0);
        cv::RNG rng(0x5A17);
        std::vector<Vec3> ref_points_s;
        for (int i = 0; i < 100; i++) {
            const double u = rng.uniform(80.0, 560.0);
            const double v = rng.uniform(80.0, 400.0);
            const double d = rng.uniform(3.0, 10.0);
            const Vec3 p = cam->pixel2camera(Vec2(u, v), d);
            ref_points_s.push_back(p);
            ref_frame->keypoints.emplace_back((float)u, (float)v, 20.0f);
            cv::Mat desc(1, 32, CV_8U);
            for (int b = 0; b < 32; b++) desc.at<uchar>(b) = (uchar)rng.uniform(0, 256);
            ref_frame->descriptors.push_back(desc);
            ref_frame->map_points.push_back(std::make_shared<vslam::MapPoint>((unsigned long)i));
        }

        // 真位姿：绕 Y 轴 5°，平移 (0.2, 0, 0.3)
        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(0.0873, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const SE3 T_true(Eigen::Quaterniond(R), Vec3(0.2, 0.0, 0.3));

        // curr 帧：ref 点经真位姿投影，加 1px 噪声
        auto curr_frame = std::make_shared<Frame>(1, 0.1);
        for (size_t i = 0; i < ref_points_s.size(); i++) {
            const Vec2 px = cam->world2pixel(ref_points_s[i], T_true);
            curr_frame->keypoints.emplace_back(
                (float)(px.x() + rng.uniform(-1.0, 1.0)),
                (float)(px.y() + rng.uniform(-1.0, 1.0)), 20.0f);
            curr_frame->descriptors.push_back(ref_frame->descriptors.row((int)i));
        }

        // 预测位姿 = 真位姿（理想情况）
        const auto matches = tracker.matchGuided(ref_frame, curr_frame, ref_points_s,
                                                 T_true, 25.0, 0.7);
        assert(matches.size() >= 90 && "正确预测下窗口匹配应召回 ≥90%");
        // 匹配的 queryIdx/trainIdx 应对齐（投影噪声小）
        int aligned = 0;
        for (const auto& m : matches)
            if (m.queryIdx == m.trainIdx) aligned++;
        assert(aligned >= 90 && "投影噪声 1px 内应保持索引对齐");
    } TEST_PASS();

    TEST("matchGuided: 预测位姿大幅偏离 → 召回骤降（回退全图匹配的依据）") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);

        auto ref_frame = std::make_shared<Frame>(0, 0.0);
        cv::RNG rng(0x5A17);
        std::vector<Vec3> ref_points_s;
        for (int i = 0; i < 100; i++) {
            const double u = rng.uniform(80.0, 560.0);
            const double v = rng.uniform(80.0, 400.0);
            const double d = rng.uniform(3.0, 10.0);
            const Vec3 p = cam->pixel2camera(Vec2(u, v), d);
            ref_points_s.push_back(p);
            ref_frame->keypoints.emplace_back((float)u, (float)v, 20.0f);
            cv::Mat desc(1, 32, CV_8U);
            for (int b = 0; b < 32; b++) desc.at<uchar>(b) = (uchar)rng.uniform(0, 256);
            ref_frame->descriptors.push_back(desc);
            ref_frame->map_points.push_back(std::make_shared<vslam::MapPoint>((unsigned long)i));
        }

        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(0.0873, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const SE3 T_true(Eigen::Quaterniond(R), Vec3(0.2, 0.0, 0.3));

        auto curr_frame = std::make_shared<Frame>(1, 0.1);
        for (size_t i = 0; i < ref_points_s.size(); i++) {
            const Vec2 px = cam->world2pixel(ref_points_s[i], T_true);
            curr_frame->keypoints.emplace_back((float)px.x(), (float)px.y(), 20.0f);
            curr_frame->descriptors.push_back(ref_frame->descriptors.row((int)i));
        }

        // 预测位姿偏离真值 0.8m（远大于 25px 搜索半径在 5m 深度处的覆盖）
        const SE3 T_wrong(Eigen::Quaterniond(R), Vec3(1.0, 0.0, 0.3));
        const auto matches = tracker.matchGuided(ref_frame, curr_frame, ref_points_s,
                                                 T_wrong, 25.0, 0.7);
        assert(matches.size() < 30 && "偏离预测必须显著降低窗口匹配召回");
    } TEST_PASS();
}

// ============================================================
// 方案 B：共视图局部地图投影匹配
// 合成局部地图点（相机前方）→ 用真位姿投影到 curr 帧特征；
// 部分特征已被首轮占用（occupied）→ 不重复关联。
// ============================================================
void test_track_local_map() {
    TEST("trackLocalMap: 投影补匹配 + occupied 防重复") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);

        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const SE3 T_cs(Eigen::Quaterniond(R), Vec3(0.1, 0.0, 0.2));

        auto curr_frame = std::make_shared<Frame>(1, 0.1);
        cv::RNG rng(0x5A17);
        std::vector<Vec3> local_points_s;
        std::vector<cv::Mat> local_descs;
        std::vector<vslam::MapPoint::Ptr> local_mps;
        for (int i = 0; i < 50; i++) {
            const double u = rng.uniform(100.0, 540.0);
            const double v = rng.uniform(100.0, 380.0);
            const double d = rng.uniform(3.0, 10.0);
            const Vec3 p = cam->pixel2camera(Vec2(u, v), d);
            local_points_s.push_back(p);
            local_descs.push_back(cv::Mat(1, 32, CV_8U));
            for (int b = 0; b < 32; b++)
                local_descs.back().at<uchar>(b) = (uchar)rng.uniform(0, 256);
            local_mps.push_back(std::make_shared<vslam::MapPoint>((unsigned long)i));
        }
        // curr 帧特征 = 局部点经 T_cs 投影（注意 world2pixel 用 T_cw；此处
        // T_cs 是子地图局部系 T_cs，语义一致）
        for (size_t i = 0; i < local_points_s.size(); i++) {
            const Vec2 px = cam->world2pixel(local_points_s[i], T_cs);
            curr_frame->keypoints.emplace_back((float)px.x(), (float)px.y(), 20.0f);
            curr_frame->descriptors.push_back(local_descs[i]);
        }

        // 前 10 个特征已被首轮占用
        std::vector<int> occupied;
        for (int i = 0; i < 10; i++) occupied.push_back(i);

        const auto result = tracker.trackLocalMap(
            curr_frame, local_points_s, local_descs, local_mps, occupied,
            T_cs, 30.0, 0.7);
        assert(result.added >= 35 && "未被占用的局部点应被投影匹配找回");
        assert(result.pts3d.size() == result.curr_feature_indices.size());
        assert(result.mps.size() == result.curr_feature_indices.size());
        for (int ti : result.curr_feature_indices)
            assert(ti >= 10 && "occupied 特征不得被重复匹配");
    } TEST_PASS();

    TEST("trackLocalMap: 空描述子/空点 → 空结果") {
        FrontendTracker tracker(makeMonocular());
        auto curr_frame = std::make_shared<Frame>(1, 0.1);
        const auto result = tracker.trackLocalMap(
            curr_frame, {}, {}, {}, {}, SE3(), 30.0, 0.7);
        assert(result.added == 0 && result.pts3d.empty());
    } TEST_PASS();
}

// ============================================================
// 方案 B：确定性精修
// refinePnP 用 solvePnP(iterative + useExtrinsicGuess)，不消耗全局 RNG。
// 验证：交替调用 trackPnP/refinePnP 与只调 trackPnP 后，全局 RNG 状态
// 保持一致（确定性等价的前提）。
// ============================================================
void test_refine_pnp_deterministic() {
    TEST("refinePnP: 不消耗全局 RNG（solvePnP iterative 确定性）") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);

        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        cv::RNG rng(0x5A17);
        for (int i = 0; i < 80; i++) {
            const double u = rng.uniform(60.0, 580.0);
            const double v = rng.uniform(60.0, 420.0);
            const double depth = rng.uniform(3.0, 12.0);
            const Vec3 p = cam->pixel2camera(Vec2(u, v), depth);
            pts3d.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
            pts2d.emplace_back((float)u, (float)v);
        }

        // 参考：只跑 trackPnP（消耗 RNG），读取其后 RNG 状态
        cv::setRNGSeed(0x5A17);
        const MotionBaseline motion;
        (void)tracker.trackPnP(pts3d, pts2d, SE3(), motion, 15, 0.3, 2.5);
        const unsigned rng_after_track = cv::theRNG().next();

        // 实验：重置后先 refinePnP 再 trackPnP，RNG 状态应仍等于上面
        cv::setRNGSeed(0x5A17);
        const SE3 initial(Eigen::Quaterniond::Identity(), Vec3::Zero());
        (void)tracker.refinePnP(pts3d, pts2d, initial, SE3(), motion,
                                15, 0.3, 2.5);
        (void)tracker.trackPnP(pts3d, pts2d, SE3(), motion, 15, 0.3, 2.5);
        const unsigned rng_after_refine = cv::theRNG().next();

        assert(rng_after_track == rng_after_refine
               && "refinePnP 不得消耗全局 RNG（否则破坏双实例确定性等价）");
    } TEST_PASS();
}

void test_refine_pnp_does_not_modify_inputs() {
    TEST("refinePnP: const 输入排序不应修改调用方数据") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        for (int i = 0; i < 30; i++) {
            const Vec3 p(-1.2 + 0.08 * i, -0.7 + 0.04 * (i % 9),
                        4.0 + 0.15 * (i % 7));
            const Vec2 px = cam->camera2pixel(p);
            pts3d.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
            pts2d.emplace_back((float)px.x(), (float)px.y());
        }
        // 打乱输入顺序，覆盖精修内部的确定性排序路径。
        std::vector<cv::Point3f> input_3d;
        std::vector<cv::Point2f> input_2d;
        for (int i = 29; i >= 0; i--) {
            input_3d.push_back(pts3d[(size_t)i]);
            input_2d.push_back(pts2d[(size_t)i]);
        }
        const auto before_3d = input_3d;
        const auto before_2d = input_2d;
        const TrackingResult result = tracker.refinePnP(
            input_3d, input_2d, SE3(), SE3(), MotionBaseline(), 6, 0.5, 2.5);
        assert(result.valid);
        assert(input_3d == before_3d && "精修不得通过 const_cast 改写 3D 输入");
        assert(input_2d == before_2d && "精修不得通过 const_cast 改写 2D 输入");
    } TEST_PASS();
}

void test_refine_pnp_scores_all_correspondences() {
    TEST("refinePnP: 非首点大误差必须计入 RMSE") {
        const vslam::Camera cam = makeMonocular();
        FrontendTracker tracker(cam);
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        for (int i = 0; i < 60; i++) {
            const Vec3 p(-1.5 + 0.05 * i, -0.8 + 0.03 * (i % 11),
                        4.0 + 0.1 * (i % 13));
            const Vec2 px = cam->camera2pixel(p);
            pts3d.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
            pts2d.emplace_back((float)px.x(), (float)px.y());
        }
        // 3D 字典序中最后一个点是非首点；只污染它，旧的全 0 索引 RMSE
        // 会重复计算首点而错误放行该精修结果。
        pts2d.back().x += 100.0f;
        const TrackingResult result = tracker.refinePnP(
            pts3d, pts2d, SE3(), SE3(), MotionBaseline(), 6, 0.5, 2.5);
        assert(!result.valid && "所有对应都必须参与 RMSE 验收");
        assert(result.pose_rmse > 2.5);
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
    test_match_guided_recall();
    test_track_local_map();
    test_refine_pnp_deterministic();
    test_refine_pnp_does_not_modify_inputs();
    test_refine_pnp_scores_all_correspondences();

    std::cout << "全部通过" << std::endl;
    return 0;
}
