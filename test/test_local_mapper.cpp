/**
 * test_local_mapper.cpp - LocalMapper 单元测试
 *
 * 覆盖 M1.5（PRODUCTION_LOCALIZATION_PLAN §5.5）：
 *   1. includeLocalBALandmark：观测数门槛
 *   2. createMapPointsFromStereo：双目单帧建点 + 正式观测
 *   3. triangulateNewPoints：两帧三角化建点
 *   4. selectLocalWindow：共视窗口选择 / 兜底最近 n 帧
 *   5. buildLocalBASnapshot：min_observed 过滤 + anchor 连通分量裁剪
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_local_mapper
 * 运行: ./build/test_local_mapper（独立 CTest）
 */

#include "vslam/local_mapper.h"
#include "vslam/atlas.h"
#include "vslam/camera.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/mappoint.h"
#include "vslam/optimizer.h"

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
using vslam::LocalMapper;
using vslam::Map;
using vslam::MapPoint;
using vslam::OptimizationSnapshot;
using vslam::SE3;
using vslam::Vec2;
using vslam::Vec3;

namespace {

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

// 为特征 i 设置相机系 3D 观测（由像素 + 深度反投影）
void setStereoPoint(const Frame::Ptr& frame, size_t i, double depth) {
    if (frame->pts_c.size() <= i) frame->pts_c.resize(i + 1, Vec3::Zero());
    const auto& kp = frame->keypoints[i];
    frame->pts_c[i] = Vec3((kp.pt.x - 320.0) / 500.0 * depth,
                           (kp.pt.y - 240.0) / 500.0 * depth, depth);
}

void test_include_local_ba_landmark() {
    TEST("includeLocalBALandmark: 观测数门槛") {
        assert(vslam::includeLocalBALandmark(3, 3));
        assert(vslam::includeLocalBALandmark(5, 3));
        assert(!vslam::includeLocalBALandmark(2, 3));
        assert(vslam::includeLocalBALandmark(0, 0));
    } TEST_PASS();
}

void test_create_map_points_from_stereo() {
    TEST("createMapPointsFromStereo: 双目单帧建点 + 正式观测") {
        const vslam::Camera cam = makeStereo();
        LocalMapper mapper(cam);
        auto map = std::make_shared<Map>();

        auto frame = std::make_shared<Frame>(0, 0.0);
        frame->pose_cs = SE3();
        frame->keypoints = {cv::KeyPoint(320, 240, 20), cv::KeyPoint(200, 150, 20),
                            cv::KeyPoint(100, 300, 20)};
        frame->map_points.resize(3, nullptr);
        setStereoPoint(frame, 0, 5.0);
        setStereoPoint(frame, 1, 8.0);
        // 第 3 个特征无深度（z=0）

        map->insertKeyFrame(frame);
        mapper.createMapPointsFromStereo(map, frame);

        assert(map->mapPointCount() == 2);
        // 第 0 点深度 5m 在 z 轴前方 → pos_s.z > 0
        assert(frame->map_points[0] && std::abs(frame->map_points[0]->pos_s.z() - 5.0) < 1e-6);
        assert(frame->map_points[1] && std::abs(frame->map_points[1]->pos_s.z() - 8.0) < 1e-6);
        assert(frame->map_points[2] == nullptr);
        // 已注册 KF 建立双向正式观测
        assert(frame->map_points[0]->observationCount() == 1);
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_stereo_point_budget_is_per_point() {
    TEST("createMapPointsFromStereo: 建点预算逐点执行且不越限") {
        const vslam::Camera cam = makeStereo();
        LocalMapper mapper(cam);
        auto map = std::make_shared<Map>();
        auto frame = std::make_shared<Frame>(0, 0.0);
        frame->pose_cs = SE3();
        frame->keypoints = {
            cv::KeyPoint(320, 240, 20), cv::KeyPoint(200, 150, 20),
            cv::KeyPoint(100, 300, 20), cv::KeyPoint(400, 240, 20)};
        frame->map_points.resize(frame->keypoints.size(), nullptr);
        for (size_t i = 0; i < frame->keypoints.size(); i++)
            setStereoPoint(frame, i, 5.0 + static_cast<double>(i));
        map->insertKeyFrame(frame);

        // 旧实现会为全部有效深度点建图；剩余配额只有两个时必须精确
        // 建两个点，不能先批量创建再超出上限。
        mapper.createMapPointsFromStereo(map, frame, 2);
        assert(map->mapPointCount() == 2);
        assert(frame->map_points[0] && frame->map_points[1]);
        assert(frame->map_points[2] == nullptr && frame->map_points[3] == nullptr);
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_tracking_hit_keeps_recent_points_fresh() {
    TEST("Map: KF 关联点命中统计可供 rolling 回收使用") {
        auto map = std::make_shared<Map>();
        auto frame = std::make_shared<Frame>(0, 0.0);
        frame->keypoints = {cv::KeyPoint(320, 240, 20)};
        frame->map_points.resize(1, nullptr);
        auto mp = std::make_shared<MapPoint>(map->nextMapPointId());
        frame->map_points[0] = mp;
        map->insertMapPoint(mp);
        map->insertKeyFrame(frame);
        assert(map->lastHitKeyframeCount(mp->id) == map->keyFrameCount());

        // 普通帧跟踪命中不增加正式 Observation，但必须刷新旁路命中计数，
        // 这样达到点 cap 时不会把持续使用的强点当成 stale 点删除。
        map->recordTrackingHit(mp->id);
        assert(map->lastHitKeyframeCount(mp->id) == map->keyFrameCount());
    } TEST_PASS();
}

void test_triangulate_new_points() {
    TEST("triangulateNewPoints: 两帧三角化建点") {
        const vslam::Camera cam = makeStereo();  // 只用其 K
        LocalMapper mapper(cam);
        auto map = std::make_shared<Map>();

        // 帧1（原点）、帧2（+x 平移 0.5m），共享 3 个匹配特征；
        // 相机2 在右侧 → 同一 3D 点投影到帧2 的 x 更大（右移 30px 视差），
        // 深度 ≈ 500*0.5/30 ≈ 8.3m；反向视差 → 射线发散会被拒绝
        auto f1 = std::make_shared<Frame>(0, 0.0);
        auto f2 = std::make_shared<Frame>(1, 0.1);
        f1->pose_cs = SE3();
        f2->pose_cs = SE3(Eigen::Quaterniond::Identity(), Vec3(0.5, 0.0, 0.0));

        const std::vector<cv::KeyPoint> kps1 = {
            cv::KeyPoint(280, 240, 20), cv::KeyPoint(360, 240, 20),
            cv::KeyPoint(320, 180, 20)};
        const std::vector<cv::KeyPoint> kps2 = {
            cv::KeyPoint(310, 240, 20), cv::KeyPoint(390, 240, 20),
            cv::KeyPoint(350, 180, 20)};
        f1->keypoints = kps1;
        f2->keypoints = kps2;
        f1->map_points.resize(3, nullptr);
        f2->map_points.resize(3, nullptr);

        std::vector<cv::DMatch> matches;
        for (int i = 0; i < 3; i++) matches.push_back(cv::DMatch(i, i, 0));

        map->insertKeyFrame(f1);
        map->insertKeyFrame(f2);
        mapper.triangulateNewPoints(map, f1, f2, matches);

        assert(map->mapPointCount() == 3);
        for (int i = 0; i < 3; i++) {
            assert(f1->map_points[i] && f2->map_points[i]);
            assert(f1->map_points[i] == f2->map_points[i]);
        }
        // 帧1 处的深度为正（相机前方）
        for (int i = 0; i < 3; i++) {
            const Vec3 pc = f1->pose_cs * f1->map_points[i]->pos_s;
            assert(pc.z() > 0);
        }
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_select_local_window() {
    TEST("selectLocalWindow: 兜底取最近 n 帧并按 id 升序") {
        const vslam::Camera cam = makeStereo();
        LocalMapper mapper(cam);
        auto map = std::make_shared<Map>();

        for (unsigned long id = 0; id < 5; id++) {
            auto kf = std::make_shared<Frame>(id, (double)id * 0.1);
            kf->pose_cs = SE3(Eigen::Quaterniond::Identity(), Vec3((double)id * 0.2, 0, 0));
            map->insertKeyFrame(kf);
        }
        // 当前帧不在 Map 中（未注册）→ 兜底最近 3 帧
        auto curr = std::make_shared<Frame>(100, 1.0);
        curr->pose_cs = SE3();
        auto window = mapper.selectLocalWindow(map, curr, 3);
        assert(window.size() == 3);
        assert(window[0]->id == 2 && window[1]->id == 3 && window[2]->id == 4);
    } TEST_PASS();
}

void test_build_local_ba_snapshot() {
    TEST("buildLocalBASnapshot: min_observed 过滤 + anchor 裁剪") {
        const vslam::Camera cam = makeStereo();
        LocalMapper mapper(cam);
        auto map = std::make_shared<Map>();
        auto atlas = std::make_shared<vslam::Atlas>();

        // 3 个 KF 共享同一个地图点（观测数 3），1 个远端 KF 无共视
        std::vector<Frame::Ptr> kfs;
        for (unsigned long id = 0; id < 4; id++) {
            auto kf = std::make_shared<Frame>(id, (double)id * 0.1);
            kf->pose_cs = SE3(Eigen::Quaterniond::Identity(), Vec3((double)id * 0.3, 0, 0));
            kf->keypoints = {cv::KeyPoint(320, 240, 20)};
            kf->map_points.resize(1, nullptr);
            map->insertKeyFrame(kf);
            kfs.push_back(kf);
        }
        // 共享点：kf0/1/2 各观测一次 → observationCount=3
        auto mp = std::make_shared<MapPoint>(map->nextMapPointId());
        mp->pos_s = Vec3(0, 0, 5);
        map->insertMapPoint(mp);
        for (int i = 0; i < 3; i++) {
            assert(map->setObservation(kfs[i], 0, mp));
        }
        // 窗口 = 全部 4 帧，anchor = kf2
        auto snap = mapper.buildLocalBASnapshot(
            map, atlas, kfs, /*anchor_kf_id=*/2, /*min_observed=*/3);

        // 远端 kf3 无共视观测 → 被连通分量裁剪掉
        assert(!snap.keyframes.empty());
        for (const auto& ks : snap.keyframes)
            assert(ks.id != 3 && "远端帧必须被 anchor 连通分量裁剪");
        assert(snap.landmarks.size() == 1 && snap.landmarks[0].id == mp->id);
        assert(!snap.observations.empty());
        assert(!snap.fixed_kf_ids.empty());
    } TEST_PASS();
}

}  // namespace

int main() {
    cv::setNumThreads(1);
    cv::setRNGSeed(0x5A17);

    std::cout << "test_local_mapper (M1.5 LocalMapper)" << std::endl;

    test_include_local_ba_landmark();
    test_create_map_points_from_stereo();
    test_stereo_point_budget_is_per_point();
    test_tracking_hit_keeps_recent_points_fresh();
    test_triangulate_new_points();
    test_select_local_window();
    test_build_local_ba_snapshot();

    std::cout << "全部通过" << std::endl;
    return 0;
}
