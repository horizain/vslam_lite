/**
 * test_resource_budget.cpp - ResourceBudget 资源预算单元测试
 *
 * 覆盖 M2.2（PRODUCTION_LOCALIZATION_PLAN §6.2/§6.3）：
 *   1. evaluate：预算判定（KF/点/描述子/snapshot/总量）
 *   2. reclaim 第 1 步：删除 0 个正式 Observation 的点
 *   3. reclaim 第 2 步：删除 observationCount<2 且超过 30 个 KF 未被命中的点
 *      （旁路统计 lastHitKeyframeCount，禁止复用 observed_count 语义）
 *   4. reclaim 第 3 步：卸载非活动 KF 原图/灰度图，保留关键点/描述子/位姿/观测
 *   5. reclaim 第 4 步：共视重叠 >0.9 + 相邻位姿差 <0.15m/3deg 的冗余 KF 剔除，
 *      回环/子地图锚点（protected）必须保留
 *   6. reclaim 第 5 步：普通历史 KF 压缩为有界 Essential Anchor 骨架，
 *      保留首尾、protected 回环端点和最近活动稠密窗口
 *   7. reclaim 第 6 步：冻结超过 2 个的非活动子地图并卸载其 KF 图像缓存
 *   8. reclaim 第 7 步：仍超预算时停止增加地图（stopped_map_growth），
 *      不得随机删除强点/锚点
 *   9. Map 新 API：removeKeyFrame 一致性、recordTrackingHit 旁路统计
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_resource_budget
 * 运行: ./build/test_resource_budget（独立 CTest）
 */

#include "vslam/resource_budget.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/mappoint.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::Frame;
using vslam::KeyframeId;
using vslam::Map;
using vslam::MapPoint;
using vslam::MapPointId;
using vslam::ResourceBudget;
using vslam::SE3;
using vslam::Vec3;

namespace {

// 构造一个带 keypoints/descriptors（可选图像）的关键帧并注册到 Map。
// feature_slots：绑定到该 KF 的 keypoint 数量（用于后续 setObservation）。
Frame::Ptr makeKeyframe(const Map::Ptr& map, unsigned long id, const Vec3& t,
                        size_t feature_slots, bool with_images) {
    auto kf = std::make_shared<Frame>(id, (double)id * 0.1);
    kf->pose_cs = SE3(Eigen::Quaterniond::Identity(), t);
    kf->keypoints.reserve(feature_slots);
    for (size_t i = 0; i < feature_slots; i++)
        kf->keypoints.emplace_back(100.0 + (double)i * 10.0, 120.0, 20.0);
    kf->map_points.resize(feature_slots, nullptr);
    kf->descriptors = cv::Mat((int)feature_slots, 32, CV_8U, cv::Scalar::all(1));
    if (with_images) {
        kf->image = cv::Mat(240, 320, CV_8U, cv::Scalar::all(128));
        kf->image_gray = cv::Mat(240, 320, CV_8U, cv::Scalar::all(128));
        kf->image_right = cv::Mat(240, 320, CV_8U, cv::Scalar::all(128));
        kf->image_right_gray = cv::Mat(240, 320, CV_8U, cv::Scalar::all(128));
    }
    map->insertKeyFrame(kf);
    return kf;
}

MapPoint::Ptr makePoint(const Map::Ptr& map) {
    auto mp = std::make_shared<MapPoint>(map->nextMapPointId());
    mp->pos_s = Vec3(1.0, 2.0, 5.0);
    mp->descriptor = cv::Mat(1, 32, CV_8U, cv::Scalar::all(3));
    map->insertMapPoint(mp);
    return mp;
}

// 给 KF 的第 feature 个 slot 建立与点的正式观测
void observe(const Map::Ptr& map, const Frame::Ptr& kf, size_t feature,
             const MapPoint::Ptr& mp) {
    assert(map->setObservation(kf, static_cast<vslam::FeatureIndex>(feature), mp));
}

bool kfImagesReleased(const Frame::Ptr& kf) {
    return kf->image.empty() && kf->image_gray.empty() &&
           kf->image_right.empty() && kf->image_right_gray.empty();
}

void test_evaluate_empty_map() {
    TEST("evaluate: 空地图在预算内") {
        ResourceBudget budget;
        auto map = std::make_shared<Map>();
        const auto s = budget.evaluate(map);
        assert(s.within_budget);
        assert(!s.over_keyframes && !s.over_points);
        assert(!s.over_descriptor && !s.over_snapshot && !s.over_total);
    } TEST_PASS();
}

void test_evaluate_over_keyframes() {
    TEST("evaluate: 关键帧超限") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 3;
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();
        for (unsigned long id = 0; id < 4; id++)
            makeKeyframe(map, id, Vec3((double)id, 0, 0), 0, false);
        const auto s = budget.evaluate(map);
        assert(s.over_keyframes);
        assert(!s.within_budget);
        assert(s.keyframes == 4);
    } TEST_PASS();
}

void test_evaluate_over_points() {
    TEST("evaluate: 地图点超限") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_points = 5;
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();
        for (int i = 0; i < 6; i++) makePoint(map);
        const auto s = budget.evaluate(map);
        assert(s.over_points);
        assert(!s.within_budget);
        assert(s.points == 6);
    } TEST_PASS();
}

void test_evaluate_over_descriptor() {
    TEST("evaluate: 描述子字节超限") {
        vslam::MapBudgetConfig cfg;
        cfg.max_descriptor_mb = 0;  // 任意描述子字节即超限
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();
        makeKeyframe(map, 0, Vec3::Zero(), 1, false);
        makePoint(map);
        const auto s = budget.evaluate(map);
        assert(s.over_descriptor);
        assert(!s.within_budget);
        assert(s.descriptor_bytes > 0);
    } TEST_PASS();
}

void test_evaluate_over_snapshot() {
    TEST("evaluate: 外部 snapshot 字节计入总量") {
        vslam::MapBudgetConfig cfg;
        cfg.max_snapshot_mb = 1;   // 1 MiB
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();
        const size_t snapshot_bytes = 2ull * 1024 * 1024;  // 2 MiB（外部在途快照）
        const auto s = budget.evaluate(map, snapshot_bytes);
        assert(s.over_snapshot);
        assert(!s.within_budget);
        assert(s.estimated_total_bytes >= snapshot_bytes);
    } TEST_PASS();
}

void test_reclaim_step1_zero_observation_points() {
    TEST("reclaim 第1步: 0 观测点删除、正式观测点保留") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_points = 1;
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();

        const auto kf = makeKeyframe(map, 0, Vec3::Zero(), 1, false);
        const auto orphan_a = makePoint(map);   // 0 观测
        const auto orphan_b = makePoint(map);   // 0 观测
        const auto observed = makePoint(map);   // 1 个正式观测
        observe(map, kf, 0, observed);

        const auto r = budget.reclaim(map);
        assert(r.removed_zero_obs_points == 2);
        assert(map->mapPointCount() == 1);
        assert(map->getMapPoint(orphan_a->id) == nullptr);
        assert(map->getMapPoint(orphan_b->id) == nullptr);
        assert(map->getMapPoint(observed->id) == observed);
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_reclaim_step2_weak_stale_points() {
    TEST("reclaim 第2步: 弱且 30 KF 未命中点删除、最近命中弱点保留、强点保留") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_points = 2;
        cfg.weak_point_stale_kf_window = 30;
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();

        // kf0: 点 A（weak）首次观测；kf30: 点 B（weak）首次观测；
        // kf31 之后 total KF = 32。A 的 last hit 距今 31 个 KF > 30 → stale；
        // B 的 last hit 距今 1 个 KF → fresh。
        const auto kf0 = makeKeyframe(map, 0, Vec3::Zero(), 2, false);
        const auto a = makePoint(map);
        observe(map, kf0, 0, a);

        const auto kf1 = makeKeyframe(map, 1, Vec3(1.0, 0, 0), 1, false);
        for (unsigned long id = 2; id < 30; id++)
            makeKeyframe(map, id, Vec3((double)id, 0, 0), 0, false);
        const auto kf30 = makeKeyframe(map, 30, Vec3(30, 0, 0), 1, false);
        const auto b = makePoint(map);
        observe(map, kf30, 0, b);
        makeKeyframe(map, 31, Vec3(31, 0, 0), 0, false);

        // 强点 C：观测数 2，即使陈旧也不得删除
        const auto c = makePoint(map);
        observe(map, kf0, 1, c);
        observe(map, kf1, 0, c);

        const auto r = budget.reclaim(map);
        assert(r.removed_weak_stale_points == 1);
        assert(map->mapPointCount() == 2);
        assert(map->getMapPoint(a->id) == nullptr);
        assert(map->getMapPoint(b->id) == b);
        assert(map->getMapPoint(c->id) == c);
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_reclaim_step3_unload_inactive_kf_images() {
    TEST("reclaim 第3步: 非活动 KF 图像卸载、几何数据保留、最近 KF 图像保留") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 100;
        cfg.max_total_estimated_mb = 1;
        cfg.overhead_bytes_per_keyframe = 0;
        cfg.overhead_bytes_per_point = 0;
        cfg.kf_image_keep_recent = 2;
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();

        std::vector<Frame::Ptr> kfs;
        for (unsigned long id = 0; id < 4; id++)
            kfs.push_back(makeKeyframe(map, id, Vec3((double)id * 3.0, 0, 0),
                                       2, /*with_images=*/true));

        const auto r = budget.reclaim(map);
        assert(r.unloaded_kf_images == 2);
        // 非活动 KF（kf0/kf1）：图像全部释放，几何数据必须保留
        for (int i = 0; i < 2; i++) {
            assert(kfs[i]->image.empty() && "非活动 KF 图像必须卸载");
            assert(kfs[i]->keypoints.size() == 2);
            assert(!kfs[i]->descriptors.empty());
            assert(kfs[i]->pose_cs.t.norm() == 3.0 * i);
        }
        // 最近窗口 KF（kf2/kf3）：图像保留
        for (int i = 2; i < 4; i++)
            assert(!kfs[i]->image.empty() && "最近 KF 图像必须保留");
        // 图像卸载不是删除 KF
        assert(map->keyFrameCount() == 4);
    } TEST_PASS();
}

void test_reclaim_step4_redundant_keyframes() {
    TEST("reclaim 第4步: 冗余 KF 剔除、锚点保护、大位移 KF 保留") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 5;
        cfg.redundant_overlap_threshold = 0.9;
        cfg.redundant_max_translation_m = 0.15;
        cfg.redundant_max_rotation_deg = 3.0;
        cfg.kf_image_keep_recent = 1;  // 只保护最后 1 个 KF 的图像/剔除窗口
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();

        // kf0/kf1：位移 0.05m、共享 10/10 点 → 冗余（kf0 剔除）
        // kf2/kf3：位移 0.05m、共享 10/10 点，但 kf2 是锚点（protected）→ 都不剔除
        // kf4/kf5：位移 2m → 不冗余
        const auto kf0 = makeKeyframe(map, 0, Vec3(0.0, 0, 0), 10, false);
        const auto kf1 = makeKeyframe(map, 1, Vec3(0.05, 0, 0), 10, false);
        const auto kf2 = makeKeyframe(map, 2, Vec3(1.0, 0, 0), 10, false);
        const auto kf3 = makeKeyframe(map, 3, Vec3(1.05, 0, 0), 10, false);
        const auto kf4 = makeKeyframe(map, 4, Vec3(3.0, 0, 0), 10, false);
        const auto kf5 = makeKeyframe(map, 5, Vec3(5.0, 0, 0), 10, false);

        // 三组各自共享的 10 个强点（obs=2）
        for (auto [ka, kb] : {std::pair{kf0, kf1}, std::pair{kf2, kf3},
                              std::pair{kf4, kf5}}) {
            for (size_t i = 0; i < 10; i++) {
                const auto mp = makePoint(map);
                observe(map, ka, i, mp);
                observe(map, kb, i, mp);
            }
        }

        std::unordered_set<KeyframeId> protected_kfs{kf2->id};  // 锚点
        size_t cull_callbacks = 0;
        const auto r = budget.reclaim(
            map, protected_kfs, {}, 0, {},
            [&](const Frame::Ptr& removed, const Frame::Ptr& replacement) {
                cull_callbacks++;
                assert(removed && replacement);
                assert(map->getKeyFrame(removed->id) == removed &&
                       "回调必须发生在 KF 真正删除之前");
                assert(removed->id == kf0->id && replacement->id == kf1->id);
            });

        assert(r.culled_redundant_keyframes == 1);
        assert(r.culled_keyframe_ids == std::vector<KeyframeId>{kf0->id});
        assert(cull_callbacks == 1);
        assert(map->getKeyFrame(kf0->id) == nullptr && "冗余 KF kf0 必须剔除");
        assert(map->getKeyFrame(kf2->id) == kf2 && "锚点 KF 必须保留");
        assert(map->getKeyFrame(kf3->id) == kf3);
        assert(map->getKeyFrame(kf4->id) == kf4);
        assert(map->getKeyFrame(kf5->id) == kf5);
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_reclaim_step6_freeze_submaps() {
    TEST("reclaim 第6步: 超过 2 个非活动子地图 → 冻结并卸载其 KF 图像缓存") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 100;
        cfg.max_descriptor_mb = 0;     // 描述子压力持续到第 6 步
        cfg.kf_image_keep_recent = 4;  // 全部 KF 都在图像保留窗口内（隔离第 3 步）
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();

        std::vector<Frame::Ptr> kfs;
        for (unsigned long id = 0; id < 4; id++)
            kfs.push_back(makeKeyframe(map, id, Vec3((double)id * 3.0, 0, 0),
                                       2, /*with_images=*/true));

        // 3 个非活动子地图：100/200/300 → 保留最新 2 个，冻结最老 1 个（100）
        std::unordered_map<unsigned long, std::vector<KeyframeId>> submap_kfs;
        submap_kfs[100] = {kfs[0]->id};
        submap_kfs[200] = {kfs[1]->id};
        submap_kfs[300] = {kfs[2]->id};
        const auto r = budget.reclaim(map, {}, submap_kfs);

        assert(r.frozen_submaps == 1);
        assert(r.unloaded_submap_kf_images == 1);
        assert(kfImagesReleased(kfs[0]) && "冻结子地图 100 的 KF 图像必须卸载");
        assert(!kfs[1]->image.empty() && "保留子地图 200 的 KF 图像必须保留");
        assert(!kfs[2]->image.empty());
        assert(!kfs[3]->image.empty());
    } TEST_PASS();
}

void test_reclaim_essential_history_compaction() {
    TEST("reclaim 第5步: 历史普通 KF 压成 Essential Anchor 骨架") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 8;
        cfg.active_dense_keyframes = 3;
        cfg.max_historical_anchors = 3;
        cfg.historical_anchor_stride = 2;
        cfg.kf_image_keep_recent = 1;
        cfg.max_active_points = 100000;
        cfg.max_descriptor_mb = 256;
        cfg.max_total_estimated_mb = 4096;
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();
        for (unsigned long id = 0; id < 12; ++id)
            makeKeyframe(map, id, Vec3(static_cast<double>(id), 0, 0), 0, false);

        std::unordered_set<KeyframeId> protected_kfs{3};
        std::unordered_map<KeyframeId, KeyframeId> replacements;
        const auto r = budget.reclaim(
            map, protected_kfs, {}, 0, {},
            [&](const Frame::Ptr& removed, const Frame::Ptr& replacement) {
                assert(removed && replacement);
                replacements.emplace(removed->id, replacement->id);
            });

        assert(r.compacted_historical_keyframes == 6);
        assert(r.culled_keyframe_ids.size() == 6);
        assert(map->keyFrameCount() == 6);
        assert(map->getKeyFrame(0) && "历史首锚必须保留");
        assert(map->getKeyFrame(3) && "回环/protected 端点必须保留");
        assert(map->getKeyFrame(8) && "历史尾锚必须保留");
        assert(map->getKeyFrame(9) && map->getKeyFrame(10) &&
               map->getKeyFrame(11) && "最近活动稠密窗口必须完整保留");
        assert(replacements.size() == r.compacted_historical_keyframes);
        assert(!r.stopped_map_growth);
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_mobile_history_compaction_scale() {
    TEST("reclaim 第5步: mobile 321 KF 压到约 220 个检索/活动锚") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 320;
        cfg.active_dense_keyframes = 120;
        cfg.max_historical_anchors = 128;
        cfg.historical_anchor_stride = 2;
        cfg.max_active_points = 100000;
        cfg.max_descriptor_mb = 256;
        cfg.max_total_estimated_mb = 4096;
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();
        for (unsigned long id = 0; id < 321; ++id)
            makeKeyframe(map, id, Vec3(static_cast<double>(id), 0, 0), 0, false);

        const auto r = budget.reclaim(map);
        assert(r.compacted_historical_keyframes == 100);
        assert(map->keyFrameCount() == 221);
        assert(map->getKeyFrame(0) && map->getKeyFrame(200));
        for (unsigned long id = 201; id < 321; ++id)
            assert(map->getKeyFrame(id) && "最近 120 KF 活动稠密窗口必须完整保留");
        assert(!r.stopped_map_growth);
    } TEST_PASS();
}

void test_reclaim_step6_freeze_submap_weak_points() {
    TEST("reclaim 第6步: 冻结子地图同时删除弱陈点（M2 遗留清理，RSS 硬门槛）") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 100;
        cfg.max_descriptor_mb = 0;      // 描述子压力持续到第 6 步
        cfg.kf_image_keep_recent = 4;   // 隔离第 3 步
        cfg.weak_point_min_observations = 2;
        cfg.weak_point_stale_kf_window = 1;
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();

        std::vector<Frame::Ptr> kfs;
        for (unsigned long id = 0; id < 4; id++)
            kfs.push_back(makeKeyframe(map, id, Vec3((double)id * 3.0, 0, 0),
                                       2, /*with_images=*/true));

        // 冻结子地图 100 的独立 Map（模拟 Submap::map）
        auto sub_map = std::make_shared<Map>();
        const auto s_kf0 = makeKeyframe(sub_map, 0, Vec3::Zero(), 3, false);
        // 弱陈点：1 个观测（observe 时 kf_count=1 → lastHit=1）；
        // 此后又插入 2 个 KF（kf_count=3）→ 1+窗口1 < 3 → 陈旧可删
        const auto weak_mp = makePoint(sub_map);
        observe(sub_map, s_kf0, 0, weak_mp);
        const auto s_kf1 = makeKeyframe(sub_map, 1, Vec3(1.0, 0, 0), 3, false);
        const auto s_kf2 = makeKeyframe(sub_map, 2, Vec3(2.0, 0, 0), 3, false);
        // 强点：2 个观测（≥ weak_point_min_observations）→ 必须保留
        const auto strong_mp = makePoint(sub_map);
        observe(sub_map, s_kf0, 1, strong_mp);
        observe(sub_map, s_kf2, 1, strong_mp);

        std::unordered_map<unsigned long, std::vector<KeyframeId>> submap_kfs;
        submap_kfs[100] = {kfs[0]->id};
        submap_kfs[200] = {kfs[1]->id};
        submap_kfs[300] = {kfs[2]->id};
        std::unordered_map<unsigned long, vslam::Map::Ptr> inactive_submaps;
        inactive_submaps[100] = sub_map;
        inactive_submaps[200] = std::make_shared<Map>();
        inactive_submaps[300] = std::make_shared<Map>();

        const auto r = budget.reclaim(map, {}, submap_kfs, 0, inactive_submaps);

        assert(r.frozen_submaps == 1);
        assert(r.unloaded_submap_kf_images == 1);
        assert(r.removed_frozen_submap_points == 1 && "弱陈点必须被删除");
        assert(sub_map->mapPointCount() == 1 && "强点必须保留");
        assert(sub_map->getMapPoint(strong_mp->id) != nullptr);
        assert(sub_map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_reclaim_step7_stop_growth() {
    TEST("reclaim 第7步: 仍超预算 → stopped_map_growth，强点不随机删除") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 10;
        cfg.max_active_points = 2;  // 3 个强点无法回收回预算内
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();

        // 3 个 KF 全部观测 3 个强点（obs=3，最近命中 → 第 1/2 步均不删）
        std::vector<Frame::Ptr> kfs;
        for (unsigned long id = 0; id < 3; id++)
            kfs.push_back(makeKeyframe(map, id, Vec3((double)id, 0, 0), 3, false));
        std::vector<MapPoint::Ptr> mps;
        for (int i = 0; i < 3; i++) {
            mps.push_back(makePoint(map));
            for (unsigned long j = 0; j < 3; j++)
                observe(map, kfs[j], (size_t)i, mps[i]);
        }

        const auto r = budget.reclaim(map);
        assert(r.stopped_map_growth && "无法回收到预算内时必须停止增加地图");
        assert(r.removed_zero_obs_points == 0 && r.removed_weak_stale_points == 0);
        assert(map->mapPointCount() == 3 && "不得随机删除强点");
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

void test_map_remove_keyframe() {
    TEST("Map::removeKeyFrame: 双向观测/共视/计数一致") {
        auto map = std::make_shared<Map>();
        const auto kf0 = makeKeyframe(map, 0, Vec3::Zero(), 1, false);
        const auto kf1 = makeKeyframe(map, 1, Vec3(0.5, 0, 0), 1, false);
        const auto mp = makePoint(map);
        observe(map, kf0, 0, mp);
        observe(map, kf1, 0, mp);
        assert(mp->observationCount() == 2);
        const uint64_t topo_before = map->topologyRevision();

        assert(map->removeKeyFrame(kf0->id));
        assert(!map->removeKeyFrame(kf0->id) && "重复删除幂等返回 false");
        assert(map->getKeyFrame(kf0->id) == nullptr);
        assert(map->keyFrameCount() == 1);
        assert(map->topologyRevision() > topo_before);
        // 点的反向观测只剩 kf1；kf0 的 slot 被清空
        assert(mp->observationCount() == 1);
        assert(kf0->map_points[0] == nullptr);
        assert(kf1->map_points[0] == mp);
        assert(map->verifyObservationConsistency());
        // 正式观测被移除后点不再与 kf0 有共视
        assert(map->sharedObservationCount(kf0->id, kf1->id) == 0);
    } TEST_PASS();
}

void test_map_last_hit_side_statistics() {
    TEST("Map::recordTrackingHit: 旁路统计（不复用 observed_count）") {
        auto map = std::make_shared<Map>();
        const auto kf0 = makeKeyframe(map, 0, Vec3::Zero(), 1, false);
        const auto never_hit = makePoint(map);
        const auto observed = makePoint(map);
        observe(map, kf0, 0, observed);

        // 从未命中：lastHit = 0（map->keyFrameCount() == 1）
        assert(map->lastHitKeyframeCount(never_hit->id) == 0);
        // 正式观测自动记录命中（kf0 插入后计数为 1）
        assert(map->lastHitKeyframeCount(observed->id) == 1);

        // 手动记录跟踪命中（未来 LK/PnP 关联路径调用；与观测数无关）
        makeKeyframe(map, 1, Vec3(1, 0, 0), 0, false);  // kf_count_ = 2
        map->recordTrackingHit(never_hit->id);
        assert(map->lastHitKeyframeCount(never_hit->id) == 2);
        assert(never_hit->observationCount() == 0 && "命中统计不得改变正式观测");

        // 点删除后旁路统计一并清理
        map->removeMapPoint(never_hit->id);
        assert(map->lastHitKeyframeCount(never_hit->id) == 0);
    } TEST_PASS();
}

void test_reclaim_full_consistency() {
    TEST("reclaim 全程: 组合场景后地图观测一致") {
        vslam::MapBudgetConfig cfg;
        cfg.max_active_keyframes = 5;   // 6 个 KF 超限 → 触发回收
        cfg.max_active_points = 10;     // 点数量不超限，隔离第 1/2 步的执行
        ResourceBudget budget(cfg);
        auto map = std::make_shared<Map>();

        // kf0/kf1 相距 0.05m，但 kf0 是锚点（protected）→ 不得剔除
        const auto kf0 = makeKeyframe(map, 0, Vec3(0.0, 0, 0), 4, true);
        const auto kf1 = makeKeyframe(map, 1, Vec3(0.05, 0, 0), 4, true);
        makeKeyframe(map, 2, Vec3(3.0, 0, 0), 0, true);
        makeKeyframe(map, 3, Vec3(4.0, 0, 0), 0, true);
        makeKeyframe(map, 4, Vec3(5.0, 0, 0), 0, true);
        makeKeyframe(map, 5, Vec3(6.0, 0, 0), 0, true);

        const auto orphan = makePoint(map);     // 0 观测 → 第 1 步删除
        const auto strong_a = makePoint(map);   // obs=2（kf0/kf1 共享）
        const auto strong_b = makePoint(map);   // obs=2（kf0/kf1 共享）
        observe(map, kf0, 1, strong_a);
        observe(map, kf1, 1, strong_a);
        observe(map, kf0, 2, strong_b);
        observe(map, kf1, 2, strong_b);

        std::unordered_set<KeyframeId> protected_kfs{
            kf0->id, kf1->id, 2, 3, 4, 5};
        const auto r = budget.reclaim(map, protected_kfs);

        assert(r.removed_zero_obs_points == 1);
        assert(map->getMapPoint(orphan->id) == nullptr);
        assert(map->getMapPoint(strong_a->id) == strong_a);
        assert(map->getMapPoint(strong_b->id) == strong_b);
        assert(r.culled_redundant_keyframes == 0 && "锚点 KF 不得被冗余剔除");
        assert(map->keyFrameCount() == 6);
        assert(map->verifyObservationConsistency());
    } TEST_PASS();
}

}  // namespace

int main() {
    cv::setNumThreads(1);
    cv::setRNGSeed(0x5A17);

    std::cout << "test_resource_budget (M2.2 ResourceBudget)" << std::endl;

    test_evaluate_empty_map();
    test_evaluate_over_keyframes();
    test_evaluate_over_points();
    test_evaluate_over_descriptor();
    test_evaluate_over_snapshot();
    test_reclaim_step1_zero_observation_points();
    test_reclaim_step2_weak_stale_points();
    test_reclaim_step3_unload_inactive_kf_images();
    test_reclaim_step4_redundant_keyframes();
    test_reclaim_essential_history_compaction();
    test_mobile_history_compaction_scale();
    test_reclaim_step6_freeze_submaps();
    test_reclaim_step6_freeze_submap_weak_points();
    test_reclaim_step7_stop_growth();
    test_map_remove_keyframe();
    test_map_last_hit_side_statistics();
    test_reclaim_full_consistency();

    std::cout << "全部通过" << std::endl;
    return 0;
}
