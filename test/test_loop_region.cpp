#include "vslam/loop_region.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace vslam;

namespace {

Camera testCamera() {
    auto camera = std::make_shared<MonocularCamera>();
    camera->fx = 400.0;
    camera->fy = 400.0;
    camera->cx = 320.0;
    camera->cy = 240.0;
    return camera;
}

cv::Mat desc(int i) {
    cv::Mat d(1, 32, CV_8U, cv::Scalar::all(0));
    d.at<unsigned char>(0, i % 32) = static_cast<unsigned char>(255);
    return d;
}

void populate(const Frame::Ptr& anchor, const Frame::Ptr& neighbor,
              const Frame::Ptr& current, int count) {
    for (int i = 0; i < count; ++i) {
        const Vec3 p(-1.0 + 0.15 * (i % 8), -0.6 + 0.15 * (i / 8),
                     4.0 + 0.1 * (i % 3));
        auto mp = std::make_shared<MapPoint>(static_cast<unsigned long>(i + 1));
        mp->pos_s = p;
        mp->descriptor = desc(i);
        anchor->map_points.push_back(mp);
        anchor->descriptors.push_back(mp->descriptor);
        neighbor->map_points.push_back(mp);
        neighbor->descriptors.push_back(mp->descriptor);
        const Vec3 pc = current->pose_cs * p;
        current->keypoints.emplace_back(
            static_cast<float>(400.0 * pc.x() / pc.z() + 320.0),
            static_cast<float>(400.0 * pc.y() / pc.z() + 240.0), 1.0F);
        current->descriptors.push_back(mp->descriptor);
    }
}

void test_region_expands_beyond_anchor() {
    auto anchor = std::make_shared<Frame>(10, 1.0);
    auto neighbor = std::make_shared<Frame>(20, 2.0);
    auto current = std::make_shared<Frame>(100, 10.0);
    LoopRegionConfig cfg;
    cfg.max_keyframes = 2;
    cfg.max_covisible_keyframes = 1;
    cfg.max_temporal_neighbors = 0;
    cfg.max_points = 32;
    cfg.min_matches = cfg.min_inliers = 8;
    cfg.min_inlier_ratio = 0.8;
    cfg.min_grid_cells = 1;
    populate(anchor, neighbor, current, 16);
    // anchor 稀疏，邻居补足剩余地图点；区域应包含邻居点。
    anchor->map_points.resize(2);
    anchor->descriptors = anchor->descriptors.rowRange(0, 2).clone();
    LoopRegionSnapshot region;
    assert(LoopRegionVerifier::build({anchor, neighbor}, anchor, 3, 7, 9, cfg,
                                     region));
    assert(region.keyframes.size() == 2);
    assert(region.points.size() == 16);
    SE3 edge;
    LoopRegionResult result;
    assert(LoopRegionVerifier::verify(region, current, testCamera(), cfg, edge,
                                      &result));
    assert(result.inliers >= 8);
}

void test_negative_and_budget_and_formula() {
    auto anchor = std::make_shared<Frame>(10, 1.0);
    auto neighbor = std::make_shared<Frame>(20, 2.0);
    auto current = std::make_shared<Frame>(100, 10.0);
    current->pose_cs = SE3(Eigen::Quaterniond(
                               Eigen::AngleAxisd(0.12, Vec3::UnitY())),
                           Vec3(0.2, -0.1, 0.15));
    populate(anchor, neighbor, current, 24);
    LoopRegionConfig cfg;
    cfg.max_keyframes = 2;
    cfg.max_points = 10;
    cfg.max_covisible_keyframes = 1;
    cfg.max_temporal_neighbors = 1;
    cfg.min_matches = cfg.min_inliers = 8;
    cfg.min_inlier_ratio = 0.75;
    cfg.min_grid_cells = 1;
    LoopRegionSnapshot region;
    assert(LoopRegionVerifier::build({anchor, neighbor}, anchor, 4, 11, 13, cfg,
                                     region));
    assert(region.keyframes.size() <= 2 && region.points.size() <= 10);
    assert(std::ranges::any_of(region.points, [&](const auto& point) {
        return point.source_keyframe_id == neighbor->id;
    }) && "dense anchor must not monopolize the region point budget");
    SE3 good_edge;
    LoopRegionResult good_result;
    assert(LoopRegionVerifier::verify(region, current, testCamera(), cfg,
                                      good_edge, &good_result));
    SE3 expected = region.anchor_pose_cs * good_result.T_cw_curr_in_loop.inverse();
    assert((good_edge.matrix() - expected.matrix()).norm() < 1e-9);
    auto bad = std::make_shared<Frame>(*current);
    for (size_t i = 0; i < bad->keypoints.size(); ++i) {
        bad->keypoints[i].pt.x += 70.0F + static_cast<float>(i * 3);
        if (i % 2 == 0) bad->keypoints[i].pt.y -= 55.0F;
    }
    SE3 edge;
    LoopRegionResult result;
    assert(!LoopRegionVerifier::verify(region, bad, testCamera(), cfg, edge,
                                       &result));
    assert(!result.accepted);
    assert(result.matches >= cfg.min_matches);
    assert(region.isBoundTo(4, 11, 13));
    assert(!region.isBoundTo(4, 12, 13));
}

} // namespace

int main() {
    test_region_expands_beyond_anchor();
    test_negative_and_budget_and_formula();
    std::cout << "test_loop_region: PASS\n";
    return 0;
}
