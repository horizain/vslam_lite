#include "vslam/loop_graph.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_map>

using vslam::Constraint;
using vslam::EssentialGraphConfig;
using vslam::KeyframeState;
using vslam::OptimizationResult;
using vslam::OptimizationSnapshot;
using vslam::SE3;
using vslam::Vec3;

namespace {

double rotationDistance(const Eigen::Quaterniond& a,
                        const Eigen::Quaterniond& b) {
    return a.angularDistance(b);
}

void test_global_pose_split() {
    const SE3 T_oc(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.31, Vec3::UnitY())),
        Vec3(1.2, -0.4, 3.1));
    const SE3 T_wo(
        Eigen::Quaterniond(Eigen::AngleAxisd(-0.27, Vec3::UnitZ())),
        Vec3(-4.0, 2.3, 0.8));
    const SE3 T_wc = T_wo * T_oc;

    const auto split = vslam::splitGlobalCameraPose(T_wc, T_oc);
    assert((split.T_wo.t - T_wo.t).norm() < 1e-12);
    assert(rotationDistance(split.T_wo.q, T_wo.q) < 1e-12);
    const SE3 recomposed = split.T_wo * split.T_oc;
    assert((recomposed.t - T_wc.t).norm() < 1e-12);
    assert(rotationDistance(recomposed.q, T_wc.q) < 1e-12);
}

void test_prefix_tail_rebase() {
    const SE3 old_endpoint(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.22, Vec3::UnitX())),
        Vec3(2.0, -1.0, 4.0));
    const SE3 new_endpoint(
        Eigen::Quaterniond(Eigen::AngleAxisd(-0.35, Vec3::UnitZ())),
        Vec3(-3.0, 0.7, 1.0));
    const SE3 old_tail(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.41, Vec3::UnitY())),
        Vec3(5.2, -2.1, 7.4));

    const SE3 new_tail = vslam::rebaseTailPose(
        old_tail, old_endpoint, new_endpoint);
    const SE3 old_relative = old_tail * old_endpoint.inverse();
    const SE3 new_relative = new_tail * new_endpoint.inverse();
    assert((old_relative.t - new_relative.t).norm() < 1e-12);
    assert(rotationDistance(old_relative.q, new_relative.q) < 1e-12);

    const Vec3 point_s(2.5, -0.7, 8.2);
    const Vec3 rebased_point = vslam::rebaseTailPoint(
        point_s, old_endpoint, new_endpoint);
    assert((new_endpoint * rebased_point - old_endpoint * point_s).norm()
           < 1e-12);
}

OptimizationSnapshot makeLongGraph() {
    OptimizationSnapshot full;
    full.submap_id = 7;
    for (unsigned long i = 0; i <= 100; ++i) {
        const Eigen::Quaterniond q(
            Eigen::AngleAxisd(0.002 * static_cast<double>(i), Vec3::UnitY()));
        full.keyframes.push_back({i, SE3(q, Vec3(-0.5 * i, 0.0, 0.1 * i))});
        if (i > 0) {
            const auto& a = full.keyframes[i - 1];
            const auto& b = full.keyframes[i];
            full.constraints.push_back(
                {i - 1, i, a.pose_cs * b.pose_cs.inverse(), 2.0, false});
        }
    }
    const auto& loop = full.keyframes[13];
    const auto& curr = full.keyframes[97];
    full.constraints.push_back(
        {13, 97, loop.pose_cs * curr.pose_cs.inverse(), 10.0, true});
    return full;
}

void test_essential_anchor_graph_is_bounded() {
    const OptimizationSnapshot full = makeLongGraph();
    EssentialGraphConfig cfg;
    cfg.max_anchors = 16;
    cfg.preferred_stride = 8;
    const auto graph = vslam::buildEssentialAnchorGraph(full, cfg);

    assert(graph.snapshot.keyframes.size() <= 16);
    assert(graph.snapshot.keyframes.front().id == 0);
    assert(graph.snapshot.keyframes.back().id == 100);
    bool has_loop_a = false;
    bool has_loop_b = false;
    for (const auto& kf : graph.snapshot.keyframes) {
        has_loop_a = has_loop_a || kf.id == 13;
        has_loop_b = has_loop_b || kf.id == 97;
    }
    assert(has_loop_a && has_loop_b);
    assert(!graph.snapshot.constraints.empty());
    for (const auto& edge : graph.snapshot.constraints) {
        bool a_is_anchor = false;
        bool b_is_anchor = false;
        for (const auto& kf : graph.snapshot.keyframes) {
            a_is_anchor = a_is_anchor || kf.id == edge.a;
            b_is_anchor = b_is_anchor || kf.id == edge.b;
        }
        assert(a_is_anchor && b_is_anchor);
    }

    // KITTI 高精度档的量级回归：1800 个逐 KF 顶点不得重新进 g2o。
    OptimizationSnapshot production_scale;
    for (unsigned long i = 0; i < 1800; ++i) {
        production_scale.keyframes.push_back(
            {i, SE3(Eigen::Quaterniond::Identity(), Vec3(-0.2 * i, 0.0, 0.0))});
        if (i > 0) {
            const auto& a = production_scale.keyframes[i - 1];
            const auto& b = production_scale.keyframes[i];
            production_scale.constraints.push_back(
                {i - 1, i, a.pose_cs * b.pose_cs.inverse(), 1.0, false});
        }
    }
    production_scale.constraints.push_back(
        {153, 1597,
         production_scale.keyframes[153].pose_cs *
             production_scale.keyframes[1597].pose_cs.inverse(),
         10.0, true});
    const auto mobile_graph = vslam::buildEssentialAnchorGraph(
        production_scale, EssentialGraphConfig{64, 6});
    assert(mobile_graph.snapshot.keyframes.size() <= 64);
}

void test_anchor_correction_propagates_to_all_prefix_keyframes() {
    OptimizationSnapshot full;
    for (unsigned long i = 0; i <= 10; ++i)
        full.keyframes.push_back({i, SE3(Eigen::Quaterniond::Identity(),
                                         Vec3(-static_cast<double>(i), 0.0, 0.0))});

    vslam::EssentialAnchorGraph graph;
    graph.full_keyframes = full.keyframes;
    graph.snapshot.keyframes = {full.keyframes.front(), full.keyframes.back()};

    OptimizationResult anchor_result;
    anchor_result.valid = true;
    anchor_result.poses.push_back({0, full.keyframes.front().pose_cs});
    const SE3 old_last_wc = full.keyframes.back().pose_cs.inverse();
    const SE3 correction(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Vec3::UnitZ())),
        Vec3(3.0, -2.0, 1.0));
    const SE3 new_last_wc = correction * old_last_wc;
    anchor_result.poses.push_back({10, new_last_wc.inverse()});

    const auto propagated = vslam::propagateAnchorCorrections(graph, anchor_result);
    assert(propagated.size() == full.keyframes.size());
    assert((propagated.front().pose_cs.t - full.keyframes.front().pose_cs.t).norm()
           < 1e-12);
    assert((propagated.back().pose_cs.t - new_last_wc.inverse().t).norm() < 1e-12);
    assert(rotationDistance(propagated.back().pose_cs.q,
                            new_last_wc.inverse().q) < 1e-12);
    // 中间帧必须获得有限、非零且小于端点的渐进校正，不能整段刚体跳变。
    const SE3 old_mid_wc = full.keyframes[5].pose_cs.inverse();
    const SE3 new_mid_wc = propagated[5].pose_cs.inverse();
    const double mid_shift = (new_mid_wc.t - old_mid_wc.t).norm();
    const double end_shift = (new_last_wc.t - old_last_wc.t).norm();
    assert(mid_shift > 0.0 && mid_shift < end_shift);
    assert(new_mid_wc.t.allFinite() && new_mid_wc.q.coeffs().allFinite());
}

void test_submap_graph_has_single_gauge_and_movable_history() {
    vslam::Atlas atlas;
    const vslam::SE3 T_ws0(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.17, vslam::Vec3::UnitY())),
        vslam::Vec3(3.0, -1.0, 2.0));
    const vslam::SE3 T_ws1(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.31, vslam::Vec3::UnitZ())),
        vslam::Vec3(14.0, 2.0, 1.0));
    const vslam::SE3 T_ws2(
        Eigen::Quaterniond(Eigen::AngleAxisd(-0.22, vslam::Vec3::UnitX())),
        vslam::Vec3(31.0, 4.0, -2.0));
    const auto id0 = atlas.createSubmap(T_ws0, true).id;
    const auto id1 = atlas.createSubmap(T_ws1, true).id;
    const auto id2 = atlas.createSubmap(T_ws2, true).id;

    atlas.addConstraint({id0, id1, T_ws0.inverse() * T_ws1,
                         1.0, vslam::AtlasConstraintType::TrackingBridge});
    atlas.addConstraint({id1, id2, T_ws1.inverse() * T_ws2,
                         1.0, vslam::AtlasConstraintType::TrackingBridge});
    const auto graph = vslam::buildSubmapGraph(atlas);

    assert(graph.keyframes.size() == 3);
    assert(graph.constraints.size() == 2);
    assert(graph.fixed_kf_ids == std::vector<vslam::KeyframeId>{id0} &&
           "Atlas 只能固定一个 gauge，历史子图必须可被后续回环重分配");
    for (const auto& state : graph.keyframes) {
        const vslam::SE3 expected = state.id == id0 ? T_ws0
            : state.id == id1 ? T_ws1 : T_ws2;
        const vslam::SE3 recovered = state.pose_cs.inverse();
        assert((recovered.t - expected.t).norm() < 1e-12);
        assert(recovered.q.angularDistance(expected.q) < 1e-12);
    }

#ifdef HAS_G2O
    // 三节点链上同时存在里程计漂移与闭环时，中间历史节点必须参与分配，
    // 不能只把活动尾节点从 30m 拉回而把历史节点钉死在 15m。
    vslam::Atlas drifted;
    const auto a = drifted.createSubmap(SE3(), true).id;
    const auto b = drifted.createSubmap(SE3(
        Eigen::Quaterniond::Identity(), Vec3(15.0, 0.0, 0.0)), true).id;
    const auto c = drifted.createSubmap(SE3(
        Eigen::Quaterniond::Identity(), Vec3(30.0, 0.0, 0.0)), true).id;
    drifted.addConstraint({a, b, SE3(
        Eigen::Quaterniond::Identity(), Vec3(10.0, 0.0, 0.0)),
        1.0, vslam::AtlasConstraintType::TrackingBridge});
    drifted.addConstraint({b, c, SE3(
        Eigen::Quaterniond::Identity(), Vec3(10.0, 0.0, 0.0)),
        1.0, vslam::AtlasConstraintType::TrackingBridge});
    drifted.addConstraint({a, c, SE3(
        Eigen::Quaterniond::Identity(), Vec3(20.0, 0.0, 0.0)),
        10.0, vslam::AtlasConstraintType::LoopClosure});
    const auto result = vslam::Optimizer::solvePoseGraph(
        vslam::buildSubmapGraph(drifted));
    assert(result.valid);
    double xa = 0.0, xb = 0.0, xc = 0.0;
    for (const auto& update : result.poses) {
        const double x = update.pose_cs.inverse().t.x();
        if (update.id == a) xa = x;
        if (update.id == b) xb = x;
        if (update.id == c) xc = x;
    }
    assert(std::abs(xa) < 1e-9);
    assert(std::abs(xb - 10.0) < 0.5 && "历史中间子图必须参与校正");
    assert(std::abs(xc - 20.0) < 0.5);
#endif
}

} // namespace

int main() {
    std::cout << "[Loop Prefix / Essential Graph]\n";
    test_global_pose_split();
    test_prefix_tail_rebase();
    test_essential_anchor_graph_is_bounded();
    test_anchor_correction_propagates_to_all_prefix_keyframes();
    test_submap_graph_has_single_gauge_and_movable_history();
    std::cout << "All loop graph tests passed\n";
    return 0;
}
