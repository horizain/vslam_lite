#include "vslam/optimizer.h"
#include "perf_monitor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef HAS_G2O
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/vertex_se3.h>      // Phase 2 位姿图：VertexSE3
#include <g2o/types/slam3d/edge_se3.h>        // Phase 2 位姿图：EdgeSE3

#include <unordered_map>
#include <unordered_set>
#endif

namespace vslam {

#ifdef HAS_G2O
// ---- 类型别名 ----
using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
// 位姿图专用：只有 6D 位姿顶点，无 3D 点（纯 pose 求解）
using PoseBlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
using PoseLinearSolverType = g2o::LinearSolverEigen<PoseBlockSolverType::PoseMatrixType>;

/// 旋转角（rad）：2*acos(|w|)，处理 q 与 -q 等价
static double rotationAngle(const SE3& pose) {
    return 2.0 * std::acos(std::clamp(std::abs(pose.q.w()), 0.0, 1.0));
}
#endif

OptimizationResult Optimizer::solveLocalBA(
    const Camera& camera,
    const OptimizationSnapshot& snap,
    int max_iterations,
    std::optional<bool> fix_points,
    size_t max_points) {

    OptimizationResult result;
    result.submap_id = snap.submap_id;
    result.base_topology_revision = snap.topology_revision;
    result.base_geometry_revision = snap.geometry_revision;

#ifndef HAS_G2O
    LOG_WARN("Local BA skipped: vslam was built without g2o");
    return result;  // valid=false
#else
    PERF_SCOPE("opt.ba");
    if (snap.keyframes.size() < 2) {
        LOG_INFO("Local BA: not enough keyframes (" << snap.keyframes.size() << ")");
        return result;
    }

    // ========================================================
    // 1. 构建优化器
    // ========================================================
    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<LinearSolverType>();
    // 防御：Cholesky 失败时避免每次迭代把整个 Hessian 写进 debug.txt
    linearSolver->setWriteDebug(false);
    auto blockSolver  = std::make_unique<BlockSolverType>(std::move(linearSolver));
    auto algorithm    = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer.setAlgorithm(algorithm);

    // ========================================================
    // 2. 添加相机参数
    // ========================================================
    auto* cam_params = new g2o::CameraParameters(
        camera->fx,
        g2o::Vector2(camera->cx, camera->cy),
        0.0);  // baseline = 0 (monocular projection edge)
    cam_params->setId(0);
    if (!optimizer.addParameter(cam_params)) {
        LOG_WARN("Local BA: camera parameter already exists");  // will use existing
    }

    // Motion-only vs 全 BA 由调用方显式指定，默认按传感器自动选择：
    // 单目三角化点的尺度由初始化（recoverPose 归一化 t）决定，若让点自由
    // 优化存在尺度 gauge 自由度（点+位姿平移同时缩放 s 重投影不变），
    // 实测会让刚插入的关键帧位姿被拉偏直至发散 → 单目必须固定点；
    // 双目/RGB-D 有绝对尺度（视差直接测深），点自由优化不会退化，
    // 还能让回环后位姿图移动的关键帧把点坐标真正收敛到重投影残差上。
    const bool points_fixed = fix_points.value_or(!camera->hasPerFrameDepth());
    const std::unordered_set<unsigned long> fixed_set(
        snap.fixed_kf_ids.begin(), snap.fixed_kf_ids.end());

    // ========================================================
    // 3. 添加位姿顶点（快照关键帧；固定集内 setFixed）
    // ========================================================
    std::unordered_map<unsigned long, size_t> kf_id_to_idx;
    for (size_t i = 0; i < snap.keyframes.size(); i++) {
        const auto& kf = snap.keyframes[i];
        // g2o 约定：VertexSE3Expmap + EdgeProjectXYZ2UV 的 estimate 为
        // T_cw（世界→相机，误差 = cam_map(T.map(P_w)) - obs）——直接喂 pose_cs。
        g2o::SE3Quat pose(kf.pose_cs.q.toRotationMatrix(), kf.pose_cs.t);
        auto* v_pose = new g2o::VertexSE3Expmap();
        v_pose->setId(static_cast<int>(i));
        v_pose->setEstimate(pose);
        // 锚定固定集（窗口最早 2 帧 = 基线长度 → 尺度锚定）
        if (fixed_set.count(kf.id)) v_pose->setFixed(true);
        optimizer.addVertex(v_pose);
        kf_id_to_idx[kf.id] = i;
    }

    // ========================================================
    // 4. 收集窗口内观测 + 添加 3D 点顶点
    // ========================================================
    struct Observation {
        size_t    kf_idx;
        size_t    kp_idx;
        Eigen::Vector2d pixel;
    };
    std::unordered_map<unsigned long, std::vector<Observation>> mp_observations;
    std::unordered_map<unsigned long, Vec3> mp_pos;
    for (const auto& lm : snap.landmarks) mp_pos.emplace(lm.id, lm.pos_s);

    for (size_t i = 0; i < snap.keyframes.size(); i++) {
        const auto& kf = snap.keyframes[i];
        for (size_t j = 0; j < kf.keypoints.size() && j < kf.map_points.size(); j++) {
            const unsigned long mp_id = kf.map_points[j];
            if (mp_id == 0) continue;
            mp_observations[mp_id].push_back({
                i, j,
                Eigen::Vector2d(kf.keypoints[j].pt.x, kf.keypoints[j].pt.y)
            });
        }
    }

    // 只保留被至少 3 帧观测到的地图点，且总量受限（BA 规模上限）
    std::unordered_set<unsigned long> selected_mps;
    if (mp_observations.size() > max_points) {
        std::vector<unsigned long> ranked(mp_observations.size());
        std::ranges::transform(mp_observations, ranked.begin(),
                               [](const auto& kv) { return kv.first; });
        std::ranges::partial_sort(ranked, ranked.begin() + (long)max_points,
                                  std::greater<unsigned long>{},
                                  [&](unsigned long id) {
                                      return mp_observations.find(id)->second.size();
                                  });
        selected_mps.insert(ranked.begin(), ranked.begin() + (long)max_points);
        LOG_WARN("BA point cap: " << mp_observations.size() << " -> "
                 << max_points);
    }

    int point_vertex_id = static_cast<int>(snap.keyframes.size());
    std::unordered_map<unsigned long, int> mp_id_to_vertex;
    for (auto& [mp_id, obs_list] : mp_observations) {
        if (obs_list.size() < 3) continue;  // 至少 3 帧观测，过滤弱观测坏点
        if (!selected_mps.empty() && !selected_mps.count(mp_id)) continue;
        auto it = mp_pos.find(mp_id);
        if (it == mp_pos.end()) continue;

        auto* v_point = new g2o::VertexPointXYZ();
        v_point->setId(point_vertex_id);
        v_point->setEstimate(it->second);
        v_point->setFixed(points_fixed);
        // 关键：点顶点必须标记 marginalized，BlockSolver 才会把它们当作
        // 3x3 路标块走 Schur 消除
        v_point->setMarginalized(true);
        optimizer.addVertex(v_point);
        mp_id_to_vertex[mp_id] = point_vertex_id;
        point_vertex_id++;
    }

    // ========================================================
    // 5. 添加重投影误差边（深度加权，近点高权重）
    // ========================================================
    const bool depth_weighted = camera->hasPerFrameDepth();
    int edge_count = 0;
    for (auto& [mp_id, obs_list] : mp_observations) {
        auto it = mp_id_to_vertex.find(mp_id);
        if (it == mp_id_to_vertex.end()) continue;
        const int point_vid = it->second;
        const auto pit = mp_pos.find(mp_id);

        for (auto& obs : obs_list) {
            auto* edge = new g2o::EdgeProjectXYZ2UV();
            edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>(
                optimizer.vertex(point_vid)));
            edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(
                optimizer.vertex(static_cast<int>(obs.kf_idx))));
            edge->setMeasurement(obs.pixel);

            double huber_delta = 5.991;  // 2 DOF, chi2 95% 阈值
            if (depth_weighted && pit != mp_pos.end()) {
                const double depth = std::max(
                    0.1, (snap.keyframes[obs.kf_idx].pose_cs * pit->second).z());
                const double w = std::clamp(100.0 / (depth * depth), 0.04, 25.0);
                edge->setInformation(w * Eigen::Matrix2d::Identity());
                huber_delta /= std::sqrt(w);
            } else {
                edge->setInformation(Eigen::Matrix2d::Identity());
            }
            edge->setParameterId(0, 0);
            auto* robust_kernel = new g2o::RobustKernelHuber();
            robust_kernel->setDelta(huber_delta);
            edge->setRobustKernel(robust_kernel);
            optimizer.addEdge(edge);
            edge_count++;
        }
    }

    if (edge_count < 10) {
        LOG_WARN("Local BA: too few edges (" << edge_count << "), skipping optimization");
        return result;
    }

    // ========================================================
    // 6. 执行优化
    // ========================================================
    optimizer.initializeOptimization();
    optimizer.optimize(max_iterations);
    optimizer.initializeOptimization();
    optimizer.optimize(max_iterations);

    // ========================================================
    // 7. 收集候选增量（不修改任何实时对象）
    // ========================================================
    result.metrics.vertices = snap.keyframes.size() + mp_id_to_vertex.size();
    result.metrics.edges = (size_t)edge_count;

    double max_correction = 0.0;
    result.poses.reserve(snap.keyframes.size());
    for (size_t i = 0; i < snap.keyframes.size(); i++) {
        auto* v_pose = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(static_cast<int>(i)));
        if (!v_pose) continue;
        const g2o::SE3Quat& opt_pose = v_pose->estimate();
        SE3 opt_cw(Eigen::Quaterniond(opt_pose.rotation()), opt_pose.translation());
        if (!opt_cw.t.allFinite() || !opt_cw.q.coeffs().allFinite()) {
            LOG_WARN("Local BA rejected non-finite pose kf#" << snap.keyframes[i].id);
            return OptimizationResult{};  // 整笔无效
        }
        max_correction = std::max(
            max_correction,
            (snap.keyframes[i].pose_cs.inverse() * opt_cw.inverse()).t.norm());
        result.poses.push_back({snap.keyframes[i].id, opt_cw});
    }
    result.metrics.max_correction = max_correction;

    result.points.reserve(mp_id_to_vertex.size());
    for (auto& [mp_id, point_vid] : mp_id_to_vertex) {
        auto* v_point = dynamic_cast<g2o::VertexPointXYZ*>(
            optimizer.vertex(point_vid));
        if (!v_point) continue;
        const Eigen::Vector3d& p = v_point->estimate();
        if (!p.allFinite()) {
            LOG_WARN("Local BA rejected non-finite point " << mp_id);
            return OptimizationResult{};  // 整笔无效
        }
        result.points.push_back({mp_id, p});
    }

    LOG_INFO("Local BA: optimized " << snap.keyframes.size() << " keyframes, "
             << result.points.size() << " points, "
             << edge_count << " edges");
    result.valid = true;
    return result;
#endif
}

OptimizationResult Optimizer::solvePoseGraph(const OptimizationSnapshot& snap) {
    OptimizationResult result;
    result.submap_id = snap.submap_id;
    result.base_topology_revision = snap.topology_revision;
    result.base_geometry_revision = snap.geometry_revision;

#ifndef HAS_G2O
    LOG_WARN("Pose graph optimization skipped: vslam was built without g2o");
    return result;
#else
    PERF_SCOPE("opt.pose_graph");
    if (snap.keyframes.size() < 2) return result;

    // 优化前完整 T_wc（快照），用于错误回环预检与事务式验收
    std::vector<SE3> old_twc;
    old_twc.reserve(snap.keyframes.size());
    for (const auto& kf : snap.keyframes) old_twc.push_back(kf.pose_cs.inverse());

    const Vec3 anchor = old_twc.front().t;
    double graph_span = 0.0;
    std::vector<double> old_steps;
    old_steps.reserve(old_twc.size() - 1);
    for (size_t i = 0; i < old_twc.size(); i++) {
        graph_span = std::max(graph_span, (old_twc[i].t - anchor).norm());
        if (i > 0) old_steps.push_back((old_twc[i].t - old_twc[i - 1].t).norm());
    }
    auto median = [](std::vector<double> values) {
        if (values.empty()) return 0.0;
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + (long)mid, values.end());
        return values[mid];
    };
    const double median_step = median(old_steps);
    const double loop_translation_limit = std::max(100.0, 2.0 * graph_span);
    const double correction_limit = std::max(100.0, 2.0 * graph_span);
    const double neighbor_step_limit = std::max(50.0, 10.0 * median_step);
    constexpr double kLoopRotationLimit = 150.0 * M_PI / 180.0;

    // 回环边残差预检（M0）：真实闭环可修正几十米漂移，但不能接受相对残差
    // 超过整张图尺度数倍或接近反向 180 度的边。
    {
        std::unordered_map<unsigned long, const KeyframeState*> kf_by_id;
        for (const auto& kf : snap.keyframes) kf_by_id.emplace(kf.id, &kf);
        for (const auto& c : snap.constraints) {
            if (!c.is_loop) continue;
            auto it_a = kf_by_id.find(c.a);
            auto it_b = kf_by_id.find(c.b);
            if (it_a == kf_by_id.end() || it_b == kf_by_id.end()) continue;
            const SE3 predicted = it_a->second->pose_cs * it_b->second->pose_cs.inverse();
            const SE3 residual = c.T_rel.inverse() * predicted;
            if (!std::isfinite(residual.t.norm())
                || residual.t.norm() > loop_translation_limit
                || rotationAngle(residual) > kLoopRotationLimit) {
                LOG_WARN("Pose graph rejected loop edge kf#" << c.a << " -> kf#" << c.b
                         << ": residual=" << residual.t.norm() << "m/"
                         << rotationAngle(residual) * 180.0 / M_PI << "deg"
                         << " limits=" << loop_translation_limit << "m/150deg");
                return result;  // valid=false
            }
        }
    }

    // ========================================================
    // 1. 构建优化器（位姿图专用 <6,6> solver：纯 6D 位姿顶点）
    // ========================================================
    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<PoseLinearSolverType>();
    auto blockSolver  = std::make_unique<PoseBlockSolverType>(std::move(linearSolver));
    auto algorithm    = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer.setAlgorithm(algorithm);

    // ========================================================
    // 2. 位姿顶点：VertexSE3，estimate 为 T_wc（g2o SE3 群运算标准语义）
    // ========================================================
    std::unordered_map<unsigned long, int> kf_vid;
    for (size_t i = 0; i < snap.keyframes.size(); i++) {
        const auto& kf = snap.keyframes[i];
        auto* v = new g2o::VertexSE3();
        v->setId(static_cast<int>(i));
        v->setEstimate(Eigen::Isometry3d(kf.pose_cs.inverse().matrix()));
        if (i == 0) v->setFixed(true);  // 最老关键帧锚定坐标系
        optimizer.addVertex(v);
        kf_vid[kf.id] = static_cast<int>(i);
    }

    auto add_edge = [&](int vid_a, int vid_b, const SE3& T_rel,
                        double weight, bool robust) {
        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, optimizer.vertex(vid_a));
        edge->setVertex(1, optimizer.vertex(vid_b));
        edge->setMeasurement(Eigen::Isometry3d(T_rel.matrix()));
        edge->setInformation(Eigen::MatrixXd::Identity(6, 6) * weight);
        if (robust) {
            auto* kernel = new g2o::RobustKernelHuber();
            kernel->setDelta(std::sqrt(12.592));  // 6 DoF chi2 95%
            edge->setRobustKernel(kernel);
        }
        optimizer.addEdge(edge);
    };

    int edge_count = 0;
    for (const auto& c : snap.constraints) {
        auto it_a = kf_vid.find(c.a);
        auto it_b = kf_vid.find(c.b);
        if (it_a == kf_vid.end() || it_b == kf_vid.end()) continue;
        add_edge(it_a->second, it_b->second, c.T_rel, c.weight, c.is_loop);
        edge_count++;
        if (c.is_loop) {
            LOG_INFO("Pose graph: loop edge kf#" << c.a << " -> kf#" << c.b
                     << " weight=" << c.weight);
        }
    }

    if (edge_count < 2) {
        LOG_WARN("Pose graph: too few edges (" << edge_count << "), skipping");
        return result;
    }

    // ========================================================
    // 3. 执行优化 + 质量验收（chi2 / 有限值 / 最大校正 / 相邻步长）
    // ========================================================
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    const double initial_chi2 = optimizer.activeRobustChi2();
    const int iterations = optimizer.optimize(100);
    optimizer.computeActiveErrors();
    const double final_chi2 = optimizer.activeRobustChi2();

    result.metrics.iterations = iterations;
    result.metrics.initial_chi2 = initial_chi2;
    result.metrics.final_chi2 = final_chi2;
    result.metrics.vertices = snap.keyframes.size();
    result.metrics.edges = (size_t)edge_count;

    if (iterations <= 0 || !std::isfinite(initial_chi2) || !std::isfinite(final_chi2)
        || final_chi2 > initial_chi2 * 1.01) {
        LOG_WARN("Pose graph rejected optimizer result: iterations=" << iterations
                 << " chi2=" << initial_chi2 << " -> " << final_chi2);
        return result;
    }
    result.metrics.converged = true;

    std::vector<SE3> optimized_twc;
    optimized_twc.reserve(snap.keyframes.size());
    double max_correction = 0.0;
    double max_neighbor_step = 0.0;
    for (size_t i = 0; i < snap.keyframes.size(); i++) {
        auto* v = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(static_cast<int>(i)));
        if (!v) return result;
        const Eigen::Isometry3d& value = v->estimate();
        SE3 Twc(Eigen::Quaterniond(value.linear()), value.translation());
        if (!Twc.t.allFinite() || !Twc.q.coeffs().allFinite()) {
            LOG_WARN("Pose graph rejected non-finite vertex #" << i);
            return result;
        }
        max_correction = std::max(
            max_correction, (old_twc[i].inverse() * Twc).t.norm());
        if (!optimized_twc.empty()) {
            max_neighbor_step = std::max(
                max_neighbor_step, (Twc.t - optimized_twc.back().t).norm());
        }
        optimized_twc.push_back(Twc);
    }
    result.metrics.max_correction = max_correction;
    result.metrics.max_neighbor_step = max_neighbor_step;

    if (max_correction > correction_limit || max_neighbor_step > neighbor_step_limit) {
        LOG_WARN("Pose graph rejected unsafe correction: max_delta=" << max_correction
                 << "m (limit " << correction_limit << "), neighbor_step="
                 << max_neighbor_step << "m (limit " << neighbor_step_limit << ")");
        return result;
    }

    result.poses.reserve(snap.keyframes.size());
    for (size_t i = 0; i < snap.keyframes.size(); i++) {
        result.poses.push_back({snap.keyframes[i].id, optimized_twc[i].inverse()});
    }

    LOG_INFO("Pose graph: optimized " << snap.keyframes.size() << " keyframes, "
             << edge_count << " edges, chi2=" << initial_chi2 << " -> " << final_chi2
             << ", max_delta=" << max_correction << "m");
    result.valid = true;
    return result;
#endif
}

} // namespace vslam
