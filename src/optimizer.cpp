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
#endif

void Optimizer::localBundleAdjustment(
    const Camera& camera,
    Map::Ptr map,
    const std::vector<Frame::Ptr>& active_kfs,
    int max_iterations,
    std::optional<bool> fix_points,
    size_t max_points) {

#ifndef HAS_G2O
    (void)camera;
    (void)map;
    (void)active_kfs;
    (void)max_iterations;
    LOG_WARN("Local BA skipped: vslam was built without g2o");
    return;
#else
    PERF_SCOPE("opt.ba");
    if (active_kfs.size() < 2) {
        LOG_INFO("Local BA: not enough keyframes (" << active_kfs.size() << ")");
        return;
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

    // ========================================================
    // 3. 添加位姿顶点（每个关键帧一个）
    // ========================================================
    std::unordered_map<unsigned long, size_t> kf_id_to_idx;

    for (size_t i = 0; i < active_kfs.size(); i++) {
        auto& kf = active_kfs[i];

        // 注意：本机 apt 版 g2o 的 VertexSE3Expmap + EdgeProjectXYZ2UV 约定
        // estimate 为 T_cw（世界→相机，误差 = cam_map(T.map(P_w)) - obs），
        // 与官方新版（T_wc）不同——实测喂 T_wc 会被反向优化导致位姿跑飞。
        // 因此直接喂 pose_cw（T_cw 语义）。
        g2o::SE3Quat pose(kf->pose_cw.q.toRotationMatrix(), kf->pose_cw.t);

        auto* v_pose = new g2o::VertexSE3Expmap();
        v_pose->setId(static_cast<int>(i));
        v_pose->setEstimate(pose);
        // 固定窗口内最早的 2 帧：仅固定第 1 帧只能锚定绝对位置，
        // 无法约束单目 BA 的尺度自由度（所有点+位姿平移同时缩放 s 时
        // 重投影不变）。固定两帧 = 固定基线长度 → 尺度锚定。
        if (i == 0 || i == 1) {
            v_pose->setFixed(true);
        }
        optimizer.addVertex(v_pose);
        kf_id_to_idx[kf->id] = i;
    }

    // ========================================================
    // 4. 收集活跃窗口内的地图点 + 添加 3D 点顶点
    // ========================================================
    struct Observation {
        size_t    kf_idx;
        size_t    kp_idx;   // index in keyframe->keypoints
        Eigen::Vector2d pixel;
    };

    std::unordered_map<unsigned long, std::vector<Observation>> mp_observations;

    for (size_t i = 0; i < active_kfs.size(); i++) {
        auto& kf = active_kfs[i];
        for (size_t j = 0; j < kf->keypoints.size(); j++) {
            auto& mp = kf->map_points[j];
            if (mp == nullptr) continue;
            // 检查这个地图点是否已被添加到优化器
            if (mp_observations.find(mp->id) == mp_observations.end()) {
                mp_observations[mp->id] = {};
            }
            mp_observations[mp->id].push_back({
                i, j,
                Eigen::Vector2d(kf->keypoints[j].pt.x, kf->keypoints[j].pt.y)
            });
        }
    }

    // 只保留被至少 3 帧观测到的地图点，且总量受限（BA 规模上限）：
    // 后段地图膨胀后（KF 上千、点数十万），窗口内点顶点上万会让每次 LM
    // 迭代的 Schur 消除呈超线性增长，单次 BA 实测可爆到 100 秒级。
    // 按观测数降序保留前 max_points（观测多 = 约束强，信息量最大），
    // 前段地图小时不触发截断，精度无损。
    std::unordered_set<unsigned long> selected_mps;
    if (mp_observations.size() > max_points) {
        // 部分排序取 top-k（C++23 ranges：按观测数降序）
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

    int point_vertex_id = static_cast<int>(active_kfs.size());  // 点 ID 从 KF 数量之后开始
    std::unordered_map<unsigned long, int> mp_id_to_vertex;

    for (auto& [mp_id, obs_list] : mp_observations) {
        if (obs_list.size() < 3) continue;  // 至少 3 帧观测，过滤弱观测坏点
        if (!selected_mps.empty() && !selected_mps.count(mp_id)) continue;

        auto mp = map->getMapPoint(mp_id);
        if (!mp) continue;

        auto* v_point = new g2o::VertexPointXYZ();
        v_point->setId(point_vertex_id);
        v_point->setEstimate(mp->pos_w);
        // 单目 motion-only（points_fixed=true）：地图点固定，只优化位姿
        v_point->setFixed(points_fixed);
        // 关键：点顶点必须标记 marginalized，BlockSolver 才会把它们当作
        // 3x3 路标块走 Schur 消除；否则被当作 6x6 位姿块，Eigen 固定尺寸
        // 断言/静默损坏 Hessian → Cholesky 失败 → 每次迭代写 debug.txt。
        v_point->setMarginalized(true);
        optimizer.addVertex(v_point);
        mp_id_to_vertex[mp_id] = point_vertex_id;
        point_vertex_id++;
    }

    // ========================================================
    // 5. 添加重投影误差边
    // ========================================================
    // 双目/RGB-D：视差测深误差 σ_z ∝ z²（z = f·b/d），近点观测精度高。
    // 信息矩阵按 1/σ_z² 加权（近点高权重），抑制远点错误点对位姿的污染；
    // Huber 阈值随权重等比缩放（delta' = delta/√w），保持像素级鲁棒语义不变。
    const bool depth_weighted = camera->hasPerFrameDepth();
    int edge_count = 0;
    for (auto& [mp_id, obs_list] : mp_observations) {
        auto it = mp_id_to_vertex.find(mp_id);
        if (it == mp_id_to_vertex.end()) continue;

        int point_vid = it->second;
        auto mp = map->getMapPoint(mp_id);

        for (auto& obs : obs_list) {
            auto* edge = new g2o::EdgeProjectXYZ2UV();

            // 边连接：点顶点(0) + 位姿顶点(1)
            edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>(
                optimizer.vertex(point_vid)));
            edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(
                optimizer.vertex(static_cast<int>(obs.kf_idx))));

            // 观测值：像素坐标
            edge->setMeasurement(obs.pixel);

            double huber_delta = 5.991;  // 2 DOF, chi2 95% 阈值
            if (depth_weighted && mp) {
                // 该观测在所属关键帧相机系下的深度（优化前估计，仅用于定权）
                const double depth = std::max(
                    0.1, (active_kfs[obs.kf_idx]->pose_cw * mp->pos_w).z());
                // 参考深度 10m 处权重为 1，近点放大、远点抑制
                const double w = std::clamp(100.0 / (depth * depth), 0.04, 25.0);
                edge->setInformation(w * Eigen::Matrix2d::Identity());
                huber_delta /= std::sqrt(w);
            } else {
                edge->setInformation(Eigen::Matrix2d::Identity());
            }

            // 关联相机参数
            edge->setParameterId(0, 0);

            // Huber 鲁棒核函数（抑制外点）
            auto* robust_kernel = new g2o::RobustKernelHuber();
            robust_kernel->setDelta(huber_delta);
            edge->setRobustKernel(robust_kernel);

            optimizer.addEdge(edge);
            edge_count++;
        }
    }

    if (edge_count < 10) {
        LOG_WARN("Local BA: too few edges (" << edge_count << "), skipping optimization");
        return;
    }

    // ========================================================
    // 6. 执行优化
    // ========================================================
    optimizer.initializeOptimization();
    optimizer.optimize(max_iterations);
    optimizer.initializeOptimization();
    optimizer.optimize(max_iterations);

    // ========================================================
    // 7. 回写优化结果
    // ========================================================
    for (size_t i = 0; i < active_kfs.size(); i++) {
        auto* v_pose = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(static_cast<int>(i)));
        if (!v_pose) continue;

        const g2o::SE3Quat& opt_pose = v_pose->estimate();
        // g2o 输出即 T_cw（与本项目 pose_cw 语义一致），直接回写
        active_kfs[i]->pose_cw = SE3(
            Eigen::Quaterniond(opt_pose.rotation()), opt_pose.translation());
    }

    for (auto& [mp_id, point_vid] : mp_id_to_vertex) {
        auto* v_point = dynamic_cast<g2o::VertexPointXYZ*>(
            optimizer.vertex(point_vid));
        if (!v_point) continue;

        auto mp = map->getMapPoint(mp_id);
        if (mp) {
            mp->pos_w = v_point->estimate();
        }
    }

    LOG_INFO("Local BA: optimized " << active_kfs.size() << " keyframes, "
             << mp_id_to_vertex.size() << " points, "
             << edge_count << " edges");
#endif
}

void Optimizer::globalBundleAdjustment(const Camera& camera, Map::Ptr map,
                                       int max_iterations,
                                       std::optional<bool> fix_points,
                                       size_t max_points) {
    // 结构同 localBA，但使用所有关键帧和地图点。
    // 全图 KF 数千时 Schur 后位姿系统上万维，单次可达分钟级；按间隔采样
    // 位姿顶点（首尾必含），未采样 KF 保持位姿图解——全局 BA 只是回环
    // 后的精修，采样不丢尺度约束（点仍按全图观测收集）。
    auto all_kfs = map->getAllKeyFrames();
    std::vector<Frame::Ptr> sampled;
    sampled.reserve(all_kfs.size() / 3 + 2);
    for (size_t i = 0; i < all_kfs.size(); i++) {
        if (i % 3 == 0) sampled.push_back(all_kfs[i]);
    }
    if (sampled.empty() || sampled.back() != all_kfs.back())
        sampled.push_back(all_kfs.back());
    localBundleAdjustment(camera, map, sampled, max_iterations, fix_points,
                          max_points);
    LOG_INFO("Global BA complete (" << all_kfs.size() << " -> "
             << sampled.size() << " keyframes)");
}

bool Optimizer::poseGraphOptimization(Map::Ptr map,
                                      const std::vector<LoopEdge>& odometry_edges,
                                      const std::vector<LoopEdge>& loop_edges) {
#ifndef HAS_G2O
    (void)map;
    (void)odometry_edges;
    (void)loop_edges;
    LOG_WARN("Pose graph optimization skipped: vslam was built without g2o");
    return false;
#else
    PERF_SCOPE("opt.pose_graph");
    auto kfs = map->getAllKeyFrames();
    if (kfs.size() < 2) return false;

    // 优化前保留完整 T_wc，用于错误回环预检和优化后事务式验收。PGO 在快照
    // Map 上运行，验收失败时只需不回写即可保证真实地图完全不受影响。
    std::vector<SE3> old_twc;
    old_twc.reserve(kfs.size());
    for (const auto& kf : kfs) old_twc.push_back(kf->pose_cw.inverse());

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
    auto rotation_angle = [](const SE3& pose) {
        return 2.0 * std::acos(std::clamp(std::abs(pose.q.w()), 0.0, 1.0));
    };

    // 回环测量必须至少与当前图预测处于同一数量级。真实闭环可以修正几十米
    // 漂移，但不能接受相对残差超过整张图尺度数倍或接近反向 180 度的边。
    for (const auto& le : loop_edges) {
        auto kf_a = map->getKeyFrame(le.a);
        auto kf_b = map->getKeyFrame(le.b);
        if (!kf_a || !kf_b) continue;
        const SE3 predicted = kf_a->pose_cw * kf_b->pose_cw.inverse();
        const SE3 residual = le.T_rel.inverse() * predicted;
        if (!std::isfinite(residual.t.norm())
            || residual.t.norm() > loop_translation_limit
            || rotation_angle(residual) > kLoopRotationLimit) {
            LOG_WARN("Pose graph rejected loop edge kf#" << le.a << " -> kf#" << le.b
                     << ": residual=" << residual.t.norm() << "m/"
                     << rotation_angle(residual) * 180.0 / M_PI << "deg"
                     << " limits=" << loop_translation_limit << "m/150deg");
            return false;
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
    // 2. 位姿顶点：VertexSE3，estimate 为 T_wc（g2o SE3 群运算标准语义）。
    //    与 localBA 的 EdgeProjectXYZ2UV 特殊 T_cw 语义无关，此处必须喂 T_wc。
    // ========================================================
    std::unordered_map<unsigned long, int> kf_vid;
    for (size_t i = 0; i < kfs.size(); i++) {
        auto& kf = kfs[i];
        auto* v = new g2o::VertexSE3();
        v->setId(static_cast<int>(i));
        v->setEstimate(Eigen::Isometry3d(kf->pose_cw.inverse().matrix()));
        if (i == 0) v->setFixed(true);  // 第一个关键帧锚定坐标系
        optimizer.addVertex(v);
        kf_vid[kf->id] = static_cast<int>(i);
    }

    // 添加一条相对位姿边：测量 T_rel 满足 X_b = X_a · T_rel
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

    // ========================================================
    // 3. 里程计边：使用关键帧插入时冻结的测量。不能从上一次优化后的
    //    位姿重算，否则会把优化结果误当成新的观测并逐次锁死。
    // ========================================================
    int edge_count = 0;
    for (const auto& edge : odometry_edges) {
        auto it_a = kf_vid.find(edge.a);
        auto it_b = kf_vid.find(edge.b);
        if (it_a == kf_vid.end() || it_b == kf_vid.end()) continue;
        add_edge(it_a->second, it_b->second, edge.T_rel, edge.weight, false);
        edge_count++;
    }

    // ========================================================
    // 4. 回环边：回环帧 a ↔ 当前帧 b（Sim3 已由传播吸收，边用 SE3 丢尺度）
    // ========================================================
    for (const auto& le : loop_edges) {
        auto it_a = kf_vid.find(le.a);
        auto it_b = kf_vid.find(le.b);
        if (it_a == kf_vid.end() || it_b == kf_vid.end()) continue;
        add_edge(it_a->second, it_b->second, le.T_rel, le.weight, true);
        edge_count++;
        LOG_INFO("Pose graph: loop edge kf#" << le.a << " -> kf#" << le.b
                 << " weight=" << le.weight);
    }

    if (edge_count < 2) {
        LOG_WARN("Pose graph: too few edges (" << edge_count << "), skipping");
        return false;
    }

    // ========================================================
    // 5. 执行优化 + 回写（estimate 是 T_wc → 取逆写回 pose_cw）
    // ========================================================
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    const double initial_chi2 = optimizer.activeRobustChi2();
    const int iterations = optimizer.optimize(100);
    optimizer.computeActiveErrors();
    const double final_chi2 = optimizer.activeRobustChi2();

    if (iterations <= 0 || !std::isfinite(initial_chi2) || !std::isfinite(final_chi2)
        || final_chi2 > initial_chi2 * 1.01) {
        LOG_WARN("Pose graph rejected optimizer result: iterations=" << iterations
                 << " chi2=" << initial_chi2 << " -> " << final_chi2);
        return false;
    }

    double max_correction = 0.0;
    double max_neighbor_step = 0.0;
    std::vector<SE3> optimized_twc;
    optimized_twc.reserve(kfs.size());
    for (size_t i = 0; i < kfs.size(); i++) {
        auto* v = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(static_cast<int>(i)));
        if (!v) return false;
        const Eigen::Isometry3d& value = v->estimate();
        SE3 Twc(Eigen::Quaterniond(value.linear()), value.translation());
        if (!Twc.t.allFinite() || !Twc.q.coeffs().allFinite()) {
            LOG_WARN("Pose graph rejected non-finite vertex #" << i);
            return false;
        }
        max_correction = std::max(
            max_correction, (old_twc[i].inverse() * Twc).t.norm());
        if (!optimized_twc.empty()) {
            max_neighbor_step = std::max(
                max_neighbor_step, (Twc.t - optimized_twc.back().t).norm());
        }
        optimized_twc.push_back(Twc);
    }

    if (max_correction > correction_limit || max_neighbor_step > neighbor_step_limit) {
        LOG_WARN("Pose graph rejected unsafe correction: max_delta=" << max_correction
                 << "m (limit " << correction_limit << "), neighbor_step="
                 << max_neighbor_step << "m (limit " << neighbor_step_limit << ")");
        return false;
    }

    for (size_t i = 0; i < kfs.size(); i++) {
        kfs[i]->pose_cw = optimized_twc[i].inverse();
    }

    LOG_INFO("Pose graph: optimized " << kfs.size() << " keyframes, "
             << edge_count << " edges, chi2=" << initial_chi2 << " -> " << final_chi2
             << ", max_delta=" << max_correction << "m");
    return true;
#endif
}

} // namespace vslam
