#include "vslam/optimizer.h"

// g2o
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

#include <set>
#include <unordered_map>

namespace vslam {

// ---- 类型别名 ----
using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

void Optimizer::localBundleAdjustment(
    const Camera& camera,
    Map::Ptr map,
    const std::vector<Frame::Ptr>& active_kfs,
    int max_iterations) {

    if (active_kfs.size() < 2) {
        LOG_INFO("Local BA: not enough keyframes (" << active_kfs.size() << ")");
        return;
    }

    // ========================================================
    // 1. 构建优化器
    // ========================================================
    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<LinearSolverType>();
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

    // 只保留被至少 2 帧观测到的地图点
    int point_vertex_id = static_cast<int>(active_kfs.size());  // 点 ID 从 KF 数量之后开始
    std::unordered_map<unsigned long, int> mp_id_to_vertex;

    for (auto& [mp_id, obs_list] : mp_observations) {
        if (obs_list.size() < 3) continue;  // 至少 3 帧观测，过滤弱观测坏点

        auto mp = map->getMapPoint(mp_id);
        if (!mp) continue;

        auto* v_point = new g2o::VertexPointXYZ();
        v_point->setId(point_vertex_id);
        v_point->setEstimate(mp->pos_w);
        // Motion-only BA：地图点固定，只优化位姿。
        // 单目三角化点的尺度由初始化（recoverPose 归一化 t）决定，
        // 若让点自由优化，存在尺度 gauge 自由度（点+位姿平移同时缩放 s
        // 重投影不变），实测会让刚插入的关键帧位姿被拉偏直至发散。
        v_point->setFixed(true);
        optimizer.addVertex(v_point);
        mp_id_to_vertex[mp_id] = point_vertex_id;
        point_vertex_id++;
    }

    // ========================================================
    // 5. 添加重投影误差边
    // ========================================================
    int edge_count = 0;
    for (auto& [mp_id, obs_list] : mp_observations) {
        auto it = mp_id_to_vertex.find(mp_id);
        if (it == mp_id_to_vertex.end()) continue;

        int point_vid = it->second;

        for (auto& obs : obs_list) {
            auto* edge = new g2o::EdgeProjectXYZ2UV();

            // 边连接：点顶点(0) + 位姿顶点(1)
            edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>(
                optimizer.vertex(point_vid)));
            edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(
                optimizer.vertex(static_cast<int>(obs.kf_idx))));

            // 观测值：像素坐标
            edge->setMeasurement(obs.pixel);
            edge->setInformation(Eigen::Matrix2d::Identity());

            // 关联相机参数
            edge->setParameterId(0, 0);

            // Huber 鲁棒核函数（抑制外点）
            auto* robust_kernel = new g2o::RobustKernelHuber();
            robust_kernel->setDelta(5.991);  // 2 DOF, chi2 95% 阈值
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
}

void Optimizer::globalBundleAdjustment(const Camera& camera, Map::Ptr map) {
    // 结构同 localBA，但使用所有关键帧和地图点
    auto all_kfs = map->getAllKeyFrames();
    localBundleAdjustment(camera, map, all_kfs);
    LOG_INFO("Global BA complete");
}

void Optimizer::poseGraphOptimization(Map::Ptr map) {
    // TODO Phase 2: 只优化关键帧位姿 + 帧间相对位姿约束 + 回环约束
    LOG_INFO("Pose Graph Optimization - TODO Phase 2");
}

} // namespace vslam
