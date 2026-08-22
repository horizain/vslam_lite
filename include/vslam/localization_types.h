#pragma once

#include "vslam/common.h"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <algorithm>

namespace vslam {

/// 定位运行模式（PRODUCTION_LOCALIZATION_PLAN §1.3）
enum class LocalizationMode {
    OdometryOnly,       ///< 仅连续局部里程计，不加载全局地图
    Mapping,            ///< 新建地图，仅用于建图任务
    LocalizationOnly,   ///< 地图只读；机器人日常运行默认模式
    MapMaintenance      ///< 对 staging 地图做受控更新，不直接覆盖生产地图
};

/// 跟踪状态机状态（§4.2）
enum class TrackingState {
    Initializing,       ///< 初始化中
    Tracking,           ///< 正常跟踪
    Degraded,           ///< 弱质量跟踪
    Relocalizing,       ///< 重定位中
    Lost,               ///< 丢失
    Stopped             ///< 已停止
};

/// 位姿失败原因码（§4.1），None 表示成功；每次拒绝都有唯一结构化原因
enum class FailureReason {
    None,                       ///< 无失败
    InvalidInput,               ///< 输入不合法（空图/尺寸类型不符等）
    TimestampRollback,          ///< 时间戳倒退或相等
    StereoUnsynchronized,       ///< 双目时间不同步
    InsufficientFeatures,       ///< 特征不足
    GeometricRejection,         ///< 几何估计被拒
    MotionDiscontinuity,        ///< 运动连续性被破坏
    RelocalizationTimeout,      ///< 重定位超时
    BackendOverloaded,          ///< 后端过载
    MapIncompatible,            ///< 地图不兼容
    ImageDegraded,              ///< M3.1：输入图像质量硬拒绝（模糊/暗/亮超限）
    InternalError               ///< 内部错误
};

/// 位姿估计输出（§4.1）
struct PoseEstimate {
    uint64_t sequence = 0;                    ///< 输入帧序号
    double timestamp = 0.0;                   ///< 帧时间戳（秒）
    SE3 T_ob;                                 ///< 机器人基座 b -> 连续 odom 系 o（不因回环跳变）
    SE3 T_wo;                                 ///< 连续 odom 系 o -> 全局地图 w（仅全局校正更新）
    SE3 T_wb;                                 ///< 机器人基座 b -> 全局地图 w（允许全局修正）
    Mat6 covariance = Mat6::Zero();           ///< T_ob 左扰动在 odom 系中的 6x6 协方差
    TrackingState state = TrackingState::Initializing;
    FailureReason reason = FailureReason::None;
    bool pose_valid = false;                  ///< 位姿是否有效（可被控制器消费）
    bool prediction_only = false;             ///< 是否仅为外推预测
    uint64_t map_generation = 0;              ///< 当前地图 generation（回环提交后递增）
    uint64_t global_correction_generation = 0;///< T_wo 原子发布代次
};

/// 回环/重定位发布全局校正时的结构化事件。控制器不消费该事件来改写 T_ob；
/// 规划/地图层用它识别 T_wb 的合法全局修正。
struct GlobalCorrectionEvent {
    uint64_t generation = 0;
    SE3 old_T_wo;
    SE3 new_T_wo;
};

// ---------- 输出质量契约：§3 硬不变量（静态部分） ----------

/// 四元数是否为单位模长（容差 1e-6，§4.3）
[[nodiscard]] inline bool isUnitQuaternion(const Eigen::Quaterniond& q) {
    return std::abs(q.coeffs().norm() - 1.0) < 1e-6;
}

/// SE3 平移/旋转分量是否全部有限
[[nodiscard]] inline bool isFinite(const SE3& T) {
    return T.t.allFinite() && T.q.coeffs().allFinite();
}

/// 时间戳是否有限（倒退/相等由状态机拒绝，不在静态谓词内）
[[nodiscard]] inline bool isValidTimestamp(double timestamp) {
    return std::isfinite(timestamp);
}

/// 6x6 矩阵是否有限、对称且正定（§3-1：非正定协方差不得发布）
[[nodiscard]] inline bool isPositiveDefiniteCovariance(const Mat6& cov) {
    if (!cov.allFinite()) return false;
    const double tol = 1e-9 * std::max(1.0, cov.norm());
    if ((cov - cov.transpose()).norm() > tol) return false;
    Eigen::SelfAdjointEigenSolver<Mat6> solver(cov);
    if (solver.info() != Eigen::Success) return false;
    return solver.eigenvalues().minCoeff() > 0.0;
}

/// 位姿发布前置静态校验：有限位姿、单位四元数、有效时间戳；
/// pose_valid=true 时还要求协方差正定（prediction_only 不影响静态校验）。
[[nodiscard]] inline bool isPublishable(const PoseEstimate& p) {
    if (!isValidTimestamp(p.timestamp)) return false;
    if (!isFinite(p.T_ob) || !isFinite(p.T_wo) || !isFinite(p.T_wb)) return false;
    if (!isUnitQuaternion(p.T_ob.q) || !isUnitQuaternion(p.T_wo.q) ||
        !isUnitQuaternion(p.T_wb.q)) return false;
    if (p.pose_valid && !isPositiveDefiniteCovariance(p.covariance)) return false;
    return true;
}

} // namespace vslam
