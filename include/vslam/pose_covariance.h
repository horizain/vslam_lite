#pragma once

#include "vslam/common.h"
#include "vslam/localization_types.h"

#include <opencv2/core.hpp>

#include <limits>
#include <vector>

namespace vslam {

/// M3.2（§7.5）位姿数值协方差配置（产品规格首版常数，非数据集标定）
struct PoseCovarianceConfig {
    double translation_perturbation = 1e-4;  ///< m：[tx,ty,tz] 中心差分步长
    double rotation_perturbation = 1e-6;     ///< rad：[rx,ry,rz]
    double min_sigma2 = 0.25;                ///< px²：σ² 下限（≈0.5px 像素噪声）
    double eigen_floor = 1e-9;               ///< H 特征值截断下限
    double degenerate_condition = 1e8;       ///< cond(H) 上限，超过判退化
};

/// 协方差估计结果：ξ=[tx,ty,tz,rx,ry,rz] 左扰动 Exp(δξ)·T 的切空间协方差。
/// covariance_cs 表达在相机系（扰动施加于 T_cs 的左侧）；
/// valid=false 表示退化/数值不可用——调用方必须回退保守占位，不得发布假精度。
struct PoseCovarianceResult {
    bool valid = false;
    bool degenerate = false;
    Mat6 covariance_cs = Mat6::Zero();
    double sigma2 = 0.0;
    double condition = std::numeric_limits<double>::infinity();
};

/// SE(3) 指数映射。切空间向量顺序固定 [tx,ty,tz,rx,ry,rz]（§7.5），
/// 与"左扰动 T' = Exp(δξ)·T、扰动在相机系表达"配套使用。
[[nodiscard]] SE3 se3Exp(const Eigen::Matrix<double, 6, 1>& xi);

/// SE(3) 左伴随 Ad_T = [[R,[t]^R],[0,R]]，满足 T·Exp(δ) = Exp(Ad_T·δ)·T。
/// 用于把相机系切空间协方差变换到 odom 系（A = Ad_{T_oc}，精确覆盖
/// 全局校正后 T_wo≠I 的情形）。
[[nodiscard]] Mat6 se3Adjoint(const SE3& T);

/// §7.5：最终 PnP 内点的中心有限差分数值协方差。
/// pose_cs 为收敛位姿（子地图局部系 T_cs）；points/pixels 为全部 3D-2D 对应，
/// inlier_indices 选择参与估计的内点（越界索引忽略）。纯函数、无 RNG、
/// 不消耗全局随机序列——可在跟踪热路径内安全调用。
[[nodiscard]] PoseCovarianceResult pnpPoseCovariance(
    const SE3& pose_cs,
    const std::vector<cv::Point3f>& points_s,
    const std::vector<cv::Point2f>& pixels,
    const std::vector<int>& inlier_indices,
    const Mat33& K,
    const PoseCovarianceConfig& config = PoseCovarianceConfig());

} // namespace vslam
