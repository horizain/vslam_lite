#include "vslam/pose_covariance.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace vslam {
namespace {

/// 反对称矩阵 [w]×
[[nodiscard]] Mat33 skew(const Vec3& w) {
    Mat33 m;
    m << 0.0, -w.z(), w.y(),
         w.z(), 0.0, -w.x(),
         -w.y(), w.x(), 0.0;
    return m;
}

/// 重投影残差 r(T) ∈ R^{2N}（投影 − 观测像素）。任一点扰动后落到相机
/// 后方（z≤ε）返回 false——该扰动方向数值无效，调用方判退化。
bool residuals(const SE3& T,
               const std::vector<cv::Point3f>& pts,
               const std::vector<cv::Point2f>& uv,
               const Mat33& K,
               Eigen::Matrix<double, Eigen::Dynamic, 1>& r) {
    const int n = static_cast<int>(pts.size());
    r.resize(2 * n);
    for (int i = 0; i < n; ++i) {
        const Vec3 p_c = T * Vec3(pts[i].x, pts[i].y, pts[i].z);
        if (!(p_c.z() > 1e-6)) return false;
        const double inv_z = 1.0 / p_c.z();
        r(2 * i) = K(0, 0) * p_c.x() * inv_z + K(0, 2) - uv[i].x;
        r(2 * i + 1) = K(1, 1) * p_c.y() * inv_z + K(1, 2) - uv[i].y;
    }
    return true;
}

} // namespace

SE3 se3Exp(const Eigen::Matrix<double, 6, 1>& xi) {
    const Vec3 rho = xi.head<3>();
    const Vec3 omega = xi.tail<3>();
    const double theta = omega.norm();
    const Mat33 omega_hat = skew(omega);
    Mat33 R;
    Mat33 V;
    if (theta < 1e-9) {
        // 小角一阶/二阶级数，避免除零且保持数值稳定
        R = Mat33::Identity() + omega_hat + 0.5 * omega_hat * omega_hat;
        V = Mat33::Identity() + 0.5 * omega_hat + (1.0 / 6.0) * omega_hat * omega_hat;
    } else {
        const double theta2 = theta * theta;
        const double a = std::sin(theta) / theta;                 // sin θ / θ
        const double b = (1.0 - std::cos(theta)) / theta2;        // (1−cos θ)/θ²
        const double c = (theta - std::sin(theta)) / (theta2 * theta); // (θ−sinθ)/θ³
        R = Mat33::Identity() + a * omega_hat + b * (omega_hat * omega_hat);
        V = Mat33::Identity() + b * omega_hat + c * (omega_hat * omega_hat);
    }
    return SE3(Eigen::Quaterniond(R), V * rho);
}

Mat6 se3Adjoint(const SE3& T) {
    const Mat33 R = T.q.toRotationMatrix();
    const Mat33 t_hat = skew(T.t);
    Mat6 Ad = Mat6::Zero();
    Ad.block<3, 3>(0, 0) = R;
    Ad.block<3, 3>(0, 3) = t_hat * R;
    Ad.block<3, 3>(3, 3) = R;
    return Ad;
}

PoseCovarianceResult pnpPoseCovariance(const SE3& pose_cs,
                                       const std::vector<cv::Point3f>& points_s,
                                       const std::vector<cv::Point2f>& pixels,
                                       const std::vector<int>& inlier_indices,
                                       const Mat33& K,
                                       const PoseCovarianceConfig& config) {
    PoseCovarianceResult out;

    // 内点收集：越界索引忽略；少于 6 个时 Jacobian 秩亏（2N<12），直接退化。
    std::vector<cv::Point3f> pts;
    std::vector<cv::Point2f> uv;
    pts.reserve(inlier_indices.size());
    uv.reserve(inlier_indices.size());
    for (const int idx : inlier_indices) {
        if (idx < 0 || idx >= static_cast<int>(points_s.size()) ||
            idx >= static_cast<int>(pixels.size()))
            continue;
        pts.push_back(points_s[idx]);
        uv.push_back(pixels[idx]);
    }
    constexpr int kMinInliersForJacobian = 6;
    if (static_cast<int>(pts.size()) < kMinInliersForJacobian) {
        out.degenerate = true;
        return out;
    }

    // 基准残差与 SSE（最终内点在收敛位姿处的 2D 重投影残差）
    Eigen::Matrix<double, Eigen::Dynamic, 1> r0;
    if (!residuals(pose_cs, pts, uv, K, r0)) {
        out.degenerate = true;
        return out;
    }
    const double sse = r0.squaredNorm();

    // §7.5：σ² = max(0.25, SSE / max(1, 2N−6))
    const int dof = std::max(1, 2 * static_cast<int>(pts.size()) - 6);
    out.sigma2 = std::max(config.min_sigma2, sse / static_cast<double>(dof));

    // 中心有限差分 Jacobian：J 列 k = ∂r/∂ξ_k，
    // 扰动顺序固定 [tx,ty,tz,rx,ry,rz]，左扰动 Exp(δξ)·T_cs。
    const int n = static_cast<int>(pts.size());
    Eigen::MatrixXd J(2 * n, 6);
    for (int k = 0; k < 6; ++k) {
        const double h = (k < 3) ? config.translation_perturbation
                                 : config.rotation_perturbation;
        Eigen::Matrix<double, 6, 1> xi_plus = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> xi_minus = Eigen::Matrix<double, 6, 1>::Zero();
        xi_plus(k) = h;
        xi_minus(k) = -h;
        Eigen::Matrix<double, Eigen::Dynamic, 1> rp;
        Eigen::Matrix<double, Eigen::Dynamic, 1> rm;
        if (!residuals(se3Exp(xi_plus) * pose_cs, pts, uv, K, rp) ||
            !residuals(se3Exp(xi_minus) * pose_cs, pts, uv, K, rm)) {
            out.degenerate = true;
            return out;
        }
        J.col(k) = (rp - rm) / (2.0 * h);
    }

    const Mat6 H = (J.transpose() * J).eval();

    // 特征分解：原始特征值上做条件数判定（§7.5 cond(H)>1e8 直接判退化），
    // 再按 eigen_floor 截断求逆。
    Eigen::SelfAdjointEigenSolver<Mat6> solver(H);
    if (solver.info() != Eigen::Success) {
        out.degenerate = true;
        return out;
    }
    const Vec6 eig = solver.eigenvalues();
    const double lambda_min = eig.minCoeff();
    const double lambda_max = eig.maxCoeff();
    if (!(lambda_min > 0.0)) {
        out.degenerate = true;
        return out;
    }
    out.condition = lambda_max / lambda_min;
    if (out.condition > config.degenerate_condition || !std::isfinite(out.condition)) {
        out.degenerate = true;
        return out;
    }
    const Vec6 eig_clamped = eig.cwiseMax(config.eigen_floor);
    const Mat6 H_inv = solver.eigenvectors() *
                       eig_clamped.cwiseInverse().asDiagonal() *
                       solver.eigenvectors().transpose();

    // Σ_c = σ²·H⁻¹；对称化后做有限/正定校验（§7.5 第 6 步）
    Mat6 cov = (out.sigma2 * H_inv).eval();
    cov = 0.5 * (cov + cov.transpose());
    if (!cov.allFinite() || !isPositiveDefiniteCovariance(cov)) {
        out.degenerate = true;
        out.covariance_cs = cov;
        return out;
    }
    out.covariance_cs = cov;
    out.valid = true;
    return out;
}

} // namespace vslam
