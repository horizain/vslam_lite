#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <opencv2/core.hpp>

namespace vslam {

// ---------- 基础类型别名 ----------
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat33 = Eigen::Matrix3d;
using Mat44 = Eigen::Matrix4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using SubmapId = unsigned long;

// ---------- 位姿表示（最小 SE3） ----------
struct SE3 {
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Vec3                t = Vec3::Zero();

    SE3() = default;
    SE3(const Eigen::Quaterniond& q_, const Vec3& t_) : q(q_), t(t_) {}

    /// 从 4x4 变换矩阵构造
    [[nodiscard]] static SE3 fromMatrix(const Mat44& T) {
        Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        return SE3(Eigen::Quaterniond(R), T.block<3, 1>(0, 3));
    }

    /// 转为 4x4 变换矩阵
    [[nodiscard]] Mat44 matrix() const {
        Mat44 T = Mat44::Identity();
        T.block<3, 3>(0, 0) = q.toRotationMatrix();
        T.block<3, 1>(0, 3) = t;
        return T;
    }

    /// 逆变换
    [[nodiscard]] SE3 inverse() const {
        Eigen::Quaterniond q_inv = q.inverse();
        return SE3(q_inv, -(q_inv * t));
    }

    /// 相机光心在世界系中的位置（仅当本变换语义为 T_cw 时使用）
    [[nodiscard]] Vec3 camera_position() const {
        return -(q.inverse() * t);
    }

    /// 组合变换：this * other
    [[nodiscard]] SE3 operator*(const SE3& other) const {
        return SE3(q * other.q, q * other.t + t);
    }

    /// 变换一个 3D 点
    [[nodiscard]] Vec3 operator*(const Vec3& p) const {
        return q * p + t;
    }
};

// ---------- 相似变换（Sim3，7 自由度，单目回环校正用） ----------
/// p' = s · (R·p) + t。s 表示两个子地图之间的尺度比，
/// 单目 VO 回环闭合时由 Umeyama 闭式求解直接给出。
struct Sim3 {
    double s = 1.0;  // 尺度（两子地图尺度比）
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Vec3 t = Vec3::Zero();

    Sim3() = default;
    Sim3(double s_, const Eigen::Quaterniond& q_, const Vec3& t_)
        : s(s_), q(q_), t(t_) {}

    /// 变换一个 3D 点：p' = s·(q·p) + t
    [[nodiscard]] Vec3 operator*(const Vec3& p) const {
        return s * (q * p) + t;
    }

    /// 4x4 变换矩阵：[sR t; 0 1]
    [[nodiscard]] Mat44 matrix() const {
        Mat44 T = Mat44::Identity();
        T.block<3, 3>(0, 0) = s * q.toRotationMatrix();
        T.block<3, 1>(0, 3) = t;
        return T;
    }

    /// 逆变换：s'=1/s, q'=q⁻¹, t' = -(q⁻¹·t)/s
    [[nodiscard]] Sim3 inverse() const {
        Eigen::Quaterniond q_inv = q.inverse();
        return Sim3(1.0 / s, q_inv, -(q_inv * t) / s);
    }

    /// 丢弃尺度 → SE3（回环边用 SE3 表示时使用，尺度已由传播吸收）
    [[nodiscard]] SE3 toSE3() const { return SE3(q, t); }

    /// 从 4x4 矩阵构造：左上 3x3 = s·R，列范数即尺度 s
    [[nodiscard]] static Sim3 fromMatrix(const Mat44& T) {
        Eigen::Matrix3d sR = T.block<3, 3>(0, 0);
        double s = sR.col(0).norm();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (s > 1e-12) R = sR / s;
        return Sim3(s, Eigen::Quaterniond(R), T.block<3, 1>(0, 3));
    }

    /// Umeyama 闭式求解：dst ≈ s·R·src + t（with_scaling=true）
    /// 点数 < 3 或点集退化（共线/共面）返回 false
    [[nodiscard]] static bool estimate(const std::vector<Vec3>& src,
                                       const std::vector<Vec3>& dst,
                                       Sim3& out) {
        if (src.size() < 3 || src.size() != dst.size()) return false;

        // 退化检查：中心化后交叉协方差的最小/最大奇异值比
        // （共线/共面 → 数值秩退化），相对阈值避免误伤小尺度点集
        Vec3 c_src = Vec3::Zero(), c_dst = Vec3::Zero();
        for (const auto& p : src) c_src += p;
        for (const auto& p : dst) c_dst += p;
        c_src /= (double)src.size();
        c_dst /= (double)dst.size();
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < src.size(); i++)
            cov += (src[i] - c_src) * (dst[i] - c_dst).transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov);
        if (svd.singularValues().maxCoeff() < 1e-12 ||
            svd.singularValues().minCoeff() < 1e-9 * svd.singularValues().maxCoeff())
            return false;

        // Eigen::umeyama 闭式解（内部 SVD），返回 4x4（src→dst）
        Eigen::MatrixXd S(3, src.size()), D(3, dst.size());
        for (size_t i = 0; i < src.size(); i++) {
            S.col(i) = src[i];
            D.col(i) = dst[i];
        }
        Mat44 T = Eigen::umeyama(S, D, true);
        if (!T.allFinite()) return false;
        out = fromMatrix(T);
        return true;
    }
};

// ---------- 日志宏 ----------
#define LOG_INFO(msg)  std::cout << "[INFO] " << msg << std::endl
#define LOG_WARN(msg)  std::cout << "[WARN] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl

} // namespace vslam
