#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace vslam {

// ---------- 基础类型别名 ----------
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat33 = Eigen::Matrix3d;
using Mat44 = Eigen::Matrix4d;

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

    /// 组合变换：this * other
    [[nodiscard]] SE3 operator*(const SE3& other) const {
        return SE3(q * other.q, q * other.t + t);
    }

    /// 变换一个 3D 点
    [[nodiscard]] Vec3 operator*(const Vec3& p) const {
        return q * p + t;
    }
};

// ---------- 日志宏 ----------
#define LOG_INFO(msg)  std::cout << "[INFO] " << msg << std::endl
#define LOG_WARN(msg)  std::cout << "[WARN] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl

} // namespace vslam
