#pragma once

#include "vslam/common.h"
#include <string>
#include <memory>

namespace vslam {

// ============================================================
// 相机类型枚举
// ============================================================
enum class CameraType {
    MONOCULAR,   // 单目
    STEREO,      // 双目
    RGBD         // RGB-D (深度)
};

// ============================================================
// 抽象基类：所有相机模型的公共接口
// ============================================================
class CameraBase {
public:
    virtual ~CameraBase() = default;
    virtual CameraType type() const = 0;

    /// 内参矩阵 K (3x3)
    virtual cv::Mat K() const = 0;

    /// 图像尺寸
    virtual int width()  const = 0;
    virtual int height() const = 0;

    /// 世界坐标 → 像素坐标
    virtual Vec2 world2pixel(const Vec3& p_w, const SE3& T_cw) const = 0;

    /// 像素坐标 → 归一化相机坐标 (z=1 or depth)
    virtual Vec3 pixel2camera(const Vec2& pixel, double depth = 1.0) const = 0;
};

// ============================================================
// 单目相机（Phase 1 主实现）
// ============================================================
class MonocularCamera : public CameraBase {
public:
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int    img_width = 0, img_height = 0;
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;  // 畸变

    MonocularCamera() = default;

    static MonocularCamera fromYaml(const std::string& path);

    CameraType type() const override { return CameraType::MONOCULAR; }

    cv::Mat K() const override {
        return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    }
    cv::Mat distCoeffs() const {
        return (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
    }

    int width()  const override { return img_width; }
    int height() const override { return img_height; }

    Vec2 world2pixel(const Vec3& p_w, const SE3& T_cw) const override;
    Vec3 pixel2camera(const Vec2& pixel, double depth = 1.0) const override;
};

// ============================================================
// 双目相机（Phase 2 预留接口）
// ============================================================
class StereoCamera : public CameraBase {
public:
    // 左右目共享的内参（通常相同）
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int    img_width = 0, img_height = 0;
    double baseline = 0;  // 基线长度（米）

    CameraType type() const override { return CameraType::STEREO; }

    cv::Mat K() const override {
        return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    }
    int width()  const override { return img_width; }
    int height() const override { return img_height; }

    Vec2 world2pixel(const Vec3& p_w, const SE3& T_cw) const override {
        (void)p_w; (void)T_cw; return Vec2::Zero();  // TODO Phase 2
    }
    Vec3 pixel2camera(const Vec2& pixel, double depth = 1.0) const override {
        (void)pixel; (void)depth; return Vec3::Zero();  // TODO Phase 2
    }

    /// 视差 → 深度
    double disparityToDepth(double disparity) const {
        return fx * baseline / disparity;
    }
};

// ============================================================
// RGB-D 相机（Phase 2 预留接口）
// ============================================================
class RGBDCamera : public CameraBase {
public:
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int    img_width = 0, img_height = 0;
    double depth_scale = 1000.0;  // 深度值到米的转换系数

    CameraType type() const override { return CameraType::RGBD; }

    cv::Mat K() const override {
        return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    }
    int width()  const override { return img_width; }
    int height() const override { return img_height; }

    Vec2 world2pixel(const Vec3& p_w, const SE3& T_cw) const override {
        (void)p_w; (void)T_cw; return Vec2::Zero();  // TODO Phase 2
    }
    Vec3 pixel2camera(const Vec2& pixel, double depth = 1.0) const override {
        (void)pixel; (void)depth; return Vec3::Zero();  // TODO Phase 2
    }
};

// ============================================================
// 类型别名：Phase 1 直接用 MonocularCamera
// ============================================================
using Camera = MonocularCamera;

} // namespace vslam
