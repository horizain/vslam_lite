#pragma once

#include "vslam/common.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <memory>

namespace vslam {

// ============================================================
// 相机类型枚举：传感器类型（本轮实现双目；RGB-D 为预留接口）
// ============================================================
enum class CameraType {
    MONOCULAR,   // 单目
    STEREO,      // 双目
    RGBD         // RGB-D (深度)
};

// ============================================================
// 抽象基类：所有相机模型的公共接口
//
// 第一性原理：相机模型 = "如何把 3D 点投影到像素" 的几何模型。
// 单目/双目/RGB-D 的区别不在于投影（都是针孔模型），而在于
// "每一帧能否直接获得带绝对尺度的深度观测"（hasPerFrameDepth）：
//   - 单目: 无单帧深度，3D 靠多帧三角化（对极几何），尺度不可观测
//   - 双目: 左右目视差 → 单帧绝对尺度深度，首帧即可建图
//   - RGB-D: 深度图直接给出单帧绝对尺度深度
// 后续 IMU 是独立于相机的传感器（提供帧间运动先验），不属于本抽象。
// ============================================================
class CameraBase {
public:
    // ---- 公共内参（所有针孔模型共享）----
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int    img_width = 0, img_height = 0;
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;  // 主目畸变
    mutable cv::Mat mapx_, mapy_;  // 去畸变 remap 表（懒初始化）

    virtual ~CameraBase() = default;
    virtual CameraType type() const = 0;

    /// 主目（左目）内参矩阵 K (3x3)
    cv::Mat K() const;
    cv::Mat distCoeffs() const;
    /// 主目是否配置了畸变（非零畸变系数）
    bool hasDistortion() const;
    /// 主目去畸变（预计算 remap 表加速；无畸变时原样返回）
    cv::Mat undistort(const cv::Mat& img) const;

    int width()  const { return img_width; }
    int height() const { return img_height; }

    /// 世界坐标 → 主目（左目）像素坐标
    Vec2 world2pixel(const Vec3& p_w, const SE3& T_cw) const;
    /// 主目像素 → 归一化相机坐标（z = depth）
    Vec3 pixel2camera(const Vec2& pixel, double depth = 1.0) const;

    // ---- 传感器能力 ----
    /// 单帧是否可直接获得带绝对尺度的深度观测（双目视差 / RGB-D 深度图）
    virtual bool hasPerFrameDepth() const { return false; }
    /// 双目基线（米）；非双目返回 0
    virtual double baseline() const { return 0.0; }
    /// 左目相机系 3D 点 → 右目像素（平行双目假设；非双目返回 Zero）
    virtual Vec2 camera2pixelRight(const Vec3& p_c) const {
        (void)p_c; return Vec2::Zero();
    }
};

// ============================================================
// 类型别名：Camera = 智能指针（承载单目/双目/RGB-D 多态）
// 后续新增 IMU 作为独立传感器类型，不在此枚举内
// ============================================================
using Camera = std::shared_ptr<CameraBase>;

// ============================================================
// 单目相机
// ============================================================
class MonocularCamera : public CameraBase {
public:
    CameraType type() const override { return CameraType::MONOCULAR; }

    /// 从 yaml 配置加载（Camera 段）
    static Camera fromYaml(const std::string& path);
};

// ============================================================
// 双目相机（校正后平行双目：右目相对左目沿 +x 平移 baseline）
// ============================================================
class StereoCamera : public CameraBase {
public:
    // 右目内参（KITTI 等校正后通常与左目相同）
    double fx_r = 0, fy_r = 0, cx_r = 0, cy_r = 0;
    double baseline_m = 0;  // 基线（米）

    CameraType type() const override { return CameraType::STEREO; }
    bool hasPerFrameDepth() const override { return true; }
    double baseline() const override { return baseline_m; }

    /// 左目相机系点 → 右目像素：p_r = K_r * (p_c - (baseline,0,0))（平行双目）
    Vec2 camera2pixelRight(const Vec3& p_c) const override;

    /// 视差 → 深度：z = fx * baseline / disparity
    double disparityToDepth(double disparity) const { return fx * baseline_m / disparity; }

    /// 从 yaml 加载（Camera 段 + Stereo.baseline），返回双目相机
    static Camera fromYaml(const std::string& path);
};

// ============================================================
// RGB-D 相机（Phase 2 预留：本轮不实现深度图读取）
// ============================================================
class RGBDCamera : public CameraBase {
public:
    double depth_scale = 1000.0;  // 深度值到米的转换系数
    CameraType type() const override { return CameraType::RGBD; }
};

} // namespace vslam
