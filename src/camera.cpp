#include "vslam/camera.h"
#include <yaml-cpp/yaml.h>

namespace vslam {

// ============================================================
// CameraBase 默认实现（针孔投影，全部基于公共内参）
// ============================================================
cv::Mat CameraBase::K() const {
    return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

cv::Mat CameraBase::distCoeffs() const {
    return (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
}

bool CameraBase::hasDistortion() const {
    return k1 != 0 || k2 != 0 || p1 != 0 || p2 != 0 || k3 != 0;
}

cv::Mat CameraBase::undistort(const cv::Mat& img) const {
    if (!hasDistortion()) return img;
    if (mapx_.empty()) {
        cv::initUndistortRectifyMap(K(), distCoeffs(), cv::Mat(), K(),
                                    cv::Size(img_width, img_height),
                                    CV_32FC1, mapx_, mapy_);
    }
    cv::Mat out;
    cv::remap(img, out, mapx_, mapy_, cv::INTER_LINEAR);
    return out;
}

Vec2 CameraBase::world2pixel(const Vec3& p_w, const SE3& T_cw) const {
    Vec3 p_c = T_cw * p_w;
    double inv_z = 1.0 / p_c.z();
    return Vec2(fx * p_c.x() * inv_z + cx,
                fy * p_c.y() * inv_z + cy);
}

Vec2 CameraBase::camera2pixel(const Vec3& p_c) const {
    double inv_z = 1.0 / p_c.z();
    return Vec2(fx * p_c.x() * inv_z + cx,
                fy * p_c.y() * inv_z + cy);
}

Vec3 CameraBase::pixel2camera(const Vec2& pixel, double depth) const {
    return Vec3((pixel.x() - cx) / fx * depth,
                (pixel.y() - cy) / fy * depth,
                depth);
}

// ============================================================
// 单目相机
// ============================================================
Camera MonocularCamera::fromYaml(const std::string& path) {
    auto cam = std::make_shared<MonocularCamera>();
    try {
        YAML::Node cfg = YAML::LoadFile(path);
        auto c = cfg["Camera"];
        cam->fx = c["fx"].as<double>();
        cam->fy = c["fy"].as<double>();
        cam->cx = c["cx"].as<double>();
        cam->cy = c["cy"].as<double>();
        cam->img_width  = c["width"].as<int>();
        cam->img_height = c["height"].as<int>();
        if (c["k1"]) {
            cam->k1 = c["k1"].as<double>();
            cam->k2 = c["k2"].as<double>();
            cam->p1 = c["p1"].as<double>();
            cam->p2 = c["p2"].as<double>();
            cam->k3 = c["k3"].as<double>();
        }
        LOG_INFO("Monocular camera loaded from: " << path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load camera from yaml: " << e.what());
    }
    return cam;
}

Camera MonocularCamera::fromEurocSensorYaml(const std::string& path) {
    auto cam = std::make_shared<MonocularCamera>();
    try {
        YAML::Node cfg = YAML::LoadFile(path);
        const auto model = cfg["camera_model"].as<std::string>();
        const auto distortion_model = cfg["distortion_model"].as<std::string>();
        if (model != "pinhole" || distortion_model != "radial-tangential") {
            LOG_WARN("Unsupported EuRoC camera model in " << path
                     << " (camera_model=" << model
                     << ", distortion_model=" << distortion_model << ")");
            return nullptr;
        }

        const auto resolution = cfg["resolution"];
        const auto intrinsics = cfg["intrinsics"];
        const auto distortion = cfg["distortion_coefficients"];
        if (!resolution || resolution.size() != 2
            || !intrinsics || intrinsics.size() != 4
            || !distortion || distortion.size() != 4) {
            LOG_WARN("Invalid EuRoC camera calibration in " << path);
            return nullptr;
        }

        // EuRoC 顺序为 [fu, fv, cu, cv]，畸变为 [k1, k2, p1, p2]。
        cam->img_width = resolution[0].as<int>();
        cam->img_height = resolution[1].as<int>();
        cam->fx = intrinsics[0].as<double>();
        cam->fy = intrinsics[1].as<double>();
        cam->cx = intrinsics[2].as<double>();
        cam->cy = intrinsics[3].as<double>();
        cam->k1 = distortion[0].as<double>();
        cam->k2 = distortion[1].as<double>();
        cam->p1 = distortion[2].as<double>();
        cam->p2 = distortion[3].as<double>();
        cam->k3 = 0.0;
        LOG_INFO("EuRoC camera loaded from: " << path
                 << " (" << cam->img_width << "x" << cam->img_height
                 << ", fx=" << cam->fx << ", fy=" << cam->fy << ")");
    } catch (const std::exception& e) {
        LOG_WARN("Failed to load EuRoC camera from " << path << ": " << e.what());
        return nullptr;
    }
    return cam;
}

// ============================================================
// 双目相机（校正后平行双目）
// ============================================================
Vec2 StereoCamera::camera2pixelRight(const Vec3& p_c) const {
    // 平行双目（KITTI 校正后）：右目相对左目沿 +x 平移 baseline，
    // 同一点在右目相机系中的坐标为 p_c - (baseline, 0, 0)。
    // 对应 calib.txt 中 P1 = K_r * [I | (-baseline, 0, 0)]。
    double inv_z = 1.0 / p_c.z();
    return Vec2(fx_r * (p_c.x() - baseline_m) * inv_z + cx_r,
                fy_r * p_c.y() * inv_z + cy_r);
}

Camera StereoCamera::fromYaml(const std::string& path) {
    auto cam = std::make_shared<StereoCamera>();
    try {
        YAML::Node cfg = YAML::LoadFile(path);
        auto c = cfg["Camera"];
        cam->fx = c["fx"].as<double>();
        cam->fy = c["fy"].as<double>();
        cam->cx = c["cx"].as<double>();
        cam->cy = c["cy"].as<double>();
        cam->img_width  = c["width"].as<int>();
        cam->img_height = c["height"].as<int>();
        if (c["k1"]) {
            cam->k1 = c["k1"].as<double>();
            cam->k2 = c["k2"].as<double>();
            cam->p1 = c["p1"].as<double>();
            cam->p2 = c["p2"].as<double>();
            cam->k3 = c["k3"].as<double>();
        }
        auto s = cfg["Stereo"];
        if (s["baseline"]) cam->baseline_m = s["baseline"].as<double>();
        // 右目内参默认与左目一致（校正后双目通常如此）
        cam->fx_r = cam->fx; cam->fy_r = cam->fy;
        cam->cx_r = cam->cx; cam->cy_r = cam->cy;
        LOG_INFO("Stereo camera loaded from: " << path
                 << " (baseline=" << cam->baseline_m << " m)");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load stereo camera from yaml: " << e.what());
    }
    return cam;
}

} // namespace vslam
