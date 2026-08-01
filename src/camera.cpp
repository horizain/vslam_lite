#include "vslam/camera.h"
#include <yaml-cpp/yaml.h>

namespace vslam {

MonocularCamera MonocularCamera::fromYaml(const std::string& path) {
    MonocularCamera cam;
    try {
        YAML::Node cfg = YAML::LoadFile(path);
        auto c = cfg["Camera"];
        cam.fx = c["fx"].as<double>();
        cam.fy = c["fy"].as<double>();
        cam.cx = c["cx"].as<double>();
        cam.cy = c["cy"].as<double>();
        cam.img_width  = c["width"].as<int>();
        cam.img_height = c["height"].as<int>();
        if (c["k1"]) {
            cam.k1 = c["k1"].as<double>();
            cam.k2 = c["k2"].as<double>();
            cam.p1 = c["p1"].as<double>();
            cam.p2 = c["p2"].as<double>();
            cam.k3 = c["k3"].as<double>();
        }
        LOG_INFO("Camera loaded from: " << path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load camera from yaml: " << e.what());
    }
    return cam;
}

Vec2 MonocularCamera::world2pixel(const Vec3& p_w, const SE3& T_cw) const {
    Vec3 p_c = T_cw * p_w;
    double inv_z = 1.0 / p_c.z();
    return Vec2(fx * p_c.x() * inv_z + cx,
                 fy * p_c.y() * inv_z + cy);
}

Vec3 MonocularCamera::pixel2camera(const Vec2& pixel, double depth) const {
    return Vec3((pixel.x() - cx) / fx * depth,
                (pixel.y() - cy) / fy * depth,
                depth);
}

} // namespace vslam
