/**
 * run_vo.cpp - 视觉里程计演示入口
 *
 * 用法：
 *   ./run_vo [dataset_path] [config.yaml]
 *
 * 示例：
 *   ./run_vo /data/kitti/sequences/00/image_0 config/default.yaml
 *   ./run_vo 0                          # 使用摄像头 0
 */

#include "vslam/vo.h"
#include "vslam/dataset.h"
#include "vslam/viewer.h"
#include "vslam/camera.h"

#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    std::string input_path;
    vslam::Dataset::Type dataset_type = vslam::Dataset::Type::KITTI;
    std::string config_path = "config/default.yaml";

    // ---- 解析命令行参数 ----
    if (argc >= 2) {
        input_path = argv[1];
        // 如果第一个参数是数字，视为摄像头索引
        if (std::all_of(input_path.begin(), input_path.end(), ::isdigit)) {
            dataset_type = vslam::Dataset::Type::CAMERA;
        }
    } else {
        std::cout << "Usage: run_vo <dataset_path|camera_index> [config.yaml]\n";
        std::cout << "  dataset_path: path to image directory\n";
        std::cout << "  camera_index: integer (e.g., 0) for live camera\n";
        std::cout << "\nExample:\n";
        std::cout << "  ./run_vo /data/kitti/00/image_0\n";
        std::cout << "  ./run_vo 0\n";
        return 1;
    }

    if (argc >= 3) {
        config_path = argv[2];
    }

    // ---- 加载相机参数 ----
    vslam::Camera camera;
    if (dataset_type != vslam::Dataset::Type::CAMERA) {
        camera = vslam::Camera::fromYaml(config_path);
    } else {
        // 摄像头模式使用默认参数（实际应通过标定获取）
        camera.fx = 500; camera.fy = 500;
        camera.cx = 320; camera.cy = 240;
        camera.img_width = 640; camera.img_height = 480;
        LOG_WARN("Using default camera params for live camera - calibrate for best results");
    }

    // ---- 创建数据源 ----
    vslam::Dataset dataset(
        dataset_type == vslam::Dataset::Type::CAMERA
            ? vslam::Dataset(std::stoi(input_path))
            : vslam::Dataset(input_path, dataset_type));

    // ---- 初始化 VO ----
    vslam::VisualOdometry vo(camera);

    // ---- 初始化可视化 ----
    vslam::Viewer viewer;
    viewer.start();

    // ---- 主循环 ----
    cv::Mat image;
    double timestamp;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    LOG_INFO("Starting VO pipeline... Press Ctrl+C or close window to exit.");

    while (dataset.nextFrame(image, timestamp) && !viewer.shouldQuit()) {
        frame_count++;

        // 运行一帧 VO
        vslam::SE3 pose = vo.addFrame(image, timestamp);

        // 更新可视化（彩色视频帧 + 绿色特征点 + 轨迹）
        auto cf = vo.currentFrame();
        viewer.updateFrame(cf->image, cf->keypoints, vo.getTrajectory());

        // 更新状态栏（格式化后传给 viewer）
        auto st = vo.getStatus();
        std::string state_str;
        switch (st.state) {
            case vslam::VisualOdometry::State::INITIALIZING: state_str = "INIT"; break;
            case vslam::VisualOdometry::State::TRACKING:     state_str = "TRACKING"; break;
            case vslam::VisualOdometry::State::LOST:         state_str = "LOST"; break;
        }
        char buf[128];
        snprintf(buf, sizeof(buf), "%s | Matches:%d Inl:%d Parallax:%.2f | MP:%lu KF:%lu",
                 state_str.c_str(), st.matches, st.inliers,
                 st.parallax, st.map_points, st.keyframes);
        viewer.setStatus(buf);

        // 打印状态
        if (frame_count % 30 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();
            double fps = frame_count * 1000.0 / elapsed;
            LOG_INFO("Frame " << frame_count << " | FPS: " << fps
                     << " | Pose: (" << pose.t.transpose() << ")"
                     << " | Map points: " << st.map_points
                     << " | Keyframes: " << st.keyframes);
        }
    }

    // ---- 清理 ----
    viewer.stop();

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    LOG_INFO("Done. Processed " << frame_count << " frames in "
             << total_time / 1000.0 << " seconds ("
             << frame_count * 1000.0 / total_time << " FPS)");
    LOG_INFO("Final map: " << vo.getMap()->mapPointCount() << " points, "
             << vo.getMap()->keyFrameCount() << " keyframes");

    return 0;
}
