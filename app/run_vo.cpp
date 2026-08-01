/**
 * run_vo.cpp - 视觉里程计演示入口
 *
 * 用法：
 *   ./run_vo [dataset_path] [config.yaml] [trajectory.txt]
 *
 * 示例：
 *   ./run_vo /data/kitti/sequences/00/image_0 config/default.yaml
 *   ./run_vo /data/kitti/sequences/00/image_0 config/default.yaml traj.txt
 *   ./run_vo 0                          # 使用摄像头 0
 *
 * 输出：运行结束后将轨迹保存为 TUM 格式
 *   timestamp tx ty tz qx qy qz qw（T_wc，相机在世界系），可直接用 EVO 评估
 */

#include "vslam/vo.h"
#include "vslam/dataset.h"
#include "vslam/viewer.h"
#include "vslam/camera.h"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>

int main(int argc, char** argv) {
    std::string input_path;
    vslam::Dataset::Type dataset_type = vslam::Dataset::Type::KITTI;
    std::string config_path = "config/default.yaml";
    std::string traj_path   = "trajectory.txt";

    // ---- 解析命令行参数 ----
    // 用法: run_vo <dataset_path|camera_index> [config.yaml] [trajectory.txt] [--tum|--euroc]
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--tum")       dataset_type = vslam::Dataset::Type::TUM;
        else if (a == "--euroc") dataset_type = vslam::Dataset::Type::EUROC;
    }
    if (argc >= 2) {
        input_path = argv[1];
        // 如果第一个参数是数字，视为摄像头索引
        if (std::all_of(input_path.begin(), input_path.end(), ::isdigit)) {
            dataset_type = vslam::Dataset::Type::CAMERA;
        }
    } else {
        std::cout << "Usage: run_vo <dataset_path|camera_index> [config.yaml] [trajectory.txt] [--tum|--euroc]\n";
        std::cout << "  dataset_path: path to image directory (KITTI) / dataset root (TUM, EuRoC)\n";
        std::cout << "  camera_index: integer (e.g., 0) for live camera\n";
        std::cout << "  --tum / --euroc: dataset format flag\n";
        std::cout << "\nExample:\n";
        std::cout << "  ./run_vo /data/kitti/00/image_0\n";
        std::cout << "  ./run_vo /data/tum/rgbd_dataset_freiburg1_xyz --tum\n";
        std::cout << "  ./run_vo /data/euroc/MH_01 --euroc\n";
        std::cout << "  ./run_vo 0\n";
        return 1;
    }

    if (argc >= 3) {
        config_path = argv[2];
    }
    if (argc >= 4) {
        traj_path = argv[3];
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

    // ---- 初始化 VO（加载 VO/Feature/Optimizer 参数）----
    vslam::VOConfig vo_cfg = vslam::VOConfig::fromYaml(config_path);
    vslam::VisualOdometry vo(camera, vo_cfg);

    // ---- 初始化可视化 ----
    vslam::Viewer viewer;
    viewer.start();

    // ---- 主循环 ----
    cv::Mat image;
    double timestamp;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    // 记录完整位姿（TUM 格式输出用）
    std::vector<std::pair<double, vslam::SE3>> traj_saved;

    LOG_INFO("Starting VO pipeline... Press Ctrl+C or close window to exit.");

    while (dataset.nextFrame(image, timestamp) && !viewer.shouldQuit()) {
        frame_count++;

        // 运行一帧 VO
        vslam::SE3 pose = vo.addFrame(image, timestamp);
        traj_saved.emplace_back(timestamp, pose);

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

    // ---- 保存轨迹（TUM 格式：time tx ty tz qx qy qz qw，位姿为 T_wc）----
    if (!traj_saved.empty()) {
        std::ofstream ofs(traj_path);
        if (ofs.is_open()) {
            ofs << std::fixed << std::setprecision(6);
            for (auto& [ts, pose_cw] : traj_saved) {
                vslam::SE3 Twc = pose_cw.inverse();  // T_cw → T_wc
                ofs << ts << " "
                    << Twc.t.x() << " " << Twc.t.y() << " " << Twc.t.z() << " "
                    << Twc.q.x() << " " << Twc.q.y() << " " << Twc.q.z() << " "
                    << Twc.q.w() << "\n";
            }
            LOG_INFO("Trajectory saved to " << traj_path
                     << " (" << traj_saved.size() << " poses, TUM format)");
        } else {
            LOG_ERROR("Cannot write trajectory to " << traj_path);
        }
    }

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    LOG_INFO("Done. Processed " << frame_count << " frames in "
             << total_time / 1000.0 << " seconds ("
             << frame_count * 1000.0 / total_time << " FPS)");
    LOG_INFO("Final map: " << vo.getMap()->mapPointCount() << " points, "
             << vo.getMap()->keyFrameCount() << " keyframes");

    return 0;
}
