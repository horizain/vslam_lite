/**
 * run_slam.cpp - 完整 SLAM 演示入口（Phase 2：VO + 回环检测 + 位姿图 + 全局 BA）
 *
 * 用法：
 *   ./run_slam [dataset_path] [config.yaml] [trajectory.txt]
 *
 * 示例：
 *   ./run_slam datasets/kitti/sequences/00 config/default.yaml slam_traj.txt
 *   ./run_slam 0                          # 使用摄像头 0
 *
 * 与 run_vo 的唯一区别：VOConfig.enable_loop_closure 强制置 true
 * （run_vo 强制置 false），便于 A/B 对比回环对 ATE 的提升。
 *
 * 输出：运行结束后保存 TUM 格式轨迹（回环校正后的全局位姿）
 *   timestamp tx ty tz qx qy qz qw（T_wc，相机在世界系），可直接用 EVO 评估
 */

#include "vslam/vo.h"
#include "vslam/dataset.h"
#include "vslam/viewer.h"
#include "vslam/camera.h"
#include "perf_monitor.h"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <format>
#include <chrono>
#include <algorithm>
#include <cstdlib>

int main(int argc, char** argv) {
    std::string input_path;
    vslam::Dataset::Type dataset_type = vslam::Dataset::Type::KITTI;
    std::string config_path = "config/default.yaml";
    std::string traj_path   = "trajectory.txt";

    // ---- 解析命令行参数（与 run_vo 相同）----
    bool headless = false;
    int max_frames = 0;  // 0 = 全程；>0 = 只处理前 N 帧（性能/回归测试用）
    int skip_frames = 0; // 跳过前 N 帧（性能分片测试用）
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--tum")       dataset_type = vslam::Dataset::Type::TUM;
        else if (a == "--euroc") dataset_type = vslam::Dataset::Type::EUROC;
        else if (a == "--headless") headless = true;
        else if (a == "--frames" && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (a == "--skip" && i + 1 < argc) skip_frames = std::atoi(argv[++i]);
        else if (a.starts_with("--")) {
            std::cerr << "Unknown or incomplete option: " << a << "\n";
            return 1;
        } else {
            positional.push_back(a);
        }
    }
    if (!positional.empty()) {
        input_path = positional[0];
        if (std::all_of(input_path.begin(), input_path.end(), ::isdigit)) {
            dataset_type = vslam::Dataset::Type::CAMERA;
        }
    } else {
        std::cout << "Usage: run_slam <dataset_path|camera_index> [config.yaml] [trajectory.txt] [--tum|--euroc]\n";
        std::cout << "  dataset_path: path to image directory (KITTI) / dataset root (TUM, EuRoC)\n";
        std::cout << "  camera_index: integer (e.g., 0) for live camera\n";
        std::cout << "  --tum / --euroc: dataset format flag\n";
        return 1;
    }
    if (positional.size() > 3) {
        std::cerr << "Too many positional arguments\n";
        return 1;
    }
    if (positional.size() >= 2) config_path = positional[1];
    if (positional.size() >= 3) traj_path = positional[2];

    // ---- 创建数据源（先创建，便于读取数据集自带标定）----
    vslam::Dataset dataset(
        dataset_type == vslam::Dataset::Type::CAMERA
            ? vslam::Dataset(std::stoi(input_path))
            : vslam::Dataset(input_path, dataset_type));

    // ---- 加载相机参数（与 run_vo 相同的优先级）----
    vslam::Camera camera;
    if (dataset_type != vslam::Dataset::Type::CAMERA && dataset.hasCalibration()) {
        camera = dataset.getCamera();
    } else if (dataset_type != vslam::Dataset::Type::CAMERA) {
        camera = dataset.isStereo()
            ? vslam::StereoCamera::fromYaml(config_path)
            : vslam::MonocularCamera::fromYaml(config_path);
    } else {
        camera = std::make_shared<vslam::MonocularCamera>();
        camera->fx = 500; camera->fy = 500;
        camera->cx = 320; camera->cy = 240;
        camera->img_width = 640; camera->img_height = 480;
        LOG_WARN("Camera mode: using DEFAULT intrinsics (500/500/320/240). "
                 "Calibrate your camera and pass config.yaml.");
    }

    // ---- 初始化 SLAM（回环强制开启；run_vo 强制关闭，A/B 对比）----
    vslam::VOConfig vo_cfg = vslam::VOConfig::fromYaml(config_path);
    vo_cfg.enable_loop_closure = true;
    vslam::VisualOdometry vo(camera, vo_cfg);
    if (!vo.loopClosureEnabled()) {
        LOG_WARN("Loop closure NOT enabled (vocab missing? run scripts/fetch_vocab.sh)");
    }

    // ---- 初始化可视化 ----
    vslam::Viewer viewer;
    if (!headless) viewer.start();

    // ---- 主循环 ----
    cv::Mat image, image_right;
    double timestamp;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    // ---- 跳过前 N 帧（性能分片测试；不计入 frame_count）----
    for (int i = 0; i < skip_frames && dataset.nextFrame(image, image_right, timestamp); i++) {
    }
    // 计时基准必须在 skip 之后，否则 Done/FPS 会包含跳帧的解码时间
    start_time = std::chrono::steady_clock::now();

    // VO 内部保留并回环修正完整 T_cw；这里只保存与有效位姿一一对应的时间戳。
    std::vector<double> valid_timestamps;

    LOG_INFO("Starting SLAM pipeline (loop closure "
             << (vo.loopClosureEnabled() ? "ON" : "OFF")
             << ")... Press Ctrl+C or close window to exit.");

    while (dataset.nextFrame(image, image_right, timestamp)
           && (max_frames == 0 || frame_count < max_frames)
           && (headless || !viewer.shouldQuit())) {
        frame_count++;

        // 运行一帧 SLAM（双目：左右目；单目：仅左目）
        vslam::SE3 pose = image_right.empty()
            ? vo.addFrame(image, timestamp)
            : vo.addFrame(image, image_right, timestamp);

        auto st = vo.getStatus();
        if (st.pose_valid) valid_timestamps.push_back(timestamp);

        std::string state_str;
        switch (st.state) {
            case vslam::VisualOdometry::State::INITIALIZING: state_str = "INIT"; break;
            case vslam::VisualOdometry::State::TRACKING:     state_str = "TRACKING"; break;
            case vslam::VisualOdometry::State::RECOVERING:   state_str = "RECOVERING"; break;
            case vslam::VisualOdometry::State::LOST:         state_str = "LOST"; break;
        }
        std::string status = std::format("{} | {}{} | Stereo:{} d:{:.1f}px z:{:.1f}m | Matches:{} Inl:{} RMSE:{:.2f} | MP:{} KF:{} SM:{} Lost:{} | Loop:{}",
                                         state_str, st.pose_method,
                                         st.pose_valid ? "" : " INVALID",
                                         st.stereo_points, st.median_disparity, st.median_depth,
                                         st.matches, st.inliers, st.pose_rmse,
                                         st.map_points, st.keyframes, st.submap_id, st.lost_frames,
                                         vo.loopClosureCount());
        if (!headless) {
            viewer.setStatus(status);
            auto cf = vo.currentFrame();
            viewer.updateFrame(cf->image, cf->keypoints, vo.getTrajectory(),
                               pose, cf->image_right);
        }

        if (frame_count % 30 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();
            double fps = frame_count * 1000.0 / elapsed;
            LOG_INFO("Frame " << frame_count << " | FPS: " << fps
                     << " | MP: " << st.map_points << " KF: " << st.keyframes
                     << " Loops: " << vo.loopClosureCount());
        }
    }

    // ---- 清理 ----
    if (!headless) viewer.stop();

    // 批量评估必须等待排队中的 BA/回环全部写回；否则末次优化、闭环计数和
    // perf 数据可能发生在轨迹保存之后，结果随线程时序变化。
    vo.finishPendingBackendWork();

    // ---- 保存轨迹（TUM 格式，回环校正后的全局位姿）----
    const auto pose_trajectory = vo.getPoseTrajectory();
    if (!pose_trajectory.empty()) {
        if (valid_timestamps.size() != pose_trajectory.size()) {
            LOG_ERROR("Trajectory timestamp/pose size mismatch: "
                      << valid_timestamps.size() << " vs " << pose_trajectory.size());
        }
        std::ofstream ofs(traj_path);
        if (ofs.is_open()) {
            ofs << std::fixed << std::setprecision(6);
            const size_t valid_count = std::min(valid_timestamps.size(),
                                                pose_trajectory.size());
            for (size_t i = 0; i < valid_count; i++) {
                vslam::SE3 Twc = pose_trajectory[i].inverse();
                ofs << valid_timestamps[i] << " "
                    << Twc.t.x() << " " << Twc.t.y() << " " << Twc.t.z() << " "
                    << Twc.q.x() << " " << Twc.q.y() << " " << Twc.q.z() << " "
                    << Twc.q.w() << "\n";
            }
            LOG_INFO("Trajectory saved to " << traj_path
                     << " (" << valid_count << "/" << valid_timestamps.size()
                     << " globally valid poses, TUM format)");
        } else {
            LOG_ERROR("Cannot write trajectory to " << traj_path);
        }
    }

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    LOG_INFO("Done. Processed " << frame_count << " frames in "
             << total_time / 1000.0 << " seconds ("
             << frame_count * 1000.0 / total_time << " FPS)");
    LOG_INFO("Loop closures: " << vo.loopClosureCount()
             << " | Final map: " << vo.getMap()->mapPointCount() << " points, "
             << vo.getMap()->keyFrameCount() << " keyframes");

    // 性能监测 dump（VSLAM_ENABLE_PERF 关闭时为空操作）
    vslam::perf_dump(traj_path + ".perf.csv");

    return 0;
}
