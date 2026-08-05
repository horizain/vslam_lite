/**
 * run_vo.cpp - 视觉里程计演示入口
 *
 * 用法：
 *   ./run_vo [dataset_path] [config.yaml] [trajectory.txt]
 *
 * 示例：
 *   ./run_vo /data/kitti/sequences/00 config/default.yaml    # 双目（自动检测 image_1）
 *   ./run_vo /data/kitti/sequences/00/image_0 config/default.yaml   # 单目（仅左目）
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
#include "perf_monitor.h"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <format>
#include <chrono>

int main(int argc, char** argv) {
    std::string input_path;
    vslam::Dataset::Type dataset_type = vslam::Dataset::Type::KITTI;
    std::string config_path = "config/default.yaml";
    std::string traj_path   = "trajectory.txt";

    // ---- 解析命令行参数 ----
    // 用法: run_vo <dataset_path|camera_index> [config.yaml] [trajectory.txt] [--tum|--euroc] [--headless]
    bool headless = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--tum")       dataset_type = vslam::Dataset::Type::TUM;
        else if (a == "--euroc") dataset_type = vslam::Dataset::Type::EUROC;
        else if (a == "--headless") headless = true;
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

    // ---- 创建数据源（先创建，便于读取数据集自带标定）----
    vslam::Dataset dataset(
        dataset_type == vslam::Dataset::Type::CAMERA
            ? vslam::Dataset(std::stoi(input_path))
            : vslam::Dataset(input_path, dataset_type));

    // ---- 加载相机参数：优先数据集标定（KITTI calib.txt），否则配置文件 ----
    vslam::Camera camera;
    if (dataset_type != vslam::Dataset::Type::CAMERA && dataset.hasCalibration()) {
        camera = dataset.getCamera();
        LOG_INFO("Using camera intrinsics from dataset calib.txt"
                 << " (type=" << (int)camera->type()
                 << " fx=" << camera->fx << " cx=" << camera->cx
                 << " cy=" << camera->cy << " img=" << camera->img_width
                 << "x" << camera->img_height << ")");
    } else if (dataset_type != vslam::Dataset::Type::CAMERA) {
        if (dataset.isStereo()) {
            camera = vslam::StereoCamera::fromYaml(config_path);
            LOG_INFO("Using stereo camera from " << config_path
                     << " (no dataset calibration found)");
        } else {
            camera = vslam::MonocularCamera::fromYaml(config_path);
            LOG_INFO("Using camera intrinsics from " << config_path
                     << " (no dataset calibration found)");
        }
    } else {
        // 摄像头模式使用默认参数（建议先用标定工具获取真实内参）
        camera = std::make_shared<vslam::MonocularCamera>();
        camera->fx = 500; camera->fy = 500;
        camera->cx = 320; camera->cy = 240;
        camera->img_width = 640; camera->img_height = 480;
        LOG_WARN("Camera mode: using DEFAULT intrinsics (500/500/320/240). "
                 "For accurate rotation/scale, calibrate your camera "
                 "(e.g. OpenCV chessboard calibration) and pass config.yaml.");
    }

    // ---- 初始化 VO（加载 VO/Feature/Optimizer 参数）----
    vslam::VOConfig vo_cfg = vslam::VOConfig::fromYaml(config_path);
    // run_vo 固定关闭回环 + 局部 BA：定位为"纯 VO 前端"（仅跟踪），
    // 与 run_slam（回环开 + 局部BA开）构成 A/B 对比基线。
    // 注意：此处强制覆盖 config 的 Optimizer.enable_local_ba/LoopClosure，
    // 如需对比"纯 VO vs 带局部BA"，请用 run_slam + 相应 config。
    vo_cfg.enable_loop_closure = false;
    vo_cfg.enable_local_ba = false;
    // 纯 VO 无 BA/回环任务，后端线程空转，一并关闭
    vo_cfg.async_backend = false;
    vslam::VisualOdometry vo(camera, vo_cfg);

    // ---- 初始化可视化（--headless 跳过，用于批量评估）----
    vslam::Viewer viewer;
    if (!headless) viewer.start();

    // ---- 主循环 ----
    cv::Mat image, image_right;
    double timestamp;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    struct TrajectoryRecord {
        double timestamp;
        vslam::SE3 pose_cw;
        vslam::VisualOdometry::Status status;
    };
    // 主 TUM 只写全局有效位姿；完整帧状态另存 debug CSV。
    std::vector<TrajectoryRecord> traj_saved;

    LOG_INFO("Starting VO pipeline... Press Ctrl+C or close window to exit.");

    while (dataset.nextFrame(image, image_right, timestamp) && (headless || !viewer.shouldQuit())) {
        frame_count++;

        // 运行一帧 VO（双目：左右目；单目：仅左目）
        vslam::SE3 pose = image_right.empty()
            ? vo.addFrame(image, timestamp)
            : vo.addFrame(image, image_right, timestamp);

        // 更新状态栏（格式化后传给 viewer）
        auto st = vo.getStatus();
        traj_saved.push_back({timestamp, pose, st});
        std::string state_str;
        switch (st.state) {
            case vslam::VisualOdometry::State::INITIALIZING: state_str = "INIT"; break;
            case vslam::VisualOdometry::State::TRACKING:     state_str = "TRACKING"; break;
            case vslam::VisualOdometry::State::RECOVERING:   state_str = "RECOVERING"; break;
            case vslam::VisualOdometry::State::LOST:         state_str = "LOST"; break;
        }
        // 更新状态栏（std::format 类型安全，替代 snprintf）
        std::string status = std::format("{} | {}{} | Stereo:{} d:{:.1f}px z:{:.1f}m | Matches:{} Inl:{} RMSE:{:.2f} | MP:{} KF:{} SM:{} Lost:{}",
                                         state_str, st.pose_method,
                                         st.pose_valid ? "" : " INVALID",
                                         st.stereo_points, st.median_disparity, st.median_depth,
                                         st.matches, st.inliers, st.pose_rmse,
                                         st.map_points, st.keyframes, st.submap_id, st.lost_frames);
        if (!headless) {
            viewer.setStatus(status);
            // 双目上下排列，避免超宽画面被压缩；单目显示视频流、特征点和世界系轨迹。
            auto cf = vo.currentFrame();
            viewer.updateFrame(cf->image, cf->keypoints, vo.getTrajectory(),
                               pose, cf->image_right);
        }

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
    if (!headless) viewer.stop();

    // ---- 保存轨迹（TUM 格式：time tx ty tz qx qy qz qw，位姿为 T_wc）----
    if (!traj_saved.empty()) {
        std::ofstream ofs(traj_path);
        if (ofs.is_open()) {
            ofs << std::fixed << std::setprecision(6);
            size_t valid_count = 0;
            for (const auto& record : traj_saved) {
                if (!record.status.pose_valid) continue;
                vslam::SE3 Twc = record.pose_cw.inverse();  // T_cw → T_wc
                ofs << record.timestamp << " "
                    << Twc.t.x() << " " << Twc.t.y() << " " << Twc.t.z() << " "
                    << Twc.q.x() << " " << Twc.q.y() << " " << Twc.q.z() << " "
                    << Twc.q.w() << "\n";
                valid_count++;
            }
            LOG_INFO("Trajectory saved to " << traj_path
                     << " (" << valid_count << "/" << traj_saved.size()
                     << " globally valid poses, TUM format)");
        } else {
            LOG_ERROR("Cannot write trajectory to " << traj_path);
        }

        const std::string debug_path = traj_path + ".debug.csv";
        std::ofstream debug_ofs(debug_path);
        if (debug_ofs.is_open()) {
            debug_ofs << "timestamp,state,tracking_valid,map_connected,pose_valid,pose_method,"
                         "submap_id,lost_frames,stereo_points,median_disparity,median_depth,"
                         "matches,inliers,inlier_ratio,pose_rmse,translation_delta,rotation_delta,"
                         "tx,ty,tz,qx,qy,qz,qw\n";
            debug_ofs << std::fixed << std::setprecision(6);
            for (const auto& record : traj_saved) {
                const auto& st = record.status;
                const vslam::SE3 Twc = record.pose_cw.inverse();
                debug_ofs << record.timestamp << "," << static_cast<int>(st.state) << ","
                          << st.tracking_valid << "," << st.map_connected << ","
                          << st.pose_valid << "," << st.pose_method << ","
                          << st.submap_id << "," << st.lost_frames << ","
                          << st.stereo_points << "," << st.median_disparity << ","
                          << st.median_depth << "," << st.matches << "," << st.inliers << ","
                          << st.inlier_ratio << "," << st.pose_rmse << ","
                          << st.translation_delta << "," << st.rotation_delta << ","
                          << Twc.t.x() << "," << Twc.t.y() << "," << Twc.t.z() << ","
                          << Twc.q.x() << "," << Twc.q.y() << "," << Twc.q.z() << ","
                          << Twc.q.w() << "\n";
            }
            LOG_INFO("Trajectory diagnostics saved to " << debug_path);
        }
    }

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    LOG_INFO("Done. Processed " << frame_count << " frames in "
             << total_time / 1000.0 << " seconds ("
             << frame_count * 1000.0 / total_time << " FPS)");
    LOG_INFO("Final map: " << vo.getMap()->mapPointCount() << " points, "
             << vo.getMap()->keyFrameCount() << " keyframes");

    // 性能监测 dump（VSLAM_ENABLE_PERF 关闭时为空操作）
    vslam::perf_dump("perf.csv");

    return 0;
}
