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
#include "vslam/localizer.h"
#include "perf_monitor.h"
#include "metrics_json.h"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <format>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstdlib>

int main(int argc, char** argv) {
    std::string input_path;
    vslam::Dataset::Type dataset_type = vslam::Dataset::Type::KITTI;
    std::string config_path = "config/default.yaml";
    std::string traj_path   = "trajectory.txt";
    std::string robot_yaml_path = "config/robot.yaml";  // M0.3 Localizer 配置（相对路径从仓库根启动）

    // ---- 解析命令行参数（与 run_vo 相同）----
    bool headless = false;
    bool use_localizer = false;  // M0.3 可选入口：用 Localizer Facade 包装同一 VO
    bool loop = false;           // M2 遗留清理：数据集 EOF 后重建循环重放（§6.5 soak 单进程连续档）
    int max_frames = 0;  // 0 = 全程；>0 = 只处理前 N 帧（性能/回归测试用）
    int skip_frames = 0; // 跳过前 N 帧（性能分片测试用）
    std::string status_csv_path;    // M1 确定性回归：逐帧状态/计数 CSV
    std::string metrics_json_path;  // 结构化指标 JSON（benchmark.py / soak_test.py 消费）
    std::string metrics_csv_path;   // M2.3（§6.4）：结构化指标 CSV
    long long deadline_ms = 100;    // 单帧延迟门限（10Hz 地面机器人默认 100ms）
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--tum")       dataset_type = vslam::Dataset::Type::TUM;
        else if (a == "--euroc") dataset_type = vslam::Dataset::Type::EUROC;
        else if (a == "--headless") headless = true;
        else if (a == "--localizer") use_localizer = true;
        else if (a == "--loop") loop = true;
        else if (a == "--status-csv" && i + 1 < argc) status_csv_path = argv[++i];
        else if (a == "--metrics-json" && i + 1 < argc) metrics_json_path = argv[++i];
        else if (a == "--metrics-csv" && i + 1 < argc) metrics_csv_path = argv[++i];
        else if (a == "--robot-yaml" && i + 1 < argc) robot_yaml_path = argv[++i];
        else if (a == "--deadline-ms" && i + 1 < argc) deadline_ms = std::atoll(argv[++i]);
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
    if (dataset_type != vslam::Dataset::Type::CAMERA &&
        dataset.totalFrames() == 0) {
        std::cerr << "Dataset contains no readable frames: " << input_path << "\n";
        return 2;
    }

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

    // ---- M0.3 可选入口：Localizer Facade（包装同一 VO，只读输出 PoseEstimate）----
    // 与旧路径完全并行、互不影响：走 Localizer 时不创建/不驱动裸 VisualOdometry。
    if (use_localizer) {
        vslam::VOConfig loc_vo_cfg = vslam::VOConfig::fromYaml(config_path);
        loc_vo_cfg.enable_loop_closure = true;  // 与 run_slam 旧路径一致（A/B 对比）
        vslam::LocalizerConfig loc_cfg = vslam::LocalizerConfig::fromYaml(robot_yaml_path);
        vslam::Localizer localizer(camera, loc_vo_cfg, loc_cfg);

        vslam::Viewer viewer;
        if (!headless) viewer.start();

        cv::Mat image, image_right;
        double timestamp;
        int frame_count = 0;
        double round_base = 0.0;   // --loop 重放偏移：保证重放时间戳单调（§4.3）
        double last_play_ts = 0.0;
        auto start_time = std::chrono::steady_clock::now();
        std::vector<std::pair<double, vslam::SE3>> valid_poses;  // timestamp + T_wc

        for (int i = 0; i < skip_frames && dataset.nextFrame(image, image_right, timestamp); i++) {
        }
        start_time = std::chrono::steady_clock::now();

        // M2 遗留清理（§6.5）：--loop 时 EOF 后重建数据集循环重放——单进程
        // 连续 2h 档的 RSS 稳态斜率必须在同一进程内测量（每轮重启进程会
        // 把词汇表加载的启动噪声算进斜率，§3.37）。
        while (true) {
            if (!dataset.nextFrame(image, image_right, timestamp)) {
                if (!loop) break;
                dataset = vslam::Dataset(input_path, dataset_type);
                round_base = last_play_ts + 1.0;  // 时间戳偏移，保持单调递增
                LOG_INFO("Dataset EOF: restarting replay (loop mode, ts offset "
                         << round_base << ")");
                continue;
            }
            if (max_frames != 0 && frame_count >= max_frames) break;
            if (!headless && viewer.shouldQuit()) break;
            frame_count++;
            const double play_ts = timestamp + round_base;
            last_play_ts = play_ts;
            vslam::PoseEstimate est = image_right.empty()
                ? localizer.processFrame(image, play_ts)
                : localizer.processFrame(image, image_right, play_ts);
            if (est.pose_valid) {
                // 轨迹约定为相机系：T_wc = T_wb · T_bc（§2）
                valid_poses.emplace_back(play_ts, est.T_wb * loc_cfg.T_bc);
            }
            if (!headless) {
                std::string state_str;
                switch (est.state) {
                    case vslam::TrackingState::Initializing: state_str = "INIT"; break;
                    case vslam::TrackingState::Tracking:     state_str = "TRACKING"; break;
                    case vslam::TrackingState::Degraded:     state_str = "DEGRADED"; break;
                    case vslam::TrackingState::Relocalizing: state_str = "RELOC"; break;
                    case vslam::TrackingState::Lost:         state_str = "LOST"; break;
                    case vslam::TrackingState::Stopped:      state_str = "STOPPED"; break;
                }
                viewer.setStatus(std::format("LOC {} | {}{} | MP:{} KF:{}",
                                             state_str,
                                             est.pose_valid ? "" : " INVALID",
                                             est.prediction_only ? " PRED" : "",
                                             localizer.mapPointCount(),
                                             localizer.keyFrameCount()));
                viewer.updateMapPoints(
                    localizer.mapPointsWorld(vslam::Viewer::kMaxMapPoints));
                viewer.updateColoredPointCloud(
                    localizer.currentStereoPointCloud(
                        vslam::Viewer::kMaxMapPoints));
            }
        }

        if (!headless) {
            // 跑完后保持 Viewer 打开，直到用户关闭窗口/按 Esc。
            LOG_INFO("Dataset finished. Viewer stays open; "
                     "close the window or press ESC to exit.");
            while (!viewer.shouldQuit())
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            viewer.stop();
        }
        localizer.stop();

        // M2.3（§6.4）：结构化指标 JSON/CSV（soak_test.py / nightly 消费）
        if (!metrics_json_path.empty()) {
            localizer.writeMetricsJson(metrics_json_path);
            LOG_INFO("Structured metrics -> " << metrics_json_path);
        }
        if (!metrics_csv_path.empty()) {
            localizer.writeMetricsCsv(metrics_csv_path);
            LOG_INFO("Structured metrics CSV -> " << metrics_csv_path);
        }

        std::ofstream ofs(traj_path);
        if (ofs.is_open()) {
            ofs << std::fixed << std::setprecision(6);
            for (const auto& [ts, Twc] : valid_poses) {
                ofs << ts << " "
                    << Twc.t.x() << " " << Twc.t.y() << " " << Twc.t.z() << " "
                    << Twc.q.x() << " " << Twc.q.y() << " " << Twc.q.z() << " "
                    << Twc.q.w() << "\n";
            }
        }
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        LOG_INFO("Done (localizer). Processed " << frame_count << " frames in "
                 << total_time / 1000.0 << " seconds ("
                 << frame_count * 1000.0 / total_time << " FPS), "
                 << valid_poses.size() << " valid poses -> " << traj_path);
        return 0;
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

    // ---- M1 确定性回归：逐帧状态 CSV（frame/state/pose_valid/kf/map revision）----
    std::ofstream status_ofs;
    if (!status_csv_path.empty()) {
        status_ofs.open(status_csv_path);
        if (status_ofs.is_open()) {
            status_ofs << "frame_id,timestamp,state,pose_valid,tracking_valid,"
                          "map_points,keyframes,submap_id,topology_revision,geometry_revision\n";
        } else {
            LOG_WARN("Cannot open status CSV: " << status_csv_path);
        }
    }

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
    std::vector<double> latencies;  // 每帧 addFrame 耗时（ms），用于延迟分位

    // ---- 结构化指标跟踪 ----
    long long deadline_miss = 0;
    long long lost_count = 0;
    double lost_duration = 0.0;
    bool in_lost = false;
    double lost_start = 0.0;
    double last_ts = 0.0;
    unsigned long prev_submap_id = 0;
    bool have_prev_submap = false;
    long long submap_reinit = 0;
    double round_base = 0.0;   // --loop 重放时间戳偏移（保持单调，§4.3）
    double last_play_ts = 0.0;

    LOG_INFO("Starting SLAM pipeline (loop closure "
             << (vo.loopClosureEnabled() ? "ON" : "OFF")
             << (loop ? ", loop replay ON" : "")
             << ")... Press Ctrl+C or close window to exit.");

    while (true) {
        if (!dataset.nextFrame(image, image_right, timestamp)) {
            if (!loop) break;
            dataset = vslam::Dataset(input_path, dataset_type);
            round_base = last_play_ts + 1.0;
            LOG_INFO("Dataset EOF: restarting replay (loop mode, ts offset "
                     << round_base << ")");
            continue;
        }
        if (max_frames != 0 && frame_count >= max_frames) break;
        if (!headless && viewer.shouldQuit()) break;
        frame_count++;
        last_ts = timestamp + round_base;
        last_play_ts = last_ts;

        // 运行一帧 SLAM（双目：左右目；单目：仅左目；--loop 重放用偏移时间戳）
        const auto t0 = std::chrono::steady_clock::now();
        vslam::SE3 pose = image_right.empty()
            ? vo.addFrame(image, last_ts)
            : vo.addFrame(image, image_right, last_ts);
        const double frame_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        latencies.push_back(frame_ms);
        if (frame_ms > static_cast<double>(deadline_ms)) deadline_miss++;

        auto st = vo.getStatus();
        if (st.pose_valid) valid_timestamps.push_back(last_ts);

        // LOST 状态迁移计数 + 时长
        if (st.state == vslam::VisualOdometry::State::LOST) {
            if (!in_lost) { in_lost = true; lost_start = timestamp; }
        } else if (in_lost) {
            in_lost = false;
            lost_count++;
            lost_duration += timestamp - lost_start;
        }
        // 子地图重建计数（submap_id 递增）
        if (!have_prev_submap) {
            prev_submap_id = st.submap_id;
            have_prev_submap = true;
        } else if (st.submap_id != prev_submap_id) {
            submap_reinit++;
            prev_submap_id = st.submap_id;
        }

        if (status_ofs.is_open()) {
            status_ofs << frame_count - 1 << "," << timestamp << ","
                       << static_cast<int>(st.state) << ","
                       << (st.pose_valid ? 1 : 0) << ","
                       << (st.tracking_valid ? 1 : 0) << ","
                       << st.map_points << "," << st.keyframes << "," << st.submap_id << ","
                       << vo.getMap()->topologyRevision() << ","
                       << vo.getMap()->geometryRevision() << "\n";
        }

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
            viewer.updateFrame(cf->image, cf->keypoints,
                               vo.getTrajectory(vslam::Viewer::kMaxTrajectoryPoints),
                               pose, cf->image_right);
            viewer.updateMapPoints(
                vo.getMapPointsWorld(vslam::Viewer::kMaxMapPoints));
            viewer.updateColoredPointCloud(
                vo.getCurrentStereoPointCloud(vslam::Viewer::kMaxMapPoints));
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

    // ---- 结构化指标 JSON（benchmark.py / 提交门消费）----
    if (!metrics_json_path.empty()) {
        vslam::RunMetrics m;
        m.frames_processed = frame_count;
        m.valid_poses = (long long)valid_timestamps.size();
        m.valid_ratio = frame_count > 0
            ? (double)valid_timestamps.size() / (double)frame_count : 0.0;
        m.fps = total_time > 0 ? (double)frame_count * 1000.0 / (double)total_time : 0.0;
        m.seconds = (double)total_time / 1000.0;
        m.deadline_ms = deadline_ms;
        m.deadline_miss = deadline_miss;
        m.deadline_miss_ratio = frame_count > 0
            ? (double)deadline_miss / (double)frame_count : 0.0;
        if (!latencies.empty()) {
            std::vector<double> sorted = latencies;
            std::ranges::sort(sorted);
            m.latency_p50_ms = vslam::percentile(sorted, 50.0);
            m.latency_p95_ms = vslam::percentile(sorted, 95.0);
            m.latency_p99_ms = vslam::percentile(sorted, 99.0);
            m.latency_max_ms = sorted.back();
        }
        if (in_lost) {  // 收尾未关闭的 LOST 段
            lost_count++;
            lost_duration += last_ts - lost_start;
        }
        m.lost_count = lost_count;
        m.lost_duration_s = lost_duration;
        m.submap_reinit = submap_reinit;
        m.loops = (long long)vo.loopClosureCount();
        m.map_points = (long long)vo.getMap()->mapPointCount();
        m.keyframes = (long long)vo.getMap()->keyFrameCount();
        const auto resources = vo.runtimeResourceSnapshot();
        m.process_rss_bytes = static_cast<long long>(resources.rss_bytes);
        m.process_thread_count = static_cast<long long>(resources.thread_count);
        m.allowed_cpu_count = static_cast<long long>(resources.allowed_cpu_count);
        const auto backend = vo.backendStats();
        m.backend_submitted = backend.submitted;
        m.backend_executed = backend.executed;
        m.backend_dropped = backend.dropped;
        m.backend_expired = backend.expired;
        m.backend_service_max_ms = backend.task_service_max_ms;
        vslam::writeRunMetricsJson(metrics_json_path, m);
        LOG_INFO("Structured metrics -> " << metrics_json_path);
    }

    // ---- 保持 Viewer 打开（仅非 headless；headless 立即退出）----
    // 数据集跑完后不立即关闭窗口：渲染线程继续显示最终地图/轨迹/点云，
    // 直到用户关闭窗口或按 Esc（renderLoop 退出时把 quit_ 置真）。
    if (!headless) {
        LOG_INFO("Dataset finished. Viewer stays open; "
                 "close the window or press ESC to exit.");
        while (!viewer.shouldQuit())
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        viewer.stop();
    }

    return 0;
}
