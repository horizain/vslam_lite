/**
 * stereo_cloud_viewer - 当前帧双目 RGB 点云查看/导出工具。
 *
 * 用法：
 *   ./build/bin/stereo_cloud_viewer \
 *       datasets/kitti/sequences/00 config/kitti00.yaml \
 *       --frames 200 --export-ply /tmp/stereo_last.ply
 *
 * 点云来自当前帧的 Frame::pts_c，不写入 MapPoint；颜色来自左目图像。
 */

#include "vslam/camera.h"
#include "vslam/dataset.h"
#include "vslam/viewer.h"
#include "vslam/vo.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string dataset_path;
    std::string config_path = "config/default.yaml";
    std::string export_ply;
    vslam::Dataset::Type dataset_type = vslam::Dataset::Type::KITTI;
    int max_frames = 0;
    bool headless = false;
};

void printUsage() {
    std::cout
        << "Usage: stereo_cloud_viewer <dataset_path> <config.yaml> [options]\n"
        << "Options:\n"
        << "  --euroc                 read EuRoC cam0/data.csv\n"
        << "  --tum                   read TUM rgb.txt\n"
        << "  --frames N              process at most N frames (0=all)\n"
        << "  --export-ply PATH       write the last valid colored cloud as ASCII PLY\n"
        << "  --headless              process/export without opening Pangolin\n";
}

bool parseOptions(int argc, char** argv, Options& options) {
    if (argc < 3) return false;
    options.dataset_path = argv[1];
    options.config_path = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--euroc") {
            options.dataset_type = vslam::Dataset::Type::EUROC;
        } else if (arg == "--tum") {
            options.dataset_type = vslam::Dataset::Type::TUM;
        } else if (arg == "--headless") {
            options.headless = true;
        } else if (arg == "--frames" && i + 1 < argc) {
            options.max_frames = std::stoi(argv[++i]);
        } else if (arg == "--export-ply" && i + 1 < argc) {
            options.export_ply = argv[++i];
        } else {
            return false;
        }
    }
    return !options.dataset_path.empty() && options.max_frames >= 0;
}

bool writePly(const std::string& path,
             const std::vector<vslam::ColoredPoint>& points) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "ply\nformat ascii 1.0\n"
        << "element vertex " << points.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "end_header\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& point : points) {
        out << point.position_w.x() << ' '
            << point.position_w.y() << ' '
            << point.position_w.z() << ' '
            << static_cast<unsigned int>(point.r) << ' '
            << static_cast<unsigned int>(point.g) << ' '
            << static_cast<unsigned int>(point.b) << '\n';
    }
    return out.good();
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseOptions(argc, argv, options)) {
        printUsage();
        return 1;
    }

    vslam::Dataset dataset(options.dataset_path, options.dataset_type);
    if (dataset.totalFrames() == 0) {
        std::cerr << "Dataset contains no readable frames: "
                  << options.dataset_path << '\n';
        return 2;
    }

    vslam::Camera camera;
    if (dataset.hasCalibration()) {
        camera = dataset.getCamera();
    } else if (dataset.isStereo()) {
        camera = vslam::StereoCamera::fromYaml(options.config_path);
    } else {
        camera = vslam::MonocularCamera::fromYaml(options.config_path);
    }

    vslam::VOConfig vo_config = vslam::VOConfig::fromYaml(options.config_path);
    vo_config.enable_loop_closure = false;
    vslam::VisualOdometry vo(camera, vo_config);
    vslam::Viewer viewer;
    if (!options.headless) viewer.start();

    std::vector<vslam::ColoredPoint> last_cloud;
    cv::Mat image, image_right;
    double timestamp = 0.0;
    int processed = 0;
    while (dataset.nextFrame(image, image_right, timestamp)) {
        if (options.max_frames != 0 && processed >= options.max_frames) break;
        if (!options.headless && viewer.shouldQuit()) break;
        ++processed;

        const vslam::SE3 pose = image_right.empty()
            ? vo.addFrame(image, timestamp)
            : vo.addFrame(image, image_right, timestamp);
        last_cloud = vo.getCurrentStereoPointCloud(vslam::Viewer::kMaxMapPoints);

        if (!options.headless) {
            const auto frame = vo.currentFrame();
            if (frame) {
                viewer.updateFrame(frame->image, frame->keypoints,
                                   vo.getTrajectory(vslam::Viewer::kMaxTrajectoryPoints),
                                   pose, frame->image_right);
            }
            viewer.updateMapPoints(
                vo.getMapPointsWorld(vslam::Viewer::kMaxMapPoints));
            viewer.updateColoredPointCloud(last_cloud);
            viewer.setStatus("STEREO CLOUD | frame=" + std::to_string(processed) +
                             " | points=" + std::to_string(last_cloud.size()));
        }
    }

    vo.finishPendingBackendWork();
    if (!options.export_ply.empty()) {
        if (!writePly(options.export_ply, last_cloud)) {
            std::cerr << "Cannot write PLY: " << options.export_ply << '\n';
            if (!options.headless) viewer.stop();
            return 3;
        }
        std::cout << "PLY written: " << options.export_ply
                  << " (" << last_cloud.size() << " points)\n";
    }
    std::cout << "Processed " << processed << " frames; last stereo cloud has "
              << last_cloud.size() << " colored points\n";

    if (!options.headless) {
        std::cout << "Viewer stays open; close the window or press ESC to exit.\n";
        while (!viewer.shouldQuit())
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        viewer.stop();
    }
    return 0;
}
