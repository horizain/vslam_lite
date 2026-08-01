#include "vslam/dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <sstream>
#include <ranges>
#include <algorithm>

namespace vslam {

Dataset::Dataset(const std::string& path, Type type)
    : type_(type), current_index_(0) {

    if (type_ == Type::KITTI) {
        loadKITTIImageList(path);
        loadCalibration(path);   // 自动读取 calib.txt 内参（各序列内参不同）
    } else if (type_ == Type::TUM) {
        loadTUMImageList(path);
    } else if (type_ == Type::EUROC) {
        loadEUROCImageList(path);
    }
}

Dataset::Dataset(int camera_index) : type_(Type::CAMERA) {
    cap_.open(camera_index);
    if (!cap_.isOpened()) {
        LOG_ERROR("Failed to open camera index " << camera_index);
    } else {
        // 使用 MJPG 格式（与 guvcview 一致，usbipd 下兼容性更好）
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  640);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        LOG_INFO("Camera opened at index " << camera_index
                 << " (" << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                 << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << ")");
        total_frames_ = -1;  // 无限
    }
}

bool Dataset::loadKITTIImageList(const std::string& path) {
    // KITTI 格式：path/sequences/XX/image_0/000000.png
    // 这里简化：path 直接指向图片目录
    cv::String pattern = path + "/*.png";
    std::vector<cv::String> files;
    cv::glob(pattern, files, false);
    if (files.empty()) {
        pattern = path + "/*.jpg";
        cv::glob(pattern, files, false);
    }

    image_paths_.assign(files.begin(), files.end());
    std::ranges::sort(image_paths_);
    total_frames_ = static_cast<int>(image_paths_.size());
    LOG_INFO("Loaded " << total_frames_ << " images from " << path);
    return !image_paths_.empty();
}

bool Dataset::loadCalibration(const std::string& image_dir) {
    // KITTI 结构：<sequences>/<seq>/image_0/，calib.txt 在 <sequences>/<seq>/calib.txt
    std::string calib_path = image_dir + "/../calib.txt";
    std::ifstream ifs(calib_path);
    if (!ifs.is_open()) {
        calib_path = image_dir + "/calib.txt";   // 兼容直接指向序列目录
        ifs.open(calib_path);
    }
    if (!ifs.is_open()) {
        LOG_WARN("KITTI: calib.txt not found (" << calib_path << "), falling back to config");
        return false;
    }
    // 找 P0 行（左目投影矩阵 3x4），提取 fx fy cx cy
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("P0:", 0) != 0) continue;
        std::istringstream iss(line.substr(3));
        std::vector<double> m;
        double v;
        while (iss >> v) m.push_back(v);
        if (m.size() >= 12) {
            camera_.fx = m[0];
            camera_.cx = m[2];
            camera_.fy = m[5];
            camera_.cy = m[6];
            // 图像尺寸：从第一帧获取（calib.txt 不含）
            if (!image_paths_.empty()) {
                cv::Mat first = cv::imread(image_paths_[0], cv::IMREAD_GRAYSCALE);
                if (!first.empty()) {
                    camera_.img_width  = first.cols;
                    camera_.img_height = first.rows;
                }
            }
            calib_loaded_ = true;
            LOG_INFO("Loaded KITTI calibration from " << calib_path
                     << " (fx=" << camera_.fx << " cx=" << camera_.cx
                     << " cy=" << camera_.cy << " img=" << camera_.img_width
                     << "x" << camera_.img_height << ")");
            return true;
        }
    }
    LOG_WARN("KITTI: no P0 in " << calib_path);
    return false;
}

bool Dataset::loadTUMImageList(const std::string& path) {
    // TUM RGB-D 官方结构：<path>/rgb.txt，每行 "timestamp rgb/<file>"，# 开头为注释
    std::ifstream ifs(path + "/rgb.txt");
    if (!ifs.is_open()) {
        LOG_WARN("TUM: cannot open " << path << "/rgb.txt");
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double ts; std::string file;
        if (!(iss >> ts >> file)) continue;
        image_paths_.push_back(path + "/" + file);
        timestamps_.push_back(ts);
    }
    total_frames_ = static_cast<int>(image_paths_.size());
    LOG_INFO("Loaded " << total_frames_ << " TUM images from " << path);
    return !image_paths_.empty();
}

bool Dataset::loadEUROCImageList(const std::string& path) {
    // EuRoC 官方结构：<path>/cam0/data.csv，每行 "timestamp,<file>"
    std::ifstream ifs(path + "/cam0/data.csv");
    if (!ifs.is_open()) {
        LOG_WARN("EuRoC: cannot open " << path << "/cam0/data.csv");
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::string ts_str, file;
        std::istringstream ss(line);
        std::getline(ss, ts_str, ',');
        std::getline(ss, file, ',');
        if (file.empty()) continue;
        image_paths_.push_back(path + "/cam0/data/" + file);
        timestamps_.push_back(std::stod(ts_str));
    }
    total_frames_ = static_cast<int>(image_paths_.size());
    LOG_INFO("Loaded " << total_frames_ << " EuRoC images from " << path);
    return !image_paths_.empty();
}

bool Dataset::nextFrame(cv::Mat& image, double& timestamp) {
    if (type_ == Type::CAMERA) {
        if (!cap_.isOpened()) return false;
        cap_ >> image;
        if (image.empty()) return false;
        timestamp = cap_.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        return true;
    }

    // 数据集模式：从预加载的路径列表读取
    if (current_index_ >= static_cast<int>(image_paths_.size())) {
        return false;
    }

    // 读取彩色图像（VO 内部转灰度，Viewer 显示彩色视频流）
    image = cv::imread(image_paths_[current_index_], cv::IMREAD_COLOR);
    // 时间戳：TUM/EuRoC 用数据集自带时间戳；KITTI 假设 10fps
    timestamp = !timestamps_.empty()
        ? timestamps_[current_index_]
        : static_cast<double>(current_index_) / 10.0;
    current_index_++;
    return !image.empty();
}

} // namespace vslam
