#include "vslam/dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <ranges>
#include <algorithm>

namespace fs = std::filesystem;

namespace vslam {

Dataset::Dataset(const std::string& path, Type type)
    : type_(type), current_index_(0),
      camera_(std::make_shared<MonocularCamera>()) {

    if (type_ == Type::KITTI) {
        loadKITTIImageList(path);
        loadCalibration(path);   // 自动读取 calib.txt 内参（各序列内参不同）
    } else if (type_ == Type::TUM) {
        loadTUMImageList(path);
    } else if (type_ == Type::EUROC) {
        loadEUROCImageList(path);
    }
}

Dataset::Dataset(int camera_index)
    : type_(Type::CAMERA), camera_(std::make_shared<MonocularCamera>()) {
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
    // KITTI 格式：path/sequences/XX/image_0/000000.png（左目）+ image_1/（右目）
    // 入参支持两种：
    //   1) 序列根目录 <seq>/（含 image_0/ image_1/ calib.txt）→ 双目
    //   2) 左目图片目录 <seq>/image_0/ → 单目（向后兼容）
    auto load_dir = [](const std::string& dir, std::vector<std::string>& out) {
        out.clear();
        if (!fs::is_directory(dir)) return;  // 目录不存在（如未配置右目）
        cv::String pattern = dir + "/*.png";
        std::vector<cv::String> files;
        cv::glob(pattern, files, false);
        if (files.empty()) {
            pattern = dir + "/*.jpg";
            cv::glob(pattern, files, false);
        }
        out.assign(files.begin(), files.end());
        std::ranges::sort(out);
    };

    // 探测 path 是否为序列根目录（含 image_0 子目录）→ 双目。
    // 注意不能用 cv::glob 探测：目录不存在时它会抛异常而非返回空。
    std::string left_dir = path;
    std::string right_dir;
    if (fs::is_directory(path + "/image_0")) {
        left_dir  = path + "/image_0";
        right_dir = path + "/image_1";
    }

    load_dir(left_dir, image_paths_);
    if (!right_dir.empty()) {
        load_dir(right_dir, right_image_paths_);
        if (right_image_paths_.size() != image_paths_.size()) {
            LOG_WARN("KITTI: left/right frame count mismatch ("
                     << image_paths_.size() << " vs " << right_image_paths_.size()
                     << "), falling back to monocular");
            right_image_paths_.clear();
        } else if (!right_image_paths_.empty()) {
            LOG_INFO("Loaded stereo sequence: " << image_paths_.size()
                     << " frames (image_0 + image_1) from " << path);
        }
    }
    total_frames_ = static_cast<int>(image_paths_.size());
    if (right_image_paths_.empty())
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

    // 解析投影矩阵行（P0 = 左目, P1 = 右目），P = K * [R|t] 的 12 元素
    auto parse_row = [](const std::string& line, std::vector<double>& m) -> bool {
        if (line.rfind("P0:", 0) != 0 && line.rfind("P1:", 0) != 0) return false;
        std::istringstream iss(line.substr(3));
        m.clear();
        double v;
        while (iss >> v) m.push_back(v);
        return m.size() >= 12;
    };

    std::vector<double> p0, p1;
    std::string line;
    while (std::getline(ifs, line)) {
        std::vector<double> m;
        if (!parse_row(line, m)) continue;
        if (line.rfind("P0:", 0) == 0) p0 = m;
        else if (line.rfind("P1:", 0) == 0) p1 = m;
        if (!p0.empty() && !p1.empty()) break;
    }
    if (p0.empty()) {
        LOG_WARN("KITTI: no P0 in " << calib_path);
        return false;
    }

    // 图像尺寸：从第一帧获取（calib.txt 不含）
    int w = 0, h = 0;
    if (!image_paths_.empty()) {
        cv::Mat first = cv::imread(image_paths_[0], cv::IMREAD_GRAYSCALE);
        if (!first.empty()) { w = first.cols; h = first.rows; }
    }

    // 双目：P1 存在且检测到右目图像流 → StereoCamera。
    // baseline 由 P1 = K_r * [I | (-baseline,0,0)] 反推：baseline = -P1[0][3] / fx_r
    if (!p1.empty() && !right_image_paths_.empty()) {
        auto sc = std::make_shared<StereoCamera>();
        sc->fx = p0[0]; sc->cx = p0[2]; sc->fy = p0[5]; sc->cy = p0[6];
        sc->img_width = w; sc->img_height = h;
        sc->fx_r = p1[0]; sc->cx_r = p1[2]; sc->fy_r = p1[5]; sc->cy_r = p1[6];
        sc->baseline_m = -p1[3] / sc->fx_r;
        camera_ = sc;
        calib_loaded_ = true;
        LOG_INFO("Loaded KITTI stereo calibration from " << calib_path
                 << " (fx=" << sc->fx << " baseline=" << sc->baseline_m
                 << "m img=" << w << "x" << h << ")");
        return true;
    }

    // 单目：仅 P0
    auto mc = std::make_shared<MonocularCamera>();
    mc->fx = p0[0]; mc->cx = p0[2]; mc->fy = p0[5]; mc->cy = p0[6];
    mc->img_width = w; mc->img_height = h;
    camera_ = mc;
    calib_loaded_ = true;
    LOG_INFO("Loaded KITTI calibration from " << calib_path
             << " (fx=" << mc->fx << " cx=" << mc->cx
             << " cy=" << mc->cy << " img=" << w << "x" << h << ")");
    return true;
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
        // 官方 data.csv 为 CRLF 行尾：剔除 \r，否则文件名拼接失败
        if (!line.empty() && line.back() == '\r') line.pop_back();
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

bool Dataset::nextFrame(cv::Mat& left, cv::Mat& right, double& timestamp) {
    if (type_ == Type::CAMERA) {
        if (!cap_.isOpened()) return false;
        cap_ >> left;
        if (left.empty()) return false;
        right.release();  // 摄像头模式默认单目
        timestamp = cap_.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        return true;
    }

    // 数据集模式：从预加载的路径列表读取
    if (current_index_ >= static_cast<int>(image_paths_.size())) {
        return false;
    }

    // 读取彩色图像（VO 内部转灰度，Viewer 显示彩色视频流）
    left = cv::imread(image_paths_[current_index_], cv::IMREAD_COLOR);
    right = right_image_paths_.empty()
        ? cv::Mat()
        : cv::imread(right_image_paths_[current_index_], cv::IMREAD_COLOR);
    // 时间戳：TUM/EuRoC 用数据集自带时间戳；KITTI 假设 10fps
    timestamp = !timestamps_.empty()
        ? timestamps_[current_index_]
        : static_cast<double>(current_index_) / 10.0;
    current_index_++;
    return !left.empty();
}

} // namespace vslam
