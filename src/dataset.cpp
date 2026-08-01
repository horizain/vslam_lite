#include "vslam/dataset.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>

namespace vslam {

Dataset::Dataset(const std::string& path, Type type)
    : type_(type), current_index_(0) {

    if (type_ == Type::KITTI) {
        loadKITTIImageList(path);
    } else if (type_ == Type::TUM) {
        LOG_WARN("TUM dataset support: TODO - parse rgb.txt and associate timestamps");
    } else if (type_ == Type::EUROC) {
        LOG_WARN("EuRoC dataset support: TODO - load from cam0/data.csv");
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
    std::sort(image_paths_.begin(), image_paths_.end());
    total_frames_ = static_cast<int>(image_paths_.size());
    LOG_INFO("Loaded " << total_frames_ << " images from " << path);
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
    // KITTI 数据集：文件名即时间戳的隐式编码
    timestamp = static_cast<double>(current_index_) / 10.0; // 假设 10fps
    current_index_++;
    return !image.empty();
}

} // namespace vslam
