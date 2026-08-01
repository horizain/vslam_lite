#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace vslam {

/// 数据集/摄像头输入抽象
class Dataset {
public:
    enum class Type {
        KITTI,      // KITTI 灰度数据集
        TUM,        // TUM RGB-D 数据集
        EUROC,      // EuRoC MAV 数据集
        CAMERA      // 实时摄像头
    };

    Dataset(const std::string& path, Type type);
    Dataset(int camera_index = 0);  // 实时摄像头

    /// 读取下一帧，返回 false 表示数据结束
    bool nextFrame(cv::Mat& image, double& timestamp);

    /// 获取相机模型（从数据集自带参数或标定文件读取）
    const vslam::Camera& getCamera() const { return camera_; }

    /// 是否从数据集加载了标定内参（否则由调用方用配置文件）
    bool hasCalibration() const { return calib_loaded_; }

    /// 获取数据集类型
    Type type() const { return type_; }

    /// 获取总帧数（摄像头返回 -1 表示无限）
    int totalFrames() const { return total_frames_; }

private:
    bool loadKITTIImageList(const std::string& path);
    /// TUM: 解析 <path>/rgb.txt（每行 "timestamp rgb/<file>"）
    bool loadTUMImageList(const std::string& path);
    /// EuRoC: 解析 <path>/cam0/data.csv（每行 "timestamp,<file>"）
    bool loadEUROCImageList(const std::string& path);
    /// KITTI: 从 <image_dir>/../calib.txt 读取 P0 内参（KITTI 图像已校正，无畸变系数）
    bool loadCalibration(const std::string& image_dir);

    Type type_;
    vslam::Camera camera_;

    // 数据集文件列表 / 时间戳 / 摄像头句柄
    std::vector<std::string> image_paths_;
    std::vector<double>      timestamps_;   // 数据集自带时间戳（KITTI 为空则按 10fps 推算）
    cv::VideoCapture cap_;

    bool calib_loaded_ = false;
    int current_index_ = 0;
    int total_frames_  = 0;
};

} // namespace vslam
