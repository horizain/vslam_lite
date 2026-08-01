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

    /// 获取数据集类型
    Type type() const { return type_; }

    /// 获取总帧数（摄像头返回 -1 表示无限）
    int totalFrames() const { return total_frames_; }

private:
    bool loadKITTIImageList(const std::string& path);

    Type type_;
    vslam::Camera camera_;

    // 数据集文件列表 / 摄像头句柄
    std::vector<std::string> image_paths_;
    cv::VideoCapture cap_;

    int current_index_ = 0;
    int total_frames_  = 0;
};

} // namespace vslam
