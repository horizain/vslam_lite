#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/feature.h"

namespace vslam {

/// 视觉里程计前端
class VisualOdometry {
public:
    enum class State {
        INITIALIZING,  // 正在初始化（等待足够视差的前两帧）
        TRACKING,      // 正常跟踪中
        LOST           // 跟丢了，需要重定位
    };

    struct Status {
        State         state        = State::INITIALIZING;
        int           matches      = 0;
        int           inliers      = 0;
        double        parallax     = 0.0;
        unsigned long map_points   = 0;
        unsigned long keyframes    = 0;
    };

    VisualOdometry(const Camera& camera);

    /// 输入一帧图像，返回当前的位姿估计
    SE3 addFrame(const cv::Mat& image, double timestamp = 0.0);

    /// 获取当前帧
    Frame::Ptr currentFrame() const { return curr_frame_; }

    /// 获取地图
    Map::Ptr getMap() const { return map_; }

    /// 获取当前状态
    State state() const { return state_; }

    /// 获取详细状态信息（用于 UI 反馈）
    Status getStatus() const { return status_; }

    /// 获取估计的轨迹（所有帧的位姿）
    std::vector<Vec3> getTrajectory() const;

    /// 是否应该创建新的关键帧
    bool needNewKeyFrame() const;

    /// 设置特征方法（0: ORB匹配, 1: LK光流）
    void setFeatureMethod(int method) { feature_method_ = method; }

private:
    void updateStatus(int matches, int inliers, double parallax);
    bool tryInitialize();
    SE3 trackFrame();
    bool tryRelocalize();               // LOST 状态重定位
    void insertKeyFrame();              // 关键帧插入 + 三角化 + BA
    void triangulateNewPoints(const Frame::Ptr& f1, const Frame::Ptr& f2,
                              const std::vector<cv::DMatch>& matches);
    static SE3 matToSE3(const cv::Mat& R, const cv::Mat& t);  // cv::Mat → SE3

    // ---- 成员变量 ----
    Camera    camera_;
    State     state_ = State::INITIALIZING;
    Map::Ptr  map_;

    Frame::Ptr ref_frame_;   // 参考帧（上一个关键帧）
    Frame::Ptr curr_frame_;  // 当前帧

    FeatureMatcher feature_matcher_;

    int feature_method_ = 0;     // 0: ORB匹配, 1: LK光流
    int num_features_   = 1000;  // 每帧提取的特征数

    // 轨迹记录
    std::vector<Vec3> trajectory_;
    unsigned long frame_count_ = 0;
    Status status_;  // 当前详细状态（初始化/跟踪/丢失信息）
};

} // namespace vslam
