#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/feature.h"

namespace vslam {

/// VO/Feature/Optimizer 参数（默认值与 config/default.yaml 一致）
struct VOConfig {
    int    num_features           = 1000;   // 每帧 ORB 特征数
    double scale_factor           = 1.2;    // 金字塔尺度因子
    int    pyramid_levels         = 8;      // 金字塔层数
    double match_ratio            = 0.7;    // 最近邻/次近邻比率阈值
    double ransac_pixel_threshold = 3.0;    // PnP RANSAC 重投影误差阈值(px)
    int    min_matches_init       = 100;    // 初始化最小匹配数
    int    min_matches_track      = 20;     // 跟踪/对极回退最小匹配数
    double keyframe_translation   = 0.5;    // 关键帧最小平移(m)，单目（尺度归一化后位移小）
    double keyframe_translation_stereo = 1.5;  // 双目/RGB-D 关键帧最小平移(m)：
                                             // 单帧即可建点（绝对尺度），KF 只需服务 BA 稀疏化；
                                             // 阈值过大 → ref 间隔远 → 高速段匹配骤减会 LOST，
                                             // 过小 → 每帧插 KF → 地图膨胀 + BA 空转
    double keyframe_rotation      = 0.2;    // 关键帧最小旋转(rad)
    int    keyframe_min_inliers   = 15;     // 跟踪内点低于该值 → 强制插入关键帧
    int    min_keyframe_interval  = 10;     // 关键帧最小帧间隔（防 weak_match 风暴）
    int    local_window_size      = 10;     // 局部 BA 滑动窗口
    int    local_ba_iterations    = 10;     // 局部 BA 迭代次数
    bool   enable_local_ba        = true;   // 局部 BA 开关（诊断/教学对比用）
    int    feature_method         = 0;      // 0: ORB匹配, 1: LK光流
    double stereo_min_depth      = 0.5;    // 双目/RGB-D 有效深度下限(m)
    double stereo_max_depth      = 50.0;   // 双目/RGB-D 有效深度上限(m)

    /// 从 yaml 配置加载（缺省字段保持默认值）
    static VOConfig fromYaml(const std::string& path);
};

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

    VisualOdometry(const Camera& camera, const VOConfig& cfg = VOConfig());

    /// 输入一帧图像（单目/RGB-D 主图），返回当前的位姿估计
    SE3 addFrame(const cv::Mat& image, double timestamp = 0.0);

    /// 输入一帧双目图像（左目 + 右目），返回当前的位姿估计
    SE3 addFrame(const cv::Mat& left, const cv::Mat& right, double timestamp);

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
    void setFeatureMethod(int method) { cfg_.feature_method = method; }

private:
    void updateStatus(int matches, int inliers, double parallax);
    SE3 addFrameImpl(const cv::Mat& left, const cv::Mat& right, double timestamp);
    /// 双目/RGB-D：用视差（或深度）为每特征点计算相机系 3D 观测 pts_c
    void computeStereoDepths();
    /// 双目/RGB-D：从当前帧 pts_c 直接创建地图点（单帧绝对尺度建点）
    void createMapPointsFromStereo(const Frame::Ptr& frame);
    bool tryInitialize();
    SE3 trackFrame();
    /// LK 光流跟踪（feature_method=1）：基于索引对齐的 map_points 做 PnP，失败回退 ORB
    SE3 trackFrameLK();
    /// 双目/RGB-D：3D-3D 位姿估计（ref 世界系点 vs 当前帧 pts_c，绝对尺度、旋转鲁棒）。
    /// PnP 失败或旋转-平移歧义（假平移跳变）时使用；成功返回 true 并写入 pose_cw
    bool tryTrack3D3D(const std::vector<cv::DMatch>& matches);
    /// 运动先验：ref→上一帧 位移（跳变保护阈值基准）
    double motionPrior() const;
    bool tryRelocalize();               // LOST 状态重定位
    void insertKeyFrame();              // 关键帧插入 + 三角化 + BA
    /// 按共视地图点数选取 Local BA 窗口（含当前关键帧，最早帧锚定）
    std::vector<Frame::Ptr> selectLocalWindow(int n) const;
    void triangulateNewPoints(const Frame::Ptr& f1, const Frame::Ptr& f2,
                              const std::vector<cv::DMatch>& matches);
    static SE3 matToSE3(const cv::Mat& R, const cv::Mat& t);  // cv::Mat → SE3

    // ---- 成员变量 ----
    Camera    camera_;
    VOConfig  cfg_;
    State     state_ = State::INITIALIZING;
    Map::Ptr  map_;

    Frame::Ptr ref_frame_;   // 参考帧（上一个关键帧）
    Frame::Ptr curr_frame_;  // 当前帧
    Frame::Ptr prev_frame_;  // 上一帧（LK 光流模式用）

    FeatureMatcher feature_matcher_;

    // 轨迹记录
    std::vector<Vec3> trajectory_;
    unsigned long frame_count_ = 0;
    unsigned long last_kf_frame_id_ = 0;  // 上一个关键帧的帧号（关键帧冷却用）
    Status status_;  // 当前详细状态（初始化/跟踪/丢失信息）
};

} // namespace vslam
