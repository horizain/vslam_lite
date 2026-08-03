#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/atlas.h"
#include "vslam/feature.h"
#include "vslam/loop_closure.h"
#include <string>

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
    int    pnp_min_inliers        = 15;     // PnP 最小内点数
    double pnp_min_inlier_ratio   = 0.3;    // PnP 最小内点比例
    double pnp_max_rmse           = 2.5;    // PnP 最大内点重投影 RMSE(px)
    int    max_tracking_failures  = 5;      // 连续失败多少帧后进入重定位
    int    max_relocalize_frames  = 20;     // 重定位失败多少帧后创建新子地图
    double max_frame_translation  = 3.0;    // 双目相邻有效帧最大平移(m)
    double max_frame_rotation     = 0.35;   // 双目相邻有效帧最大旋转(rad)
    double keyframe_translation   = 0.5;    // 关键帧最小平移(m)，单目（尺度归一化后位移小）
    double keyframe_translation_stereo = 0.9;  // 双目/RGB-D 关键帧最小平移(m)：
                                             // 单帧即可建点（绝对尺度），KF 只需服务 BA 稀疏化；
                                             // 阈值过大 → ref 间隔远 → 高速段匹配骤减会 LOST，
                                             // 过小 → 每帧插 KF → 地图膨胀 + BA 空转
    double keyframe_rotation      = 0.2;    // 关键帧最小旋转(rad)
    int    keyframe_min_inliers   = 15;     // 跟踪内点低于该值 → 强制插入关键帧
    int    min_keyframe_interval  = 10;     // 关键帧最小帧间隔（防 weak_match 风暴）
    int    max_keyframe_interval  = 15;     // 最长关键帧间隔（防尺度低估导致参考帧过旧）
    int    local_window_size      = 10;     // 局部 BA 滑动窗口
    int    local_ba_iterations    = 10;     // 局部 BA 迭代次数
    bool   enable_local_ba        = true;   // 局部 BA 开关（诊断/教学对比用）
    int    feature_method         = 0;      // 0: ORB匹配, 1: LK光流
    double stereo_min_depth       = 0.5;    // 双目/RGB-D 有效深度下限(m)
    double stereo_max_depth       = 35.0;   // 双目/RGB-D 位姿估计深度上限(m)
    int    stereo_min_points      = 40;     // 建图所需最少有效双目点
    int    rigid_min_inliers      = 15;     // 3D-3D 最少 RANSAC 内点
    double rigid_min_inlier_ratio = 0.5;    // 3D-3D 最小内点比例
    double rigid_ransac_threshold = 0.25;   // 3D-3D RANSAC 距离阈值(m)
    double rigid_max_rmse         = 0.25;   // 3D-3D 刚体拟合最大 RMSE(m)

    // ---- 回环检测 (Phase 2) ----
    bool   enable_loop_closure   = false;   // run_slam 默认开、run_vo 默认关（A/B 对比）
    std::string vocab_path       = "";      // 词袋词典路径（config/ORBvoc.dbow3）
    double min_score             = 0.3;     // 词袋候选最低分
    int    temporal_window       = 30;      // 跳过最近 N 个关键帧（时间过滤）
    int    detection_interval    = 10;      // 每 N 个关键帧检测一次回环
    double pnp_inlier_ratio      = 0.7;     // 几何验证最小内点比例
    int    min_loop_inliers      = 30;      // 几何验证最小内点数
    int    loop_cooldown_frames  = 200;     // 回环校正冷却（帧）：防止同区域连续校正
    int    global_ba_iterations  = 20;      // 回环后全局 BA 迭代次数

    /// 从 yaml 配置加载（缺省字段保持默认值）
    static VOConfig fromYaml(const std::string& path);
};

/// 视觉里程计前端
class VisualOdometry {
public:
    enum class State {
        INITIALIZING,  // 正在初始化（等待足够视差的前两帧）
        TRACKING,      // 正常跟踪中
        RECOVERING,    // 短时跟踪退化，保留位姿并尝试恢复
        LOST           // 跟丢了，需要重定位
    };

    struct Status {
        State         state        = State::INITIALIZING;
        int           matches      = 0;
        int           inliers      = 0;
        double        parallax     = 0.0;
        unsigned long map_points   = 0;
        unsigned long keyframes    = 0;
        unsigned long submap_id    = 0;
        int           lost_frames  = 0;
        bool          tracking_valid = false;
        bool          map_connected = true;
        bool          pose_valid   = false;
        std::string   pose_method  = "NONE";
        int           stereo_points = 0;
        double        median_disparity = 0.0;
        double        median_depth = 0.0;
        double        inlier_ratio = 0.0;
        double        pose_rmse    = 0.0;
        double        translation_delta = 0.0;
        double        rotation_delta = 0.0;
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

    /// 获取轻量 Atlas（用于查看子地图数量和状态）
    Atlas::Ptr getAtlas() const { return atlas_; }

    /// 获取当前状态
    State state() const { return state_; }

    /// 获取详细状态信息（用于 UI 反馈）
    Status getStatus() const { return status_; }

    /// 获取估计轨迹中的相机位置（世界系 C_w）
    std::vector<Vec3> getTrajectory() const;

    /// 是否应该创建新的关键帧
    bool needNewKeyFrame() const;

    /// 设置特征方法（0: ORB匹配, 1: LK光流）
    void setFeatureMethod(int method) { cfg_.feature_method = method; }

    /// (Phase 2) 启用回环检测：配置参数 + 加载词袋词典。
    /// 返回 false 表示词典加载失败（回环保持关闭，VO 不受影响）。
    bool enableLoopClosure(const std::string& vocab_path);

    /// (Phase 2) 回环是否已启用（词典加载成功）
    bool loopClosureEnabled() const { return loop_closure_enabled_; }

    /// (Phase 2) 已闭合的回环次数（状态栏/评估用）
    unsigned long loopClosureCount() const { return loop_closure_count_; }

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
    /// 检查双目相邻有效帧运动是否合理，并返回平移/旋转增量。
    bool validateMotion(const SE3& pose_cw, double& translation, double& rotation) const;
    /// 计算 PnP 内点的重投影 RMSE。
    double pnpReprojectionRmse(const std::vector<cv::Point3f>& pts3d,
                               const std::vector<cv::Point2f>& pts2d,
                               const cv::Mat& rvec, const cv::Mat& tvec,
                               const std::vector<int>& inliers) const;
    bool tryRelocalize();               // LOST 状态重定位
    void createSubmap();                // 长时间丢失后锚定全局位姿并新建子地图
    void insertKeyFrame();              // 关键帧插入 + 三角化 + BA
    /// (Phase 2) 回环校正：Sim3 传播 + 位姿图优化 + 全局 BA + 轨迹更新
    void handleLoopCorrection(const Sim3& sim3_loop_to_curr,
                              const Frame::Ptr& kf_curr, const Frame::Ptr& kf_loop);
    /// 按共视地图点数选取 Local BA 窗口（含当前关键帧，最早帧锚定）
    std::vector<Frame::Ptr> selectLocalWindow(int n) const;
    void triangulateNewPoints(const Frame::Ptr& f1, const Frame::Ptr& f2,
                              const std::vector<cv::DMatch>& matches);
    static SE3 matToSE3(const cv::Mat& R, const cv::Mat& t);  // cv::Mat → SE3

    // ---- 成员变量 ----
    Camera    camera_;
    VOConfig  cfg_;
    State     state_ = State::INITIALIZING;
    Atlas::Ptr atlas_;
    Map::Ptr  map_;

    Frame::Ptr ref_frame_;   // 参考帧（上一个关键帧）
    Frame::Ptr curr_frame_;  // 当前帧
    Frame::Ptr prev_frame_;  // 上一帧（LK 光流模式用）

    FeatureMatcher feature_matcher_;

    // 轨迹记录：相机光心在世界系中的位置 C_w（不是 T_cw.t）
    std::vector<Vec3> trajectory_;
    std::vector<unsigned long> traj_frame_ids_;  // 轨迹点对应帧号（回环校正用）

    // ---- Phase 2 回环状态 ----
    std::unique_ptr<LoopClosure> loop_closure_;
    bool loop_closure_enabled_ = false;
    unsigned long loop_closure_count_ = 0;   // 已闭合回环次数
    unsigned long last_loop_kf_id_ = 0;      // 上次回环校正的当前帧号（冷却基准）

    unsigned long frame_count_ = 0;
    unsigned long last_kf_frame_id_ = 0;  // 上一个关键帧的帧号（关键帧冷却用）
    SE3 last_valid_pose_cw_;
    bool has_last_valid_pose_ = false;
    int tracking_failures_ = 0;
    int relocalization_frames_ = 0;
    Status status_;  // 当前详细状态（初始化/跟踪/丢失信息）
};

} // namespace vslam
