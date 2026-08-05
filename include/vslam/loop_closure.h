#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/camera.h"
#include "vslam/feature.h"
#include <memory>
#include <mutex>
#include <string>

namespace vslam {

/// 回环检测模块（Phase 2）
///
/// 数据流：词袋检测(DBoW3) → 候选过滤（时间窗 + 分数）→ PnP 几何验证 → SE3 回环约束。
/// 只负责"检测 + 验证"，位姿/地图校正由 VisualOdometry::handleLoopCorrection 完成。
class LoopClosure {
public:
    LoopClosure();
    ~LoopClosure();  // Impl 在 .cpp 中定义，析构函数须在此处定义

    /// 配置回环参数 + 相机内参（PnP 几何验证需要）
    /// @param top_candidates   词袋查询返回候选数（Top-N，提高召回）
    /// @param pos_prior_dist_m 位置先验距离阈值(m)：新 KF 与该阈值内的历史
    ///                          KF（间隔足够远）直接成为候选，靠 PnP 验证把关
    /// @param pos_prior_gap    位置先验最小关键帧间隔（防刚走过的路）
    void setParams(double min_score, int temporal_window, int min_loop_inliers,
                   double pnp_inlier_ratio, double ransac_pixel_threshold,
                   const Camera& camera,
                   int top_candidates = 20,
                   double pos_prior_dist_m = 25.0,
                   int pos_prior_gap = 100);

    /// 从文件加载预训练词袋词典（.txt / .dbow3），并初始化数据库
    bool loadVocabulary(const std::string& vocab_path);

    /// 将关键帧加入数据库（词袋向量入库 + 缓存，供回查）
    void addKeyFrame(Frame::Ptr kf);

    /// 检测回环：返回候选关键帧列表（按优先级排序，可空）。
    /// 候选来源：① DBoW3 Top-N 词袋候选（分数过滤 + 时间窗）按分数降序；
    ///            ② 位置先验候选（世界系距离近 + 间隔足够），词袋候选验证
    ///               失败时兜底——轨迹自交区域词袋分低但几何验证仍可成功。
    /// 调用方依次做 PnP 几何验证，第一个通过的即回环。
    std::vector<Frame::Ptr> detectLoop(Frame::Ptr kf);

    /// 几何一致性验证：ORB 匹配 → 3D-2D PnP → 内点判定。
    /// 成功时输出 T_loop_curr，满足 T_wc_curr = T_wc_loop * T_loop_curr，
    /// 可直接作为位姿图中 loop → curr 的 EdgeSE3 测量。
    bool verifyLoop(Frame::Ptr kf_curr, Frame::Ptr kf_loop,
                    SE3& T_loop_curr);

private:
    /// DBoW3 数据库/缓存访问互斥（异步后端：主线程 addKeyFrame 与
    /// 后台线程 detectLoop 并发；verifyLoop 不碰内部状态无需此锁）
    mutable std::mutex mutex_;
#ifdef HAS_DBOW3
    class Impl;  // DBoW3 具体状态，在 .cpp 中定义
    std::unique_ptr<Impl> impl_;
#endif

    // ---- 非 DBoW3 状态（几何验证）----
    double min_score_            = 0.3;   // 词袋候选最低分（归一化 0~1）
    int    temporal_window_      = 30;    // 跳过最近 N 个关键帧（时间过滤）
    int    min_loop_inliers_     = 30;    // 几何验证最小内点数
    double pnp_inlier_ratio_     = 0.7;   // 几何验证最小内点比例
    double ransac_pixel_threshold_ = 3.0; // PnP RANSAC 重投影阈值(px)
    int    top_candidates_       = 20;    // 词袋查询 Top-N（召回扩宽）
    double position_prior_dist_m_ = 25.0; // 位置先验距离阈值(m)
    int    position_prior_gap_   = 100;   // 位置先验最小关键帧间隔
    Camera camera_;
    FeatureMatcher matcher_;
};

} // namespace vslam
