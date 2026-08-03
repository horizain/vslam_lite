#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/camera.h"
#include "vslam/feature.h"
#include <memory>
#include <string>

namespace vslam {

/// 回环检测模块（Phase 2）
///
/// 数据流：词袋检测(DBoW3) → 候选过滤（时间窗 + 分数）→ PnP 几何验证 → Sim3 输出。
/// 只负责"检测 + 验证"，位姿/地图校正由 VisualOdometry::handleLoopCorrection 完成。
class LoopClosure {
public:
    LoopClosure();
    ~LoopClosure();  // Impl 在 .cpp 中定义，析构函数须在此处定义

    /// 配置回环参数 + 相机内参（PnP 几何验证需要）
    void setParams(double min_score, int temporal_window, int min_loop_inliers,
                   double pnp_inlier_ratio, double ransac_pixel_threshold,
                   const Camera& camera);

    /// 从文件加载预训练词袋词典（.txt / .dbow3），并初始化数据库
    bool loadVocabulary(const std::string& vocab_path);

    /// 将关键帧加入数据库（词袋向量入库 + 缓存，供回查）
    void addKeyFrame(Frame::Ptr kf);

    /// 检测回环：返回候选关键帧；nullptr 表示无回环。
    /// 候选过滤：DBoW3 Top-5 → 时间窗（跳过刚走过的路）→ 分数阈值。
    Frame::Ptr detectLoop(Frame::Ptr kf);

    /// 几何一致性验证：ORB 匹配 → 3D-2D PnP → 内点判定 → Sim3 求解。
    /// 成功时输出 sim3_loop_to_curr（回环帧相机系 → 当前帧相机系，含尺度比）。
    bool verifyLoop(Frame::Ptr kf_curr, Frame::Ptr kf_loop,
                    Sim3& sim3_loop_to_curr);

private:
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
    Camera camera_;
    FeatureMatcher matcher_;
};

} // namespace vslam
