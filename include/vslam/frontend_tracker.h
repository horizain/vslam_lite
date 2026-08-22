#pragma once

#include "vslam/camera.h"
#include "vslam/common.h"
#include "vslam/feature.h"
#include "vslam/frame.h"
#include "vslam/mappoint.h"
#include "vslam/pose_covariance.h"
#include "vslam/tracking_quality.h"

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vslam {

/// 前端跟踪配置（M1.4：从 VOConfig 抽出的跟踪相关字段快照；§5.5）。
/// 由 VisualOdometry 构造时从 cfg_ 填充一次，运行期不变。
struct TrackerConfig {
    // ORB 提取
    int    num_features = 1000;
    double scale_factor = 1.2;
    int    pyramid_levels = 8;
    int    orb_max_bands = 8;
    bool   stereo_reverse_prune = true;
    // 匹配 / PnP
    double match_ratio = 0.7;
    double ransac_pixel_threshold = 3.0;
    int    min_matches_track = 20;
    int    pnp_min_inliers = 15;
    double pnp_min_inlier_ratio = 0.3;
    double pnp_max_rmse = 2.5;
    double max_frame_translation = 3.0;
    double max_frame_rotation = 0.35;
    // 运动模型引导匹配（方案 A）：用预测位姿把参考帧 3D 点投影到当前帧，
    // 只在投影点邻域内做描述子匹配，替代全图 BF。通用默认值：25px 覆盖
    // 典型 10Hz 双目帧间像素位移；视角/帧率差异大的场景可通过配置调整。
    bool   guided_match = true;
    double guided_search_radius_px = 25.0;
    // 共视图局部地图投影匹配（方案 B）：首轮 PnP 成功后把共视 KF 的地图点
    // 投影进当前帧补充 3D-2D 对应，再精修位姿。通用默认值：共视≥2 的 KF、
    // 最多 400 个点、30px 搜索半径，均为量级预算而非数据集标定。
    bool   local_map_tracking = true;
    int    local_map_min_shared = 2;
    int    local_map_max_points = 400;
    double local_map_search_radius_px = 30.0;
    // 双目深度
    double stereo_min_depth = 0.5;
    double stereo_max_depth = 35.0;
    int    stereo_min_points = 40;
    // 3D-3D
    int    rigid_min_inliers = 15;
    double rigid_min_inlier_ratio = 0.5;
    double rigid_ransac_threshold = 0.25;
    double rigid_max_rmse = 0.25;
    // 关键帧提议
    double keyframe_translation = 0.5;
    double keyframe_translation_stereo = 0.9;
    int    keyframe_max_count = 1800;
    double keyframe_rotation = 0.2;
    int    keyframe_min_inliers = 15;
    int    min_keyframe_interval = 10;
    int    max_keyframe_interval = 15;
    // M3.1（§7.2）：输入图像与特征分布质量门（代码默认关闭以兼容旧配置）
    QualityConfig quality;
};

/// 双目深度统计（M1.4，值对象）
struct StereoStats {
    int    stereo_points = 0;
    double median_disparity = 0.0;
    double median_depth = 0.0;
};

/// 帧首快照的参考帧数据（M1.4，值对象）：跟踪只消费版本绑定的点坐标，
/// 不在中途读实时地图点（后端写回在锁内进行，快照保证帧级一致）。
struct RefView {
    bool has_ref = false;
    SE3 ref_pose_cs;                       // 参考帧位姿（子地图局部系 T_cs）
    SE3 T_ws;                              // 活动子地图锚（世界组合用）
    std::vector<Vec3> ref_points_s;        // 参考帧点局部坐标拷贝（索引对齐）
    std::vector<MapPoint::Ptr> ref_mps;    // 参考帧 map_points（索引对齐）
    // 共视图局部地图（方案 B）：参考 KF 及其共视 KF 的地图点快照。
    // 局部地图投影匹配用它补 3D-2D 对应；与 ref 点同一"版本绑定拷贝"规则。
    std::vector<Vec3> local_points_s;      // 局部地图点局部坐标（索引对齐）
    std::vector<cv::Mat> local_descs;      // 局部地图点描述子（索引对齐）
    std::vector<MapPoint::Ptr> local_mps;  // 局部地图点指针（关联写引用用）
};

/// 运动基线/门限（M1.4，值对象）：由调用方（VO）从自身状态计算后传入。
/// baseline_twc 为空表示跳过连续性验收（单目/无上一有效位姿）。
struct MotionBaseline {
    std::optional<SE3> baseline_twc;
    double max_translation = 0.0;
    double max_rotation = 0.0;
    // 方案 A：匀速模型预测的当前帧位姿（子地图局部系 T_cs）。由 VO 用
    // 相邻有效帧相对运动外推得到；为空时前端退化为全图 BF 匹配。
    std::optional<SE3> predicted_pose_cs;
};

/// 跟踪结果（M1.4，§5.5）：FrontendTracker 只输出结果，不写地图/状态/
/// 切换子地图。调用方（VO）负责应用位姿、关联、状态转换。
struct TrackingResult {
    SE3 pose_cs;                           // 子地图局部系位姿（T_cs）
    bool valid = false;                    // 位姿通过全部质量验收
    std::string method = "NONE";           // PNP / 3D3D / LK_PNP / EPIPOLAR
    int matches = 0;                       // 匹配数（ORB 跟踪）
    int inliers = 0;                       // 成功路径的内点数（EPIPOLAR=匹配数）
    double inlier_ratio = 0.0;             // 最近一次几何估计的验收质量
    double pose_rmse = 0.0;
    double translation_delta = 0.0;        // 相对运动基线平移（验收用）
    double rotation_delta = 0.0;
    bool recovering = false;               // 需要进入 RECOVERING（跟踪失败/大跳变）
    std::vector<int> pnp_inlier_indices;   // PnP 内点在 pts3d 中的索引（关联用）
    std::vector<cv::DMatch> match_pairs;   // 本帧 ORB 匹配，供关键帧建点复用
    // 关联的地图点：trainIdx → MapPoint（普通帧不产生正式观测）
    std::vector<std::pair<int, MapPoint::Ptr>> associations;
    // 与 associations 逐位对齐的"版本绑定快照坐标"（pos_s 拷贝）。跟踪在
    // 锁外读 live mp->pos_s 违反 M1.4 快照约定（后端 BA 可改写坐标）——
    // 局部地图精修等后续步骤必须复用这份快照，而不是 live 指针坐标。
    std::vector<Vec3> association_points_s;
    // M3.2（§7.5）：接受位姿的数值协方差——ξ=[tx,ty,tz,rx,ry,rz] 左扰动
    // Exp(δξ)·pose_cs 的切空间协方差（相机系）。valid=false 表示退化/数值
    // 不可用，调用方必须回退保守占位（单位阵 ×弱质量系数），不得发布假精度。
    Mat6 pose_covariance = Mat6::Zero();
    bool pose_covariance_valid = false;
};

/// 3D-3D 刚体估计结果（M1.4，值对象）
struct RigidResult {
    bool ok = false;
    SE3 pose_cs;                           // T_cw（世界→相机）
    int inliers = 0;
    size_t total = 0;
    double ratio = 0.0;
    double rmse = std::numeric_limits<double>::infinity();
    std::vector<unsigned char> inlier_mask;  // 掩码（与输入点对齐，关联用）
};

/// 共视图局部地图投影匹配结果（方案 B，值对象）
struct LocalMapTrackResult {
    int added = 0;                         // 新增 3D-2D 对应数
    std::vector<cv::Point3f> pts3d;        // 局部地图点（子地图局部系）
    std::vector<cv::Point2f> pts2d;        // 当前帧像素
    std::vector<MapPoint::Ptr> mps;        // 与 pts2d 对齐的地图点指针
    std::vector<int> curr_feature_indices; // 已占用的当前帧特征索引（防重复）
};

/// 关键帧提议（M1.4，§5.5）
struct KeyframeProposal {
    bool need = false;
    double translation = 0.0;   // 相对参考帧平移(m)
    double rotation = 0.0;      // 相对参考帧旋转(rad)
    bool weak_match = false;    // 内点衰减触发
    bool max_interval = false;  // 最远 KF 间隔触发
};

/// 关键帧提议输入（M1.4，值对象）
struct KeyframeInput {
    Frame::Ptr curr_frame;
    SE3 ref_pose_cs;                 // 帧首快照参考位姿（版本绑定）
    int inliers = 0;                 // 最近跟踪内点数
    unsigned long last_kf_frame_id = 0;   // 上一关键帧帧号（冷却）
    unsigned long map_keyframe_count = 0; // 当前活动子地图 KF 数（规模硬顶）
};

/// 前端跟踪模块（M1.4，§5.5）：初始化几何、ORB/LK 跟踪、PnP/3D-3D、
/// 双目深度与关键帧提议。只输出 TrackingResult / KeyframeProposal /
/// StereoStats，不执行 BA/回环，不写 Map/Atlas，不创建线程。
class FrontendTracker {
public:
    FrontendTracker(const Camera& camera, const TrackerConfig& config = TrackerConfig());

    /// M3.1（§7.2）：输入图像与特征分布质量评估（纯函数聚合入口）。
    /// raw_gray 必须是 CLAHE 增强前的原始灰度（统计反映传感器输入）；
    /// 非空且 8-bit 单通道时做像素级判定，否则只按特征分布降级。
    /// 只输出 QualityVerdict，不改变任何跟踪决策。
    [[nodiscard]] QualityVerdict assessFrameQuality(
        const cv::Mat& raw_gray,
        const std::vector<cv::KeyPoint>& keypoints,
        int image_cols, int image_rows) const;

    /// M3.2（§7.5）：最终 PnP 内点的数值位姿协方差（相机系左扰动切空间）。
    /// 纯函数聚合：pnpPoseCovariance + 相机内参；退化时 valid=false，
    /// 调用方不得把该结果当作真实精度发布。
    [[nodiscard]] PoseCovarianceResult estimatePnPCovariance(
        const SE3& pose_cs,
        const std::vector<cv::Point3f>& points_s,
        const std::vector<cv::Point2f>& pixels,
        const std::vector<int>& inlier_indices) const;

    /// 双目/RGB-D：视差（或深度）→ 每特征点相机系 3D 观测 pts_c（写 frame）。
    /// 单目或无右图时不写 pts_c。返回深度统计。
    [[nodiscard]] StereoStats computeStereoDepths(const Frame::Ptr& frame) const;
    /// 使用调用方已经转换好的原始灰度图，避免同一帧再次 BGR→灰度。
    [[nodiscard]] StereoStats computeStereoDepths(
        const Frame::Ptr& frame,
        const cv::Mat& left_gray,
        const cv::Mat& right_gray) const;

    /// PnP 核心（ORB/LK 共用）：3D-2D → solvePnPRansac → PoseGate 验收。
    /// 返回 TrackingResult（pose/valid/quality/pnp_inlier_indices）；
    /// 不关联地图点（由调用方按 inlier 索引映射）。
    [[nodiscard]] TrackingResult trackPnP(
        const std::vector<cv::Point3f>& pts3d,
        const std::vector<cv::Point2f>& pts2d,
        const SE3& T_ws,
        const MotionBaseline& motion,
        int min_inliers, double min_ratio, double max_rmse) const;

    /// 确定性位姿精修（局部地图补匹配后复用）：在给定初始位姿上用
    /// cv::solvePnP（iterative + useExtrinsicGuess）做纯几何精修，不做
    /// RANSAC——不消耗全局 RNG，与 solvePnPRansac 交替调用时保持两路
    /// 确定性同步（test_localizer_contract 等价性要求）。外点由调用方
    /// 提前过滤（首轮内点 + 窗口匹配新增），这里只负责数值优化。
    [[nodiscard]] TrackingResult refinePnP(
        const std::vector<cv::Point3f>& pts3d,
        const std::vector<cv::Point2f>& pts2d,
        const SE3& initial_pose_cs, const SE3& T_ws,
        const MotionBaseline& motion,
        int min_inliers, double min_ratio, double max_rmse) const;

    /// 3D-3D 刚体拟合（Kabsch，RANSAC 内点）：纯几何，不验收。
    /// 通过 RANSAC 内点/比例/退化检查时返回 ok=true；max_rmse 由调用方验收。
    [[nodiscard]] RigidResult estimateRigid3D3D(
        const std::vector<cv::Point3f>& pts_w,
        const std::vector<cv::Point3f>& pts_c,
        int min_inliers, double min_ratio) const;

    /// ORB 跟踪全流程：匹配 → 3D-2D 对应 → PnP → 双目 3D-3D →
    /// 单目对极回退 → RECOVERING。返回完整 TrackingResult（含关联）。
    [[nodiscard]] TrackingResult trackOrb(
        const Frame::Ptr& curr_frame, const Frame::Ptr& ref_frame,
        const RefView& ref, const MotionBaseline& motion,
        const StereoStats& stereo) const;

    /// 方案 A：运动模型引导匹配。用预测位姿把参考帧有 3D 点的特征投影到
    /// 当前帧，只在投影点邻域（网格桶窗口）内做描述子比率匹配，替代全图 BF。
    /// queryIdx 属于 ref_frame，trainIdx 属于 curr_frame（与 match 同一约定）。
    /// 返回空表示预测失效（调用方回退全图匹配）。
    [[nodiscard]] std::vector<cv::DMatch> matchGuided(
        const Frame::Ptr& ref_frame, const Frame::Ptr& curr_frame,
        const std::vector<Vec3>& ref_points_s,   // 版本绑定快照坐标（索引对齐）
        const SE3& T_cs_pred,                    // 预测当前帧位姿（子地图局部系）
        double search_radius_px, double ratio_thresh) const;

    /// 方案 B：共视图局部地图投影匹配。把局部地图点投影进当前帧（用首轮
    /// PnP 位姿），窗口内做描述子比率匹配，补充 3D-2D 对应。occupied_features
    /// 是已被首轮匹配占用的当前帧特征索引，避免同一特征匹配多个点。
    [[nodiscard]] LocalMapTrackResult trackLocalMap(
        const Frame::Ptr& curr_frame,
        const std::vector<Vec3>& local_points_s,
        const std::vector<cv::Mat>& local_descs,
        const std::vector<MapPoint::Ptr>& local_mps,
        const std::vector<int>& occupied_features,
        const SE3& T_cs,                        // 当前帧位姿（子地图局部系）
        double search_radius_px, double ratio_thresh) const;

    /// 关键帧提议（原 needNewKeyFrame 判定，§5.5）。
    [[nodiscard]] KeyframeProposal proposeKeyFrame(const KeyframeInput& input) const;

private:
    Camera camera_;
    mutable FeatureMatcher matcher_;  // 仅供前端线程使用；const 方法内可匹配
    TrackerConfig cfg_;

    /// 构建当前帧特征点的像素网格索引（cell = search_radius_px），
    /// 供投影点窗口搜索用。返回 cell → keypoint 索引列表。
    [[nodiscard]] std::vector<std::vector<int>> buildKeypointGrid(
        const Frame::Ptr& frame, double cell_size_px) const;
    /// 在窗口候选集中做描述子比率匹配（BF Hamming），返回 best match 与
    /// best/second 距离（比率测试用）。单候选时退化为"距离 < max_dist 即接受"
    /// ——局部地图精修不做 RANSAC 外点过滤，窗口匹配必须设距离上限，
    /// 防止随机近邻拉偏位姿（max_dist 与 quickMatchCount 默认 64 一致）。
    [[nodiscard]] bool windowRatioMatch(
        const cv::Mat& query_desc, const cv::Mat& train_desc,
        const std::vector<int>& candidates,
        double ratio_thresh, float max_dist, int& best_train_idx,
        float& best_dist, float& second_dist) const;
};

} // namespace vslam
