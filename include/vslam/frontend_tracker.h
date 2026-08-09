#pragma once

#include "vslam/camera.h"
#include "vslam/common.h"
#include "vslam/feature.h"
#include "vslam/frame.h"
#include "vslam/mappoint.h"

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
    // 匹配 / PnP
    double match_ratio = 0.7;
    double ransac_pixel_threshold = 3.0;
    int    min_matches_track = 20;
    int    pnp_min_inliers = 15;
    double pnp_min_inlier_ratio = 0.3;
    double pnp_max_rmse = 2.5;
    double max_frame_translation = 3.0;
    double max_frame_rotation = 0.35;
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
};

/// 运动基线/门限（M1.4，值对象）：由调用方（VO）从自身状态计算后传入。
/// baseline_twc 为空表示跳过连续性验收（单目/无上一有效位姿）。
struct MotionBaseline {
    std::optional<SE3> baseline_twc;
    double max_translation = 0.0;
    double max_rotation = 0.0;
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
    // 关联的地图点：trainIdx → MapPoint（普通帧不产生正式观测）
    std::vector<std::pair<int, MapPoint::Ptr>> associations;
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

    /// 双目/RGB-D：视差（或深度）→ 每特征点相机系 3D 观测 pts_c（写 frame）。
    /// 单目或无右图时不写 pts_c。返回深度统计。
    [[nodiscard]] StereoStats computeStereoDepths(const Frame::Ptr& frame) const;

    /// PnP 核心（ORB/LK 共用）：3D-2D → solvePnPRansac → PoseGate 验收。
    /// 返回 TrackingResult（pose/valid/quality/pnp_inlier_indices）；
    /// 不关联地图点（由调用方按 inlier 索引映射）。
    [[nodiscard]] TrackingResult trackPnP(
        const std::vector<cv::Point3f>& pts3d,
        const std::vector<cv::Point2f>& pts2d,
        const SE3& T_ws,
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

    /// 关键帧提议（原 needNewKeyFrame 判定，§5.5）。
    [[nodiscard]] KeyframeProposal proposeKeyFrame(const KeyframeInput& input) const;

private:
    Camera camera_;
    mutable FeatureMatcher matcher_;  // 仅供前端线程使用；const 方法内可匹配
    TrackerConfig cfg_;
};

} // namespace vslam
