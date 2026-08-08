#pragma once

#include "vslam/camera.h"
#include "vslam/common.h"
#include "vslam/feature.h"
#include "vslam/frame.h"
#include "vslam/map.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace vslam {

/// 重定位结果（M1.2，§5.3）：Relocalizer 只负责候选的几何验证，返回纯结果，
/// 不切换 Atlas、不写 Map、不写轨迹。提交方（VisualOdometry）在独占锁内再次
/// 检查 expected Map / submap / KF 对象身份与 geometry revision。
struct RelocalizationResult {
    bool accepted = false;          // 是否有候选通过几何验证
    unsigned long submap_id = 0;    // 最佳候选所在子地图
    uint64_t geometry_revision = 0; // 验证时的几何版本（提交方身份检查用）
    Map::Ptr map;                   // 最佳候选所属 Map 实例身份
    Frame::Ptr kf;                  // 最佳候选关键帧（live 对象身份）
    SE3 T_cs;                       // 相机位姿（候选子地图局部系 T_cs）
    int inliers = 0;                // PnP 内点数
    size_t total = 0;               // PnP 输入 3D-2D 对应数
    double rmse = std::numeric_limits<double>::infinity();  // 内点重投影 RMSE(px)
    int quick_checked = 0;          // 检查过的候选数（诊断/日志）
    int quick_passed = 0;           // 通过粗筛进入全量匹配的候选数
};

/// 单候选几何验证的 3D-2D 对应供应（M1.2，§5.3）：调用方（VO）在 map_mutex_
/// 读锁内检查 Map/submap/KF 身份与 geometry revision 并收集点。Relocalizer
/// 不持锁、不访问 Atlas/Map 内部状态。回调返回 false 表示候选已失效，跳过。
struct RelocalizationPointSet {
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    Map::Ptr map;                    // 候选所属 Map 实例身份
    uint64_t geometry_revision = 0;  // 收集时的几何版本
    unsigned long submap_id = 0;     // 候选所在子地图
};

/// 重定位几何验证模块（M1.2，§5.3）。
///
/// 固定流程：候选按给定顺序 → ORB 粗筛（quickMatchCount）→ 全量 ORB 匹配 →
/// PnP 几何验证（solvePnPRansac + 内点重投影 RMSE）→ 保留内点最多的候选，
/// 首个内点达标的候选即返回。只返回 RelocalizationResult；是否接受/提交由
/// 调用方的 PoseGate 与事务决定。不访问 Map/Atlas/Viewer/日志全局状态，
/// 不创建线程。
class Relocalizer {
public:
    /// @param camera          相机内参（PnP 用 K）
    /// @param num_features    ORB 特征预算（内部 FeatureMatcher 配置）
    /// @param scale_factor    ORB 金字塔尺度因子
    /// @param pyramid_levels  ORB 金字塔层数
    /// @param orb_max_bands   ORB 分带上限（1=单带，确定性）
    Relocalizer(const Camera& camera,
                int num_features = 1000, double scale_factor = 1.2,
                int pyramid_levels = 8, int orb_max_bands = 8);

    /// 一次重定位查询（值对象）。curr_frame 必须已含描述子（LOST 前的
    /// 提取已由调用方完成）。
    struct Query {
        using PointProvider = std::function<bool(
            unsigned long submap_id, const Frame::Ptr& kf,
            const std::vector<cv::DMatch>& matches, RelocalizationPointSet& out)>;

        Frame::Ptr curr_frame;       // 待重定位的当前帧（已含描述子）
        std::vector<std::pair<unsigned long, Frame::Ptr>> candidates;  // (submap_id, kf)
        PointProvider supply_points; // 锁内供应 3D-2D 对应与身份/版本

        double match_ratio = 0.7;            // ORB 比率测试阈值
        double ransac_pixel_threshold = 3.0; // PnP RANSAC 重投影阈值(px)
        int    min_inliers = 20;             // 重定位最小内点数
        double min_ratio = 0.4;              // 最小内点比例
        double max_rmse = 2.5;               // 最大内点重投影 RMSE(px)
        int    min_matches = 30;             // 全量匹配数下限（早期剔除）
        int    min_pts3d = 10;               // 3D 点下限（早期剔除）
        int    quick_threshold = 20;         // 粗筛匹配数下限
        int    quick_subset = 256;           // 粗筛描述子子集行数
        double quick_dist_thresh = 64.0;     // 粗筛匹配距离阈值
    };

    /// 候选按给定顺序逐个粗筛 + 几何验证，返回内点最多的通过候选。
    /// 首个候选使内点达到 min_inliers 即停止（与 vo.cpp 原 tryRelocalize 一致）。
    [[nodiscard]] RelocalizationResult relocalize(const Query& query);

    /// 单个候选的几何验证（relocalize 内部调用；单独暴露供测试/复用）。
    /// 通过几何门限时返回 accepted=true 的结果，否则返回 accepted=false。
    [[nodiscard]] RelocalizationResult verifyCandidate(
        unsigned long submap_id, const Frame::Ptr& kf,
        const Query& query);

    /// cv::Mat(rvec, tvec) → SE3（从 VisualOdometry::matToSE3 迁移，公式不变）
    [[nodiscard]] static SE3 matToSE3(const cv::Mat& R, const cv::Mat& t);

private:
    Camera camera_;
    FeatureMatcher matcher_;
};

} // namespace vslam
