#pragma once

#include "vslam/camera.h"
#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/map.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace vslam {

/// 历史地点的有限局部快照。地图点位置和 descriptor 在构建时深拷贝；
/// keyframes 只作为地点身份/诊断句柄，验证路径不读取其可变内容。
struct LoopRegionConfig {
    size_t max_keyframes = 12;
    size_t max_points = 800;
    size_t max_covisible_keyframes = 4;
    size_t max_temporal_neighbors = 7;
    size_t temporal_window = 30;

    double descriptor_ratio = 0.80;
    int max_hamming_distance = 96;
    int min_matches = 8;
    int min_inliers = 20;
    double min_inlier_ratio = 0.40;
    int ransac_iterations = 200;
    double ransac_pixel_threshold = 3.0;
    double ransac_confidence = 0.99;
    double max_reprojection_rmse = 2.5;
    double min_positive_depth_ratio = 0.95;
    int grid_columns = 8;
    int grid_rows = 6;
    int min_grid_cells = 12;
};

struct LoopRegionPoint {
    MapPointId id = 0;
    Vec3 pos_s = Vec3::Zero();
    cv::Mat descriptor;
    bool from_pts_c = false;
    KeyframeId source_keyframe_id = 0;
    FeatureIndex source_feature_index = 0;
};

struct LoopRegionKeyframe {
    KeyframeId id = 0;
};

struct LoopRegionSnapshot {
    SubmapId submap_id = 0;
    uint64_t topology_revision = 0;
    uint64_t geometry_revision = 0;
    KeyframeId anchor_id = 0;
    SE3 anchor_pose_cs;
    std::vector<LoopRegionKeyframe> keyframes;
    std::vector<LoopRegionPoint> points;

    [[nodiscard]] bool isBoundTo(SubmapId id, uint64_t topology,
                                 uint64_t geometry) const {
        return submap_id == id && topology_revision == topology &&
               geometry_revision == geometry;
    }
};

struct LoopRegionResult {
    bool accepted = false;
    int matches = 0;
    int inliers = 0;
    double inlier_ratio = 0.0;
    double reprojection_rmse = std::numeric_limits<double>::infinity();
    double positive_depth_ratio = 0.0;
    int grid_cells = 0;
    KeyframeId supporting_keyframe_id = 0; // PnP 内点中占比最高的历史帧
    SE3 T_cw_curr_in_loop;
};

/// 历史局部区域的构建与几何验证。
///
/// 该类不获取 VO 的 map_mutex：build 的调用方必须在持有 Map 读锁时提供
/// 关键帧列表，verify 只消费已经深拷贝的快照。这样锁协议由调用方明确管理，
/// 且快照可以绑定子地图身份以及拓扑/几何 revision 做提交前 stale 检查。
class LoopRegionVerifier {
public:
    /// 在 anchor 周围选择有限共视邻居和时间邻居，并深拷贝地图点描述子。
    /// keyframes 应来自同一子地图且由调用方在 map 读锁内取得。
    [[nodiscard]] static bool build(
        const std::vector<Frame::Ptr>& keyframes, const Frame::Ptr& anchor,
        SubmapId submap_id, uint64_t topology_revision,
        uint64_t geometry_revision, const LoopRegionConfig& config,
        LoopRegionSnapshot& out);

    /// Map 便捷重载；Map 不记录子地图归属，因此调用方应只传入对应子地图的
    /// Map，并在持有该 Map 读锁时调用。
    [[nodiscard]] static bool build(
        const Map& map, const Frame::Ptr& anchor, SubmapId submap_id,
        const LoopRegionConfig& config, LoopRegionSnapshot& out);

    /// 使用区域点 descriptor 做 KNN ratio + mutual + 唯一匹配，再做 PnP RANSAC。
    /// 成功时输出 T_loop_curr = anchor.pose_cs *
    /// T_cw_curr_in_loop.inverse()，即回环边的 loop→current 测量。
    [[nodiscard]] static bool verify(const LoopRegionSnapshot& region,
                                     const Frame::Ptr& current,
                                     const Camera& camera,
                                     const LoopRegionConfig& config,
                                     SE3& T_loop_curr,
                                     LoopRegionResult* result = nullptr);

    [[nodiscard]] static bool verify(const LoopRegionSnapshot& region,
                                     const Frame::Ptr& current,
                                     const Camera& camera,
                                     SE3& T_loop_curr) {
        return verify(region, current, camera, LoopRegionConfig{}, T_loop_curr,
                      nullptr);
    }
};

} // namespace vslam
