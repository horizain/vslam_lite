#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include "vslam/camera.h"
#include "vslam/feature.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vslam {

/// 回环检测模块（Phase 2）
///
/// 数据流：DBoW3 或紧凑二进制签名召回 → 候选过滤（时间窗 + 分数）→
/// 历史地点聚类/时序假设跟踪 → PnP 几何验证 → SE3 回环约束。
/// 只负责"检测 + 验证"，位姿/地图校正由 VisualOdometry::handleLoopCorrection 完成。
class LoopClosure {
public:
    /// 回环候选的身份边界。
    ///
    /// Frame::pose_cs 只在所属子地图局部坐标系内有意义；调用方必须使用
    /// submap_id 将候选交给对应的 Atlas 子地图，而不能默认从当前 Map 查找。
    struct LoopCandidate {
        Frame::Ptr frame;
        SubmapId submap_id = 0;
        double score = 0.0;
        // 同一历史地点时间邻域内的候选成员；frame 是当前几何验证代表。
        // 保留成员而不在这里展开地图查询，供后续 region verifier 使用。
        std::vector<Frame::Ptr> cluster_members;
        int temporal_support = 0;
        bool mature = false;
    };

    using SubmapPoses = std::unordered_map<SubmapId, SE3>;

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
                   int pos_prior_gap = 100,
                   double max_reprojection_rmse = 2.5,
                   double min_positive_depth_ratio = 0.95,
                   int min_grid_cells = 12);

    /// 从文件加载预训练词袋词典（.txt / .dbow3），并初始化数据库
    bool loadVocabulary(const std::string& vocab_path);

    /// 加载与 DBoW3 完全相同的二进制层次词表和 TF-IDF/L1 量化语义，但将
    /// 百万节点存成连续 POD，且不复制 Vocabulary/预分配百万倒排表。
    bool loadFlatVocabulary(const std::string& vocab_path,
                            size_t max_keyframes = 256);

    /// 启用实验性紧凑检索：每个 KF 仅保留 global+2x2 的 ORB bit-statistics
    /// 签名，线性扫描有界历史，不加载 97 万词 DBoW3 树。该模式尚未通过
    /// 长序列回环召回门；它只替代候选召回，
    /// 后续时间窗、地点聚类和 PnP/残差/正深度/网格接受门完全复用。
    bool enableCompactRetrieval(size_t max_keyframes = 256);

    /// 将关键帧加入数据库（词袋向量入库 + 缓存，供回查）。
    /// @param submap_id 关键帧所属子地图，不能用当前活动子地图替代。
    /// @param T_ws 加入时的子地图→世界位姿快照；仅作没有最新 Atlas 快照时
    ///             的回退值，位置先验优先使用 detectLoop 传入的最新位姿。
    void addKeyFrame(Frame::Ptr kf, SubmapId submap_id, const SE3& T_ws);

    /// 地图预算剔除关键帧后同步回环索引生命周期。DBoW3 不支持单条删除，
    /// 因此实现会批量清理缓存并只用仍存活的 BoW 重建数据库，避免失效条目
    /// 占据 Top-N 候选和强引用绕过地图内存硬预算。
    void removeKeyFrames(const std::vector<KeyframeId>& keyframe_ids);

    /// 当前由回环模块持有并可参与检索的关键帧数（资源指标/回归测试）。
    [[nodiscard]] size_t indexedKeyFrameCount() const;

    /// 检索特征的载荷字节数（不含 Frame 强引用和容器开销）。用于验证紧凑
    /// 后端的确定性内存上界；DBoW3 返回缓存 BowVector 的保守估计。
    [[nodiscard]] size_t retrievalIndexBytes() const;

    /// 检测回环：返回候选关键帧列表（按优先级排序，可空）。
    /// 候选来源：① DBoW3 Top-N 词袋候选（分数过滤 + 时间窗）按地点聚类；
    ///            ② 位置先验候选（世界系距离近 + 间隔足够），词袋候选验证
    ///               失败时兜底——轨迹自交区域词袋分低但几何验证仍可成功。
    /// 调用方依次对每个地点代表做 PnP 几何验证，第一个通过的即回环；
    /// 每个 LoopCandidate.cluster_members 保留该地点的历史邻域成员，供
    /// 后续 region verifier 扩展使用。BoW 召回量受 top_candidates 配置，
    /// 与位置先验合并去重后每次最多返回 12 个历史地点。
    /// @param query_submap_id 当前查询帧所属子地图。
    /// @param query_T_ws 当前子地图→世界位姿。
    /// @param latest_submap_poses Atlas 中各子地图最新的子地图→世界位姿；
    ///        位置先验使用该表重算双方世界光心，禁止把 pose_cs 局部光心当世界坐标。
    std::vector<LoopCandidate> detectLoop(
        Frame::Ptr kf, SubmapId query_submap_id, const SE3& query_T_ws,
        const SubmapPoses& latest_submap_poses);

    /// 几何一致性验证：ORB 匹配 → 3D-2D PnP → 内点判定。
    /// 成功时输出 T_loop_curr，满足 T_wc_curr = T_wc_loop * T_loop_curr，
    /// 可直接作为位姿图中 loop → curr 的 EdgeSE3 测量。
    bool verifyLoop(Frame::Ptr kf_curr, Frame::Ptr kf_loop,
                    SE3& T_loop_curr);

private:
    class Impl;  // 检索后端具体状态，在 .cpp 中定义
    std::unique_ptr<Impl> impl_;

    // ---- 非 DBoW3 状态（几何验证）----
    double min_score_            = 0.3;   // 词袋候选最低分（归一化 0~1）
    int    temporal_window_      = 30;    // 跳过最近 N 个关键帧（时间过滤）
    int    min_loop_inliers_     = 30;    // 几何验证最小内点数
    double pnp_inlier_ratio_     = 0.7;   // 几何验证最小内点比例
    double ransac_pixel_threshold_ = 3.0; // PnP RANSAC 重投影阈值(px)
    int    top_candidates_       = 20;    // 词袋召回池（聚类后几何验证仍有硬上限）
    double position_prior_dist_m_ = 25.0; // 位置先验距离阈值(m)
    int    position_prior_gap_   = 100;   // 位置先验最小关键帧间隔
    double max_reprojection_rmse_ = 2.5;
    double min_positive_depth_ratio_ = 0.95;
    int min_grid_cells_ = 12;
    Camera camera_;
    FeatureMatcher matcher_;
    // 只保护本模块状态；不持有地图锁，也不从锁内回调 Map。
    mutable std::mutex mutex_;
};

} // namespace vslam
