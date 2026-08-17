#pragma once

#include "vslam/common.h"
#include "vslam/point_cloud.h"
#include "vslam/camera.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/atlas.h"
#include "vslam/feature.h"
#include "vslam/loop_closure.h"
#include "vslam/loop_region.h"
#include "vslam/loop_graph.h"
#include "vslam/optimizer.h"
#include "vslam/backend_committer.h"
#include "vslam/backend_scheduler.h"
#include "vslam/frontend_tracker.h"
#include "vslam/local_mapper.h"
#include "vslam/pose_gate.h"
#include "vslam/relocalizer.h"
#include "vslam/resource_budget.h"
#include "vslam/runtime_resources.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <string>
#include <unordered_set>

namespace vslam {

/// 回环事务的身份边界；候选发现后只携带 Map/submap/KF 身份进入事务。
/// 最终验证/前缀快照在 map 共享锁下重做，求解在锁外；提交时必须在独占锁下
/// 联合复验 Map/Submap、geometry revision 和全部前缀对象身份，再重基尾段。
struct LoopCorrectionContext {
    Map::Ptr map;                 // 当前活动子地图
    Map::Ptr loop_map;            // 候选所属子地图（同图时等于 map）
    SubmapId submap_id = 0;
    SubmapId loop_submap_id = 0;
    uint64_t topology_revision = 0;
    uint64_t geometry_revision = 0;
    KeyframeId curr_kf_id = 0;
    KeyframeId loop_kf_id = 0;
    Frame::Ptr curr_kf;
    Frame::Ptr loop_kf;
    bool preverified = false;
    SE3 preverified_T_loop_curr;
    uint64_t preverified_geometry_revision = 0;
    double preverified_reference_time = 0.0;
};

/// VO/Feature/Optimizer 参数（默认值与 config/default.yaml 一致）
struct VOConfig {
    int    num_features           = 1000;   // 每帧 ORB 特征数
    double scale_factor           = 1.2;    // 金字塔尺度因子
    int    pyramid_levels         = 8;      // 金字塔层数
    double match_ratio            = 0.7;    // 最近邻/次近邻比率阈值
    double ransac_pixel_threshold = 3.0;    // PnP RANSAC 重投影误差阈值(px)
    int    orb_max_bands          = 8;      // ORB 外层并行分带上限（1=单带）
    int    opencv_threads         = 0;      // OpenCV 线程数（0=保持库默认值）
    RuntimeResourceConfig runtime_resources; // 进程 CPU/RSS 资源契约
    int    max_backend_task_age_ms = 500;   // 后台任务最大排队年龄
    bool   reuse_keyframe_matches = true;   // 关键帧建点复用跟踪匹配
    bool   stereo_reverse_prune = true;     // 反向 LK 只验证正向有效点
    bool   use_raw_stereo_gray = true;      // 双目 LK 使用未做 CLAHE 的灰度
    bool   copy_gray_before_clahe = true;   // 避免增强原地改写调用方图像
    int    rng_seed               = 0;      // M1 确定性：非 0 时构造调用 cv::setRNGSeed(rng_seed)
                                            // （固定 solvePnPRansac 等内部 RNG，见 config/deterministic.yaml）
    int    min_matches_init       = 100;    // 初始化最小匹配数
    int    min_init_inliers       = 20;     // 初始化最小对极几何内点数
    double min_parallax           = 0.0175; // 单目初始化最小三角化角（弧度，约1°）
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
                                             // 跳过插入（关键帧稀疏化，压地图规模）
    int    keyframe_max_count      = 1800;  // 关键帧规模软阈值：超过后平移阈值放大 1.5 倍
                                             // （长序列/大场景防 KF 无限增长）
    double keyframe_rotation      = 0.2;    // 关键帧最小旋转(rad)
    int    keyframe_min_inliers   = 15;     // 跟踪内点低于该值 → 强制插入关键帧
    int    min_keyframe_interval  = 10;     // 关键帧最小帧间隔（防 weak_match 风暴）
    int    max_keyframe_interval  = 15;     // 最长关键帧间隔（防尺度低估导致参考帧过旧）
    int    local_window_size      = 10;     // 局部 BA 滑动窗口
    int    local_ba_iterations    = 10;     // 局部 BA 迭代次数
    int    local_ba_passes        = 1;      // 每次 Local BA 的优化轮数（1=实时）
    size_t local_ba_max_points    = 1500;   // Local BA 地图点硬上限
    double local_ba_max_correction = 1.0;   // 局部 BA 单次最大位姿校正(m)
    bool   enable_local_ba        = true;   // 局部 BA 开关（诊断/教学对比用）
    int    feature_method         = 0;      // 0: ORB匹配, 1: LK光流
    double stereo_min_depth       = 0.5;    // 双目/RGB-D 有效深度下限(m)
    double stereo_max_depth       = 35.0;   // 双目/RGB-D 位姿估计深度上限(m)
    int    stereo_min_points      = 40;     // 建图所需最少有效双目点
    int    rigid_min_inliers      = 15;     // 3D-3D 最少 RANSAC 内点
    double rigid_min_inlier_ratio = 0.5;    // 3D-3D 最小内点比例
    double rigid_ransac_threshold = 0.25;   // 3D-3D RANSAC 距离阈值(m)
    double rigid_max_rmse         = 0.25;   // 3D-3D 刚体拟合最大 RMSE(m)

    // ---- 前端跟踪增强（方案 A/B）----
    bool   guided_match            = true;  // 方案 A：运动模型引导匹配开关
    double guided_search_radius_px = 25.0;  // 引导匹配投影邻域搜索半径(px)
    bool   local_map_tracking      = true;  // 方案 B：共视图局部地图投影匹配开关
    int    local_map_min_shared    = 2;     // 局部地图共视 KF 最小共视点数
    int    local_map_max_points    = 400;   // 局部地图点预算（按共视降序截断）
    double local_map_search_radius_px = 30.0; // 局部地图投影搜索半径(px)

    // ---- 回环检测 (Phase 2) ----
    bool   enable_loop_closure   = false;   // run_slam 默认开、run_vo 默认关（A/B 对比）
    std::string loop_retrieval_backend = "dbow3"; // dbow3 | flat_dbow3 | compact_binary
    size_t loop_compact_max_keyframes = 256;       // flat/compact 索引硬上限
    std::string vocab_path       = "";      // 词袋词典路径（config/ORBvoc.dbow3）
    double min_score             = 0.3;     // 词袋候选最低分
    int    temporal_window       = 30;      // 跳过最近 N 个关键帧（时间过滤）
    int    detection_interval    = 5;       // 每 N 个关键帧检测一次，覆盖短重访窗口
    double pnp_inlier_ratio      = 0.7;     // 几何验证最小内点比例
    int    min_loop_inliers      = 50;      // 与各 YAML 生产档一致
    int    loop_cooldown_kfs     = 20;      // 回环校正冷却（关键帧数）：防止同区域连续校正（D 曾改 12，实测同区域 43 KF 内连续回环叠加拉扯变形，回退）
    int    loop_top_candidates   = 20;      // 词袋召回池；聚类后与位置先验合计硬限 12 地点
    int    loop_mature_verification_limit = 0; // 0=验证全部候选；mobile 用有限成熟地点级联
    double loop_position_prior_dist = 40.0; // 位置先验距离阈值(m)：轨迹自交区域补召回（A3 放宽）
    int    loop_position_prior_gap   = 100; // 位置先验最小关键帧间隔
    LoopRegionConfig loop_region;            // 单 KF PnP 失败后的有限历史区域验证
    int    global_ba_iterations  = 20;      // 回环后全局 BA 迭代次数
    int    pose_graph_iterations = 20;      // Essential/Submap Graph 最大迭代数
    int    pose_graph_max_anchors = 256;    // 单子图 PGO 锚点硬上限
    int    pose_graph_anchor_stride = 8;    // 普通 KF 首选抽样步长（回环端点必留）

    // ---- 异步后端（P2-1）----
    bool   async_backend         = false;   // Local BA / 回环检测 / 回环校正在后台线程执行，
                                            // 前端只做跟踪不等待。struct 默认 false；config 默认
                                            // true（§3.16 覆盖式队列+跳过活动参考写回后净收益
                                            // 转正：FPS+23%/LOST-97%）；false = 全同步（旧行为）

    /// 从 yaml 配置加载（缺省字段保持默认值）
    static VOConfig fromYaml(const std::string& path);

    // ---- M2.2 遗留清理：地图资源预算（§6.2/§6.3）----
    MapBudgetConfig map_budget;   // 首版参数见 §6.2 MapBudget 段（default.yaml）
};

/// 单目两帧初始化的几何质量（归一化相机坐标）。
/// parallax_rad 是两帧光线夹角的中位数，不是 recoverPose 的平移范数。
/// positive_depth_ratio 是三角化点在两帧中均为正深度的比例。
struct MonocularInitializationQuality {
    double parallax_rad = 0.0;
    double positive_depth_ratio = 0.0;
    int triangulated_points = 0;
    bool accepted = false;
};

/// 用真实三角化几何判断单目初始化是否退化。
/// normalized_ref/curr 为 K^{-1}[u,v,1] 的前两维，relative_rotation 和
/// relative_translation 满足 p_c2 = R p_c1 + t。平移尺度不可观测，因此
/// 不能把 ||t|| 当作视差；近零基线会表现为光线夹角接近零。
[[nodiscard]] MonocularInitializationQuality assessMonocularInitialization(
    const std::vector<Vec2>& normalized_ref,
    const std::vector<Vec2>& normalized_curr,
    const Mat33& relative_rotation,
    const Vec3& relative_translation,
    double min_parallax_rad,
    double min_positive_depth_ratio = 0.7);

/// 后端在一帧跟踪期间校正参考 KF 时，保持该帧相对参考帧的 T_ca 不变，
/// 把当前帧从旧局部几何重基到新参考位姿。
[[nodiscard]] SE3 rebaseAnchoredFramePose(
    const SE3& frame_pose_cs,
    const SE3& old_ref_pose_cs,
    const SE3& new_ref_pose_cs);

/// Local BA 改写锚点时重基持久轨迹的相对位姿，使已发布的世界位姿不变：
/// T_ca_new * anchor_new == T_ca_old * anchor_old。
[[nodiscard]] SE3 rebaseTrajectoryAnchor(
    const SE3& T_ca_old,
    const SE3& old_anchor_pose_cs,
    const SE3& new_anchor_pose_cs);

/// Atlas 刚体锚从 old_T_ws 校正到 new_T_ws 时，按 alpha∈[0,1] 将校正
/// 分布到子地图内部轨迹。alpha=0 保持滚动边界世界位姿，alpha=1 接受
/// 完整新锚校正，避免整张子图平移后在连接处产生单帧跳变。
[[nodiscard]] SE3 rebaseTrajectoryForSubmapAnchor(
    const SE3& T_ca_old,
    const SE3& anchor_pose_cs,
    const SE3& old_T_ws,
    const SE3& new_T_ws,
    double alpha);

/// 连续激活段内的渐进校正比例；旧激活段始终为 0，避免重激活同一子图
/// 时把多段不连续访问误当作一条时间轴。
[[nodiscard]] double submapTrajectoryCorrectionAlpha(
    unsigned long frame_id,
    unsigned long segment_start_frame_id,
    unsigned long first_frame_id,
    unsigned long endpoint_frame_id);

/// 用记录内的相对位姿和锚点快照组合世界位姿。live KF 被预算剔除时，
/// 调用方传入 FramePoseRecord::anchor_pose_cs 仍可得到原轨迹而非单位位姿。
[[nodiscard]] SE3 composeAnchoredWorldPose(
    const SE3& T_ca,
    const SE3& anchor_pose_cs,
    const SE3& T_ws);

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
    ~VisualOdometry();  // 异步后端：join 后台线程（P2-1）

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

    /// 获取轨迹尾部，供实时 Viewer 避免每帧组合完整历史。
    std::vector<Vec3> getTrajectory(size_t max_points) const;

    /// 收集全部子地图地图点在世界系中的坐标（p_w = T_ws · p_s），供实时
    /// 3D Viewer 绘制点云。max_points 为可视化硬上限；总点超出时对点集
    /// 均匀抽样，保证新旧子地图的点都可见。调用方无需持锁。
    std::vector<Vec3> getMapPointsWorld(size_t max_points) const;

    /// 获取当前帧双目匹配后的有效点，并从当前左目图像采样 RGB 颜色。
    /// 点从相机系转换到世界系；单目/无有效 pts_c 时返回空向量。
    std::vector<ColoredPoint> getCurrentStereoPointCloud(size_t max_points) const;

    /// M4：轨迹记录——普通帧不再保存全局 T_cw，只记录相对锚定关键帧的
    /// 局部运动；世界位姿读时组合 T_cw = T_ca · (anchor pose_cs · T_ws⁻¹)。
    /// 回环/子地图对齐只更新锚点（KF 局部位姿 / T_ws），轨迹自动跟随，
    /// 无需全量插值重写（删除 §3.16 的轨迹重写路径）。
    struct FramePoseRecord {
        unsigned long frame_id = 0;     // 帧号（轨迹帧号与评估对齐）
        unsigned long submap_id = 0;    // 所属子地图（T_ws 查找）
        unsigned long anchor_kf_id = 0; // 锚定关键帧（关键帧自身 = 自己）
        SE3 T_ca;                       // 相对锚定关键帧的局部运动（T_ca：anchor→camera）
        SE3 anchor_pose_cs;             // 最近事务同步的锚点位姿；KF 被预算剔除后的兜底
        bool valid = false;
    };

    /// 获取有效帧的完整位姿轨迹（世界系 T_cw，由锚定记录组合）
    std::vector<SE3> getPoseTrajectory() const {
        return composePoseTrajectory();
    }

    /// 是否应该创建新的关键帧（点预算耗尽仍允许 KF 提议；只有 KF/描述子/
    /// 快照/总量预算真正阻塞时才返回 false）。
    bool needNewKeyFrame();

    /// 设置特征方法（0: ORB匹配, 1: LK光流）
    void setFeatureMethod(int method) { cfg_.feature_method = method; }

    /// (Phase 2) 启用回环检测：配置参数 + 加载词袋词典。
    /// 返回 false 表示词典加载失败（回环保持关闭，VO 不受影响）。
    bool enableLoopClosure(const std::string& vocab_path);

    /// (Phase 2) 回环是否已启用（词典加载成功）
    bool loopClosureEnabled() const { return loop_closure_enabled_; }

    /// (Phase 2) 已闭合的回环次数（状态栏/评估用）
    unsigned long loopClosureCount() const { return loop_closure_count_; }

    /// 连续 odom 相机位姿 T_oc；只由前端相邻运动积分，回环不得改写。
    [[nodiscard]] SE3 continuousCameraPose() const;
    /// 全局校正 T_wo，满足 T_wc = T_wo * T_oc。
    [[nodiscard]] SE3 globalCorrection() const;
    /// T_wo 结构化发布代次（初始坐标建立为 0，显著全局修正后递增）。
    [[nodiscard]] uint64_t globalCorrectionGeneration() const {
        return global_correction_generation_.load(std::memory_order_relaxed);
    }

    /// M2.3：后台调度器只读统计（§6.4 backend 指标；Localizer 指标采集用）
    [[nodiscard]] BackendSchedulerStats backendStats() const {
        return backend_scheduler_.stats();
    }

    /// M2.3 遗留清理：在途 Local BA 快照字节估算（§6.4 map_snapshot_bytes；
    /// 后台 Local BA 提交后更新；0 = 尚无快照）
    [[nodiscard]] long long mapSnapshotBytes() const {
        return map_snapshot_bytes_.load(std::memory_order_relaxed);
    }

    /// 当前活动 Map 的预算快照；调用方无需自行绕过 map_mutex_。
    [[nodiscard]] BudgetStatus mapBudgetStatus() const;

    /// 进程级资源观测（RSS、线程数、允许 CPU 数）。
    [[nodiscard]] RuntimeResourceSnapshot runtimeResourceSnapshot() const {
        return RuntimeResources::snapshot();
    }

    /// M2.3 遗留清理：§6.3 第 6 步——预算耗尽或点配额已满是否停止增加点
    /// （Localizer 据此上报 Degraded；不等价于停止关键帧）
    [[nodiscard]] bool mapGrowthStopped() const {
        return map_growth_stopped_.load(std::memory_order_relaxed);
    }

    /// M2：前端只读快照（§14.1-6）——每帧开头捕获一次，整帧使用同一版本数据。
    /// 跟踪不再在途中读实时地图点坐标（避免跨版本/部分提交观测）。
    struct TrackingSnapshot {
        Map::Ptr map;                        // 捕获时的 Map 实例身份
        uint64_t topology_revision = 0;   // 捕获时的地图版本（诊断/审计用）
        uint64_t geometry_revision = 0;
        unsigned long submap_id = 0;
        SE3 T_ws;                         // 活动子地图锚（M3：世界组合唯一权威）
        bool has_ref = false;             // 是否有参考帧
        unsigned long ref_kf_id = 0;      // 参考关键帧 id（M4：轨迹锚点）
        SE3 ref_pose_cs;                  // 参考帧位姿（子地图局部系 T_cs）
        std::vector<std::shared_ptr<MapPoint>> ref_mps;  // 参考帧 map_points（索引对齐）
        std::vector<Vec3> ref_points_s;   // 对应局部坐标拷贝（版本绑定）
        // 方案 B：共视图局部地图（参考 KF 及其共视 KF 的地图点快照，索引对齐）。
        // 只在参考 KF 变化时由 captureTrackingSnapshot 刷新一次（covisibleKeyframes
        // 是 O(KF²) 全量扫描，不能每帧都做）；锁内拷贝保证与几何版本一致。
        std::vector<Vec3> local_points_s;
        std::vector<cv::Mat> local_descs;
        std::vector<std::shared_ptr<MapPoint>> local_mps;
        unsigned long local_map_kf_id = 0; // 局部地图对应的参考 KF id（缓存判据）
    };

    /// 等待异步后端队列排空并停止线程。批量评估必须先调用，再读取最终轨迹/统计。
    void finishPendingBackendWork();

    /// M1.1：位姿验收已迁移至 PoseGate（pose_gate.h）。保留 acceptPose 作
    /// 为 VO 侧的决策入口（计算运动基线后转调 PoseGate::acceptPoseCandidate）。

private:
    void updateStatus(int matches, int inliers, double parallax);
    SE3 addFrameImpl(const cv::Mat& left, const cv::Mat& right, double timestamp);
    /// M0：统一位姿验收（正常跟踪 reloc_mode=false / 重定位 reloc_mode=true）。
    /// 重定位基线 = 丢失期匀速外推的期望位姿（与 createSubmap 同一外推规则），
    /// 平移门限随外推位移放宽至 max(50m, 3×位移)、旋转门限 60°；正常跟踪基线 =
    /// 上一有效位姿（max_frame_translation/max_frame_rotation）。单目（尺度归一化，
    /// 位移无物理意义）或无上一有效位姿 → 跳过连续性，仅几何验收。
    bool acceptPose(const SE3& candidate_pose_world, int inliers, size_t total,
                    double rmse, bool reloc_mode, PoseQuality& quality,
                    int min_inliers_override = -1,
                    double min_ratio_override = -1.0,
                    double max_rmse_override = -1.0) const;
    /// M5：重定位运动基线（世界系 T_wc，丢失期匀速外推）。const：
    /// 只读 last_valid_pose_world_/per_frame_motion_ 等成员，供 const
    /// acceptPose 与 tryRelocalize 共用。
    SE3 relocBaselineWorld() const;
    /// M5：求解 Atlas 子地图约束图——节点 = 子地图 T_ws，边 = 约束
    /// （T_ws_b = T_ws_a · T_rel）；首个子地图固定锚定世界系。
    /// 成功时更新全部子地图 T_ws 并返回 true。调用方必须持有
    /// map_mutex_ 独占锁；本函数不获取该锁，也不允许锁内回调。
    bool solveAtlasConstraints();
    /// M1.1：跟踪/重定位共用的运动连续性已迁移至 PoseGate::checkMotionContinuity。
    /// 双目/RGB-D：为当前帧的每个特征点计算相机系 3D 观测 pts_c
    /// （M1.4：转调 FrontendTracker::computeStereoDepths 后应用深度统计）
    void computeStereoDepths(const cv::Mat& left_gray = cv::Mat(),
                             const cv::Mat& right_gray = cv::Mat());
    /// M1.4：正常跟踪的运动基线（世界系 T_wc = 上一有效位姿，门限
    /// max_frame_translation/rotation），与 acceptPose 的正常跟踪基线同一规则；
    /// 单目/无上一有效位姿 → 空基线（跳过连续性）。
    [[nodiscard]] MotionBaseline normalMotionBaseline() const;
    /// 双目/RGB-D：从当前帧 pts_c 直接创建地图点（单帧绝对尺度建点）
    void createMapPointsFromStereo(const Frame::Ptr& frame);
    /// M4：组合锚定轨迹为世界系 T_cw（getPoseTrajectory 的实现；
    /// 无并发时（评估）也可直接使用）
    std::vector<SE3> composePoseTrajectory() const;
    /// 前端安全点同步 Atlas 对活动子地图的锚点校正。后台只允许改 Atlas；
    /// snap_/last_valid_pose_world_ 由前端线程在 map_mutex_ 下重基，避免异步
    /// 回环把旧局部位姿和新世界锚混用。调用方须持 map_mutex_ 读/写锁。
    void syncFrontendAnchor(SubmapId submap_id, const SE3& T_ws);
    /// 同子图全局 PGO 在两帧之间发布时，把上一有效世界位姿重基到新参考
    /// 几何；只改变全局基线，不改变 per_frame_motion_/连续 T_oc。
    void syncFrontendGeometry(const TrackingSnapshot& old_snapshot,
                              const TrackingSnapshot& new_snapshot);
    /// 组合单条锚定记录为世界位姿（调用方必须已持 map_mutex_ 读/写锁）
    SE3 composeRecordWorld(const FramePoseRecord& rec) const;
    bool tryInitialize();
    /// M1.4：ORB 跟踪转调 FrontendTracker::trackOrb（匹配 → PnP → 3D-3D →
    /// 对极回退 → RECOVERING），只负责应用 TrackingResult（位姿/关联/状态）
    SE3 trackFrame();
    /// LK 光流跟踪（feature_method=1）：基于索引对齐的 map_points 做 PnP，失败回退 ORB。
    /// M1.4：PnP 核心转调 FrontendTracker::trackPnP
    SE3 trackFrameLK();
    bool tryRelocalize();               // LOST 状态重定位
    void createSubmap();                // 长时间丢失后锚定全局位姿并新建子地图
    /// 子地图重建成功后与历史轨迹做 Umeyama 刚体对齐（双目），
    /// 消除丢失期位移未知导致的锚点残留偏差。延迟到子地图有 ≥3 个
    /// 关键帧时在 insertKeyFrame 中触发（重建瞬间 KF 数不足无法拟合）
    void alignSubmapToTrajectory();
    bool insertKeyFrame();              // 关键帧插入；帧内几何过期时重基并跳过
    /// (Phase 2) 回环事务：独占 map 锁内重新验证、快照、位姿图/全局 BA 与原子提交。
    /// 低频回环会在该事务期间阻塞前端 KF/Local BA 写入，避免阶段间 stale 饥饿。
    /// 返回 true 仅表示该候选已成功提交；false 时调用方应继续尝试后续候选。
    bool handleLoopCorrection(const LoopCorrectionContext& context);
    /// 按共视地图点数选取 Local BA 窗口（含当前关键帧，最早帧锚定）
    std::vector<Frame::Ptr> selectLocalWindow(int n) const;
    void triangulateNewPoints(const Frame::Ptr& f1, const Frame::Ptr& f2,
                              const std::vector<cv::DMatch>& matches);

    // ---- 异步后端（P2-1）----
    /// M1.3：后台任务生命周期与优先级已迁移至 BackendScheduler（backend_scheduler.h，
    /// §5.4）。本函数是调度器 worker 的任务分发入口（LocalBA / LoopClosure）。
    void runBackendTask(BackendTask& task);
    /// 提交任务到后台调度器（覆盖式单任务槽，永不阻塞）
    void submitBackendTask(BackendTask task);
    /// 资源边缘化只在 Map 锁内合并待删 id；DBoW clear/rebuild 由后台 worker
    /// 锁外执行，避免索引重建卡住跟踪。同步档在 Map 锁释放后执行同一路径。
    void drainLoopKeyframeCleanup();

    // ---- M1：Optimizer 只读快照 / Result / 提交 ----
    /// 构建 Local BA 只读快照（调用方需持 map_mutex_ 读锁；窗口内已被
    /// 清理/地图重建的 KF 自动剔除）。不深拷贝任何实时对象。
    /// @param min_observed  只收集观测数 ≥ 该值的点；Local BA 与优化器统一用 3
    OptimizationSnapshot buildLocalBASnapshot(
        const std::vector<Frame::Ptr>& window,
        KeyframeId anchor_kf_id,
        int min_observed = 0) const;
    /// 执行窗口 Local BA（唯一入口）：正式弱观测过滤 → 快照 → 求解 →
    /// 提交（跳过活动参考帧）；失败结果不在同一窗口重复求解。
    void runWindowLocalBA(const std::vector<Frame::Ptr>& window,
                          KeyframeId anchor_kf_id,
                          const Map::Ptr& expected_map = nullptr,
                          std::optional<unsigned long> expected_submap_id = std::nullopt);
    /// 应用 Local BA 结果：唯一提交路径（BackendCommitter::commit，M2）——
    /// stale 检查 + 质量验收 + 锁内原子写回（跳过 skip_pose 活动参考帧）。
    /// 返回明确提交状态；stale/验收失败 → 不提交，实时状态逐项不变。
    CommitStatus applyLocalBAResult(
        const OptimizationResult& result,
        const std::unordered_set<unsigned long>& skip_pose,
        const Map::Ptr& expected_map,
        unsigned long expected_submap_id);

    // ---- M2：前端只读快照 ----
    /// 捕获前端只读快照（调用方需持 map_mutex_ 读锁；每帧开头一次）。
    /// 方案 B：局部地图按（参考 KF id, geometry revision）缓存，只在任一
    /// 变化时重新收集——covisibleKeyframes 是 O(KF²) 全量扫描，不能每帧做。
    TrackingSnapshot captureTrackingSnapshot();

    /// 快照一个关键帧：pose_cs 拷贝；keep_points 为空则全部 map_points 深拷贝，
    /// 否则只深拷贝 keep_points 中的点（其余置 nullptr，与 keypoints 索引对齐）。
    /// with_descriptors=false 用于 BA 快照：g2o 只消费关键点像素与点位姿，
    /// 不读描述子——全图回环快照省掉 ~70MB 描述子拷贝（P1，见 §3.16）。
    /// snap_cache 保证同一真点的多个引用复用同一快照点对象，避免 BA 精修值
    /// 被同 id 的其他快照点覆盖，导致写回后的点/位姿不一致。
    static Frame::Ptr snapshotFrame(
        const Frame::Ptr& kf,
        std::unordered_map<unsigned long, MapPoint::Ptr>& snap_cache,
        const std::unordered_set<unsigned long>* keep_points = nullptr,
        bool with_descriptors = true);
    /// 后台执行 Local BA（快照隔离，优化不触碰地图集合）
    void runBackendLocalBA(const BackendTask& task);
    /// 后台执行回环检测 → 验证 → 校正（候选来自词袋+位置先验）
    void runBackendLoopClosure(const Frame::Ptr& curr_kf);

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
    /// M1.2：重定位候选几何验证（tryRelocalize 转调；只返回结果，不提交）
    Relocalizer relocalizer_;
    /// M1.4：前端跟踪（ORB/LK/PnP/3D-3D/双目深度/关键帧提议；§5.5，
    /// 只输出 TrackingResult/KeyframeProposal，不写地图/不执行 BA/回环）
    FrontendTracker frontend_tracker_;
    /// M1.5：局部建图（关键帧/建点/正式观测/Local BA 快照/共视窗口；§5.5，
    /// 只通过 Map API 操作，不持有 VO 状态）
    LocalMapper local_mapper_;

    // 轨迹记录（M4）：锚定关键帧 + 局部运动，世界位姿读时组合。
    // 回环校正/子地图对齐只更新锚点，轨迹自动跟随，无需全量插值重写。
    std::vector<FramePoseRecord> pose_records_;
    std::vector<LoopEdge> odometry_edges_;   // KF 插入时冻结，不能从优化后位姿重算
    std::vector<LoopEdge> loop_edges_;       // 累积保留历史回环约束
    std::vector<cv::DMatch> last_tracking_matches_;
    KeyframeId last_tracking_ref_id_ = 0;
    KeyframeId last_tracking_curr_id_ = 0;

    // ---- 回环状态 ----
    std::unique_ptr<LoopClosure> loop_closure_;
    bool loop_closure_enabled_ = false;
    std::atomic<unsigned long> loop_closure_count_{0};  // 已闭合回环次数（前后端共享）
    /// 当前活动子地图累计插入 KF 序号；资源压缩只删容器元素，不回退该序号。
    std::atomic<unsigned long> active_keyframe_serial_{0};
    std::atomic<unsigned long> last_loop_keyframe_serial_{0};

    // 最近一次 Local BA 成功发布的几何版本。帧尾若发现本帧快照恰好被
    // 该提交跨越，则按 live reference 记录相对位姿，避免在途帧跟随局部
    // 精修发生跳变；位姿图/Atlas 校正不设置此标记，仍允许全局轨迹闭合。
    Map::Ptr last_local_ba_commit_map_;
    SubmapId last_local_ba_commit_submap_id_ = 0;
    uint64_t last_local_ba_commit_geometry_revision_ = 0;

    // ---- 异步后端同步原语 ----
    // P1：map_mutex_ 为读写锁。前端读点/位姿持共享锁（trackFrame/needNewKeyFrame 等），
    // 后端 BA 写回与前端 KF 插入持独占锁——读路径并发、写路径独占，缩短前端等待。
    // 写者饥饿由前端周期性独占锁（insertKeyFrame）天然消解。
    mutable std::shared_mutex map_mutex_;  // 保护地图集合/KF 位姿/点坐标/edges 的读写
    mutable std::mutex traj_mutex_;  // 保护 pose_records_（getter 为 const）
    mutable std::mutex output_pose_mutex_;  // 保护 T_oc/T_wo 发布快照
    /// M1.3：单后台线程 + 覆盖式单任务槽（§5.4）。锁序：scheduler 的槽锁绝不
    /// 与 map/traj 嵌套持锁（worker 在槽锁外执行任务分发）。
    BackendScheduler backend_scheduler_;

    unsigned long frame_count_ = 0;
    unsigned long last_kf_frame_id_ = 0;  // 上一个关键帧的帧号（关键帧冷却用）
    unsigned long active_trajectory_segment_start_frame_id_ = 0;
    // 当前 Atlas 激活段的首帧。重激活旧子图时，旧段轨迹保持已发布世界位姿，
    // 只让本次连续跟踪段从边界零校正渐进吸收新的跨图闭环。
    SE3 last_valid_pose_world_;
    bool has_last_valid_pose_ = false;
    SE3 per_frame_motion_;              // 相邻有效帧相对运动（Twc：X_cur = X_last·T_rel），
                                        // LOST 期匀速外推新子地图锚点用
    bool has_per_frame_motion_ = false;
    SE3 continuous_pose_oc_;            // camera -> continuous odom（控制连续输出）
    SE3 global_correction_T_wo_;        // odom -> world（回环/重定位全局校正）
    bool has_continuous_pose_ = false;
    uint64_t last_valid_geometry_revision_ = 0;
    std::atomic<uint64_t> global_correction_generation_{0};
    bool submap_needs_alignment_ = false;  // 子地图重建后待对齐标记（≥3 KF 时执行）
    int tracking_failures_ = 0;
    int relocalization_frames_ = 0;
    Status status_;  // 当前详细状态（初始化/跟踪/丢失信息）
    TrackingSnapshot snap_;  // 前端只读快照（M2，每帧开头捕获）
    // 方案 B：局部地图缓存（captureTrackingSnapshot 内按 ref KF + 几何版本刷新）
    unsigned long snap_local_map_kf_id_ = 0;
    uint64_t snap_local_map_geo_rev_ = std::numeric_limits<uint64_t>::max();
    std::vector<Vec3> snap_local_points_s_;
    std::vector<cv::Mat> snap_local_descs_;
    std::vector<std::shared_ptr<MapPoint>> snap_local_mps_;

    // ---- M2.2 遗留清理：§6.3 地图资源预算运行时接线 ----
    ResourceBudget map_budget_;   ///< 预算引擎（默认 §6.2 参数；VOConfig.map_budget 覆写）
    std::atomic<bool> map_growth_stopped_{false};   ///< 第 7 步：仍超预算停止建图
    std::atomic<long long> map_snapshot_bytes_{0};  ///< 在途 Local BA 快照字节估算
    std::mutex loop_cleanup_mutex_;
    std::vector<KeyframeId> pending_loop_cleanup_ids_;
    std::atomic<bool> loop_cleanup_pending_{false};
    /// 点预算已满时回收长期未命中的旧点，为后续 KF 让出逐点配额；
    /// 调用方必须持 map_mutex_ 独占锁。
    void reclaimStaleMapPoints();
    void enforceMapBudget();      ///< KF 插入后触发；调用方必须已持 map_mutex_ 独占锁
};

} // namespace vslam
