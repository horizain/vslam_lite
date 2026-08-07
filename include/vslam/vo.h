#pragma once

#include "vslam/common.h"
#include "vslam/camera.h"
#include "vslam/frame.h"
#include "vslam/map.h"
#include "vslam/atlas.h"
#include "vslam/feature.h"
#include "vslam/loop_closure.h"
#include "vslam/optimizer.h"
#include "vslam/backend_committer.h"
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

/// 异步后端任务（P2-1）：Local BA / 回环检测+校正，由后台线程执行。
struct BackendTask {
    enum class Type { LocalBA, LoopClosure };
    Type type = Type::LocalBA;
    std::vector<Frame::Ptr> window;   // LocalBA：窗口关键帧（快照在后台锁内构造）
    Frame::Ptr curr_kf;               // LoopClosure：当前关键帧
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
                                             // 跳过插入（关键帧稀疏化，压地图规模）
    int    keyframe_max_count      = 1800;  // 关键帧规模硬顶：超过后平移阈值放大 1.5 倍
                                             // （长序列/大场景防 KF 无限增长）
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
    int    loop_cooldown_kfs     = 20;      // 回环校正冷却（关键帧数）：防止同区域连续校正（D 曾改 12，实测同区域 43 KF 内连续回环叠加拉扯变形，回退）
    int    loop_top_candidates   = 20;      // 词袋查询候选数（Top-N，提高召回）
    double loop_position_prior_dist = 40.0; // 位置先验距离阈值(m)：轨迹自交区域补召回（A3 放宽）
    int    loop_position_prior_gap   = 100; // 位置先验最小关键帧间隔
    int    global_ba_iterations  = 20;      // 回环后全局 BA 迭代次数

    // ---- 异步后端（P2-1）----
    bool   async_backend         = false;   // Local BA / 回环检测 / 回环校正在后台线程执行，
                                            // 前端只做跟踪不等待。struct 默认 false；config 默认
                                            // true（§3.16 覆盖式队列+跳过活动参考写回后净收益
                                            // 转正：FPS+23%/LOST-97%）；false = 全同步（旧行为）

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

    /// M4：轨迹记录——普通帧不再保存全局 T_cw，只记录相对锚定关键帧的
    /// 局部运动；世界位姿读时组合 T_cw = T_ca · (anchor pose_cs · T_ws⁻¹)。
    /// 回环/子地图对齐只更新锚点（KF 局部位姿 / T_ws），轨迹自动跟随，
    /// 无需全量插值重写（删除 §3.16 的轨迹重写路径）。
    struct FramePoseRecord {
        unsigned long frame_id = 0;     // 帧号（轨迹帧号与评估对齐）
        unsigned long submap_id = 0;    // 所属子地图（T_ws 查找）
        unsigned long anchor_kf_id = 0; // 锚定关键帧（关键帧自身 = 自己）
        SE3 T_ca;                       // 相对锚定关键帧的局部运动（T_ca：anchor→camera）
        bool valid = false;
    };

    /// 获取有效帧的完整位姿轨迹（世界系 T_cw，由锚定记录组合）
    std::vector<SE3> getPoseTrajectory() const {
        return composePoseTrajectory();
    }

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

    /// M2：前端只读快照（§14.1-6）——每帧开头捕获一次，整帧使用同一版本数据。
    /// 跟踪不再在途中读实时地图点坐标（避免跨版本/部分提交观测）。
    struct TrackingSnapshot {
        uint64_t topology_revision = 0;   // 捕获时的地图版本（诊断/审计用）
        uint64_t geometry_revision = 0;
        unsigned long submap_id = 0;
        SE3 T_ws;                         // 活动子地图锚（M3：世界组合唯一权威）
        bool has_ref = false;             // 是否有参考帧
        unsigned long ref_kf_id = 0;      // 参考关键帧 id（M4：轨迹锚点）
        SE3 ref_pose_cs;                  // 参考帧位姿（子地图局部系 T_cs）
        std::vector<std::shared_ptr<MapPoint>> ref_mps;  // 参考帧 map_points（索引对齐）
        std::vector<Vec3> ref_points_s;   // 对应局部坐标拷贝（版本绑定）
    };

    /// 等待异步后端队列排空并停止线程。批量评估必须先调用，再读取最终轨迹/统计。
    void finishPendingBackendWork();

    /// M0：位姿验收结果（正常跟踪/重定位共用，供状态栏与日志）
    struct PoseQuality {
        double inlier_ratio = 0.0;
        double pose_rmse    = 0.0;
        double translation  = 0.0;   // 相对运动基线的平移 (m)
        double rotation     = 0.0;   // 相对运动基线的旋转 (rad)
        bool   geometric_ok = false; // 内点/比例/RMSE 达标
        bool   motion_ok    = false; // 连续性达标（无运动基线时恒 true）
    };

    /// M0 位姿验收纯函数：几何（内点/比例/RMSE）+ 连续性（相对运动基线平移/旋转）。
    /// baseline_twc 为空 → 跳过连续性（无运动模型场景，如初始化首帧/单目）。
    /// 验收失败由调用方保证实时状态完全不变（不写位姿、不换子地图）。
    [[nodiscard]] static bool acceptPoseCandidate(
        const SE3& candidate_pose_cs, int inliers, size_t total, double rmse,
        int min_inliers, double min_ratio, double max_rmse,
        const std::optional<SE3>& baseline_twc,
        double max_translation, double max_rotation,
        PoseQuality& quality);

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
    /// 成功时更新全部子地图 T_ws 并返回 true。
    bool solveAtlasConstraints();
    /// 跟踪/重定位共用：计算候选位姿相对运动基线的平移/旋转（角度），
    /// 并判定是否在 [max_translation, max_rotation] 门限内（validateMotion 的通用版）。
    static bool checkMotionContinuity(const SE3& candidate_pose_cs,
                                      const SE3& baseline_twc,
                                      double max_translation, double max_rotation,
                                      double& translation, double& rotation);
    /// 双目/RGB-D：为当前帧的每个特征点计算相机系 3D 观测 pts_c
    void computeStereoDepths();
    /// 双目/RGB-D：从当前帧 pts_c 直接创建地图点（单帧绝对尺度建点）
    void createMapPointsFromStereo(const Frame::Ptr& frame);
    /// M4：组合锚定轨迹为世界系 T_cw（getPoseTrajectory 的实现；
    /// 无并发时（评估）也可直接使用）
    std::vector<SE3> composePoseTrajectory() const;
    /// 组合单条锚定记录为世界位姿（调用方必须已持 map_mutex_ 读/写锁）
    SE3 composeRecordWorld(const FramePoseRecord& rec) const;
    bool tryInitialize();
    SE3 trackFrame();
    /// LK 光流跟踪（feature_method=1）：基于索引对齐的 map_points 做 PnP，失败回退 ORB
    SE3 trackFrameLK();
    /// 双目/RGB-D：3D-3D 位姿估计（ref 世界系点 vs 当前帧 pts_c，绝对尺度、旋转鲁棒）。
    /// PnP 失败或旋转-平移歧义（假平移跳变）时使用；成功返回 true 并写入 pose_cs
    bool tryTrack3D3D(const std::vector<cv::DMatch>& matches);
    /// 计算 PnP 内点的重投影 RMSE。
    double pnpReprojectionRmse(const std::vector<cv::Point3f>& pts3d,
                               const std::vector<cv::Point2f>& pts2d,
                               const cv::Mat& rvec, const cv::Mat& tvec,
                               const std::vector<int>& inliers) const;
    bool tryRelocalize();               // LOST 状态重定位
    void createSubmap();                // 长时间丢失后锚定全局位姿并新建子地图
    /// 子地图重建成功后与历史轨迹做 Umeyama 刚体对齐（双目），
    /// 消除丢失期位移未知导致的锚点残留偏差。延迟到子地图有 ≥3 个
    /// 关键帧时在 insertKeyFrame 中触发（重建瞬间 KF 数不足无法拟合）
    void alignSubmapToTrajectory();
    void insertKeyFrame();              // 关键帧插入 + 三角化 + BA
    /// (Phase 2) 回环校正：位姿图优化 + 地图点/逐帧轨迹同步 + 全局 BA
    void handleLoopCorrection(const SE3& T_loop_curr,
                              const Frame::Ptr& kf_curr, const Frame::Ptr& kf_loop);
    /// 按共视地图点数选取 Local BA 窗口（含当前关键帧，最早帧锚定）
    std::vector<Frame::Ptr> selectLocalWindow(int n) const;
    void triangulateNewPoints(const Frame::Ptr& f1, const Frame::Ptr& f2,
                              const std::vector<cv::DMatch>& matches);
    static SE3 matToSE3(const cv::Mat& R, const cv::Mat& t);  // cv::Mat → SE3

    // ---- 异步后端（P2-1）----
    /// 后台线程主循环：取任务 → 快照（锁内）→ 锁外优化 → 写回（锁内）
    void backendLoop();
    /// 提交任务到有界队列（满则调用方同步执行，防止任务无限堆积）
    void submitBackendTask(BackendTask task);
    void startBackend();
    void stopBackend();

    // ---- M1：Optimizer 只读快照 / Result / 提交 ----
    /// 构建 Local BA 只读快照（调用方需持 map_mutex_ 读锁；窗口内已被
    /// 清理/地图重建的 KF 自动剔除）。不深拷贝任何实时对象。
    /// @param min_observed  只收集观测数 ≥ 该值的点（C：弱观测垃圾点过滤，
    ///                       0 = 不过滤）
    OptimizationSnapshot buildLocalBASnapshot(
        const std::vector<Frame::Ptr>& window, int min_observed = 0) const;
    /// 执行窗口 Local BA（唯一入口）：快照 → 求解 → 提交（跳过活动参考帧）。
    /// C：提交被质量门限拒绝（大校正 = 弱观测点拖偏）时，剔除 observed<3
    /// 的点重建快照重试一次——垃圾段不再长期不被修复。
    void runWindowLocalBA(const std::vector<Frame::Ptr>& window);
    /// 应用 Local BA 结果：唯一提交路径（BackendCommitter::commit，M2）——
    /// stale 检查 + 质量验收 + 锁内原子写回（跳过 skip_pose 活动参考帧）。
    /// 返回是否提交成功；stale/验收失败 → 不提交，实时状态逐项不变。
    bool applyLocalBAResult(const OptimizationResult& result,
                            const std::unordered_set<unsigned long>& skip_pose);

    // ---- M2：前端只读快照 ----
    /// 捕获前端只读快照（调用方需持 map_mutex_ 读锁；每帧开头一次）
    TrackingSnapshot captureTrackingSnapshot() const;

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
    void runBackendLocalBA(const std::vector<Frame::Ptr>& window);
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

    // 轨迹记录（M4）：锚定关键帧 + 局部运动，世界位姿读时组合。
    // 回环校正/子地图对齐只更新锚点，轨迹自动跟随，无需全量插值重写。
    std::vector<FramePoseRecord> pose_records_;
    std::vector<LoopEdge> odometry_edges_;   // KF 插入时冻结，不能从优化后位姿重算
    std::vector<LoopEdge> loop_edges_;       // 累积保留历史回环约束

    // ---- 回环状态 ----
    std::unique_ptr<LoopClosure> loop_closure_;
    bool loop_closure_enabled_ = false;
    std::atomic<unsigned long> loop_closure_count_{0};  // 已闭合回环次数（前后端共享）
    std::atomic<unsigned long> last_loop_kf_count_{0};  // 上次回环校正时的 KF 数（冷却基准）

    // ---- 异步后端同步原语 ----
    // P1：map_mutex_ 为读写锁。前端读点/位姿持共享锁（trackFrame/needNewKeyFrame 等），
    // 后端 BA 写回与前端 KF 插入持独占锁——读路径并发、写路径独占，缩短前端等待。
    // 写者饥饿由前端周期性独占锁（insertKeyFrame）天然消解。
    mutable std::shared_mutex map_mutex_;  // 保护地图集合/KF 位姿/点坐标/edges 的读写
    mutable std::mutex traj_mutex_;  // 保护 pose_records_（getter 为 const）
    std::mutex backend_mutex_;   // 保护任务队列（锁顺序：backend → map/traj，无嵌套持锁）
    std::condition_variable backend_cv_;
    std::deque<BackendTask> backend_tasks_;
    std::thread backend_thread_;
    std::atomic<bool> backend_stop_{false};
    std::atomic<bool> backend_running_{false};

    unsigned long frame_count_ = 0;
    unsigned long last_kf_frame_id_ = 0;  // 上一个关键帧的帧号（关键帧冷却用）
    SE3 last_valid_pose_world_;
    bool has_last_valid_pose_ = false;
    SE3 per_frame_motion_;              // 相邻有效帧相对运动（Twc：X_cur = X_last·T_rel），
                                        // LOST 期匀速外推新子地图锚点用
    bool has_per_frame_motion_ = false;
    bool submap_needs_alignment_ = false;  // 子地图重建后待对齐标记（≥3 KF 时执行）
    int tracking_failures_ = 0;
    int relocalization_frames_ = 0;
    Status status_;  // 当前详细状态（初始化/跟踪/丢失信息）
    TrackingSnapshot snap_;  // 前端只读快照（M2，每帧开头捕获）
};

} // namespace vslam
