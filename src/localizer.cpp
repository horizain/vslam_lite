#include "vslam/localizer.h"
#include "vslam/resource_budget.h"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vslam {

// ============================================================
// LocalizerConfig
// ============================================================

LocalizerConfig LocalizerConfig::fromYaml(const std::string& path) {
    LocalizerConfig cfg;
    const YAML::Node root = YAML::LoadFile(path);
    if (!root["Robot"]) return cfg;

    const auto& robot = root["Robot"];
    if (robot["mode"]) {
        const std::string m = robot["mode"].as<std::string>();
        if (m == "OdometryOnly")      cfg.mode = LocalizationMode::OdometryOnly;
        else if (m == "Mapping")      cfg.mode = LocalizationMode::Mapping;
        else if (m == "LocalizationOnly") cfg.mode = LocalizationMode::LocalizationOnly;
        else if (m == "MapMaintenance")   cfg.mode = LocalizationMode::MapMaintenance;
    }
    if (robot["stereo_max_time_diff_s"]) {
        cfg.stereo_max_time_diff_s = robot["stereo_max_time_diff_s"].as<double>();
    }
    if (root["Robot"]["Runtime"]) {
        // M2.1（§6.1/§6.2）：异步输入 + 输入队列容量
        const auto& rt = root["Robot"]["Runtime"];
        if (rt["enable_async_input"])
            cfg.enable_async_input = rt["enable_async_input"].as<bool>();
        if (rt["input_queue_capacity"])
            cfg.input_queue_capacity = rt["input_queue_capacity"].as<int>();
        // M2.3（§6.2/§6.4）：跟踪硬期限与指标采集开关
        if (rt["tracking_deadline_ms"])
            cfg.tracking_deadline_ms = rt["tracking_deadline_ms"].as<long long>();
        if (rt["enable_metrics"])
            cfg.enable_metrics = rt["enable_metrics"].as<bool>();
    }
    if (robot["T_bc"]) {
        const auto& tbc = robot["T_bc"];
        if (tbc["translation"] && tbc["quaternion"] &&
            tbc["translation"].size() == 3 && tbc["quaternion"].size() == 4) {
            const auto& tr = tbc["translation"];
            const auto& qr = tbc["quaternion"];
            const Vec3 t(tr[0].as<double>(), tr[1].as<double>(), tr[2].as<double>());
            // 约定 xyzw；不做静默归一化，非单位四元数由构造校验拒绝（§4.3）
            Eigen::Quaterniond q(qr[3].as<double>(), qr[0].as<double>(),
                                 qr[1].as<double>(), qr[2].as<double>());
            cfg.T_bc = SE3(q, t);
        }
    }
    return cfg;
}

// ============================================================
// Localizer
// ============================================================

Localizer::Localizer(const Camera& camera, const VOConfig& vo_cfg,
                     const LocalizerConfig& cfg)
    : camera_(camera), vo_cfg_(vo_cfg), cfg_(cfg),
      input_queue_(static_cast<size_t>(cfg.input_queue_capacity)),
      metrics_(cfg.tracking_deadline_ms) {
    if (!camera_)
        throw std::invalid_argument("Localizer: camera is null");
    if (!isFinite(cfg_.T_bc) || !isUnitQuaternion(cfg_.T_bc.q))
        throw std::invalid_argument(
            "Localizer: T_bc must be finite with unit quaternion (norm error < 1e-6)");
    vo_ = std::make_unique<VisualOdometry>(camera_, vo_cfg_);
    // M2.1（§6.1）：异步模式启动跟踪 worker（传感器回调只入队）
    if (cfg_.enable_async_input) {
        worker_ = std::thread(&Localizer::trackingLoop, this);
        worker_started_.store(true);
    }
}

Localizer::~Localizer() { stop(); }

bool Localizer::validateInput(const cv::Mat& left, const cv::Mat& right,
                              double timestamp, double right_timestamp,
                              double last_ts, bool has_last_ts,
                              FailureReason& reason) const {
    if (!cfg_.enable_input_validation) {
        reason = FailureReason::None;
        return true;
    }
    // 时间戳严格递增；倒退或相等直接拒绝，不调用 VO（§4.3）
    if (!std::isfinite(timestamp)) {
        reason = FailureReason::InvalidInput;
        return false;
    }
    if (has_last_ts && timestamp <= last_ts) {
        reason = FailureReason::TimestampRollback;
        return false;
    }
    // 图像非空、与标定尺寸一致；不允许静默 resize（§4.3）
    if (left.empty()) {
        reason = FailureReason::InvalidInput;
        return false;
    }
    if (camera_->img_width > 0 &&
        (left.cols != camera_->img_width || left.rows != camera_->img_height)) {
        reason = FailureReason::InvalidInput;
        return false;
    }
    // 双目：右目非空时必须与左目尺寸、类型一致（§4.3）
    if (!right.empty()) {
        if (right.type() != left.type() || right.size() != left.size()) {
            reason = FailureReason::InvalidInput;
            return false;
        }
        // 双目时间差初值上限 1 ms（§4.3）；right_timestamp < 0 = 视为同步
        if (right_timestamp >= 0.0 &&
            std::abs(right_timestamp - timestamp) > cfg_.stereo_max_time_diff_s) {
            reason = FailureReason::StereoUnsynchronized;
            return false;
        }
    }
    reason = FailureReason::None;
    return true;
}

PoseEstimate Localizer::processFrame(const cv::Mat& left, const cv::Mat& right,
                                     double timestamp, double right_timestamp) {
    // M2.1：异步模式下 processFrame 不可用（会与 worker 并发驱动 VO）
    if (cfg_.enable_async_input) return stoppedOutput();
    if (stopped_) return stoppedOutput();

    FailureReason reason;
    if (!validateInput(left, right, timestamp, right_timestamp,
                       last_timestamp_, has_last_timestamp_, reason)) {
        const PoseEstimate out = rejectedOutput(timestamp, reason);
        // M2.3（§6.4）：拒绝帧也计入 pose_rejected 与对应 FailureReason
        if (cfg_.enable_metrics) metrics_.recordPose(out);
        return out;
    }
    return processValidFrame(left, right, timestamp);
}

PoseEstimate Localizer::processFrame(const cv::Mat& image, double timestamp) {
    return processFrame(image, cv::Mat(), timestamp);
}

bool Localizer::submitFrame(const cv::Mat& left, const cv::Mat& right,
                            double timestamp, double right_timestamp) {
    // M2.1（§6.1）：传感器回调只校验并入队，不运行 ORB/PnP，永不阻塞。
    if (!cfg_.enable_async_input || stopped_) return false;
    FailureReason reason;
    // 提交侧按"已提交的最大时间戳"校验（与 worker 处理进度解耦）
    if (!validateInput(left, right, timestamp, right_timestamp,
                       last_submitted_timestamp_, has_last_submitted_, reason))
        return false;
    SensorPacket pkt;
    pkt.sequence = ++packet_seq_;
    pkt.timestamp = timestamp;
    pkt.right_timestamp = right_timestamp;
    pkt.left = left;
    pkt.right = right;
    input_queue_.push(std::move(pkt));  // 满丢最旧、保最新（永不阻塞）
    last_submitted_timestamp_ = timestamp;
    has_last_submitted_ = true;
    return true;
}

PoseEstimate Localizer::latestPose() const {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return latest_estimate_;
}

size_t Localizer::inputQueueHighWaterMark() const {
    return input_queue_.highWaterMark();
}

size_t Localizer::inputQueueDroppedCount() const {
    return input_queue_.droppedCount();
}

void Localizer::trackingLoop() {
    // M1 确定性：OpenCV RNG 是线程局部的——异步 worker 线程必须按配置
    // 单独播种（与 VisualOdometry 构造函数同一规则；rng_seed=0 保持默认）
    if (vo_cfg_.rng_seed != 0)
        cv::setRNGSeed(static_cast<unsigned>(vo_cfg_.rng_seed));
    // M2.1（§6.1）：跟踪 worker——阻塞在队列上（空闲不自旋），
    // 消费按 FIFO；stop 后排空剩余帧（丢最旧语义保证均为最新帧）再退出。
    while (true) {
        SensorPacket pkt;
        if (!input_queue_.pop(pkt)) break;
        processValidFrame(pkt.left, pkt.right, pkt.timestamp);
    }
}

PoseEstimate Localizer::processValidFrame(const cv::Mat& left, const cv::Mat& right,
                                          double timestamp) {
    const uint64_t seq = ++sequence_;
    const double dt = has_last_timestamp_ ? std::max(0.0, timestamp - last_timestamp_) : 0.0;

    // M2.3（§6.4）：单帧跟踪延迟（含输入校验后的 VO 处理；异步模式为
    // worker 内处理耗时，排队等待不计入）
    const auto t0 = std::chrono::steady_clock::now();
    // 第一阶段直接委托 VO（§4.1：不修改算法）
    const SE3 T_cw = right.empty()
        ? vo_->addFrame(left, timestamp)
        : vo_->addFrame(left, right, timestamp);
    const VisualOdometry::Status st = vo_->getStatus();
    const double frame_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // 只对通过验收的帧推进时间戳（§4.3：拒绝帧不污染后续比较）
    last_timestamp_ = timestamp;
    has_last_timestamp_ = true;

    // M0 质量映射：pose_valid → Full，否则 Failed；Weak 细分留给 M3 质量门。
    FrameQuality q = st.pose_valid ? FrameQuality::Full : FrameQuality::Failed;
    // M2.2 遗留清理（§6.3 第 6 步）：预算耗尽停止建图期间，跟踪质量压为
    // Weak——状态机连续 2 帧后进入 Degraded；reason 上报 BackendOverloaded。
    const bool budget_stopped = vo_->mapGrowthStopped();
    if (q == FrameQuality::Full && budget_stopped) q = FrameQuality::Weak;

    PoseEstimate out;
    out.sequence = seq;
    out.timestamp = timestamp;
    const SE3 T_wc = T_cw.inverse();
    out.T_wb = T_wc * cfg_.T_bc.inverse();  // §2：T_wb = T_wc · T_bc⁻¹
    out.T_ob = out.T_wb;                    // M0：odom 系 = 全局系，M6 后由 T_wo 分离
    // M2.1：状态机写入 + 结果发布在 result_mutex_ 内（异步模式 worker 写、
    // 调用方 state()/latestPose() 并发读；同步模式单线程无竞争）
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        const StateMachineOutput sm_out = state_machine_.on_tracking(q, dt, seq);
        out.covariance = Mat6::Identity();  // M0 占位；M3 数值 Jacobian 替换
        if (sm_out.quality == FrameQuality::Weak) out.covariance *= 4.0;  // §4.2 弱质量 ×4
        out.state = sm_out.state;
        out.reason = sm_out.reason;
        // §6.3 第 6 步（M2.2 遗留清理）：预算停止建图期间计入 BackendOverloaded
        // （reason 覆盖：质量压弱已使状态进入 Degraded，原因码由预算决定）
        if (budget_stopped && out.pose_valid) out.reason = FailureReason::BackendOverloaded;
        out.pose_valid = sm_out.pose_valid;
        out.prediction_only = sm_out.prediction_only;
        latest_estimate_ = out;
    }
    out.map_generation = vo_->getMap()->topologyRevision();
    feedFrameMetrics(out, frame_ms);
    return out;
}

void Localizer::feedFrameMetrics(const PoseEstimate& out, double frame_ms) {
    if (!cfg_.enable_metrics) return;
    // §6.4（M2 遗留清理）：frame latency/deadline 只统计跟踪帧——重定位/
    // LOST 期间的单帧耗时是恢复过程（词袋查询/PnP 候选验证），不是跟踪
    // 延迟，不应计入 §6.5 的 p99 < 图像周期×0.8 门限。frames_processed
    // 同步定义为跟踪帧数（valid_ratio 分母一致）。
    const bool tracking_frame =
        out.state == TrackingState::Initializing ||
        out.state == TrackingState::Tracking ||
        out.state == TrackingState::Degraded;
    if (tracking_frame) metrics_.recordFrameLatency(frame_ms);
    const VisualOdometry::Status st = vo_->getStatus();
    const double features = vo_->currentFrame()
        ? static_cast<double>(vo_->currentFrame()->keypoints.size()) : -1.0;
    metrics_.recordTracking(features, static_cast<double>(st.stereo_points),
                            static_cast<double>(st.inliers), st.inlier_ratio,
                            st.pose_rmse);
    metrics_.recordPose(out);
    if (has_last_metric_state_) {
        metrics_.recordStateChange(last_metric_state_, out.state, out.timestamp);
    }
    last_metric_state_ = out.state;
    has_last_metric_state_ = true;
    // 输入计数：同步 = 每帧恰好 1 收 1 处理；异步 = 提交总数含队列丢弃
    if (cfg_.enable_async_input) {
        metrics_.recordInput(static_cast<long long>(packet_seq_),
                             static_cast<long long>(input_queue_.droppedCount()),
                             static_cast<long long>(input_queue_.highWaterMark()));
    } else {
        metrics_.recordInput(static_cast<long long>(sequence_), 0, 0);
    }
}

void Localizer::feedFinalMetrics() {
    if (!cfg_.enable_metrics || !vo_) return;
    metrics_.recordBackend(vo_->backendStats());
    metrics_.recordLoopCommitted(static_cast<long long>(vo_->loopClosureCount()));
    const Map::Ptr map = vo_->getMap();
    long long observations = 0;
    for (const auto& mp : map->getAllMapPoints())
        observations += static_cast<long long>(mp->observationCount());
    metrics_.recordMap(static_cast<long long>(map->keyFrameCount()),
                       static_cast<long long>(map->mapPointCount()), observations,
                       static_cast<long long>(ResourceBudget::descriptorBytes(map)),
                       static_cast<long long>(ResourceBudget::imageBytes(map)),
                       vo_->mapSnapshotBytes(),
                       static_cast<long long>(
                           ResourceBudget{}.evaluate(map).estimated_total_bytes));
}

PoseEstimate Localizer::rejectedOutput(double timestamp, FailureReason reason) const {
    PoseEstimate out;
    out.timestamp = timestamp;
    out.state = state_machine_.state();
    out.reason = reason;
    out.pose_valid = false;
    out.map_generation = vo_ ? vo_->getMap()->topologyRevision() : 0;
    return out;
}

PoseEstimate Localizer::stoppedOutput() const {
    PoseEstimate out;
    out.state = TrackingState::Stopped;
    out.pose_valid = false;
    out.reason = FailureReason::None;
    return out;
}

void Localizer::stop() {
    if (stopped_) return;
    stopped_ = true;
    // M2.1：先停输入队列并 join 跟踪 worker（§6.2 shutdown_timeout_ms=2000 内），
    // 再排空 VO 后台任务；锁序：worker 在队列锁外执行 VO，无嵌套持锁。
    input_queue_.stop();
    if (worker_started_.load() && worker_.joinable()) worker_.join();
    if (vo_) vo_->finishPendingBackendWork();
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        state_machine_.stop();
    }
    // M2.3（§6.4）：收尾 backend/loop/map 最终统计（含 LOST 段收口）
    if (cfg_.enable_metrics && has_last_metric_state_) {
        metrics_.recordStateChange(last_metric_state_, TrackingState::Stopped,
                                   last_timestamp_);
    }
    feedFinalMetrics();
}

TrackingState Localizer::state() const {
    // M2.1：异步模式与 worker 并发读 → 与状态机写入共用 result_mutex_
    std::lock_guard<std::mutex> lock(result_mutex_);
    return state_machine_.state();
}

uint64_t Localizer::mapTopologyRevision() const {
    return vo_ ? vo_->getMap()->topologyRevision() : 0;
}

uint64_t Localizer::mapGeometryRevision() const {
    return vo_ ? vo_->getMap()->geometryRevision() : 0;
}

size_t Localizer::mapPointCount() const {
    return vo_ ? vo_->getMap()->mapPointCount() : 0;
}

size_t Localizer::keyFrameCount() const {
    return vo_ ? vo_->getMap()->keyFrameCount() : 0;
}

MetricsSnapshot Localizer::metricsSnapshot() const {
    // M2.3（§6.4）：backend/loop/map 统计在快照时现算（collector 只存值；
    // metrics_ 为 mutable：const 快照可刷新）。snapshot_bytes 由调用方
    // 经 collector 上报（M2.3 未接，保持 -1）。
    if (cfg_.enable_metrics && vo_) {
        metrics_.recordBackend(vo_->backendStats());
        metrics_.recordLoopCommitted(static_cast<long long>(vo_->loopClosureCount()));
        const Map::Ptr map = vo_->getMap();
        long long observations = 0;
        for (const auto& mp : map->getAllMapPoints())
            observations += static_cast<long long>(mp->observationCount());
        metrics_.recordMap(static_cast<long long>(map->keyFrameCount()),
                           static_cast<long long>(map->mapPointCount()), observations,
                           static_cast<long long>(ResourceBudget::descriptorBytes(map)),
                           static_cast<long long>(ResourceBudget::imageBytes(map)),
                           vo_->mapSnapshotBytes(),
                           static_cast<long long>(
                               ResourceBudget{}.evaluate(map).estimated_total_bytes));
    }
    return metrics_.snapshot();
}

void Localizer::writeMetricsJson(const std::string& path) const {
    if (!cfg_.enable_metrics) return;
    (void)metricsSnapshot();  // 刷新 backend/loop/map 后输出
    metrics_.writeJson(path);
}

void Localizer::writeMetricsCsv(const std::string& path) const {
    if (!cfg_.enable_metrics) return;
    (void)metricsSnapshot();
    metrics_.writeCsv(path);
}

} // namespace vslam
