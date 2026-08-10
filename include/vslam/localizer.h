#pragma once

#include "vslam/bounded_queue.h"
#include "vslam/localization_types.h"
#include "vslam/metrics.h"
#include "vslam/sensor_packet.h"
#include "vslam/tracking_state_machine.h"
#include "vslam/vo.h"
#include "vslam/camera.h"

#include <opencv2/core.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace vslam {

/// Localizer Facade 配置（M0.3，§4；M2.1 增补 §6.1/§6.2 异步输入）
struct LocalizerConfig {
    LocalizationMode mode = LocalizationMode::LocalizationOnly;  ///< 运行模式（§1.3）
    SE3 T_bc = SE3();                  ///< 相机 -> 基座 固定标定（§2/§4.3，不允许回环跳变）
    double stereo_max_time_diff_s = 0.001;  ///< §4.3 双目时间差初值上限 1 ms
    bool enable_input_validation = true;    ///< §4.3 输入硬检查开关（测试可关闭）

    // ---- M2.1 异步输入（§6.1/§6.2）----
    bool enable_async_input = false;   ///< 传感器回调只校验并入队（跟踪 worker 消费）；
                                       ///< false = 保持 M0 同步路径（processFrame）
    int input_queue_capacity = 3;      ///< 输入队列固定容量（§6.2 首版参数）

    // ---- M2.3 结构化指标（§6.4）----
    bool enable_metrics = true;        ///< 内部 MetricsCollector 采集开关
    long long tracking_deadline_ms = 80;  ///< 单帧跟踪硬期限（§6.2：10 Hz 平台 80 ms）

    /// 从 robot.yaml 的 Robot 段加载（缺省字段保持默认值；T_bc 不做静默归一化，
    /// 非单位四元数由 Localizer 构造时按 §4.3 拒绝）
    static LocalizerConfig fromYaml(const std::string& path);
};

/// 定位服务 Facade（M0.3）：包装 VisualOdometry + TrackingStateMachine + 输入硬检查。
///
/// 第一阶段不修改 VO 算法；调用方不再直接读取 Frame/Map/VisualOdometry 内部状态。
/// 同步模式（默认，M0）：processFrame 在调用方线程内同步驱动（§4.4：M0 不增加线程）。
/// 异步模式（M2.1，§6.1）：enable_async_input=true 时启动一个跟踪 worker；
/// submitFrame 只做 §4.3 输入校验并入队（满丢最旧、保最新、永不阻塞），
/// 结果经 latestPose() 轮询获取。两种模式不得混用。
/// 坐标契约（§2）：输出 T_wb = T_wc · T_bc⁻¹；M0 尚未接入 ESKF，T_ob = T_wb
/// （odom 系与全局系重合），M6 后再由 T_wo 分离，届时回环只更新 T_wo。
class Localizer {
public:
    /// @param camera  双目/单目相机（内参、图像尺寸即 §4.3 输入尺寸契约）
    /// @param vo_cfg  VO 参数（沿用 config/default.yaml）
    /// @param cfg     Localizer 参数（robot.yaml）
    /// 构造时校验 T_bc：四元数归一化误差 <1e-6、平移有限（§4.3），否则抛
    /// std::invalid_argument。异步模式构造即启动跟踪 worker。
    Localizer(const Camera& camera, const VOConfig& vo_cfg,
              const LocalizerConfig& cfg = LocalizerConfig());
    ~Localizer();

    Localizer(const Localizer&) = delete;
    Localizer& operator=(const Localizer&) = delete;

    /// 处理一帧双目图像（同步模式；right 为空 = 单目路径）。
    /// right_timestamp < 0 表示未提供右目时间戳（视为与左目同步，跳过 1ms 检查）。
    /// §4.3 输入硬检查失败时直接拒绝，不调用 VO，Map revision 逐项不变。
    /// 异步模式下不可用（返回无效输出；请使用 submitFrame）。
    PoseEstimate processFrame(const cv::Mat& left, const cv::Mat& right,
                              double timestamp, double right_timestamp = -1.0);

    /// 处理一帧单目图像（同步模式；§4.3 输入硬检查同上）
    PoseEstimate processFrame(const cv::Mat& image, double timestamp);

    /// 异步模式（§6.1）：传感器回调入口——只做 §4.3 输入校验并入队，
    /// 不运行 ORB/PnP，永不阻塞（队列满丢最旧帧）。返回 true = 已入队。
    /// 同步模式下恒返回 false。
    bool submitFrame(const cv::Mat& left, const cv::Mat& right,
                     double timestamp, double right_timestamp = -1.0);

    /// 异步模式：最近一次跟踪结果（worker 写入；轮询用）。同步模式下
    /// 返回最后一次 processFrame 的结果。
    [[nodiscard]] PoseEstimate latestPose() const;

    /// §6.4：输入队列深度峰值（<= input_queue_capacity）
    [[nodiscard]] size_t inputQueueHighWaterMark() const;
    /// §6.4：输入队列因满而丢弃的帧数
    [[nodiscard]] size_t inputQueueDroppedCount() const;

    /// 停止定位服务：异步模式先停队列、join 跟踪 worker（§6.2
    /// shutdown_timeout_ms=2000 内完成），再排空 VO 后台任务并停止状态机。
    /// 幂等，可重复调用（§4.4：重复 stop 不崩溃）。
    void stop();

    /// 当前状态机状态（线程安全；异步模式可与 worker 并发调用）
    [[nodiscard]] TrackingState state() const;

    /// 地图 revision / 规模只读观测（测试与监控断言：非法输入不改 Map；
    /// 异步模式下为原子计数诊断读）
    [[nodiscard]] uint64_t mapTopologyRevision() const;
    [[nodiscard]] uint64_t mapGeometryRevision() const;
    [[nodiscard]] size_t mapPointCount() const;
    [[nodiscard]] size_t keyFrameCount() const;

    // ---- M2.3 结构化指标（§6.4）----
    /// §6.4：指标快照（线程安全；含实时 map 字节/规模统计与 backend 计数）
    [[nodiscard]] MetricsSnapshot metricsSnapshot() const;
    /// §6.4：输出指标 JSON/CSV（soak_test.py / nightly 消费；禁止正则解析日志）
    void writeMetricsJson(const std::string& path) const;
    void writeMetricsCsv(const std::string& path) const;

private:
    bool validateInput(const cv::Mat& left, const cv::Mat& right,
                       double timestamp, double right_timestamp,
                       double last_ts, bool has_last_ts,
                       FailureReason& reason) const;
    PoseEstimate processValidFrame(const cv::Mat& left, const cv::Mat& right,
                                   double timestamp);
    PoseEstimate rejectedOutput(double timestamp, FailureReason reason) const;
    PoseEstimate stoppedOutput() const;
    void trackingLoop();  // 异步模式跟踪 worker 主循环
    /// M2.3：喂入每帧指标（latency/tracking/pose/state change/input 计数）
    void feedFrameMetrics(const PoseEstimate& out, double frame_ms);
    /// M2.3：喂入后端/回环/地图最终统计（stop 与快照时调用）
    void feedFinalMetrics();

    Camera camera_;
    VOConfig vo_cfg_;
    LocalizerConfig cfg_;
    std::unique_ptr<VisualOdometry> vo_;
    TrackingStateMachine state_machine_;

    // ---- M2.1 异步输入（§6.1）----
    BoundedQueue<SensorPacket> input_queue_;
    std::thread worker_;
    std::atomic<bool> worker_started_{false};
    uint64_t packet_seq_ = 0;
    double last_submitted_timestamp_ = 0.0;  // 提交侧时间戳（仅传感器线程访问）
    bool has_last_submitted_ = false;
    /// 保护 latest_estimate_ 与 state_machine_ 的并发读写
    /// （异步模式：worker 写入，调用方读 state()/latestPose()）
    mutable std::mutex result_mutex_;
    PoseEstimate latest_estimate_;

    uint64_t sequence_ = 0;
    double last_timestamp_ = 0.0;  // 处理侧时间戳（同步=调用方线程；异步=worker）
    bool has_last_timestamp_ = false;
    bool stopped_ = false;

    // M2.3（§6.4）：结构化指标采集（enable_metrics=false 时为空实现；
    // mutable：const metricsSnapshot() 可刷新 backend/map 统计）
    mutable MetricsCollector metrics_;
    TrackingState last_metric_state_ = TrackingState::Initializing;
    bool has_last_metric_state_ = false;
};

} // namespace vslam
