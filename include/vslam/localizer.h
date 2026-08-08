#pragma once

#include "vslam/localization_types.h"
#include "vslam/tracking_state_machine.h"
#include "vslam/vo.h"
#include "vslam/camera.h"

#include <opencv2/core.hpp>
#include <memory>
#include <string>

namespace vslam {

/// Localizer Facade 配置（M0.3，§4）
struct LocalizerConfig {
    LocalizationMode mode = LocalizationMode::LocalizationOnly;  ///< 运行模式（§1.3）
    SE3 T_bc = SE3();                  ///< 相机 -> 基座 固定标定（§2/§4.3，不允许回环跳变）
    double stereo_max_time_diff_s = 0.001;  ///< §4.3 双目时间差初值上限 1 ms
    bool enable_input_validation = true;    ///< §4.3 输入硬检查开关（测试可关闭）

    /// 从 robot.yaml 的 Robot 段加载（缺省字段保持默认值；T_bc 不做静默归一化，
    /// 非单位四元数由 Localizer 构造时按 §4.3 拒绝）
    static LocalizerConfig fromYaml(const std::string& path);
};

/// 定位服务 Facade（M0.3）：包装 VisualOdometry + TrackingStateMachine + 输入硬检查。
///
/// 第一阶段不修改 VO 算法；调用方不再直接读取 Frame/Map/VisualOdometry 内部状态。
/// 不新增线程——所有权在调用方线程内同步驱动（§4.4：M0 不增加线程）。
/// 坐标契约（§2）：输出 T_wb = T_wc · T_bc⁻¹；M0 尚未接入 ESKF，T_ob = T_wb
/// （odom 系与全局系重合），M6 后再由 T_wo 分离，届时回环只更新 T_wo。
class Localizer {
public:
    /// @param camera  双目/单目相机（内参、图像尺寸即 §4.3 输入尺寸契约）
    /// @param vo_cfg  VO 参数（沿用 config/default.yaml）
    /// @param cfg     Localizer 参数（robot.yaml）
    /// 构造时校验 T_bc：四元数归一化误差 <1e-6、平移有限（§4.3），否则抛
    /// std::invalid_argument。
    Localizer(const Camera& camera, const VOConfig& vo_cfg,
              const LocalizerConfig& cfg = LocalizerConfig());
    ~Localizer();

    Localizer(const Localizer&) = delete;
    Localizer& operator=(const Localizer&) = delete;

    /// 处理一帧双目图像（right 为空 = 单目路径）。
    /// right_timestamp < 0 表示未提供右目时间戳（视为与左目同步，跳过 1ms 检查）。
    /// §4.3 输入硬检查失败时直接拒绝，不调用 VO，Map revision 逐项不变。
    PoseEstimate processFrame(const cv::Mat& left, const cv::Mat& right,
                              double timestamp, double right_timestamp = -1.0);

    /// 处理一帧单目图像（§4.3 输入硬检查同上）
    PoseEstimate processFrame(const cv::Mat& image, double timestamp);

    /// 停止定位服务并 join 后台线程；幂等，可重复调用（§4.4：重复 stop 不崩溃）。
    void stop();

    /// 当前状态机状态
    [[nodiscard]] TrackingState state() const;

    /// 地图 revision / 规模只读观测（测试与监控断言：非法输入不改 Map）
    [[nodiscard]] uint64_t mapTopologyRevision() const;
    [[nodiscard]] uint64_t mapGeometryRevision() const;
    [[nodiscard]] size_t mapPointCount() const;
    [[nodiscard]] size_t keyFrameCount() const;

private:
    bool validateInput(const cv::Mat& left, const cv::Mat& right,
                       double timestamp, double right_timestamp,
                       FailureReason& reason) const;
    PoseEstimate processValidFrame(const cv::Mat& left, const cv::Mat& right,
                                   double timestamp);
    PoseEstimate rejectedOutput(double timestamp, FailureReason reason) const;
    PoseEstimate stoppedOutput() const;

    Camera camera_;
    VOConfig vo_cfg_;
    LocalizerConfig cfg_;
    std::unique_ptr<VisualOdometry> vo_;
    TrackingStateMachine state_machine_;
    uint64_t sequence_ = 0;
    double last_timestamp_ = 0.0;
    bool has_last_timestamp_ = false;
    bool stopped_ = false;
};

} // namespace vslam
