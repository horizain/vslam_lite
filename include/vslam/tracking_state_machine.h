#pragma once

#include "vslam/localization_types.h"

#include <cstdint>

namespace vslam {

/// 单帧跟踪质量输入（§4.2：完整质量 / 弱质量 / 几何失败）
enum class FrameQuality {
    Full,      ///< 完整质量：inlier ratio/RMSE/网格分布等全部通过
    Weak,      ///< 弱质量：仅达到弱门槛；发布时协方差需至少放大 4 倍
    Failed     ///< 几何失败：只能发布预测值（prediction_only）
};

/// 状态机输出：状态迁移 + 本帧发布注释（§4.2、§3-2）
struct StateMachineOutput {
    TrackingState state = TrackingState::Initializing;
    FrameQuality quality = FrameQuality::Failed;
    bool pose_valid = false;
    bool prediction_only = false;
    FailureReason reason = FailureReason::None;
    uint64_t sequence = 0;
};

/// 确定性有限状态机（§4.2），不使用隐含布尔组合。
///
/// 生命周期：Initializing → Tracking ⇄ Degraded →(失败) Relocalizing →(超时) Lost，
/// Relocalizing/Lost 经有效重定位回到 Tracking；任意状态 stop() → Stopped。
/// 输出语义（§4.2/§3-2）：
///   - Full/Weak 帧：pose_valid=true、prediction_only=false；Weak 帧协方差需 ×4。
///   - Failed 帧：prediction_timeout 内只发布预测（pose_valid=true、prediction_only=true），
///     超过后 pose_valid=false。
///   - Initializing/Relocalizing/Lost 不发布有效位姿。
class TrackingStateMachine {
public:
    struct Params {
        int init_required_frames = 3;            ///< Initializing→Tracking 连续完整验收帧数
        int weak_to_degraded_frames = 2;         ///< Tracking→Degraded 连续弱质量帧数
        int recover_required_frames = 3;         ///< Degraded→Tracking 连续完整质量帧数
        int max_tracking_failures = 5;           ///< Tracking/Degraded→Relocalizing 连续失败帧数
        int relocalization_max_frames = 20;      ///< Relocalizing→Lost 帧数上限
        double relocalization_max_seconds = 2.0; ///< Relocalizing→Lost 时长上限（秒）
        double prediction_timeout_s = 0.5;       ///< 预测超过该时长后 pose_valid=false
    };

    TrackingStateMachine();
    explicit TrackingStateMachine(Params params);

    /// 当前状态
    [[nodiscard]] TrackingState state() const;

    /// 输入一帧普通跟踪质量结果；Initializing/Tracking/Degraded 正常推进，
    /// Relocalizing/Lost 中仅推进超时时钟，不离开当前状态。
    StateMachineOutput on_tracking(FrameQuality quality, double dt, uint64_t sequence = 0);

    /// 输入一次重定位尝试结果；success 表示重定位几何验收与运动连续性均通过。
    /// 仅在 Relocalizing/Lost 中有效，其余状态为无操作。
    StateMachineOutput on_relocalization(bool success, double dt, uint64_t sequence = 0);

    /// 任意状态 → Stopped；幂等，可重复调用（§4.4：重复 stop 不崩溃）。
    StateMachineOutput stop();

private:
    StateMachineOutput emit(TrackingState state, FrameQuality quality, FailureReason reason,
                            uint64_t sequence, bool pose_valid, bool prediction_only) const;
    StateMachineOutput process_tracking_or_degraded(FrameQuality quality, double dt,
                                                    uint64_t sequence);
    bool advance_reloc_clock(double dt);
    void reset_counters();

    Params params_;
    TrackingState state_ = TrackingState::Initializing;
    bool stopped_ = false;

    int full_streak_ = 0;          ///< 连续 Full 帧（初始化/恢复计数）
    int weak_streak_ = 0;          ///< Tracking 中连续 Weak 帧
    int fail_streak_ = 0;          ///< Tracking/Degraded 中连续 Failed 帧
    int reloc_frames_ = 0;         ///< Relocalizing 累计帧
    double reloc_time_ = 0.0;      ///< Relocalizing 累计时长（秒）
    double prediction_time_ = 0.0; ///< 连续预测时长（秒）
};

} // namespace vslam
