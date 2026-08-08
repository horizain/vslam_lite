#include "vslam/tracking_state_machine.h"

#include <algorithm>
#include <cmath>

namespace vslam {

TrackingStateMachine::TrackingStateMachine() : TrackingStateMachine(Params()) {}

TrackingStateMachine::TrackingStateMachine(Params params) : params_(params) {}

TrackingState TrackingStateMachine::state() const { return state_; }

void TrackingStateMachine::reset_counters() {
    full_streak_ = 0;
    weak_streak_ = 0;
    fail_streak_ = 0;
    reloc_frames_ = 0;
    reloc_time_ = 0.0;
    prediction_time_ = 0.0;
}

StateMachineOutput TrackingStateMachine::emit(TrackingState state, FrameQuality quality,
                                              FailureReason reason, uint64_t sequence,
                                              bool pose_valid, bool prediction_only) const {
    StateMachineOutput out;
    out.state = state;
    out.quality = quality;
    out.reason = reason;
    out.sequence = sequence;
    out.pose_valid = pose_valid;
    out.prediction_only = prediction_only;
    return out;
}

bool TrackingStateMachine::advance_reloc_clock(double dt) {
    reloc_frames_++;
    reloc_time_ += std::max(0.0, dt);
    return reloc_frames_ >= params_.relocalization_max_frames ||
           reloc_time_ >= params_.relocalization_max_seconds;
}

StateMachineOutput TrackingStateMachine::on_tracking(FrameQuality quality, double dt,
                                                     uint64_t sequence) {
    if (stopped_)
        return emit(TrackingState::Stopped, FrameQuality::Failed, FailureReason::None,
                    sequence, false, false);

    switch (state_) {
        case TrackingState::Initializing: {
            if (quality == FrameQuality::Full) {
                full_streak_++;
                if (full_streak_ >= params_.init_required_frames) {
                    state_ = TrackingState::Tracking;
                    reset_counters();
                    return emit(TrackingState::Tracking, FrameQuality::Full, FailureReason::None,
                                sequence, true, false);
                }
                return emit(TrackingState::Initializing, FrameQuality::Full, FailureReason::None,
                            sequence, false, false);
            }
            // 弱/失败帧中断初始化连续计数
            full_streak_ = 0;
            return emit(TrackingState::Initializing, quality, FailureReason::None,
                        sequence, false, false);
        }

        case TrackingState::Tracking:
        case TrackingState::Degraded:
            return process_tracking_or_degraded(quality, dt, sequence);

        case TrackingState::Relocalizing: {
            const bool timed_out = advance_reloc_clock(dt);
            if (timed_out) {
                state_ = TrackingState::Lost;
                reset_counters();
                return emit(TrackingState::Lost, FrameQuality::Failed,
                            FailureReason::RelocalizationTimeout, sequence, false, false);
            }
            return emit(TrackingState::Relocalizing, FrameQuality::Failed, FailureReason::None,
                        sequence, false, false);
        }

        case TrackingState::Lost:
            return emit(TrackingState::Lost, FrameQuality::Failed, FailureReason::None,
                        sequence, false, false);

        case TrackingState::Stopped:
            return emit(TrackingState::Stopped, FrameQuality::Failed, FailureReason::None,
                        sequence, false, false);
    }
    return emit(TrackingState::Stopped, FrameQuality::Failed, FailureReason::None,
                sequence, false, false);
}

StateMachineOutput TrackingStateMachine::process_tracking_or_degraded(FrameQuality quality,
                                                                      double dt,
                                                                      uint64_t sequence) {
    const bool in_degraded = (state_ == TrackingState::Degraded);

    if (quality == FrameQuality::Full) {
        full_streak_++;
        weak_streak_ = 0;
        fail_streak_ = 0;
        prediction_time_ = 0.0;
        if (in_degraded && full_streak_ >= params_.recover_required_frames) {
            state_ = TrackingState::Tracking;
            reset_counters();
            return emit(TrackingState::Tracking, FrameQuality::Full, FailureReason::None,
                        sequence, true, false);
        }
        return emit(state_, FrameQuality::Full, FailureReason::None, sequence, true, false);
    }

    if (quality == FrameQuality::Weak) {
        weak_streak_++;
        full_streak_ = 0;
        fail_streak_ = 0;
        prediction_time_ = 0.0;
        if (!in_degraded && weak_streak_ >= params_.weak_to_degraded_frames) {
            state_ = TrackingState::Degraded;
            reset_counters();
            return emit(TrackingState::Degraded, FrameQuality::Weak, FailureReason::None,
                        sequence, true, false);
        }
        return emit(state_, FrameQuality::Weak, FailureReason::None, sequence, true, false);
    }

    // Failed：几何失败，只允许发布预测值（§4.2）
    fail_streak_++;
    full_streak_ = 0;
    weak_streak_ = 0;
    prediction_time_ += std::max(0.0, dt);
    const bool within_window = prediction_time_ <= params_.prediction_timeout_s;

    if (fail_streak_ >= params_.max_tracking_failures) {
        state_ = TrackingState::Relocalizing;
        reset_counters();
        return emit(TrackingState::Relocalizing, FrameQuality::Failed,
                    FailureReason::GeometricRejection, sequence, within_window, within_window);
    }
    return emit(state_, FrameQuality::Failed, FailureReason::GeometricRejection,
                sequence, within_window, within_window);
}

StateMachineOutput TrackingStateMachine::on_relocalization(bool success, double dt,
                                                           uint64_t sequence) {
    if (stopped_)
        return emit(TrackingState::Stopped, FrameQuality::Failed, FailureReason::None,
                    sequence, false, false);

    switch (state_) {
        case TrackingState::Relocalizing: {
            if (success) {
                state_ = TrackingState::Tracking;
                reset_counters();
                return emit(TrackingState::Tracking, FrameQuality::Full, FailureReason::None,
                            sequence, true, false);
            }
            const bool timed_out = advance_reloc_clock(dt);
            if (timed_out) {
                state_ = TrackingState::Lost;
                reset_counters();
                return emit(TrackingState::Lost, FrameQuality::Failed,
                            FailureReason::RelocalizationTimeout, sequence, false, false);
            }
            return emit(TrackingState::Relocalizing, FrameQuality::Failed, FailureReason::None,
                        sequence, false, false);
        }

        case TrackingState::Lost: {
            if (success) {
                state_ = TrackingState::Tracking;
                reset_counters();
                return emit(TrackingState::Tracking, FrameQuality::Full, FailureReason::None,
                            sequence, true, false);
            }
            return emit(TrackingState::Lost, FrameQuality::Failed, FailureReason::None,
                        sequence, false, false);
        }

        default:
            // 非重定位状态收到重定位结果：无操作，不产生新位姿
            return emit(state_, FrameQuality::Failed, FailureReason::None, sequence, false, false);
    }
}

StateMachineOutput TrackingStateMachine::stop() {
    stopped_ = true;
    state_ = TrackingState::Stopped;
    reset_counters();
    return emit(TrackingState::Stopped, FrameQuality::Failed, FailureReason::None, 0,
                false, false);
}

} // namespace vslam
