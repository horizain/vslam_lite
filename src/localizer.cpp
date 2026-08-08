#include "vslam/localizer.h"

#include <yaml-cpp/yaml.h>

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
    : camera_(camera), vo_cfg_(vo_cfg), cfg_(cfg) {
    if (!camera_)
        throw std::invalid_argument("Localizer: camera is null");
    if (!isFinite(cfg_.T_bc) || !isUnitQuaternion(cfg_.T_bc.q))
        throw std::invalid_argument(
            "Localizer: T_bc must be finite with unit quaternion (norm error < 1e-6)");
    vo_ = std::make_unique<VisualOdometry>(camera_, vo_cfg_);
}

Localizer::~Localizer() { stop(); }

bool Localizer::validateInput(const cv::Mat& left, const cv::Mat& right,
                              double timestamp, double right_timestamp,
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
    if (has_last_timestamp_ && timestamp <= last_timestamp_) {
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
    if (stopped_) return stoppedOutput();

    FailureReason reason;
    if (!validateInput(left, right, timestamp, right_timestamp, reason))
        return rejectedOutput(timestamp, reason);
    return processValidFrame(left, right, timestamp);
}

PoseEstimate Localizer::processFrame(const cv::Mat& image, double timestamp) {
    return processFrame(image, cv::Mat(), timestamp);
}

PoseEstimate Localizer::processValidFrame(const cv::Mat& left, const cv::Mat& right,
                                          double timestamp) {
    const uint64_t seq = ++sequence_;
    const double dt = has_last_timestamp_ ? std::max(0.0, timestamp - last_timestamp_) : 0.0;

    // 第一阶段直接委托 VO（§4.1：不修改算法）
    const SE3 T_cw = right.empty()
        ? vo_->addFrame(left, timestamp)
        : vo_->addFrame(left, right, timestamp);
    const VisualOdometry::Status st = vo_->getStatus();

    // 只对通过验收的帧推进时间戳（§4.3：拒绝帧不污染后续比较）
    last_timestamp_ = timestamp;
    has_last_timestamp_ = true;

    // M0 质量映射：pose_valid → Full，否则 Failed；Weak 细分留给 M3 质量门。
    const FrameQuality q = st.pose_valid ? FrameQuality::Full : FrameQuality::Failed;
    const StateMachineOutput sm_out = state_machine_.on_tracking(q, dt, seq);

    PoseEstimate out;
    out.sequence = seq;
    out.timestamp = timestamp;
    const SE3 T_wc = T_cw.inverse();
    out.T_wb = T_wc * cfg_.T_bc.inverse();  // §2：T_wb = T_wc · T_bc⁻¹
    out.T_ob = out.T_wb;                    // M0：odom 系 = 全局系，M6 后由 T_wo 分离
    out.covariance = Mat6::Identity();      // M0 占位；M3 数值 Jacobian 替换
    if (sm_out.quality == FrameQuality::Weak) out.covariance *= 4.0;  // §4.2 弱质量 ×4
    out.state = sm_out.state;
    out.reason = sm_out.reason;
    out.pose_valid = sm_out.pose_valid;
    out.prediction_only = sm_out.prediction_only;
    out.map_generation = vo_->getMap()->topologyRevision();
    return out;
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
    if (vo_) vo_->finishPendingBackendWork();
    state_machine_.stop();
}

TrackingState Localizer::state() const { return state_machine_.state(); }

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

} // namespace vslam
