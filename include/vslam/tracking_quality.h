#pragma once

#include "vslam/localization_types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace vslam {

/// M3.1（§7.2）图像与特征分布质量门配置。
/// 阈值为产品规格首版参数，不是数据集标定值；代码默认 enabled=false 以兼容
/// 旧配置（缺省段保持旧行为），产品 profile 在 yaml 中显式开启。
struct QualityConfig {
    bool   enabled = false;
    int    grid_cols = 8;
    int    grid_rows = 6;
    int    min_features_tracking = 300;
    int    min_occupied_cells = 12;
    double hard_reject_blur_variance = 10.0;   ///< Laplacian 方差 < 该值 → 输入硬拒绝
    double degraded_blur_variance = 60.0;      ///< 方差 [hard, degraded) → 至多 Degraded
    double max_dark_ratio = 0.80;              ///< gray <= 5 像素占比上限（> 拒绝）
    double max_bright_ratio = 0.80;            ///< gray >= 250 像素占比上限（> 拒绝）
};

/// 输入图像质量原始统计。仅 8-bit 单通道可评估（§7.2：其他类型必须先显式
/// 转换并测试）；不可评估输入不做像素级判定，只可能被特征分布门降级。
struct ImageQualityStats {
    bool   assessable = false;
    double blur_variance = 0.0;   ///< 全图 Laplacian 方差
    double dark_ratio = 0.0;      ///< gray <= 5 像素占比
    double bright_ratio = 0.0;    ///< gray >= 250 像素占比
};

/// 单帧质量判定带（§7.1 第 1/2 层：图像质量 + 特征分布）。
/// M3.1 只做置信度分层，不改变任何几何验收阈值。
enum class QualityBand {
    Full,        ///< 高置信：可发布完整质量视觉位姿
    Degraded,    ///< 弱质量：允许跟踪，但状态至多 Degraded、协方差 ×4
    HardReject,  ///< 输入硬拒绝：本帧不产生有效位姿，走失败/恢复路径
};

/// 质量判定结果（值对象）：band + 结构化原因码 + 诊断统计。
struct QualityVerdict {
    QualityBand band = QualityBand::Full;
    FailureReason reason = FailureReason::None;  ///< 仅 HardReject 时非 None
    ImageQualityStats image;                     ///< 图像统计（诊断/指标用）
    int features = 0;                            ///< 当前帧特征数
    int occupied_cells = 0;                      ///< 网格占用格子数
};

/// 计算图像质量统计：Laplacian 方差 + 暗/亮像素占比。
/// 非 CV_8UC1（含空图）返回 assessable=false；纯读操作，无 RNG、无全局状态。
[[nodiscard]] ImageQualityStats assessImageQuality(const cv::Mat& gray);

/// 统计特征点在 cols×rows 网格上的占用格子数。越界点忽略；
/// 点落在网格边界时按 floor 划分（右/下边界归最后一格）。
[[nodiscard]] int countOccupiedGridCells(
    const std::vector<cv::KeyPoint>& keypoints,
    int image_cols, int image_rows, int grid_cols, int grid_rows);

/// 统一判定入口（纯函数，值进出）：
///   - config 关闭 → Full 旁路（行为与旧版本一致）；
///   - 图像可评估且 方差<hard 或 暗/亮占比超限 → HardReject+ImageDegraded；
///   - 否则按 方差<degraded / 特征数不足 / 网格稀疏 降为 Degraded（原因码 None，
///     弱质量仍可发布位姿）；全部通过 → Full。
[[nodiscard]] QualityVerdict classifyTrackingQuality(
    const ImageQualityStats& image_stats,
    int feature_count, int occupied_cells,
    const QualityConfig& config = QualityConfig());

} // namespace vslam
