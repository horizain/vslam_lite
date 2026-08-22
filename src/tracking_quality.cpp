#include "vslam/tracking_quality.h"

#include <opencv2/imgproc.hpp>

namespace vslam {

ImageQualityStats assessImageQuality(const cv::Mat& gray) {
    ImageQualityStats stats;
    // §7.2：图像阈值只适用于 8-bit 灰度；其他类型必须先显式转换并测试。
    if (gray.empty() || gray.depth() != CV_8U || gray.channels() != 1)
        return stats;
    stats.assessable = true;

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar lap_mean;
    cv::Scalar stddev;
    cv::meanStdDev(lap, lap_mean, stddev);
    stats.blur_variance = stddev.val[0] * stddev.val[0];

    const double total = static_cast<double>(gray.total());
    if (total <= 0.0) {
        stats.assessable = false;
        return stats;
    }
    // 单阈值化统计暗/亮占比（gray<=5 / gray>=250），避免全直方图开销。
    const int dark_pixels =
        cv::countNonZero(gray <= static_cast<uint8_t>(5));
    const int bright_pixels =
        cv::countNonZero(gray >= static_cast<uint8_t>(250));
    stats.dark_ratio = static_cast<double>(dark_pixels) / total;
    stats.bright_ratio = static_cast<double>(bright_pixels) / total;
    return stats;
}

int countOccupiedGridCells(const std::vector<cv::KeyPoint>& keypoints,
                           int image_cols, int image_rows,
                           int grid_cols, int grid_rows) {
    if (image_cols <= 0 || image_rows <= 0 ||
        grid_cols <= 0 || grid_rows <= 0 || keypoints.empty())
        return 0;
    std::vector<bool> occupied(static_cast<size_t>(grid_cols) *
                               static_cast<size_t>(grid_rows), false);
    int count = 0;
    for (const auto& kp : keypoints) {
        const float x = kp.pt.x;
        const float y = kp.pt.y;
        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        if (x < 0.0f || y < 0.0f ||
            x >= static_cast<float>(image_cols) ||
            y >= static_cast<float>(image_rows))
            continue;  // 越界点忽略
        int col = static_cast<int>(x * grid_cols / image_cols);
        int row = static_cast<int>(y * grid_rows / image_rows);
        col = std::min(col, grid_cols - 1);  // 右/下边界归最后一格
        row = std::min(row, grid_rows - 1);
        const size_t idx = static_cast<size_t>(row) * grid_cols + col;
        if (!occupied[idx]) {
            occupied[idx] = true;
            ++count;
        }
    }
    return count;
}

QualityVerdict classifyTrackingQuality(const ImageQualityStats& image_stats,
                                       int feature_count, int occupied_cells,
                                       const QualityConfig& config) {
    QualityVerdict verdict;
    verdict.band = QualityBand::Full;
    verdict.image = image_stats;
    verdict.features = feature_count;
    verdict.occupied_cells = occupied_cells;

    // 开关关闭：与旧版本行为完全一致（Full 旁路，不产生任何拒绝）。
    if (!config.enabled)
        return verdict;

    // 第 1 层（§7.2 图像质量）：方差 <10 或 暗/亮占比 >0.8 → 输入硬拒绝。
    // 边界语义："<10" 与 ">0.8" 为严格比较，恰在阈值上不拒绝。
    if (image_stats.assessable) {
        if (image_stats.blur_variance < config.hard_reject_blur_variance ||
            image_stats.dark_ratio > config.max_dark_ratio ||
            image_stats.bright_ratio > config.max_bright_ratio) {
            verdict.band = QualityBand::HardReject;
            verdict.reason = FailureReason::ImageDegraded;
            return verdict;
        }
    }
    // 不可评估图像不做像素级判定（§7.2），只可能被下方特征分布门降级。

    // 第 2 层（特征分布）：模糊退化带 / 特征不足 / 网格稀疏 → 至多 Degraded。
    // 不设置失败原因码——弱质量帧允许跟踪并发布位姿（置信度压弱）。
    if ((image_stats.assessable &&
         image_stats.blur_variance < config.degraded_blur_variance) ||
        feature_count < config.min_features_tracking ||
        occupied_cells < config.min_occupied_cells) {
        verdict.band = QualityBand::Degraded;
    }
    return verdict;
}

} // namespace vslam
