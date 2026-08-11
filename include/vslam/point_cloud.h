#pragma once

#include "vslam/common.h"

#include <cstdint>

namespace vslam {

/// 当前帧双目深度点及其左目图像颜色（坐标为世界系）。
/// 颜色只服务于可视化，不写入 MapPoint，避免改变地图/优化数据模型。
struct ColoredPoint {
    Vec3 position_w = Vec3::Zero();
    uint8_t r = 255;
    uint8_t g = 255;
    uint8_t b = 255;
};

} // namespace vslam
