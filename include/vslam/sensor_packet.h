#pragma once

#include "vslam/common.h"

#include <opencv2/core.hpp>

#include <cstdint>

namespace vslam {

/// 传感器帧包（M2.1，§6.1）：输入队列的传输单元。
///
/// 图像是浅拷贝（共享同一像素缓冲）：提交方必须保证提交后不再复用/修改
/// 该缓冲，直到该帧被消费者取走或队列满时被丢弃（此时缓冲即释放）。
/// 队列满时按"丢最旧、保最新"策略丢弃队首包（§6.1）。
struct SensorPacket {
    uint64_t sequence = 0;         ///< 包序号（入队序）
    double timestamp = 0.0;        ///< 左目/主目时间戳（秒）
    double right_timestamp = -1.0; ///< 右目时间戳；<0 = 未提供（视为与左目同步）
    cv::Mat left;                  ///< 左目/单目图像
    cv::Mat right;                 ///< 右目图像（单目为空）
};

} // namespace vslam
