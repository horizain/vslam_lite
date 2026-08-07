#pragma once

#include <compare>
#include <cstdint>

namespace vslam {

using KeyframeId = unsigned long;
using FeatureIndex = std::uint32_t;
using MapPointId = unsigned long;

/// 一个地图点在一个关键帧特征上的正式观测。
/// 关键帧 id 与特征索引共同构成唯一键；不存在“无点 id”。
struct Observation {
    KeyframeId keyframe_id = 0;
    FeatureIndex feature_index = 0;

    auto operator<=>(const Observation&) const = default;
};

} // namespace vslam
