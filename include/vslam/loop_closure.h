#pragma once

#include "vslam/common.h"
#include "vslam/frame.h"
#include <memory>
#include <string>

namespace vslam {

/// 回环检测模块（Phase 2）
/// 利用 DBoW3 词袋模型在关键帧中检测回环
class LoopClosure {
public:
    LoopClosure();
    ~LoopClosure();  // 显式声明析构，在 .cpp 中定义

    /// 从文件加载预训练词袋词典
    bool loadVocabulary(const std::string& vocab_path);

    /// 将关键帧加入数据库
    void addKeyFrame(Frame::Ptr kf);

    /// 检测回环：返回匹配的关键帧 ID，0 表示无回环
    unsigned long detectLoop(Frame::Ptr kf);

    /// 几何一致性验证：通过 3D-2D 对应确认回环
    bool verifyLoop(Frame::Ptr kf1, Frame::Ptr kf2);

private:
#ifdef HAS_DBOW3
    // 前向声明 DBoW3 类型
    // #include <DBoW3/DBoW3.h> 在 .cpp 中由 HAS_DBOW3 守卫
    class Impl;
    std::unique_ptr<Impl> impl_;
#endif
};

} // namespace vslam
