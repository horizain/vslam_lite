/**
 * test_tracking_quality.cpp - M3.1 前端质量门单元测试
 *
 * 覆盖 PRODUCTION_LOCALIZATION_PLAN §7.2：
 *   1. assessImageQuality：Laplacian 模糊方差 + 暗/亮像素占比统计
 *   2. countOccupiedGridCells：8×6 特征网格占用
 *   3. classifyTrackingQuality：HardReject / Degraded / Full 三档判定与边界
 *   4. FrontendTracker::assessFrameQuality：聚合入口 + 开关旁路
 *   5. VO 故障注入：纯黑帧硬拒绝不崩溃、revision 不变、恢复后继续跟踪（§7.6 子集）
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_tracking_quality
 * 运行: ./build/test_tracking_quality（独立 CTest）
 */

#include "vslam/camera.h"
#include "vslam/frontend_tracker.h"
#include "vslam/tracking_quality.h"
#include "vslam/vo.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using namespace vslam;

namespace {

constexpr int kW = 1241;   // KITTI 尺寸，网格覆盖测试用
constexpr int kH = 376;

/// 生成结构化合成灰度图：中灰底 + 随机形状（ORB 可重复提取/匹配），
/// 方差远高于退化带、暗/亮占比接近 0
cv::Mat makeTexturedImage(int rows, int cols, unsigned seed = 7) {
    cv::Mat img(rows, cols, CV_8UC1, cv::Scalar(128));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> gray(40, 215);
    std::uniform_int_distribution<int> x(10, cols - 60);
    std::uniform_int_distribution<int> y(10, rows - 60);
    std::uniform_int_distribution<int> size(8, 40);
    for (int i = 0; i < 260; ++i) {
        const int g = gray(rng);
        if (i % 3 == 0)
            cv::rectangle(img, cv::Rect(x(rng), y(rng), size(rng), size(rng)),
                          cv::Scalar(g), -1);
        else if (i % 3 == 1)
            cv::circle(img, cv::Point(x(rng), y(rng)), size(rng) / 2,
                       cv::Scalar(g), -1);
        else
            cv::rectangle(img, cv::Rect(x(rng), y(rng), size(rng), size(rng) / 2 + 2),
                          cv::Scalar(g), 2);
    }
    return img;
}

/// 在图上撒 n 个网格均匀分布的关键点
std::vector<cv::KeyPoint> spreadKeypoints(int rows, int cols, int per_cell) {
    std::vector<cv::KeyPoint> kps;
    const double cw = static_cast<double>(cols) / 8.0;
    const double ch = static_cast<double>(rows) / 6.0;
    for (int gr = 0; gr < 6; ++gr) {
        for (int gc = 0; gc < 8; ++gc) {
            for (int k = 0; k < per_cell; ++k) {
                kps.emplace_back(
                    cv::Point2f(static_cast<float>(gc * cw + cw * (k + 1) / (per_cell + 1)),
                                static_cast<float>(gr * ch + ch / 2.0)),
                    1.0f);
            }
        }
    }
    return kps;
}

} // namespace

void test_image_quality_stats() {
    TEST("assessImageQuality: 纹理图方差高、暗亮占比低") {
        const cv::Mat img = makeTexturedImage(300, 400);
        const ImageQualityStats st = assessImageQuality(img);
        assert(st.assessable);
        assert(st.blur_variance > 60.0);
        assert(st.dark_ratio < 0.01);
        assert(st.bright_ratio < 0.01);
    } TEST_PASS();

    TEST("assessImageQuality: 纯黑帧暗占比=1、方差≈0") {
        const cv::Mat img = cv::Mat::zeros(200, 320, CV_8UC1);
        const ImageQualityStats st = assessImageQuality(img);
        assert(st.assessable);
        assert(st.blur_variance < 10.0);
        assert(st.dark_ratio > 0.99);
        assert(st.bright_ratio < 0.01);
    } TEST_PASS();

    TEST("assessImageQuality: 纯白帧亮占比=1") {
        const cv::Mat img(200, 320, CV_8UC1, cv::Scalar(255));
        const ImageQualityStats st = assessImageQuality(img);
        assert(st.assessable);
        assert(st.bright_ratio > 0.99);
        assert(st.dark_ratio < 0.01);
    } TEST_PASS();

    TEST("assessImageQuality: 高斯模糊后方差显著下降") {
        cv::Mat img = makeTexturedImage(300, 400);
        cv::Mat blurred;
        cv::GaussianBlur(img, blurred, cv::Size(31, 31), 9.0);
        const ImageQualityStats s1 = assessImageQuality(img);
        const ImageQualityStats s2 = assessImageQuality(blurred);
        assert(s2.blur_variance < s1.blur_variance);
        assert(s2.blur_variance < 60.0);  // 进入退化/拒绝带
    } TEST_PASS();

    TEST("assessImageQuality: 非 8 位单通道不可评估且不做像素判定") {
        cv::Mat img(100, 100, CV_16UC1, cv::Scalar(1000));
        const ImageQualityStats st = assessImageQuality(img);
        assert(!st.assessable);
        cv::Mat bgr(100, 100, CV_8UC3, cv::Scalar(10, 20, 30));
        const ImageQualityStats st3 = assessImageQuality(bgr);
        assert(!st3.assessable);
        const ImageQualityStats st_empty = assessImageQuality(cv::Mat());
        assert(!st_empty.assessable);
    } TEST_PASS();
}

void test_grid_coverage() {
    TEST("countOccupiedGridCells: 均匀点铺满 48 格") {
        const auto kps = spreadKeypoints(kH, kW, 3);
        const int cells = countOccupiedGridCells(kps, kW, kH, 8, 6);
        assert(cells == 48);
    } TEST_PASS();

    TEST("countOccupiedGridCells: 单点只占一格、空输入为 0") {
        std::vector<cv::KeyPoint> one = {cv::KeyPoint(10.f, 10.f, 1.f)};
        assert(countOccupiedGridCells(one, kW, kH, 8, 6) == 1);
        assert(countOccupiedGridCells({}, kW, kH, 8, 6) == 0);
    } TEST_PASS();

    TEST("countOccupiedGridCells: 右下角边界点归属最后一格、越界忽略") {
        std::vector<cv::KeyPoint> edge = {cv::KeyPoint(float(kW - 1), float(kH - 1), 1.f)};
        assert(countOccupiedGridCells(edge, kW, kH, 8, 6) == 1);
        std::vector<cv::KeyPoint> out = {cv::KeyPoint(-5.f, 10.f, 1.f),
                                         cv::KeyPoint(10.f, float(kH + 3), 1.f)};
        assert(countOccupiedGridCells(out, kW, kH, 8, 6) == 0);
    } TEST_PASS();
}

void test_classification() {
    QualityConfig cfg;  // §7.2 首版参数默认值
    cfg.enabled = true;  // 代码默认关闭（兼容旧配置）；质量门测试显式开启
    const ImageQualityStats sharp{true, 500.0, 0.0, 0.0};
    const ImageQualityStats soft{true, 30.0, 0.0, 0.0};     // [10,60) 退化带
    const ImageQualityStats blur{true, 5.0, 0.0, 0.0};      // <10 硬拒
    const ImageQualityStats dark{true, 500.0, 0.9, 0.0};
    const ImageQualityStats bright{true, 500.0, 0.0, 0.9};

    TEST("classifyTrackingQuality: 全部通过 → Full") {
        const QualityVerdict v =
            classifyTrackingQuality(sharp, 600, 40, cfg);
        assert(v.band == QualityBand::Full);
        assert(v.reason == FailureReason::None);
    } TEST_PASS();

    TEST("classifyTrackingQuality: 方差 [10,60) → Degraded 不拒绝") {
        const QualityVerdict v = classifyTrackingQuality(soft, 600, 40, cfg);
        assert(v.band == QualityBand::Degraded);
        assert(v.reason == FailureReason::None);
    } TEST_PASS();

    TEST("classifyTrackingQuality: 方差 <10 → HardReject+ImageDegraded") {
        const QualityVerdict v = classifyTrackingQuality(blur, 600, 40, cfg);
        assert(v.band == QualityBand::HardReject);
        assert(v.reason == FailureReason::ImageDegraded);
    } TEST_PASS();

    TEST("classifyTrackingQuality: 暗/亮占比 >0.8 → HardReject") {
        assert(classifyTrackingQuality(dark, 600, 40, cfg).band ==
               QualityBand::HardReject);
        assert(classifyTrackingQuality(bright, 600, 40, cfg).reason ==
               FailureReason::ImageDegraded);
    } TEST_PASS();

    TEST("classifyTrackingQuality: 特征不足或网格稀疏 → Degraded") {
        assert(classifyTrackingQuality(sharp, 299, 40, cfg).band ==
               QualityBand::Degraded);
        assert(classifyTrackingQuality(sharp, 600, 11, cfg).band ==
               QualityBand::Degraded);
        // 两者都不产生失败原因码：弱质量仍可发布位姿
        assert(classifyTrackingQuality(sharp, 299, 40, cfg).reason ==
               FailureReason::None);
    } TEST_PASS();

    TEST("classifyTrackingQuality: 边界值 10/60/0.8 语义") {
        // 方差恰为 10 → 退化（<10 才硬拒）
        assert(classifyTrackingQuality(ImageQualityStats{true, 10.0, 0, 0}, 600, 40, cfg)
                   .band == QualityBand::Degraded);
        // 方差恰为 60 → Full
        assert(classifyTrackingQuality(ImageQualityStats{true, 60.0, 0, 0}, 600, 40, cfg)
                   .band == QualityBand::Full);
        // 占比恰为 0.8 → 不拒（>0.8 才拒）
        assert(classifyTrackingQuality(ImageQualityStats{true, 500.0, 0.8, 0}, 600, 40, cfg)
                   .band == QualityBand::Full);
    } TEST_PASS();

    TEST("classifyTrackingQuality: 硬拒绝优先于退化降级") {
        const ImageQualityStats blur_dark{true, 5.0, 0.9, 0.0};
        const QualityVerdict v = classifyTrackingQuality(blur_dark, 10, 2, cfg);
        assert(v.band == QualityBand::HardReject);
        assert(v.reason == FailureReason::ImageDegraded);
    } TEST_PASS();

    TEST("classifyTrackingQuality: 开关关闭 → 一律 Full 旁路") {
        QualityConfig off;
        off.enabled = false;
        assert(classifyTrackingQuality(blur, 10, 2, off).band == QualityBand::Full);
    } TEST_PASS();

    TEST("classifyTrackingQuality: 图像不可评估时仅按特征分布降级") {
        const ImageQualityStats na{};  // assessable=false
        assert(classifyTrackingQuality(na, 600, 40, cfg).band == QualityBand::Full);
        assert(classifyTrackingQuality(na, 100, 40, cfg).band == QualityBand::Degraded);
    } TEST_PASS();
}

void test_tracker_entry() {
    TEST("assessFrameQuality: 聚合入口按配置判定并可开关") {
        auto cam = std::make_shared<MonocularCamera>();
        cam->fx = 500; cam->fy = 500; cam->cx = 320; cam->cy = 240;
        cam->img_width = 640; cam->img_height = 480;
        TrackerConfig tc;
        tc.quality.enabled = true;  // 代码默认关闭；聚合入口测试显式开启
        FrontendTracker tracker(cam, tc);

        const cv::Mat black = cv::Mat::zeros(240, 320, CV_8UC1);
        const auto kps = spreadKeypoints(240, 320, 5);
        const QualityVerdict rej =
            tracker.assessFrameQuality(black, kps, 320, 240);
        assert(rej.band == QualityBand::HardReject);
        assert(rej.reason == FailureReason::ImageDegraded);

        TrackerConfig tc_off;
        tc_off.quality.enabled = false;
        FrontendTracker tracker_off(cam, tc_off);
        const QualityVerdict bypass =
            tracker_off.assessFrameQuality(black, kps, 320, 240);
        assert(bypass.band == QualityBand::Full);
        assert(bypass.features == static_cast<int>(kps.size()));
    } TEST_PASS();
}

/// §7.6 故障注入子集：运行中注入纯黑帧。
/// 场景用带深度变化的 3D 块（projectPoints 渲染），避免平面+纯平移的
/// 本质矩阵退化配置（与 test_vo 单目初始化测试同思路）。
void test_vo_black_frame_fault() {
    TEST("VO 黑帧故障注入: 硬拒绝→pose_valid=false→恢复后继续跟踪") {
        auto cam = std::make_shared<MonocularCamera>();
        cam->fx = 350; cam->fy = 350;
        cam->cx = kW / 2.0; cam->cy = kH / 2.0;
        cam->img_width = kW; cam->img_height = kH;
        const cv::Mat K = cam->K();

        struct Blk { cv::Point3f c; float sx, sy; int gray; };
        std::vector<Blk> blks;
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> dx(-3.5, 3.5), dy(-1.5, 1.5),
            dz(3.0, 7.0), ds(0.25, 0.7);
        // 唯一灰度：避免不同块角点产生完全相同的 ORB 描述子（歧义会让
        // ratio 测试把正确匹配也拒掉，重定位/跟踪都会退化）
        for (int i = 0; i < 90; ++i)
            blks.push_back({cv::Point3f(dx(rng), dy(rng), dz(rng)),
                            (float)ds(rng) * 2.f, (float)ds(rng), 64 + i * 2});
        const auto render = [&](double tx) {
            cv::Mat img(kH, kW, CV_8UC1, cv::Scalar(64));
            const cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            const cv::Mat tvec = (cv::Mat_<double>(3, 1) << -tx, 0.0, 0.0);
            for (const auto& b : blks) {
                std::vector<cv::Point3f> corners = {
                    {b.c.x - b.sx / 2, b.c.y - b.sy / 2, b.c.z},
                    {b.c.x + b.sx / 2, b.c.y - b.sy / 2, b.c.z},
                    {b.c.x + b.sx / 2, b.c.y + b.sy / 2, b.c.z},
                    {b.c.x - b.sx / 2, b.c.y + b.sy / 2, b.c.z}};
                std::vector<cv::Point2f> px;
                cv::projectPoints(corners, rvec, tvec, K, cv::Mat(), px);
                std::vector<cv::Point> poly;
                for (const auto& q : px) poly.emplace_back(cvRound(q.x), cvRound(q.y));
                cv::fillConvexPoly(img, poly, cv::Scalar(b.gray));
            }
            return img;
        };

        VOConfig cfg;
        cfg.feature_method = 0;
        cfg.quality.enabled = true;  // 显式开启（代码默认关闭以兼容旧配置）
        VisualOdometry vo(cam, cfg);

        bool initialized = false;
        // 先连续跟踪足够多的帧（覆盖 min_keyframe_interval，插入第二个 KF），
        // 保证黑帧注入点距离最新关键帧只有一步运动——重定位失败只可能
        // 归因于故障处理本身，而不是场景基线过大。
        for (int i = 0; i < 14; ++i) {
            vo.addFrame(render(0.25 * i), i * 0.1);
            initialized = initialized ||
                          (vo.state() == VisualOdometry::State::TRACKING &&
                           vo.getStatus().pose_valid);
        }
        assert(initialized);  // 非空工作证明：故障前系统已正常跟踪

        const uint64_t topo_before = vo.getMap()->topologyRevision();
        const uint64_t geom_before = vo.getMap()->geometryRevision();

        // 注入纯黑帧：必须硬拒绝——无效位姿、结构化原因码、Map revision 不变
        vo.addFrame(cv::Mat(kH, kW, CV_8UC1, cv::Scalar(0)), 14 * 0.1);
        const VisualOdometry::Status st_bad = vo.getStatus();
        assert(!st_bad.pose_valid);
        assert(st_bad.failure_reason == FailureReason::ImageDegraded);
        assert(vo.getMap()->topologyRevision() == topo_before);
        assert(vo.getMap()->geometryRevision() == geom_before);

        // 故障消失后恢复跟踪（§7.6：恢复或明确 Lost；此处应快速恢复）
        bool recovered = false;
        for (int k = 15; k < 23 && !recovered; ++k) {
            vo.addFrame(render(0.25 * k), k * 0.1);
            recovered = vo.getStatus().pose_valid;
        }
        assert(recovered);
    } TEST_PASS();
}

int main() {
    std::cout << "== test_tracking_quality ==" << std::endl;
    test_image_quality_stats();
    test_grid_coverage();
    test_classification();
    test_tracker_entry();
    test_vo_black_frame_fault();
    std::cout << "ALL PASSED" << std::endl;
    return 0;
}
