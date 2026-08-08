/**
 * test_relocalizer.cpp - Relocalizer 候选几何验证单元测试
 *
 * 覆盖 M1.2（PRODUCTION_LOCALIZATION_PLAN §5.3）：
 *   1. matToSE3：cv::Mat(rvec/tvec, R) → SE3 转换
 *   2. relocalize：候选粗筛 + ORB 匹配 + PnP 验证，恢复已知位姿（identity）
 *   3. verifyCandidate：单候选几何验证（阈值/内点/版本绑定）
 *   4. 拒绝路径：空候选、stale（身份不符）、弱几何（点不足）、
 *      不相关描述子（粗筛拦截）
 *
 * 编译: cmake -DBUILD_TESTS=ON .. && make test_relocalizer
 * 运行: ./build/test_relocalizer（独立 CTest）
 */

#include "vslam/relocalizer.h"
#include "vslam/camera.h"
#include "vslam/feature.h"
#include "vslam/frame.h"
#include "vslam/mappoint.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

// 简单的测试辅助宏（与 test_vo.cpp 保持一致）
#define TEST(name) \
    std::cout << "  TEST: " << name << " ... ";

#define TEST_PASS() \
    std::cout << "PASSED" << std::endl;

#define TEST_FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl;

using vslam::Frame;
using vslam::RelocalizationPointSet;
using vslam::RelocalizationResult;
using vslam::Relocalizer;
using vslam::SE3;
using vslam::Vec2;
using vslam::Vec3;

namespace {

vslam::Camera makeCamera() {
    auto cam = std::make_shared<vslam::MonocularCamera>();
    cam->fx = 500.0;
    cam->fy = 500.0;
    cam->cx = 320.0;
    cam->cy = 240.0;
    cam->img_width = 640;
    cam->img_height = 480;
    return cam;
}

// 构造合成场景：随机噪声图提取 ORB；候选 KF 的 map_points[i] = 把该特征点
// 以深度 5m 反投影得到的 3D 点（位姿=identity 时投影回原像素，PnP 应精确恢复）。
// curr 帧复制相同关键点/描述子 → ORB 匹配全部一一对应。
void buildScene(const vslam::Camera& cam, Frame::Ptr& kf, Frame::Ptr& curr) {
    cv::RNG rng(0x5A17);
    cv::Mat img(480, 640, CV_8U);
    rng.fill(img, cv::RNG::UNIFORM, 0, 255);

    vslam::FeatureMatcher fm;
    fm.setParams(1000, 1.2, 8, 1);  // 单带提取，确定性
    kf = std::make_shared<Frame>(10, 0.0);
    kf->image_gray = img;
    fm.extract(kf);
    assert(kf->keypoints.size() >= 50 && "noise image must yield enough features");
    for (size_t i = 0; i < kf->keypoints.size(); i++) {
        const auto& kp = kf->keypoints[i];
        const Vec3 p = cam->pixel2camera(Vec2(kp.pt.x, kp.pt.y), 5.0);
        auto mp = std::make_shared<vslam::MapPoint>(i);
        mp->pos_s = p;
        kf->map_points[i] = mp;
    }
    curr = std::make_shared<Frame>(11, 0.1);
    curr->keypoints = kf->keypoints;
    curr->descriptors = kf->descriptors.clone();
}

// 构造 3D-2D 对应供应回调（模拟 VO 在锁内收集点）。
// return_stale=true 模拟候选身份/版本失效；max_pts 限制供应的点数（弱几何测试）。
using PointProvider = Relocalizer::Query::PointProvider;
PointProvider makeProvider(const Frame::Ptr& kf, const Frame::Ptr& curr,
                           bool return_stale = false, int max_pts = 0) {
    return [kf, curr, return_stale, max_pts](
               unsigned long submap_id, const Frame::Ptr& cand,
               const std::vector<cv::DMatch>& matches,
               RelocalizationPointSet& out) -> bool {
        if (return_stale || cand != kf) return false;
        out.geometry_revision = 7;
        out.submap_id = submap_id;
        int n = 0;
        for (const auto& m : matches) {
            const auto& mp = cand->map_points[m.queryIdx];
            if (mp && (max_pts <= 0 || n < max_pts)) {
                out.pts3d.emplace_back((float)mp->pos_s.x(),
                                       (float)mp->pos_s.y(),
                                       (float)mp->pos_s.z());
                out.pts2d.push_back(curr->keypoints[m.trainIdx].pt);
                ++n;
            }
        }
        return true;
    };
}

void assert_identity(const SE3& T, double tol) {
    assert(T.t.norm() < tol);
    assert(T.q.angularDistance(Eigen::Quaterniond::Identity()) < tol);
}

void test_matToSE3_identity() {
    TEST("matToSE3: 零 rvec/tvec → identity + 平移") {
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat t = (cv::Mat_<double>(3, 1) << 1.0, 2.0, 3.0);
        const SE3 s = Relocalizer::matToSE3(rvec, t);
        assert((s.t - Vec3(1.0, 2.0, 3.0)).norm() < 1e-9);
        assert(s.q.angularDistance(Eigen::Quaterniond::Identity()) < 1e-9);
    } TEST_PASS();
}

void test_matToSE3_rotation() {
    TEST("matToSE3: 已知旋转矩阵（绕 Z 90°）") {
        const cv::Mat Rcv = (cv::Mat_<double>(3, 3)
                             << 0, -1, 0, 1, 0, 0, 0, 0, 1);
        Eigen::Matrix3d Rm;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Rm(i, j) = Rcv.at<double>(i, j);
        cv::Mat t = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
        const SE3 s = Relocalizer::matToSE3(Rcv, t);
        assert((s.q.toRotationMatrix() - Rm).norm() < 1e-6);
    } TEST_PASS();
}

void test_relocalize_success() {
    TEST("relocalize: 单候选恢复 identity 位姿") {
        const vslam::Camera cam = makeCamera();
        Frame::Ptr kf, curr;
        buildScene(cam, kf, curr);

        Relocalizer reloc(cam);
        Relocalizer::Query q;
        q.curr_frame = curr;
        q.candidates = {{0, kf}};
        q.supply_points = makeProvider(kf, curr);

        const RelocalizationResult res = reloc.relocalize(q);
        assert(res.accepted && "synthetic scene must relocalize");
        assert(res.inliers >= 20 && res.inliers > 0);
        assert(res.kf == kf);
        assert(res.submap_id == 0);
        assert(res.geometry_revision == 7);
        assert_identity(res.T_cs, 1e-2);
        assert(res.quick_checked == 1 && res.quick_passed == 1);
    } TEST_PASS();
}

void test_verify_candidate_recovers_pose() {
    TEST("verifyCandidate: 单候选几何验证恢复 identity") {
        const vslam::Camera cam = makeCamera();
        Frame::Ptr kf, curr;
        buildScene(cam, kf, curr);

        Relocalizer reloc(cam);
        Relocalizer::Query q;
        q.curr_frame = curr;
        q.supply_points = makeProvider(kf, curr);

        const RelocalizationResult res = reloc.verifyCandidate(0, kf, q);
        assert(res.accepted);
        assert_identity(res.T_cs, 1e-2);
    } TEST_PASS();
}

void test_relocalize_empty_candidates() {
    TEST("relocalize: 空候选 → 不接受") {
        const vslam::Camera cam = makeCamera();
        Frame::Ptr kf, curr;
        buildScene(cam, kf, curr);

        Relocalizer reloc(cam);
        Relocalizer::Query q;
        q.curr_frame = curr;
        q.supply_points = makeProvider(kf, curr);

        const RelocalizationResult res = reloc.relocalize(q);
        assert(!res.accepted);
        assert(res.quick_checked == 0);
    } TEST_PASS();
}

void test_relocalize_stale_candidate() {
    TEST("relocalize: 候选身份失效（stale）→ 跳过且不接受") {
        const vslam::Camera cam = makeCamera();
        Frame::Ptr kf, curr;
        buildScene(cam, kf, curr);

        Relocalizer reloc(cam);
        Relocalizer::Query q;
        q.curr_frame = curr;
        q.candidates = {{0, kf}};
        q.supply_points = makeProvider(kf, curr, /*return_stale=*/true);

        const RelocalizationResult res = reloc.relocalize(q);
        assert(!res.accepted);
        assert(res.kf == nullptr && res.inliers == 0);
    } TEST_PASS();
}

void test_relocalize_weak_geometry() {
    TEST("relocalize: 3D 点不足（弱几何）→ 不接受") {
        const vslam::Camera cam = makeCamera();
        Frame::Ptr kf, curr;
        buildScene(cam, kf, curr);

        Relocalizer reloc(cam);
        Relocalizer::Query q;
        q.curr_frame = curr;
        q.candidates = {{0, kf}};
        q.supply_points = makeProvider(kf, curr, false, /*max_pts=*/3);

        const RelocalizationResult res = reloc.relocalize(q);
        assert(!res.accepted && "3 个点低于 min_pts3d=10 必须拒绝");
    } TEST_PASS();
}

void test_relocalize_unrelated_descriptors() {
    TEST("relocalize: 描述子不相关（粗筛拦截）→ 不接受") {
        const vslam::Camera cam = makeCamera();
        Frame::Ptr kf, curr;
        buildScene(cam, kf, curr);

        // 用另一幅噪声图的特征作为当前帧 → 与候选 KF 描述子完全不相关
        Frame::Ptr other = std::make_shared<Frame>(12, 0.2);
        cv::RNG rng(0xBEAD);
        cv::Mat img(480, 640, CV_8U);
        rng.fill(img, cv::RNG::UNIFORM, 0, 255);
        other->image_gray = img;
        vslam::FeatureMatcher fm;
        fm.setParams(1000, 1.2, 8, 1);
        fm.extract(other);
        assert(!other->descriptors.empty());

        Relocalizer reloc(cam);
        Relocalizer::Query q;
        q.curr_frame = other;
        q.candidates = {{0, kf}};
        q.supply_points = makeProvider(kf, other);

        // 粗筛是宽松的距离过滤（可放行部分噪声描述子），真正的拒绝发生在
        // 全量 ORB 匹配（比率测试后匹配数 < min_matches=30）→ 绝不误接受。
        const RelocalizationResult res = reloc.relocalize(q);
        assert(!res.accepted && "不相关描述子不得被误接受为重定位");
        assert(res.quick_checked == 1);
    } TEST_PASS();
}

void test_relocalize_min_inlier_gate() {
    TEST("relocalize: 提高 min_inliers 门槛 → 相同场景拒绝") {
        const vslam::Camera cam = makeCamera();
        Frame::Ptr kf, curr;
        buildScene(cam, kf, curr);

        Relocalizer reloc(cam);
        Relocalizer::Query q;
        q.curr_frame = curr;
        q.candidates = {{0, kf}};
        q.supply_points = makeProvider(kf, curr);
        q.min_inliers = 100000;  // 超过任何候选的内点数 → 必拒绝

        const RelocalizationResult res = reloc.relocalize(q);
        assert(!res.accepted);
    } TEST_PASS();
}

}  // namespace

int main() {
    cv::setNumThreads(1);
    cv::setRNGSeed(0x5A17);

    std::cout << "test_relocalizer (M1.2 Relocalizer)" << std::endl;

    test_matToSE3_identity();
    test_matToSE3_rotation();
    test_relocalize_success();
    test_verify_candidate_recovers_pose();
    test_relocalize_empty_candidates();
    test_relocalize_stale_candidate();
    test_relocalize_weak_geometry();
    test_relocalize_unrelated_descriptors();
    test_relocalize_min_inlier_gate();

    std::cout << "全部通过" << std::endl;
    return 0;
}
