#include "vslam/mappoint.h"
#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>

namespace vslam {

MapPoint::Ptr MapPoint::create(unsigned long id,
                                const Vec2& px1, const Vec2& px2,
                                const SE3& T1, const SE3& T2,
                                const cv::Mat& K) {
    auto mp = std::make_shared<MapPoint>(id);

    // 构建投影矩阵 P = K [R|t]
    cv::Mat T1_cv = (cv::Mat_<double>(3, 4) <<
        T1.q.toRotationMatrix()(0,0), T1.q.toRotationMatrix()(0,1), T1.q.toRotationMatrix()(0,2), T1.t.x(),
        T1.q.toRotationMatrix()(1,0), T1.q.toRotationMatrix()(1,1), T1.q.toRotationMatrix()(1,2), T1.t.y(),
        T1.q.toRotationMatrix()(2,0), T1.q.toRotationMatrix()(2,1), T1.q.toRotationMatrix()(2,2), T1.t.z());

    cv::Mat T2_cv = (cv::Mat_<double>(3, 4) <<
        T2.q.toRotationMatrix()(0,0), T2.q.toRotationMatrix()(0,1), T2.q.toRotationMatrix()(0,2), T2.t.x(),
        T2.q.toRotationMatrix()(1,0), T2.q.toRotationMatrix()(1,1), T2.q.toRotationMatrix()(1,2), T2.t.y(),
        T2.q.toRotationMatrix()(2,0), T2.q.toRotationMatrix()(2,1), T2.q.toRotationMatrix()(2,2), T2.t.z());

    cv::Mat P1 = K * T1_cv;
    cv::Mat P2 = K * T2_cv;

    // 三角化
    std::vector<cv::Point2f> pts1 = {cv::Point2f(static_cast<float>(px1.x()), static_cast<float>(px1.y()))};
    std::vector<cv::Point2f> pts2 = {cv::Point2f(static_cast<float>(px2.x()), static_cast<float>(px2.y()))};

    cv::Mat points_4d;
    cv::triangulatePoints(P1, P2, pts1, pts2, points_4d);

    // 齐次坐标转 3D
    cv::Mat x = points_4d.col(0);
    x /= x.at<float>(3);
    mp->pos_s = Vec3(x.at<float>(0), x.at<float>(1), x.at<float>(2));

    // B2：最小视差角检查——两光心到点的射线夹角过小（近共线）时三角化
    // 退化，深度的数值误差放大成垃圾点（污染 PnP/BA）。0.1° 只拒绝
    // 严重退化（匹配误差导致的爆炸深度点）；实测 0.5° 会误伤 KITTI
    // 正常三角化（Triangulated -72%，ATE 42.8→91.5 回归）。
    const Vec3 c1 = -(T1.q.inverse() * T1.t);
    const Vec3 c2 = -(T2.q.inverse() * T2.t);
    const Vec3 d1 = (mp->pos_s - c1).normalized();
    const Vec3 d2 = (mp->pos_s - c2).normalized();
    const double cos_angle = std::clamp(d1.dot(d2), -1.0, 1.0);
    const double angle_deg = std::acos(cos_angle) * 180.0 / M_PI;
    if (angle_deg < 0.1) return nullptr;  // 退化三角化拒绝

    return mp;
}

} // namespace vslam
