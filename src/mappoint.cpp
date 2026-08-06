#include "vslam/mappoint.h"
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

    return mp;
}

} // namespace vslam
