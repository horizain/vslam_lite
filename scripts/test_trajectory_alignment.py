#!/usr/bin/env python3
"""轨迹评估工具的轻量回归测试；直接运行即可，无需测试框架。"""

import numpy as np

import evaluate_ate
import euroc_gt_to_tum
import kitti_gt_to_tum
import plot_traj


def transformed_points():
    src = np.array([
        [-1.0, 0.2, 2.0],
        [0.5, -2.0, 1.0],
        [2.0, 1.5, -0.5],
        [-0.3, 3.0, 1.2],
        [1.1, -0.7, 4.0],
    ])
    angle = np.deg2rad(90.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    scale = 2.75
    translation = np.array([4.0, -3.5, 1.25])
    dst = scale * (rotation @ src.T).T + translation
    return src, dst, rotation, scale, translation


def assert_umeyama(module):
    src, dst, expected_r, expected_s, expected_t = transformed_points()
    rotation, translation, scale = module.umeyama(src, dst)
    aligned = scale * (rotation @ src.T).T + translation
    rmse = np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1)))
    assert rmse < 1e-10, (module.__name__, rmse)
    assert np.linalg.det(rotation) > 0.0
    assert np.allclose(rotation, expected_r, atol=1e-10)
    assert np.isclose(scale, expected_s, atol=1e-10)
    assert np.allclose(translation, expected_t, atol=1e-10)


def assert_matched_timeline():
    # 第二个估计帧超出容差，应被跳过；返回时间轴必须仍对应第 1、3 帧。
    est = [
        (0.00, np.array([0.0, 0.0, 0.0]), np.zeros(4)),
        (0.40, np.array([1.0, 0.0, 0.0]), np.zeros(4)),
        (1.01, np.array([2.0, 0.0, 0.0]), np.zeros(4)),
    ]
    gt = [
        (0.02, np.array([0.0, 0.0, 0.0]), np.zeros(4)),
        (1.00, np.array([2.0, 0.0, 0.0]), np.zeros(4)),
    ]
    est_pts, gt_pts, matched_t = plot_traj.match_timestamps(est, gt, tol=0.05)
    assert len(est_pts) == len(gt_pts) == len(matched_t) == 2
    assert np.allclose(matched_t, [0.02, 1.00])
    assert np.allclose(est_pts[:, 0], [0.0, 2.0])


def assert_evaluator_timestamp_matching_is_one_to_one():
    est = [
        (0.000, np.zeros(3), np.array([0, 0, 0, 1.])),
        (0.009, np.ones(3), np.array([0, 0, 0, 1.])),
        (0.100, np.full(3, 2.), np.array([0, 0, 0, 1.])),
    ]
    gt = [
        (0.000, np.zeros(3), np.array([0, 0, 0, 1.])),
        (0.100, np.full(3, 2.), np.array([0, 0, 0, 1.])),
    ]
    pairs = evaluate_ate.match_timestamps(est, gt, tolerance=0.02)
    assert len(pairs) == 2
    assert [pair[1][0] for pair in pairs] == [0.0, 0.1]


def assert_stereo_se3_does_not_hide_scale_error():
    src, dst, _, _, _ = transformed_points()
    scaled_src = 0.5 * dst
    rotation, translation, scale = evaluate_ate.alignment(scaled_src, dst, "se3")
    aligned = scale * (rotation @ scaled_src.T).T + translation
    assert scale == 1.0
    assert np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))) > 0.1
    _, _, sim3_scale = evaluate_ate.alignment(scaled_src, dst, "sim3")
    assert np.isclose(sim3_scale, 2.0)


def assert_kitti_large_rotation_quaternions():
    for axis in np.eye(3):
        # 180 度旋转：R = 2 aa^T - I，必走 trace <= 0 分支。
        rotation = 2.0 * np.outer(axis, axis) - np.eye(3)
        quat = kitti_gt_to_tum.rot_to_quat(rotation)
        assert np.isclose(np.linalg.norm(quat), 1.0)
        assert np.allclose(np.abs(quat[:3]), axis)
        assert abs(quat[3]) < 1e-12


def assert_euroc_camera_extrinsic_composition():
    # 机体在世界原点、姿态为单位阵时，相机 T_WC 应恰好等于 T_BC。
    fields = ["1000000000", "0", "0", "0", "1", "0", "0", "0"]
    t_bc = np.eye(4)
    t_bc[:3, 3] = [0.1, -0.2, 0.3]
    timestamp, t_wc = euroc_gt_to_tum.row_to_camera_pose(fields, t_bc)
    assert np.isclose(timestamp, 1.0)
    assert np.allclose(t_wc, t_bc)


def main():
    assert_umeyama(evaluate_ate)
    assert_umeyama(plot_traj)
    assert_matched_timeline()
    assert_evaluator_timestamp_matching_is_one_to_one()
    assert_stereo_se3_does_not_hide_scale_error()
    assert_kitti_large_rotation_quaternions()
    assert_euroc_camera_extrinsic_composition()
    print("trajectory alignment regression tests: PASSED")


if __name__ == "__main__":
    main()
