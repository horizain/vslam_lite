#!/usr/bin/env python3
"""
evaluate_ate.py - 轨迹质量评估（无需 EVO）

按时间戳一一关联两条 TUM 轨迹，输出 ATE、逐帧 RPE 与连续性指标。
双目/RGB-D 默认只做 SE3 对齐，不允许全局缩放掩盖尺度错误；单目请显式
传入 ``--alignment sim3``。

用法:
  python3 scripts/evaluate_ate.py <est.tum> <gt.tum>
  python3 scripts/evaluate_ate.py <est.tum> <gt.tum> --alignment sim3
"""
import argparse
import sys

import numpy as np


def load_tum(path):
    """TUM: timestamp tx ty tz qx qy qz qw。"""
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = [float(x) for x in line.split()]
            if len(p) < 8:
                continue
            q = np.asarray(p[4:8], dtype=float)
            q_norm = np.linalg.norm(q)
            if q_norm <= np.finfo(float).eps:
                continue
            data.append((p[0], np.asarray(p[1:4], dtype=float), q / q_norm))
    return data


def quat_to_mat(q):
    """TUM 四元数 (x,y,z,w) -> 旋转矩阵。"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def umeyama(src, dst):
    """Sim3: dst ~= s R src + t。"""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src/dst 必须是形状相同的 (N,3) 点集")
    if len(src) < 3:
        raise ValueError("Umeyama 对齐至少需要 3 个点")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_centered = src - mu_s
    dst_centered = dst - mu_d
    covariance = src_centered.T @ dst_centered / len(src)
    U, singular_values, Vt = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        signs[-1] = -1.0
    R = Vt.T @ np.diag(signs) @ U.T
    var_s = np.sum(src_centered ** 2) / len(src)
    if var_s <= np.finfo(float).eps:
        raise ValueError("源点集方差为 0，无法估计对齐")
    scale = np.dot(singular_values, signs) / var_s
    return R, mu_d - scale * R @ mu_s, scale


def match_timestamps(est, gt, tolerance=0.02):
    """单调、一一的最近时间戳关联，避免一帧 GT 被重复使用。"""
    if not est or not gt:
        return []
    gt_times = np.asarray([item[0] for item in gt])
    pairs = []
    last_gt = -1
    for est_item in est:
        pos = int(np.searchsorted(gt_times, est_item[0]))
        candidates = [idx for idx in (pos - 1, pos)
                      if last_gt < idx < len(gt)]
        if not candidates:
            continue
        idx = min(candidates, key=lambda k: abs(gt_times[k] - est_item[0]))
        if abs(gt_times[idx] - est_item[0]) <= tolerance:
            pairs.append((est_item, gt[idx]))
            last_gt = idx
    return pairs


def alignment(src, dst, mode):
    """返回 dst ~= scale * R * src + t。"""
    if mode == "none":
        return np.eye(3), np.zeros(3), 1.0
    R, _, sim3_scale = umeyama(src, dst)
    scale = sim3_scale if mode == "sim3" else 1.0
    return R, dst.mean(0) - scale * R @ src.mean(0), scale


def rotation_angle_deg(R):
    value = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def relative_pose_errors(pairs, scale):
    """相邻已关联帧的完整 SE3 RPE；Sim3 模式只缩放估计平移。"""
    trans_errors, rot_errors = [], []
    for (est0, gt0), (est1, gt1) in zip(pairs, pairs[1:]):
        Re0, Re1 = quat_to_mat(est0[2]), quat_to_mat(est1[2])
        Rg0, Rg1 = quat_to_mat(gt0[2]), quat_to_mat(gt1[2])
        de_t = Re0.T @ (scale * (est1[1] - est0[1]))
        dg_t = Rg0.T @ (gt1[1] - gt0[1])
        de_R = Re0.T @ Re1
        dg_R = Rg0.T @ Rg1
        trans_errors.append(np.linalg.norm(de_t - dg_t))
        rot_errors.append(rotation_angle_deg(de_R.T @ dg_R))
    return np.asarray(trans_errors), np.asarray(rot_errors)


def continuity_metrics(est):
    positions = np.asarray([item[1] for item in est])
    if len(positions) < 2:
        return np.array([]), np.array([])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    rotations = np.asarray([
        rotation_angle_deg(quat_to_mat(a[2]).T @ quat_to_mat(b[2]))
        for a, b in zip(est, est[1:])
    ])
    return steps, rotations


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Timestamp-associated ATE/RPE evaluator")
    ap.add_argument("est")
    ap.add_argument("gt")
    ap.add_argument("--alignment", choices=("se3", "sim3", "none"), default="se3",
                    help="双目默认 se3；单目使用 sim3")
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--json", action="store_true",
                    help="仅输出机器可读 JSON 摘要（benchmark.py 消费，不做正则解析）")
    args = ap.parse_args(argv)

    est, gt = load_tum(args.est), load_tum(args.gt)
    pairs = match_timestamps(est, gt, args.max_time_diff)
    if len(pairs) < 3:
        print(f"错误: 有效时间戳匹配仅 {len(pairs)} 对", file=sys.stderr)
        return 1

    est_pts = np.asarray([pair[0][1] for pair in pairs])
    gt_pts = np.asarray([pair[1][1] for pair in pairs])
    R, t, scale = alignment(est_pts, gt_pts, args.alignment)
    aligned = scale * (R @ est_pts.T).T + t
    ate = np.linalg.norm(aligned - gt_pts, axis=1)
    rpe_t, rpe_r = relative_pose_errors(pairs, scale)
    steps, step_rot = continuity_metrics(est)
    est_len = float(steps.sum())
    gt_len = float(np.linalg.norm(np.diff(gt_pts, axis=0), axis=1).sum())
    coverage = len(pairs) / len(gt) * 100.0

    summary = {
        "matched": len(pairs), "est_total": len(est), "gt_total": len(gt),
        "coverage_pct": coverage,
        "ate_rmse": rmse(ate), "ate_mean": float(np.mean(ate)),
        "ate_std": float(np.std(ate)), "ate_max": float(np.max(ate)),
        "ate_p95": float(np.percentile(ate, 95)),
        "rpe_trans_rmse": rmse(rpe_t), "rpe_trans_mean": float(np.mean(rpe_t)),
        "rpe_trans_max": float(np.max(rpe_t)),
        "rpe_rot_rmse": rmse(rpe_r), "rpe_rot_mean": float(np.mean(rpe_r)),
        "rpe_rot_max": float(np.max(rpe_r)),
        "est_len": est_len, "gt_len": gt_len, "len_ratio": est_len / gt_len if gt_len else float("nan"),
        "jumps_3m": int(np.count_nonzero(steps > 3.0)) if len(steps) else 0,
        "jumps_5m": int(np.count_nonzero(steps > 5.0)) if len(steps) else 0,
        "jumps_10m": int(np.count_nonzero(steps > 10.0)) if len(steps) else 0,
        "step_p95": float(np.percentile(steps, 95)) if len(steps) else float("nan"),
        "rot_step_p99": float(np.percentile(step_rot, 99)) if len(step_rot) else float("nan"),
    }

    if args.json:
        import json
        print(json.dumps(summary, indent=2))
        return 0

    print(f"时间戳匹配: {len(pairs)} (估计 {len(est)} / 真值 {len(gt)}, "
          f"覆盖率={coverage:.2f}%)")
    print(f"对齐模式: {args.alignment.upper()}  scale={scale:.6f} "
          f"旋转={rotation_angle_deg(R):.3f}deg 平移={np.linalg.norm(t):.3f}m")
    print(f"轨迹长度: 估计={est_len:.3f}m 匹配GT={gt_len:.3f}m 比值={est_len / gt_len:.6f}")
    print(f"ATE  RMSE = {rmse(ate):.3f} m")
    print(f"ATE  Mean = {np.mean(ate):.3f} m")
    print(f"ATE  Std  = {np.std(ate):.3f} m")
    print(f"ATE  Max  = {np.max(ate):.3f} m")
    print(f"ATE  P95  = {np.percentile(ate, 95):.3f} m")
    print(f"RPE  Trans RMSE = {rmse(rpe_t):.3f} m/frame")
    print(f"RPE  Trans Mean/Max = {np.mean(rpe_t):.3f}/{np.max(rpe_t):.3f} m")
    print(f"RPE  Rot RMSE = {rmse(rpe_r):.3f} deg/frame")
    print(f"RPE  Rot Mean/Max = {np.mean(rpe_r):.3f}/{np.max(rpe_r):.3f} deg")
    if len(steps):
        print(f"Step P95/P99/Max = {np.percentile(steps, 95):.3f}/"
              f"{np.percentile(steps, 99):.3f}/{np.max(steps):.3f} m")
        print("Step jumps >3m/>5m/>10m = "
              f"{np.count_nonzero(steps > 3.0)}/{np.count_nonzero(steps > 5.0)}/"
              f"{np.count_nonzero(steps > 10.0)}")
        print(f"Rotation step P99/Max = {np.percentile(step_rot, 99):.3f}/"
              f"{np.max(step_rot):.3f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
