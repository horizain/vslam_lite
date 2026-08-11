#!/usr/bin/env python3
"""
compare_trajectories.py - M1 确定性回归对比（§5.6）

比较两个 TUM 轨迹文件（timestamp tx ty tz qx qy qz qw）与可选的状态 CSV，
报告：
  - 轨迹行数 / 时间戳序列是否一致
  - 最大平移差 (m)、最大旋转差 (rad)（默认 1e-6 m / 1e-8 rad）
  - 状态 CSV（frame_id,state,pose_valid,...,map revision）逐行序列是否一致

用法：
  python3 scripts/compare_trajectories.py A.tum B.tum
  python3 scripts/compare_trajectories.py A.tum B.tum --status A.csv B.csv
  python3 scripts/compare_trajectories.py A.tum B.tum --tol-trans 1e-9 --tol-rot 1e-10
"""

import argparse
import math
import sys


def load_tum(path):
    """解析 TUM 轨迹：返回 [(timestamp, tx, ty, tz, qx, qy, qz, qw)]。"""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            rows.append(tuple(float(p) for p in parts))
    return rows


def quat_diff_angle(qa, qb):
    """两四元数相对旋转角（rad）。qa/qb = (x, y, z, w)（TUM 顺序）。

    相对旋转 q_rel = qa⁻¹ ⊗ qb，用 angle = 2·atan2(‖v_rel‖, |w_rel|) 计算——
    对相同的四元数精确给出 0，避免 acos 在夹角≈0 时的病态放大小角误差。
    """
    xa, ya, za, wa = qa
    xb, yb, zb, wb = qb
    w = wa * wb + xa * xb + ya * yb + za * zb
    x = wa * xb - xa * wb - ya * zb + za * yb
    y = wa * yb + xa * zb - ya * wb - za * xb
    z = wa * zb - xa * yb + ya * xb - za * wb
    return 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(w))


def compare_traj(a, b, tol_trans, tol_rot):
    if not a or not b:
        print(f"[FAIL] 轨迹不得为空: {len(a)} vs {len(b)}")
        return False
    if len(a) != len(b):
        print(f"[FAIL] 行数不一致: {len(a)} vs {len(b)}")
        return False
    ok = True
    max_t = 0.0
    max_r = 0.0
    for i, (ra, rb) in enumerate(zip(a, b)):
        ta = ra[1:4]
        tb = rb[1:4]
        dt = math.sqrt(sum((ta[j] - tb[j]) ** 2 for j in range(3)))
        dr = quat_diff_angle(ra[4:8], rb[4:8])
        max_t = max(max_t, dt)
        max_r = max(max_r, dr)
        if dt > tol_trans or dr > tol_rot:
            ok = False
            print(f"[FAIL] 帧 {i}: 平移差 {dt:.3e} m (> {tol_trans:.1e}) "
                  f"或旋转差 {dr:.3e} rad (> {tol_rot:.1e})")
    print(f"max_translation_diff = {max_t:.3e} m (tol {tol_trans:.1e})")
    print(f"max_rotation_diff   = {max_r:.3e} rad (tol {tol_rot:.1e})")
    print("[PASS] 轨迹一致" if ok else "[FAIL] 轨迹不一致")
    return ok


def compare_status(a, b):
    """逐行比较状态 CSV（忽略首行表头）。"""
    with open(a, "r") as f:
        rows_a = [l.strip() for l in f if l.strip() and not l.startswith("frame_id")]
    with open(b, "r") as f:
        rows_b = [l.strip() for l in f if l.strip() and not l.startswith("frame_id")]
    if not rows_a or not rows_b:
        print(f"[FAIL] 状态序列不得为空: {len(rows_a)} vs {len(rows_b)}")
        return False
    if len(rows_a) != len(rows_b):
        print(f"[FAIL] 状态行数不一致: {len(rows_a)} vs {len(rows_b)}")
        return False
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
        if ra != rb:
            print(f"[FAIL] 状态行 {i} 不一致:\n  A: {ra}\n  B: {rb}")
            return False
    print(f"[PASS] 状态序列一致 ({len(rows_a)} 行: frame/state/pose_valid/kf/revision)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traj_a", help="轨迹 A（TUM 格式）")
    parser.add_argument("traj_b", help="轨迹 B（TUM 格式）")
    parser.add_argument("--status", nargs=2, metavar=("CSV_A", "CSV_B"),
                        help="两个逐帧状态 CSV，逐行比较")
    parser.add_argument("--tol-trans", type=float, default=1e-6,
                        help="最大平移差门限（m），默认 1e-6")
    parser.add_argument("--tol-rot", type=float, default=1e-8,
                        help="最大旋转差门限（rad），默认 1e-8")
    args = parser.parse_args()

    a = load_tum(args.traj_a)
    b = load_tum(args.traj_b)
    traj_ok = compare_traj(a, b, args.tol_trans, args.tol_rot)

    status_ok = True
    if args.status:
        status_ok = compare_status(args.status[0], args.status[1])

    sys.exit(0 if (traj_ok and status_ok) else 1)


if __name__ == "__main__":
    main()
