#!/usr/bin/env python3
"""
evaluate_ate.py - 轻量 ATE 评估（无需 EVO）

输入两条 TUM 格式轨迹（估计 + 真值），做 Sim3 对齐（Umeyama 算法），
输出绝对轨迹误差（ATE）的 RMSE / Mean / Max。

单目 VO 无真实尺度，因此 Sim3 对齐包含尺度估计——这是评估前必须的一步。

用法:
  python3 scripts/evaluate_ate.py <est.tum> <gt.tum>
"""
import sys
import numpy as np


def load_tum(path):
    """TUM 格式: timestamp tx ty tz qx qy qz qw → (t, pos, quat) 列表"""
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = [float(x) for x in line.split()]
            if len(p) < 8:
                continue
            data.append((p[0], np.array(p[1:4]), np.array(p[4:8])))
    return data


def umeyama(src, dst):
    """Sim3 对齐: dst ≈ s·R·src + t。src/dst: (N,3) 点集
    使用 Σ_xy=(x-μ_x)ᵀ(y-μ_y)/N 时，旋转为 V·S·Uᵀ；
    S 的最后一项同时用于反射修正和尺度计算。"""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src/dst 必须是形状相同的 (N,3) 点集")
    if len(src) < 3:
        raise ValueError("Umeyama 对齐至少需要 3 个点")

    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_centered = src - mu_s
    dst_centered = dst - mu_d
    covariance = src_centered.T @ dst_centered / n
    U, singular_values, Vt = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        signs[-1] = -1.0
    R = Vt.T @ np.diag(signs) @ U.T
    var_s = np.sum((src - mu_s) ** 2) / n   # 方差（含 1/N）
    if var_s <= np.finfo(float).eps:
        raise ValueError("源点集方差为 0，无法估计 Sim3")
    s = np.dot(singular_values, signs) / var_s
    t = mu_d - s * R @ mu_s
    return R, t, s


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    est = load_tum(sys.argv[1])
    gt = load_tum(sys.argv[2])
    n = min(len(est), len(gt))          # 按帧对应（时间戳对不齐时取公共帧数）
    if n == 0:
        print("错误: 轨迹为空")
        sys.exit(1)

    est_pts = np.array([e[1] for e in est[:n]])
    gt_pts = np.array([g[1] for g in gt[:n]])

    # Sim3 对齐（估计 → 真值）
    R, t, s = umeyama(est_pts, gt_pts)
    aligned = (s * (R @ est_pts.T).T + t)

    # ATE
    err = np.linalg.norm(aligned - gt_pts, axis=1)
    print(f"轨迹帧数: {n}")
    print(f"Sim3 对齐: scale={s:.4f} 旋转={np.linalg.norm(R - np.eye(3)):.4f}(偏离 I) 平移={np.linalg.norm(t):.2f}")
    print(f"ATE  RMSE = {np.sqrt(np.mean(err**2)):.3f} m")
    print(f"ATE  Mean = {np.mean(err):.3f} m")
    print(f"ATE  Max  = {np.max(err):.3f} m")
    print(f"ATE  Std  = {np.std(err):.3f} m")


if __name__ == "__main__":
    main()
