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
    Umeyama: Σ_xy = (1/N)·Σ(y-μ_y)(x-μ_x)ᵀ = U·D·Vᵀ
             s = trace(D)/σ_x²，σ_x² = (1/N)·Σ||x-μ_x||²  （注意 1/N！）"""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (src - mu_s).T @ (dst - mu_d) / n
    U, D, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0:            # 反射修正（保证纯旋转）
        R = U @ np.diag([1, 1, -1]) @ Vt
    var_s = np.sum((src - mu_s) ** 2) / n   # 方差（含 1/N）
    s = np.trace(np.diag(D)) / var_s        # 尺度
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
