#!/usr/bin/env python3
"""
plot_traj.py - 绘制 TUM 轨迹 + 叠加真值 + ATE 准确率评估

用法:
  python3 scripts/plot_traj.py <est.tum> <gt.tum> [--out figure.png]

输入两条 TUM 格式轨迹（timestamp tx ty tz qx qy qz qw，位姿为 T_wc）。
按时间戳最近邻匹配帧对 → Sim3 对齐（Umeyama）→ 输出 ATE（RMSE/Mean/Max/Std）
与相对误差（ATE_RMSE / 轨迹长度），并绘制：
  1) XY 俯视叠加图（估计 + 真值）
  2) 3D 轨迹
  3) 逐帧 ATE 误差曲线
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（无 CJK 字体环境时标签显示为方块，可改用英文）
for fp in ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break


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


def quat_to_mat(q):
    """TUM 四元数 (x,y,z,w) → 3x3 旋转矩阵"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def umeyama(src, dst):
    """Sim3 对齐: dst ≈ s·R·src + t（与 evaluate_ate.py 一致）"""
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
    var_s = np.sum(src_centered ** 2) / n
    if var_s <= np.finfo(float).eps:
        raise ValueError("源点集方差为 0，无法估计 Sim3")
    s = np.dot(singular_values, signs) / var_s
    t = mu_d - s * R @ mu_s
    return R, t, s


def match_timestamps(est, gt, tol=0.05):
    """按时间戳最近邻匹配帧对，返回位置数组和每个匹配对应的 GT 时间戳。"""
    gt_t = np.array([g[0] for g in gt])
    gt_pos = np.array([g[1] for g in gt])
    e_idx, g_idx = [], []
    for i, e in enumerate(est):
        k = np.argmin(np.abs(gt_t - e[0]))
        if abs(gt_t[k] - e[0]) <= tol:
            e_idx.append(i)
            g_idx.append(k)
    g_idx = np.asarray(g_idx, dtype=int)
    return (np.array([est[i][1] for i in e_idx]),
            gt_pos[g_idx], gt_t[g_idx])


def main():
    ap = argparse.ArgumentParser(description="Plot TUM trajectory + GT overlay + ATE")
    ap.add_argument("est", help="estimated trajectory (.tum)")
    ap.add_argument("gt", help="ground truth (.tum)")
    ap.add_argument("--out", default=None, help="output figure path (default: <est>.plot.png)")
    args = ap.parse_args()

    est = load_tum(args.est)
    gt = load_tum(args.gt)
    if not est or not gt:
        print("错误: 轨迹为空")
        return 1
    out = args.out or (os.path.splitext(args.est)[0] + ".plot.png")

    e_pts, g_pts, matched_t = match_timestamps(est, gt)
    n = len(e_pts)
    if n < 3:
        print(f"错误: 有效匹配帧对仅 {n} 个")
        return 1

    # ---- ATE（Sim3 对齐）----
    R, t, s = umeyama(e_pts, g_pts)
    aligned = s * (R @ e_pts.T).T + t
    err = np.linalg.norm(aligned - g_pts, axis=1)
    rmse, mean, mx, std = (np.sqrt(np.mean(err ** 2)), np.mean(err),
                           np.max(err), np.std(err))
    est_len = np.sum(np.linalg.norm(np.diff(e_pts, axis=0), axis=1))
    gt_len = np.sum(np.linalg.norm(np.diff(g_pts, axis=0), axis=1))

    print(f"匹配帧数: {n} (估计 {len(est)} / 真值 {len(gt)})")
    print(f"Sim3 对齐: scale={s:.4f}  平移={np.linalg.norm(t):.2f}")
    print(f"轨迹长度: 估计 {est_len:.1f}m  真值 {gt_len:.1f}m")
    print(f"ATE  RMSE = {rmse:.3f} m")
    print(f"ATE  Mean = {mean:.3f} m")
    print(f"ATE  Max  = {mx:.3f} m")
    print(f"ATE  Std  = {std:.3f} m")
    print(f"相对误差  = {rmse / gt_len * 100:.2f}% (ATE_RMSE / 真值轨迹长度)")

    # ---- 绘制 ----
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(g_pts[:, 0], g_pts[:, 1], "-", color="tab:gray", lw=1.8, label="Ground Truth")
    ax1.plot(e_pts[:, 0], e_pts[:, 1], "-", color="tab:blue", lw=1.2, label="Estimate (raw)")
    ax1.plot(aligned[:, 0], aligned[:, 1], "-", color="tab:red", lw=1.2, label="Estimate (Sim3 aligned)")
    ax1.scatter(g_pts[0, 0], g_pts[0, 1], marker="o", s=60, color="k", zorder=5)
    ax1.scatter(g_pts[-1, 0], g_pts[-1, 1], marker="s", s=60, color="k", zorder=5)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_title("XY 俯视轨迹叠加")
    ax1.legend(fontsize=8); ax1.axis("equal"); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot(g_pts[:, 0], g_pts[:, 1], g_pts[:, 2], color="tab:gray", lw=1.5, label="GT")
    ax2.plot(aligned[:, 0], aligned[:, 1], aligned[:, 2], color="tab:red", lw=1.0, label="Est (aligned)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.set_title("3D 轨迹")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(1, 3, 3)
    # 横轴直接使用 match_timestamps 接受的同一组索引，避免中间存在未匹配
    # 估计帧时，简单截断全部时间戳造成误差与时间错位。
    ax3.plot(matched_t, err, color="tab:red", lw=1.0)
    ax3.axhline(rmse, color="tab:purple", ls="--", lw=1.0, label=f"RMSE {rmse:.2f} m")
    ax3.axhline(mx, color="tab:orange", ls=":", lw=1.0, label=f"Max {mx:.2f} m")
    ax3.set_xlabel("时间 [s]"); ax3.set_ylabel("ATE [m]")
    ax3.set_title("逐帧 ATE 误差")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.suptitle(f"{os.path.basename(args.est)} vs {os.path.basename(args.gt)}"
                 f"  |  ATE RMSE={rmse:.2f}m  ({rmse / gt_len * 100:.2f}%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"图已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
