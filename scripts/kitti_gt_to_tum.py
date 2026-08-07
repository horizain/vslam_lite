#!/usr/bin/env python3
"""
kitti_gt_to_tum.py - KITTI 真值位姿 → TUM 轨迹格式

KITTI datasets/kitti/poses/XX.txt 每行是 3x4 矩阵 [R|t]（T_wc：相机系 → 世界系），
TUM 格式每行: timestamp tx ty tz qx qy qz qw（也是 T_wc，与 run_vo 输出一致）

用法:
  python3 scripts/kitti_gt_to_tum.py <poses.txt> <output.tum> [fps=10]
"""
import sys
import numpy as np


def rot_to_quat(R):
    """旋转矩阵 → 四元数 (x, y, z, w)，Shepperd's method，数值稳定"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s,
                      0.25 * s])
    else:
        i = np.argmax(np.diag(R))
        j = (i + 1) % 3
        k = (j + 1) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q /= np.linalg.norm(q)
    # 保证 w > 0（四元数 q 与 -q 等价，统一符号便于比较）
    if q[3] < 0:
        q = -q
    return q


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    with open(sys.argv[1]) as f:
        lines = [l for l in f if l.strip()]

    with open(sys.argv[2], "w") as out:
        for i, line in enumerate(lines):
            m = np.array([float(x) for x in line.split()]).reshape(3, 4)
            R, t = m[:3, :3], m[:3, 3]
            q = rot_to_quat(R)
            ts = i / fps  # KITTI 无时间戳，按 10fps 换算（与 run_vo 的假时间戳一致）
            out.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                      f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    print(f"转换完成: {len(lines)} 帧 → {sys.argv[2]}")


if __name__ == "__main__":
    main()
