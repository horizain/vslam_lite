#!/usr/bin/env python3
"""EuRoC 状态真值转换为 cam0 的 TUM 轨迹。

EuRoC ``state_groundtruth_estimate0/data.csv`` 给出机体系 ``T_WB``，
``cam0/sensor.yaml`` 给出相机到机体的 ``T_BC``（文件字段 ``T_BS``）。
输出严格使用 ``T_WC = T_WB * T_BC``，可与本项目输出的相机轨迹比较。

用法：
  python3 scripts/euroc_gt_to_tum.py <mav0目录> <output.tum>
"""

import re
import sys
from pathlib import Path

import numpy as np

from kitti_gt_to_tum import rot_to_quat


def quat_wxyz_to_rot(q):
    """四元数 (w,x,y,z) 转旋转矩阵。"""
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def load_t_bc(sensor_yaml):
    """从 EuRoC sensor.yaml 读取 T_BS；cam0 中 S 即 camera。"""
    text = Path(sensor_yaml).read_text(encoding="utf-8")
    start = text.find("T_BS:")
    if start < 0:
        raise ValueError(f"缺少 T_BS: {sensor_yaml}")
    match = re.search(r"data\s*:\s*\[([^]]+)\]", text[start:], re.DOTALL)
    if not match:
        raise ValueError(f"T_BS 缺少 data: {sensor_yaml}")
    values = [float(value) for value in match.group(1).replace(",", " ").split()]
    if len(values) != 16:
        raise ValueError(f"T_BS 应有 16 个元素，实际 {len(values)}")
    return np.asarray(values).reshape(4, 4)


def row_to_camera_pose(fields, t_bc):
    """CSV 一行与 T_BC 合成为 (timestamp_s, T_WC)。"""
    timestamp = int(fields[0].strip()) * 1e-9
    position = np.asarray([float(value) for value in fields[1:4]])
    rotation = quat_wxyz_to_rot([float(value) for value in fields[4:8]])
    t_wb = np.eye(4)
    t_wb[:3, :3] = rotation
    t_wb[:3, 3] = position
    return timestamp, t_wb @ t_bc


def convert(mav0_dir, output_path):
    mav0 = Path(mav0_dir)
    csv_path = mav0 / "state_groundtruth_estimate0" / "data.csv"
    t_bc = load_t_bc(mav0 / "cam0" / "sensor.yaml")
    count = 0
    with csv_path.open(encoding="utf-8") as source, Path(output_path).open("w") as out:
        for line in source:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.strip().split(",")
            if len(fields) < 8:
                continue
            timestamp, t_wc = row_to_camera_pose(fields, t_bc)
            q = rot_to_quat(t_wc[:3, :3])
            t = t_wc[:3, 3]
            out.write(f"{timestamp:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                      f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
            count += 1
    return count


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    count = convert(sys.argv[1], sys.argv[2])
    print(f"转换完成: {count} 个真值状态 -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
