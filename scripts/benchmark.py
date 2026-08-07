#!/usr/bin/env python3
"""
benchmark.py - VSLAM 全程基准测试套件

对同一数据集+配置跑 N 次完整 SLAM（默认 2 次），汇总：
  - 全程耗时 / FPS（解析 run_slam 输出）
  - ATE（RMSE/Mean/Max，复用 evaluate_ate.py）
  - 跟踪稳定性：LOST / 子地图重建 / 闭环次数（解析日志）
  - 后端性能：BA 最大单次耗时（perf.csv，如有）

用法:
  # 单配置基准（跑 2 次取均值）
  python3 scripts/benchmark.py datasets/kitti/sequences/00 config/kitti00.yaml /tmp/bench_async \
      --expected-frames 4541

  # EuRoC 单目（自动传 --euroc，Sim3 对齐）
  python3 scripts/benchmark.py datasets/euroc/V1_01_easy/mav0 config/default.yaml \
      /tmp/bench_euroc --format euroc --alignment sim3 --expected-frames 2912

  # 对比两个配置（各跑 3 次，输出对比表）
  python3 scripts/benchmark.py datasets/kitti/sequences/00 config/kitti00.yaml /tmp/bench_ab \
      --runs 3 --compare config/kitti00_sync.yaml

  # 指定可执行文件 / 真值
  python3 scripts/benchmark.py ... --bin build/bin/run_slam --gt kitti_gt.txt

输出：<out_dir>/results_A.json（对照配置另写 results_B.json）+ 控制台汇总表。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RE_FPS = re.compile(r"Done\. Processed (\d+) frames in ([\d.]+) seconds \(([\d.]+) FPS\)")
RE_LOST = re.compile(r"Tracking lost")
RE_ANCHOR = re.compile(r"creating anchored submap")
RE_LOOP = re.compile(r"Loop closed!")
RE_BA_CAP = re.compile(r"BA point cap")
RE_BACKEND = re.compile(r"Async backend ENABLED")
RE_FINAL_MAP = re.compile(r"Final map: (\d+) points, (\d+) keyframes")
RE_MATCHED = re.compile(r"时间戳匹配:\s*(\d+)")


def run_once(args, out_path, run_dir):
    """跑一次全程，返回指标 dict。"""
    log_path = str(Path(run_dir) / "run.log")
    cmd = [args.bin, args.dataset, args.config, out_path, "--headless"]
    if args.format != "kitti":
        cmd.append(f"--{args.format}")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=args.cwd, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"run_slam 失败(returncode={proc.returncode})，见 {log_path}")

    m = {}
    log = open(log_path).read()
    fps_m = RE_FPS.search(log)
    m["frames"] = int(fps_m.group(1)) if fps_m else -1
    m["seconds"] = float(fps_m.group(2)) if fps_m else -1.0
    m["fps"] = float(fps_m.group(3)) if fps_m else -1.0
    m["lost"] = len(RE_LOST.findall(log))
    m["submap_reinit"] = len(RE_ANCHOR.findall(log))
    m["loops"] = len(RE_LOOP.findall(log))
    m["async"] = bool(RE_BACKEND.search(log))
    final_map = RE_FINAL_MAP.search(log)
    m["map_points"] = int(final_map.group(1)) if final_map else -1
    m["keyframes"] = int(final_map.group(2)) if final_map else -1
    if m["frames"] <= 0:
        raise RuntimeError(f"未解析到完整运行统计，见 {log_path}")
    if args.expected_frames and m["frames"] != args.expected_frames:
        raise RuntimeError(f"帧数不完整: {m['frames']} != {args.expected_frames}")
    with open(out_path) as trajectory:
        m["valid_poses"] = sum(
            1 for line in trajectory if line.strip() and not line.startswith("#"))
    m["valid_ratio"] = m["valid_poses"] / m["frames"]

    # ATE（无真值则跳过）
    if args.gt and Path(out_path).exists():
        ate_out = subprocess.run(
            [sys.executable, args.evaluate, out_path, args.gt,
             "--alignment", args.alignment], capture_output=True, text=True)
        if ate_out.returncode != 0:
            raise RuntimeError(f"轨迹评估失败: {ate_out.stderr.strip()}")
        ate_text = ate_out.stdout
        matched = RE_MATCHED.search(ate_text)
        m["matched_poses"] = int(matched.group(1)) if matched else -1
        for key in ("RMSE", "Mean", "Max"):
            mm = re.search(rf"ATE\s+{key}\s*=\s*([\d.]+)", ate_text)
            m[f"ate_{key.lower()}"] = float(mm.group(1)) if mm else -1.0
        for name, pattern in {
            "rpe_trans_rmse": r"RPE\s+Trans RMSE\s*=\s*([\d.]+)",
            "rpe_rot_rmse": r"RPE\s+Rot RMSE\s*=\s*([\d.]+)",
            "jumps_10m": r"Step jumps >3m/>5m/>10m\s*=\s*\d+/\d+/(\d+)",
        }.items():
            match = re.search(pattern, ate_text)
            m[name] = float(match.group(1)) if match else -1.0

    # 性能数据与轨迹同目录输出，避免复制/删除工作区已有 perf.csv。
    perf_path = out_path + ".perf.csv"
    if Path(perf_path).exists():
        try:
            for line in open(perf_path):
                parts = line.strip().split(",")
                if parts and parts[0] == "opt.ba":
                    m["ba_max_ms"] = float(parts[5])
                if parts and parts[0] == "loop.global_ba":
                    m["global_ba_max_ms"] = float(parts[5])
        except (OSError, IndexError, ValueError):
            pass
    return m


def summarize(results):
    """多轮 → {指标: (mean, std)}"""
    out = {}
    keys = [k for k in results[0] if k != "async"]
    for k in keys:
        vals = [r[k] for r in results]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        out[k] = (mean, std)
    return out


def fmt(v, nd=2):
    mean, std = v
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="VSLAM 全程基准测试")
    ap.add_argument("dataset")
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--runs", type=int, default=2, help="每配置运行次数（默认 2）")
    ap.add_argument("--compare", default=None, help="对比配置（同 runs 次数）")
    ap.add_argument("--bin", default="build/bin/run_slam")
    ap.add_argument("--gt", default=None, help="真值轨迹（TUM），给则评估 ATE")
    ap.add_argument("--alignment", choices=("se3", "sim3", "none"), default="se3",
                    help="双目默认 se3；单目请显式选择 sim3")
    ap.add_argument("--format", choices=("kitti", "tum", "euroc"), default="kitti",
                    help="数据集格式；非 KITTI 时向 run_slam 传对应标志")
    ap.add_argument("--expected-frames", type=int, default=0,
                    help="必须完整处理的帧数；0 表示不校验")
    ap.add_argument("--cwd", default=".", help="工作目录（默认仓库根，perf.csv 在此生成）")
    ap.add_argument("--evaluate", default="scripts/evaluate_ate.py")
    args = ap.parse_args()

    # 子进程 cwd 可能不同；输入、二进制和评估脚本都转换为绝对路径。
    args.cwd = str(Path(args.cwd).resolve())
    args.dataset = str(Path(args.dataset).resolve())
    args.bin = str(Path(args.bin).resolve())
    args.evaluate = str(Path(args.evaluate).resolve())
    args.gt = str(Path(args.gt).resolve()) if args.gt else None
    args.out_dir = str(Path(args.out_dir).resolve())
    args.config = str(Path(args.config).resolve())
    args.compare = str(Path(args.compare).resolve()) if args.compare else None

    configs = [("A", args.config)]
    if args.compare:
        configs.append(("B", args.compare))

    all_results = {}
    for label, cfg in configs:
        results = []
        for i in range(args.runs):
            run_dir = str(Path(args.out_dir) / f"{label}_run{i+1}")
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            out_path = str(Path(run_dir) / "traj.txt")
            print(f"[bench] {label} run {i+1}/{args.runs} config={cfg} ...",
                  flush=True)
            results.append(run_once(args, out_path, run_dir))
        all_results[label] = summarize(results)
        with open(Path(args.out_dir) / f"results_{label}.json", "w") as f:
            json.dump({k: list(v) for k, v in all_results[label].items()}, f,
                      indent=2)

    # ---- 输出 ----
    metric_names = [
        ("fps", "FPS", "%.2f"), ("seconds", "耗时(s)", "%.1f"),
        ("valid_poses", "有效位姿", "%.0f"),
        ("valid_ratio", "有效位姿率", "%.2f"),
        ("matched_poses", "GT匹配位姿", "%.0f"),
        ("ate_rmse", "ATE RMSE(m)", "%.2f"), ("ate_mean", "ATE Mean(m)", "%.2f"),
        ("ate_max", "ATE Max(m)", "%.2f"),
        ("rpe_trans_rmse", "RPE trans(m/f)", "%.2f"),
        ("rpe_rot_rmse", "RPE rot(deg/f)", "%.2f"),
        ("jumps_10m", ">10m 跳变", "%.1f"), ("lost", "LOST 次数", "%.1f"),
        ("submap_reinit", "子地图重建", "%.1f"), ("loops", "闭环次数", "%.1f"),
        ("map_points", "最终地图点", "%.0f"), ("keyframes", "最终关键帧", "%.0f"),
        ("ba_max_ms", "Local BA max(ms)", "%.0f"),
        ("global_ba_max_ms", "全局 BA max(ms)", "%.0f"),
    ]
    print(f"\n===== Benchmark: {args.dataset} =====")
    print(f"{'指标':<18}" + "".join(f"{l:>22}" for l, _ in configs))
    for key, name, nd in metric_names:
        row = []
        for label, _ in configs:
            v = all_results[label].get(key)
            row.append(fmt(v, 0 if nd == "%.0f" else 2) if v else "-")
        print(f"{name:<18}" + "".join(f"{r:>22}" for r in row))

    # 提升百分比（B vs A，仅双侧都有）
    if len(configs) == 2:
        print("\n===== 相对提升（B vs A）=====")
        for key, name, nd in metric_names:
            a = all_results["A"].get(key)
            b = all_results["B"].get(key)
            if not a or not b or a[0] == 0:
                continue
            pct = (b[0] - a[0]) / a[0] * 100
            better = ("↓" if pct < 0 else "↑") if key not in ("fps",) else ("↑" if pct > 0 else "↓")
            print(f"{name:<18} {pct:+.1f}% {better}")


if __name__ == "__main__":
    main()
