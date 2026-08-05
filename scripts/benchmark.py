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
  python3 scripts/benchmark.py datasets/sequences/00 config/kitti00.yaml /tmp/bench_async

  # 对比两个配置（各跑 3 次，输出对比表）
  python3 scripts/benchmark.py datasets/sequences/00 config/kitti00.yaml /tmp/bench_ab \
      --runs 3 --compare config/kitti00_sync.yaml

  # 指定可执行文件 / 真值
  python3 scripts/benchmark.py ... --bin build/bin/run_slam --gt kitti_gt.txt

输出：<out_dir>/results.json + 控制台汇总表。
"""
import argparse
import json
import os
import re
import subprocess
import sys

RE_FPS = re.compile(r"Done\. Processed (\d+) frames in ([\d.]+) seconds \(([\d.]+) FPS\)")
RE_LOST = re.compile(r"Tracking lost")
RE_ANCHOR = re.compile(r"creating anchored submap")
RE_LOOP = re.compile(r"Loop closed!")
RE_BA_CAP = re.compile(r"BA point cap")
RE_BACKEND = re.compile(r"Async backend ENABLED")


def run_once(args, out_path, run_dir):
    """跑一次全程，返回指标 dict。"""
    log_path = os.path.join(run_dir, "run.log")
    cmd = [args.bin, args.dataset, args.config, out_path, "--headless"]
    with open(log_path, "w") as logf:
        subprocess.run(cmd, cwd=args.cwd, stdout=logf, stderr=subprocess.STDOUT)

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

    # ATE（无真值则跳过）
    if args.gt and os.path.exists(out_path):
        ate_out = subprocess.run(
            [sys.executable, args.evaluate, out_path, args.gt],
            capture_output=True, text=True).stdout
        for key in ("RMSE", "Mean", "Max"):
            mm = re.search(rf"ATE\s+{key}\s*=\s*([\d.]+)", ate_out)
            m[f"ate_{key.lower()}"] = float(mm.group(1)) if mm else -1.0

    # perf.csv（run_slam 退出时 dump 到 cwd；拷贝走避免互相覆盖）
    perf_path = os.path.join(args.cwd, "perf.csv")
    if os.path.exists(perf_path):
        try:
            import shutil
            dst = os.path.join(run_dir, "perf.csv")
            shutil.copy2(perf_path, dst)
            os.remove(perf_path)
            for line in open(dst):
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
    ap.add_argument("--cwd", default=".", help="工作目录（默认仓库根，perf.csv 在此生成）")
    ap.add_argument("--evaluate", default="scripts/evaluate_ate.py")
    args = ap.parse_args()

    configs = [("A", args.config)]
    if args.compare:
        configs.append(("B", args.compare))

    all_results = {}
    for label, cfg in configs:
        results = []
        for i in range(args.runs):
            run_dir = os.path.join(args.out_dir, f"{label}_run{i+1}")
            os.makedirs(run_dir, exist_ok=True)
            out_path = os.path.join(run_dir, "traj.txt")
            print(f"[bench] {label} run {i+1}/{args.runs} config={cfg} ...",
                  flush=True)
            results.append(run_once(args, out_path, run_dir))
        all_results[label] = summarize(results)
        with open(os.path.join(args.out_dir, f"results_{label}.json"), "w") as f:
            json.dump({k: list(v) for k, v in all_results[label].items()}, f,
                      indent=2)

    # ---- 输出 ----
    metric_names = [
        ("fps", "FPS", "%.2f"), ("seconds", "耗时(s)", "%.1f"),
        ("ate_rmse", "ATE RMSE(m)", "%.2f"), ("ate_mean", "ATE Mean(m)", "%.2f"),
        ("ate_max", "ATE Max(m)", "%.2f"), ("lost", "LOST 次数", "%.1f"),
        ("submap_reinit", "子地图重建", "%.1f"), ("loops", "闭环次数", "%.1f"),
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
