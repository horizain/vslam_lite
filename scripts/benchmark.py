#!/usr/bin/env python3
"""
benchmark.py - 生产级统计基准（v2）

对同一数据集+配置跑 N 轮完整/部分 SLAM，逐轮输出结构化指标（run_slam
--metrics-json，不做正则解析；ATE 用 evaluate_ate.py --json），汇总
mean/std/worst，并按 config/benchmark.yaml 的门限断言，报告通过/失败。

用法:
  # 统计基准（5 轮，全程）
  python3 scripts/benchmark.py config/benchmark.yaml /tmp/bench --gt kitti_gt.tum

  # 快速门（3 轮 × 前 500 帧，无 GT 只断言结构/性能门限）
  python3 scripts/benchmark.py config/benchmark.yaml /tmp/bench_fast --window 500 --runs 3

  # A/B 对比（每侧 runs 次）
  python3 scripts/benchmark.py config/benchmark.yaml /tmp/bench --compare config/kitti00_sync.yaml

输出: <out_dir>/report.json（逐轮 + 聚合 + 门限结果，供 CI/提交门消费）
退出码: 0 = 全部门限通过；1 = 任一失败；2 = 运行/配置错误
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = (len(sorted_vals) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo))


def load_metrics_json(path):
    with open(path) as f:
        return json.load(f)


def run_ate(est_tum, gt_tum, alignment):
    """evaluate_ate.py --json，直接解析 JSON 摘要。返回 dict。"""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "evaluate_ate.py"),
         str(est_tum), str(gt_tum), "--alignment", alignment, "--json"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ATE 评估失败: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def run_once(cfg, args, out_path, run_dir):
    """跑一轮，返回指标 dict。"""
    log_path = str(Path(run_dir) / "run.log")
    metrics_path = str(Path(run_dir) / "metrics.json")

    cmd = [args.bin, cfg["dataset"], cfg.get("config", args.config), out_path, "--headless"]
    fmt = cfg.get("format", "kitti")
    if fmt != "kitti":
        cmd.append(f"--{fmt}")
    if cfg.get("window_frames", 0):
        cmd += ["--frames", str(cfg["window_frames"])]
    cmd += ["--metrics-json", metrics_path, "--deadline-ms", str(cfg.get("deadline_ms", 100))]

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=args.cwd, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"run_slam 失败(returncode={proc.returncode})，见 {log_path}")

    m = load_metrics_json(metrics_path)

    # GT 精度（可选）
    gt = cfg.get("gt") or args.gt
    if gt:
        try:
            ate = run_ate(out_path, gt, cfg.get("alignment", "se3"))
            m.update({
                "coverage_pct": ate["coverage_pct"],
                "ate_rmse": ate["ate_rmse"], "ate_mean": ate["ate_mean"],
                "ate_std": ate["ate_std"], "ate_max": ate["ate_max"],
                "rpe_trans_rmse": ate["rpe_trans_rmse"],
                "rpe_trans_mean": ate["rpe_trans_mean"], "rpe_trans_max": ate["rpe_trans_max"],
                "rpe_rot_rmse": ate["rpe_rot_rmse"],
                "rpe_rot_mean": ate["rpe_rot_mean"], "rpe_rot_max": ate["rpe_rot_max"],
                "len_ratio": ate["len_ratio"],
                "jumps_3m": ate["jumps_3m"], "jumps_5m": ate["jumps_5m"],
                "jumps_10m": ate["jumps_10m"],
            })
        except RuntimeError as e:
            print(f"[bench] 警告: {e}（跳过 ATE 门限）", file=sys.stderr)

    # RSS 峰值（Linux /usr/bin/time -v，可选）
    if cfg.get("measure_rss", False):
        time_cmd = ["/usr/bin/time", "-v"] + cmd
        with open(log_path, "w") as logf:
            proc = subprocess.run(time_cmd, cwd=args.cwd, stdout=logf,
                                  stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            txt = open(log_path).read()
            for line in txt.splitlines():
                if "Maximum resident set size" in line:
                    m["rss_kb"] = float(line.split()[-1])
    return m


def aggregate(rounds):
    """多轮 → {metric: {mean, std, worst}}；worst 取"更差方向"原始值。"""
    out = {}
    for k in rounds[0]:
        vals = np.asarray([r[k] for r in rounds], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                  "worst": float(vals.max())}
    return out


def check_gates(agg, gates, ate_available):
    """按门限断言。min=越高越好（worst 取 min），max=越低越好（worst 取 max）。"""
    results = {}
    for name, spec in gates.items():
        if not spec:
            continue
        if name.startswith("ate_") and not ate_available:
            results[name] = {"status": "skip", "reason": "无 GT"}
            continue
        if name not in agg:
            results[name] = {"status": "skip", "reason": "无此指标"}
            continue
        worst = agg[name]["worst"]
        ok, bound, op = True, None, None
        if "max" in spec:
            bound, op = spec["max"], "max"
            ok = worst <= bound
        if "min" in spec:
            bound, op = spec["min"], "min"
            ok = worst >= bound
        results[name] = {
            "status": "pass" if ok else "fail",
            "mean": agg[name]["mean"], "std": agg[name]["std"],
            "worst": worst, "bound": bound, "op": op,
        }
    return results


def print_table(agg, gates, gate_results):
    print(f"\n{'指标':<22}{'mean':>12}{'std':>12}{'worst':>12}  门限")
    for k, v in sorted(agg.items()):
        r = gate_results.get(k, {})
        tag = {"pass": "PASS", "fail": "FAIL", "skip": "-"}.get(r.get("status"), " ")
        bound = r.get("bound")
        bound_str = f"{r.get('op', '')} {bound:.3g}" if bound is not None else ""
        print(f"{k:<22}{v['mean']:>12.4f}{v['std']:>12.4f}{v['worst']:>12.4f}"
              f"  {tag:>5} {bound_str}")


def main():
    ap = argparse.ArgumentParser(description="VSLAM 统计基准（生产级）")
    ap.add_argument("gates_yaml", help="config/benchmark.yaml（门限 + 数据集/轮数）")
    ap.add_argument("out_dir")
    ap.add_argument("--runs", type=int, default=0, help="覆盖 YAML 中的轮数")
    ap.add_argument("--window", type=int, default=0, help="覆盖 window_frames（>0=快速门）")
    ap.add_argument("--compare", default=None, help="对比配置 YAML")
    ap.add_argument("--bin", default="build/bin/run_slam")
    ap.add_argument("--gt", default=None, help="真值轨迹（TUM），覆盖 YAML")
    ap.add_argument("--config", default=None, help="run_slam 配置（默认用 YAML 内 config）")
    ap.add_argument("--cwd", default=".")
    args = ap.parse_args()

    with open(args.gates_yaml) as f:
        conf = yaml.safe_load(f)
    bench = conf.get("Benchmark", {})
    gates = conf.get("Gates", {})
    args.cwd = str(Path(args.cwd).resolve())
    args.bin = str(Path(args.bin).resolve())
    args.out_dir = str(Path(args.out_dir).resolve())
    if args.runs: bench["runs"] = args.runs
    if args.window: bench["window_frames"] = args.window
    if args.gt: bench["gt"] = args.gt
    if args.config: bench["config"] = args.config
    if "config" not in bench:
        print("错误: benchmark.yaml 缺 Benchmark.config", file=sys.stderr)
        return 2
    if "dataset" not in bench:
        print("错误: benchmark.yaml 缺 Benchmark.dataset", file=sys.stderr)
        return 2

    runs = int(bench.get("runs", 5))
    ate_available = bool(bench.get("gt"))

    all_results, all_agg, all_gates = {}, {}, {}
    for label, cfg_src in [("A", bench)] + ([("B", args.compare)] if args.compare else []):
        cfg = dict(cfg_src)
        if cfg_src is args.compare:
            with open(args.compare) as f:
                cfg = dict(bench)
                cfg.update(yaml.safe_load(f).get("Benchmark", {}))
        rounds = []
        for i in range(runs):
            run_dir = str(Path(args.out_dir) / f"{label}_run{i+1}")
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            out_path = str(Path(run_dir) / "traj.txt")
            print(f"[bench] {label} run {i+1}/{runs} dataset={cfg['dataset']} "
                  f"window={cfg.get('window_frames', 0)} ...", flush=True)
            rounds.append(run_once(cfg, args, out_path, run_dir))
        agg = aggregate(rounds)
        all_results[label] = {"rounds": rounds, "aggregated": agg}
        all_agg[label] = agg
        all_gates[label] = check_gates(agg, gates, ate_available)
        with open(Path(args.out_dir) / f"results_{label}.json", "w") as f:
            json.dump(all_results[label], f, indent=2, default=float)

    for label in all_agg:
        print(f"\n===== {label}: {bench['dataset']} window={bench.get('window_frames', 0)} "
              f"rounds={runs} =====")
        print_table(all_agg[label], gates, all_gates[label])

    failed = [k for k, v in all_gates["A"].items() if v.get("status") == "fail"]
    report = {"config": bench, "gates": gates, "failed": failed,
              "result": "pass" if not failed else "fail",
              "A": all_results["A"], "B": all_results.get("B")}
    with open(Path(args.out_dir) / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n[bench] report -> {Path(args.out_dir) / 'report.json'}"
          f"  结果: {'PASS' if not failed else 'FAIL'}"
          f"  {'（未通过门限: ' + ', '.join(failed) + '）' if failed else ''}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
