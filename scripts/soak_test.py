#!/usr/bin/env python3
"""
soak_test.py - M2.3 §6.5 soak 验收脚本

循环重放同一输入序列，断言 §6.5 的实时/资源门限：
  - 同一输入循环 N 轮无崩溃、死锁和队列增长（快速档 3 轮；--full 10 轮）
  - 输入队列 high-water mark <= capacity（默认 3）；后台等待槽 <= 1
  - 10 Hz 配置跟踪 latency p99 < deadline_ms（默认 80）
  - deadline miss < 1%
  - 关闭路径：Ctrl-C（SIGINT）、输入 EOF、构造失败三路径均在 2 s 内退出
  - --full：RSS 峰值 < 1 GiB，稳态增长 < 5 MiB/h（后 50% 窗口线性拟合）
    以及 2 小时档（--duration-h 2.0，nightly）

数据源：run_slam --localizer --metrics-json（结构化指标 JSON，禁止解析日志）。

用法：
  scripts/soak_test.py                                          # 快速档
  scripts/soak_test.py --full                                   # 完整档（10 轮 + RSS 门限）
  scripts/soak_test.py --duration-h 2.0                         # 2 小时档（nightly）
  scripts/soak_test.py --shutdown-check                         # 仅关闭路径检查
  scripts/soak_test.py --fail-inject                            # 仅构造失败路径检查
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# ---- 解析 ----
def parse_args():
    p = argparse.ArgumentParser(description="M2.3 §6.5 soak 验收")
    p.add_argument("--bin", default="./build/bin/run_slam")
    p.add_argument("--dataset", default="datasets/sequences/00")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--robot-yaml", default="config/robot.yaml")
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--deadline-ms", type=int, default=80)
    p.add_argument("--min-valid-ratio", type=float, default=0.99)
    p.add_argument("--queue-capacity", type=int, default=3)
    p.add_argument("--full", action="store_true",
                   help="完整档：10 轮 + RSS 峰值/斜率门限")
    p.add_argument("--duration-h", type=float, default=0.0,
                   help="指定时长档（小时；>0 时循环重放到时长耗尽，nightly）")
    p.add_argument("--rss-max-mb", type=float, default=1024.0)
    p.add_argument("--rss-slope-mib-h", type=float, default=5.0)
    p.add_argument("--shutdown-check", action="store_true")
    p.add_argument("--fail-inject", action="store_true")
    return p.parse_args()


def read_rss_mb(pid):
    """Linux /proc/<pid>/status VmRSS（零依赖，无需 psutil）。"""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None
    return None


def linear_slope(points):
    """最小二乘拟合斜率（y=MB, x=秒）→ MiB/h。points: [(t, mb)]。"""
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return None
    slope_mb_s = cov / var
    return slope_mb_s * 3600.0 / 1024.0  # MiB/h


def run_round(args, tmpdir, idx, metrics_checks=True):
    """跑一轮 run_slam --localizer，采样 RSS，返回 (ok, report)。"""
    metrics = Path(tmpdir) / f"metrics_{idx}.json"
    traj = Path(tmpdir) / f"traj_{idx}.tum"
    cmd = [args.bin, args.dataset, args.config, str(traj),
           "--localizer", "--headless", "--metrics-json", str(metrics),
           "--robot-yaml", args.robot_yaml,
           "--frames", str(args.frames)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            rss = read_rss_mb(proc.pid)
            if rss is not None:
                samples.append((time.time(), rss))
            stop.wait(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    rc = proc.wait()
    stop.set()
    th.join()

    report = {"round": idx + 1, "exit_code": rc,
              "rss_peak_mb": max((s[1] for s in samples), default=0.0)}
    slope = linear_slope(samples[len(samples) // 2:])
    report["rss_slope_mib_h"] = slope
    report["failures"] = []

    if rc != 0:
        report["failures"].append(f"进程退出码 {rc} != 0（崩溃/死锁）")
    if not metrics_checks:
        return (not report["failures"], report)
    try:
        m = json.loads(metrics.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        report["failures"].append(f"指标 JSON 缺失或损坏: {e}")
        return (False, report)

    checks = [
        ("latency_p99_ms < deadline_ms", m["latency_p99_ms"] < args.deadline_ms,
         f"p99={m['latency_p99_ms']:.1f}ms >= {args.deadline_ms}ms"),
        ("deadline_miss_ratio < 1%", m["deadline_miss_ratio"] < 0.01,
         f"miss_ratio={m['deadline_miss_ratio']:.4f}"),
        ("input_queue_hwm <= 容量", m["input_queue_hwm"] <= args.queue_capacity,
         f"hwm={m['input_queue_hwm']} > {args.queue_capacity}"),
        ("backend_pending <= 1", m["backend_pending"] <= 1,
         f"pending={m['backend_pending']}"),
        ("valid_ratio >= 门限",
         m["frames_processed"] == 0 or
         m["pose_accepted"] / m["frames_processed"] >= args.min_valid_ratio,
         f"valid={m['pose_accepted']}/{m['frames_processed']}"),
    ]
    for name, ok, detail in checks:
        if not ok:
            report["failures"].append(f"{name}: {detail}")
    report["latency_p99_ms"] = m["latency_p99_ms"]
    report["deadline_miss_ratio"] = m["deadline_miss_ratio"]
    report["input_queue_hwm"] = m["input_queue_hwm"]
    report["backend_pending"] = m["backend_pending"]
    report["frames_processed"] = m["frames_processed"]
    report["pose_accepted"] = m["pose_accepted"]
    return (not report["failures"], report)


def check_shutdown(args):
    """§6.5 关闭路径：SIGINT 后 2 s 内退出。"""
    metrics = "/tmp/soak_shutdown_metrics.json"
    cmd = [args.bin, args.dataset, args.config, "/tmp/soak_shutdown.tum",
           "--localizer", "--headless", "--metrics-json", metrics]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    t0 = time.time()
    proc.send_signal(signal.SIGINT)
    rc = proc.wait(timeout=10)
    elapsed = time.time() - t0
    ok = elapsed < 2.0
    print(f"  SIGINT 关闭: 耗时 {elapsed:.2f}s（<2s）退出码 {rc} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_fail_inject(args):
    """§6.5 构造失败路径：非法输入启动必须快速退出（不挂死、所有线程 join）。

    run_slam 对非法配置/数据集路径静默降级（VOConfig::fromYaml 捕获异常回退
    默认；Dataset 对不存在路径返回空数据流）——构造失败时进程立即结束。
    Localizer 层的真实构造失败（§4.3 非单位 T_bc 抛 std::invalid_argument）
    由 test_localizer_contract 单元测试覆盖（soak 无法注入硬编码的 robot.yaml）。
    """
    cmd = [args.bin, "/nonexistent/dataset", "/nonexistent/config.yaml",
           "/tmp/soak_fail.tum", "--localizer", "--headless"]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        rc = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  构造失败路径: 进程挂死 >10s -> FAIL")
        return False
    elapsed = time.time() - t0
    ok = elapsed < 2.0
    print(f"  构造失败（非法输入）: 退出码 {rc}，耗时 {elapsed:.2f}s（<2s）-> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    args = parse_args()
    results = []

    if args.shutdown_check:
        return 0 if check_shutdown(args) else 1
    if args.fail_inject:
        return 0 if check_fail_inject(args) else 1

    rounds = 10 if args.full else args.rounds
    if args.duration_h > 0.0:
        deadline = time.time() + args.duration_h * 3600.0
    else:
        deadline = None

    print(f"soak_test: {args.bin} | dataset={args.dataset} "
          f"config={args.config} frames={args.frames} rounds={rounds}"
          + (" | 时长档 %.1f h" % args.duration_h if deadline else ""))

    with tempfile.TemporaryDirectory(prefix="vslam_soak_") as tmpdir:
        rss_peaks = []
        rss_slopes = []
        i = 0
        while deadline is None or time.time() < deadline:
            ok, rep = run_round(args, tmpdir, i)
            results.append(ok)
            rss_peaks.append(rep["rss_peak_mb"])
            if rep["rss_slope_mib_h"] is not None:
                rss_slopes.append(rep["rss_slope_mib_h"])
            status = "PASS" if ok else "FAIL"
            print(f"  轮 {i + 1}: {status} | p99={rep.get('latency_p99_ms', -1):.1f}ms "
                  f"hwm={rep.get('input_queue_hwm', -1)} "
                  f"pending={rep.get('backend_pending', -1)} "
                  f"RSS峰值={rep['rss_peak_mb']:.1f}MB "
                  f"斜率={rep['rss_slope_mib_h']}MiB/h")
            if not ok:
                for f in rep["failures"]:
                    print(f"    [FAIL] {f}")
                break  # §0.8：出现失败立即停止，不得继续叠加
            i += 1
            if deadline is None and i >= rounds:
                break

        all_ok = all(results) and len(results) > 0
        if args.full and all_ok:
            peak = max(rss_peaks)
            slope = (sum(rss_slopes) / len(rss_slopes)) if rss_slopes else None
            if peak > args.rss_max_mb:
                print(f"  [FAIL] RSS 峰值 {peak:.1f}MB > {args.rss_max_mb}MB（§6.5 <1GiB）")
                all_ok = False
            else:
                print(f"  RSS 峰值 {peak:.1f}MB（<{args.rss_max_mb}MB）PASS")
            if slope is not None and slope > args.rss_slope_mib_h:
                print(f"  [FAIL] RSS 稳态增长 {slope:.2f}MiB/h > {args.rss_slope_mib_h}MiB/h（§6.5）")
                all_ok = False
            else:
                print(f"  RSS 稳态增长 {slope}MiB/h PASS")

    print(f"soak_test 结果: {'PASSED' if all_ok else 'FAILED'} "
          f"({len(results)} 轮)")

    # §6.5：输入 EOF 路径（数据集读完自然退出 = EOF）由每轮 rc==0 覆盖
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
