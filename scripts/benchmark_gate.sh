#!/usr/bin/env bash
# ============================================================
# benchmark_gate.sh - 提交基准门（L0~L2）
#
# 每次代码修改提交前必须通过：
#   L0  构建 + ctest（单元测试）
#   L1  确定性回归：deterministic.yaml + KITTI 00 前 1000 帧，
#       轨迹/状态序列与 scripts/benchmark/reference/ 逐位一致
#   L2  统计基准：benchmark.py 快速档（window=500, runs=3），
#       门限断言（config/benchmark.yaml）
#
# 用法:
#   scripts/benchmark_gate.sh            # 快速档（提交门默认，~8-10 min）
#   scripts/benchmark_gate.sh --full     # 完整档（全程 + YAML 轮数，~30 min）
#   scripts/benchmark_gate.sh --update-reference   # 行为合法变更时更新 L1 参考
#
# 退出码: 0 = 通过；非 0 = 失败（阻止提交）。紧急情况可用
#   git commit --no-verify（不推荐，需在 PR 说明）。
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL=0
UPDATE_REF=0
for a in "$@"; do
    case "$a" in
        --full) FULL=1 ;;
        --update-reference) UPDATE_REF=1 ;;
        *) echo "未知参数: $a" >&2; exit 2 ;;
    esac
done

cd "$ROOT"
TMP="$(mktemp -d /tmp/vslam_gate.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
BUILD_DIR="$TMP/build"
FAILED=0

# ---- L0: Python 门限逻辑单测 ----
echo "===== [gate] L0 Python 门限/soak 单测 ====="
if ! python3 scripts/test_benchmark.py; then
    echo "[gate] Python 门限/soak 单测失败" >&2
    exit 1
fi

# ---- L0: 构建 + 单元测试 ----
echo "===== [gate] L0 全新配置 + 构建 ====="
cmake -S . -B "$BUILD_DIR" -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j
echo "===== [gate] L0 单元测试 (ctest) ====="
if ! ctest --test-dir "$BUILD_DIR" --output-on-failure --no-tests=error; then
    echo "[gate] L0 单元测试失败" >&2
    exit 1
fi

# ---- L1: 确定性回归 ----
echo "===== [gate] L1 确定性回归 (KITTI 00 前 1000 帧, deterministic.yaml) ====="
if ! "$BUILD_DIR/bin/run_slam" datasets/kitti/sequences/00 config/deterministic.yaml \
        "$TMP/traj.txt" --headless --frames 1000 --status-csv "$TMP/status.csv" \
        >/dev/null 2>&1; then
    echo "[gate] L1 确定性运行失败" >&2
    exit 1
fi
if [ "$UPDATE_REF" = "1" ]; then
    cp "$TMP/traj.txt" scripts/benchmark/reference/pose.txt
    cp "$TMP/status.csv" scripts/benchmark/reference/status.csv
    echo "[gate] L1 参考已更新（随本次提交一起提交参考文件）"
else
    if ! python3 scripts/compare_trajectories.py "$TMP/traj.txt" \
            scripts/benchmark/reference/pose.txt \
            --status "$TMP/status.csv" scripts/benchmark/reference/status.csv; then
        echo "[gate] L1 确定性回归失败：轨迹/状态序列与参考不一致。" >&2
        echo "[gate] 若为合法的行为变更，请运行 scripts/benchmark_gate.sh --update-reference" >&2
        echo "[gate] 并把更新的参考文件一并提交。" >&2
        FAILED=1
    fi
fi

# ---- L2: 统计基准 + 门限断言 ----
if [ "$FULL" = "1" ]; then
    echo "===== [gate] L2 统计基准（完整档） ====="
    if ! python3 scripts/benchmark.py config/benchmark.yaml "$TMP/bench" \
            --bin "$BUILD_DIR/bin/run_slam"; then
        echo "[gate] L2 统计基准未通过门限" >&2
        FAILED=1
    fi
else
    echo "===== [gate] L2 统计基准（快速档: window=500, runs=3） ====="
    if ! python3 scripts/benchmark.py config/benchmark.yaml "$TMP/bench" \
            --window 500 --runs 3 --bin "$BUILD_DIR/bin/run_slam"; then
        echo "[gate] L2 统计基准未通过门限" >&2
        FAILED=1
    fi
fi

if [ "$FAILED" = "1" ]; then
    echo "[gate] ===== 提交门 FAILED =====" >&2
    exit 1
fi
echo "[gate] ===== 提交门 PASSED ====="
exit 0
