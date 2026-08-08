# VSLAM 基准评估方案（生产级）

> 目标：让**每次代码修改**在提交前都能被客观、可复现、有统计意义的基准兜底，
> 不再依赖"正则解析日志 + 最好一轮"的离线验收。
> 状态：L0/L1/L2 已落地（M1.6）；L3 长期/故障属 M7，未实施。

---

## 1. 分层评估（L0~L3）

| 层 | 内容 | 频率 | 门限来源 |
|----|------|------|----------|
| **L0 单元测试** | `ctest`（test_vo + 契约/状态机/PoseGate/Facade） | 每次提交 | `CMakeLists.txt` 注册的测试 |
| **L1 确定性回归** | `deterministic.yaml` + KITTI 00 前 1000 帧，轨迹/状态序列与 `scripts/benchmark/reference/` 逐位一致 | 每次提交 | `compare_trajectories.py` |
| **L2 统计基准** | `benchmark.py`：N 轮（默认 5）全程/分片，输出 mean/std/worst，按 `config/benchmark.yaml` 门限断言 | 提交（快速档）/ 发布（完整档） | `config/benchmark.yaml` |
| **L3 长期/故障** | 24h soak、ASan/UBSan、故障注入、相机断流/重启 | 发布/nightly | §11（M7，未实施） |

**提交门**（L0+L1+L2 快速档）：`scripts/benchmark_gate.sh`，由
`.githooks/pre-commit` 在每次 `git commit` 前自动执行（`git config core.hooksPath .githooks`）。

---

## 2. 为什么这样设计

### 2.1 评估的"准确性"来自哪里

1. **结构化指标，不用正则解析**：`run_slam --metrics-json` 输出单次运行的结构化
   JSON（延迟分位、有效位姿率、LOST 次数/时长、子地图重建、回环、地图规模、
   deadline miss）。`benchmark.py` 直接读 JSON；ATE 用 `evaluate_ate.py --json`
   输出机读摘要。彻底替代旧版对自然语言日志的正则抓取（易碎、无法给分位/时长）。
2. **多轮统计，禁止挑最好一轮**：`runs≥5`，汇总 `mean/std/worst`；门限断言在
   **worst 一轮**上（`max`=越低越好取最大、`min`=越高越好取最小），A/B 至少 5 轮
   才算数（§0.1）。
3. **逐帧延迟分布**：p50/p95/p99/max + deadline miss 率，而不是只看全程平均 FPS
   ——平均 FPS 会掩盖回环/BA 造成的单帧尖峰。
4. **确定性回归兜底重构**：Strangler Fig 拆分（M1）必须保持轨迹与状态序列
   逐位一致，L1 用 `deterministic.yaml`（单线程 + 固定 RNG + 异步关闭）保证可复现。

### 2.2 门限语义

`config/benchmark.yaml` 的 `Gates` 段，每个指标：
- `min: X` → 越高越好，worst = 各轮最小值，断言 `worst >= X`
- `max: X` → 越低越好，worst = 各轮最大值，断言 `worst <= X`
- 无 GT 时 `ate_*` 门限自动 `skip`（提供 GT 后启用）

首版门限是当前平台/数据集的**标定起点**；按 §0.1，不能只按单个 KITTI 序列调参，
后续以 ≥3 类场景的机器人实录数据统一标定。

---

## 3. 指标清单

### 3.1 结构化运行指标（`run_slam --metrics-json`）

| 指标 | 含义 | 对应 §11.4 硬门槛 |
|------|------|--------------------|
| `valid_ratio` | 有效位姿率（pose_valid / 帧数） | 位姿可用率 ≥99.9%（M7，当前门限放宽至 99%） |
| `latency_p50/p95/p99/max` | 单帧 addFrame 延迟分位 | 跟踪延迟 p99 < 图像周期×0.8 |
| `deadline_miss_ratio` | 超过 deadline_ms 的帧占比 | <1% |
| `lost_count` / `lost_duration_s` | LOST 迁移次数与累计时长 | 连续运行无死锁；恢复时延 |
| `submap_reinit` | 子地图重建次数 | 长时丢失兜底 |
| `loops` | 已闭合回环 | 回环提交 precision 100% |
| `map_points` / `keyframes` | 最终地图规模 | RSS 预算（M2 落地） |

### 3.2 精度指标（`evaluate_ate.py --json`，需 GT）

| 指标 | 含义 | 门限（M3 阶段门 §7.6） |
|------|------|------------------------|
| `ate_rmse/mean/std/max/p95` | 轨迹绝对误差统计 | worst≤40m，std≤8m |
| `rpe_trans_rmse/mean/max` | 逐帧平移误差 | - |
| `rpe_rot_rmse/mean/max` | 逐帧旋转误差 | - |
| `coverage_pct` | 匹配帧覆盖率 | 每轮 ≥99% |
| `len_ratio` | 估计/真值路径长度比 | 1.0±容差 |
| `jumps_3m/5m/10m` | 连续帧跳变计数 | >10m 必须为 0 |

---

## 4. 用法

```bash
# 提交门（快速档，~8-10 min）：构建 + ctest + L1 + L2
scripts/benchmark_gate.sh

# 完整档（全程 + 5 轮，发布前）：~30 min
scripts/benchmark_gate.sh --full

# 行为合法变更后更新 L1 参考（把参考文件随变更一起提交）
scripts/benchmark_gate.sh --update-reference

# 单独跑统计基准
python3 scripts/benchmark.py config/benchmark.yaml /tmp/bench --gt /tmp/kitti_gt.tum
python3 scripts/benchmark.py config/benchmark.yaml /tmp/bench_fast --window 500 --runs 3

# 构建 GT（KITTI 需 poses/，prepare_kitti.sh 获取）
python3 scripts/kitti_gt_to_tum.py datasets/sequences/00/poses /tmp/kitti_gt.tum
```

输出：`<out_dir>/report.json`（逐轮 + mean/std/worst + 门限结果，供 CI/提交门消费）；
退出码 0 = 通过，1 = 门限失败，2 = 配置/运行错误。

---

## 5. 提交门策略

- `.githooks/pre-commit` 调用 `benchmark_gate.sh`（快速档）；失败则退出码非 0，git 拒绝提交。
- 安装：`git config core.hooksPath .githooks`（本仓库已配置，见 `.git/config`）。
- 合法行为变更（新特性/修 bug 改变输出）时：先 `benchmark_gate.sh --update-reference`
  更新参考，再把更新的 `scripts/benchmark/reference/*` 与代码变更同一次提交。
- 紧急绕过：`git commit --no-verify`（不推荐；PR 需说明理由，并在随后补跑完整基准）。

---

## 6. 已知边界

- L2 门限为当前平台标定起点；FPS/RSS 与机器相关，跨机器需重新标定（§0.1）。
- 未测 RSS（`measure_rss` 需 Linux `/usr/bin/time -v`，默认关）。
- L3（24h soak、ASan/UBSan、故障注入、相机断流）属 §11，未实施；到 M7 时把本方案
  接到 CI（每次提交 L0+L1、nightly L2+L3）。
