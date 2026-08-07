# 调试评估数据（§3.21 回环校正跳变问题）

> 历史档案：以下轨迹和结论对应旧提交 `ff09804`/§3.21，用于复现当时的问题，
> 不是当前 HEAD 的状态报告。Phase 0/1 已移除相关硬编码与快照外传播路径；当前代码
> 的当前双轮 KITTI/EuRoC benchmark 见 `docs/DEVELOPMENT_LOG.md` §3.23；这里不追加
> 大体积运行日志和轨迹，以免继续扩大历史档案。

KITTI 00 全程轨迹与真值，用于复现/验证回环校正跳变问题。
评估命令（从仓库根目录）：

```bash
python3 scripts/evaluate_ate.py data/eval/<traj>.txt data/eval/kitti_00_gt.tum --alignment se3
```

## 文件说明

| 文件 | 对应问题/状态 |
|------|--------------|
| `kitti_00_gt.tum` | KITTI 00 真值（scripts/kitti_gt_to_tum.py 生成，4541 帧） |
| `m4l_sync_baseline_traj.txt` | sync 基线（M4 时代）：ATE 42.8m、旋转 16.5°、跳变 3 次全为 LOST 空洞（好结果对照） |
| `abcd13_async_racetraj.txt` | async 回环写回竞争：44.8m 连续帧跳变（写回前/后帧跨版本组合） |
| `abcd20_propagation_traj.txt` | 快照后插入 KF 传播修复后：85.4m 跳变（传播误用） |
| `abcd21_traj.txt` | 旧 commit 失败保护后：**仍 85.4m 跳变（历史问题复现）** |
| `abcd22_traj.txt` | 与 abcd22 日志配套的轨迹 |
| `abcd22_pgo_evidence.log` | PGO 校正正确性铁证：kf153 固定 (17.96,-1.94,90.07)，kf1597 从 (51.13,-16.78,114.08) 拉回 (17.48,-1.92,90.00)——回环 #1 方向正确，跳变源于快照外 KF |

## 历史问题（旧提交 `ff09804`，已归档）

回环 #1（静止段 kf#153→1597）PGO 校正正确（1597 拉回 153 附近），但
**快照后插入的 KF（1599 等）不在 PGO 图中、保持校正前位姿 → 与校正后的
锚点 KF 之间产生 44.8~85.4m 连续帧跳变**。已尝试：

1. loop_skip 保护端点写回 → 位姿-点失配（T_ca 44m），放弃；
2. 快照后 KF 按锚定校正量传播位姿+独有点 → 传播方向/范围问题产生 85m 跳变；
3. commit 非 COMMITTED 时跳过传播 → 85m 跳变仍在（abcd21）。

疑似方向：传播的校正量 `C = new⁻¹·old` 左乘语义与 KF 位姿/点的坐标系
组合是否正确；或传播范围（应限于回环端点附近的"漂移段"而非全部快照后 KF）。

## 复现旧版本

```bash
# 在独立 worktree/checkout 的 `ff09804` 中运行以下命令；这是旧提交的旧目录结构，
# 不要用当前 HEAD 生成“旧问题”结论：
./build/bin/run_slam datasets/kitti/sequences/00 config/default.yaml /tmp/repro.txt --headless
python3 scripts/evaluate_ate.py /tmp/repro.txt data/eval/kitti_00_gt.tum --alignment se3
# 期望看到 Step jumps >10m 含 85m 级连续帧跳变（t≈159.8→159.9）
```

当前 HEAD 的结果以开发日志 §3.23 的命令和摘要为准；原始运行产物保存在仓库外，
不覆盖这些历史轨迹。
