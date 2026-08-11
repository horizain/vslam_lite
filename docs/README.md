# 文档索引

本文档是仓库文档入口。当前实现以源码、配置和本页的“当前规范”为准；带有“历史/设计稿”
标记的文件保留实验背景，不覆盖现行接口。

## 当前规范与使用

| 文档 | 用途 |
|---|---|
| [`../README.md`](../README.md) | 构建、运行、验证的最短入口 |
| [`TUTORIAL.md`](TUTORIAL.md) | 面向初学者的算法、坐标系和逐文件教程 |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | 依赖与离线构建说明 |
| [`THREADING_DESIGN.md`](THREADING_DESIGN.md) | 当前锁序、快照与异步后端约束；旧双队列方案已标为历史 |
| [`PRODUCTION_LOCALIZATION_PLAN.md`](PRODUCTION_LOCALIZATION_PLAN.md) | 从教学 SLAM 演进到机器人长期定位组件的当前实施规格、固定算法方向和量化门槛 |

## 历史设计与计划

| 文档 | 状态 |
|---|---|
| [`PHASE2_DESIGN.md`](PHASE2_DESIGN.md) | 历史 Phase 2 设计稿；部分目标已实现，部分 Sim3/跨子地图方案仍未落地，不能当作 API 说明 |
| [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) | 历史 KITTI 基线与改进计划；数字依赖具体数据集、词典和构建配置 |
| [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) | 追加式历史记录，保留旧字段和实验结论；当前回环快照/Essential Graph 见 §3.41，Flat-DBoW3/mobile 全程证据见 §3.43 |
| [`../data/eval/README.md`](../data/eval/README.md) | 旧提交 `ff09804` 的回环跳变复现档案；当前 HEAD 指标见开发日志 §3.23 |

## 当前命名速查

| 概念 | 当前名称/语义 |
|---|---|
| 局部关键帧位姿 | `Frame::pose_cs` = `T_cs`（子地图→相机） |
| 子地图到世界 | `Submap::T_ws` |
| 局部地图点 | `MapPoint::pos_s` |
| 正式观测 | `Observation{keyframe_id, feature_index}`，由 `Map` API 维护 |
| 世界→相机 | `T_cw = T_cs * T_ws.inverse()`（派生量，不是 `pose_cw` 字段） |
| 轨迹输出 | TUM 行中的相机世界位姿 `T_wc` |

代码或旧日志中的 `pose_cw`、`pos_w`、`observed_count` 属于旧模型/历史记录，不能作为
新代码的字段名。

## 验证入口

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
git diff --check
```

当前验证边界见开发日志 §3.43：mobile KITTI 00 已完成两轮 4541 帧与回环 oracle，
但有效率、ATE、召回和 p99 均未通过产品门，ARM 实机也未测。CTest 与数据集基准是两条
独立证据，不能互相替代；最终 `benchmark_gate.sh --full` 也因 ATE std、>10m 跳变和
p99 三项失败。不能从较低 RSS 推导出实时性或定位质量已经合格。
