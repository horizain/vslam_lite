# vslam_lite

教学级 C++23 视觉里程计 / SLAM。当前代码支持单目、双目与可选的 DBoW3
回环模块；`run_vo` 是纯 VO 对比入口，`run_slam` 是回环入口。

## 快速开始

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure

./build/bin/run_vo datasets/kitti/sequences/00 config/default.yaml vo.txt --headless
./build/bin/run_slam datasets/kitti/sequences/00 config/kitti00.yaml slam.txt --headless

# EuRoC：传 mav0 根目录；标定从 cam0/sensor.yaml 自动读取
./build/bin/run_slam datasets/euroc/V1_01_easy/mav0 config/default.yaml \
    euroc.txt --euroc --headless
```

`run_slam` 还支持 `--frames N`、`--skip N`，以及 `--tum` / `--euroc`；EuRoC 示例路径为
`datasets/euroc/V1_01_easy/mav0`。当前仅支持 EuRoC 的 pinhole + radial-tangential
相机模型；程序会在送入前端前按 `sensor.yaml` 去畸变。真值与完整基准命令见
[`scripts/euroc_gt_to_tum.py`](scripts/euroc_gt_to_tum.py) 和
[`scripts/benchmark.py`](scripts/benchmark.py)。
KITTI 数据准备脚本默认写入 `datasets/kitti/sequences` 和 `datasets/kitti/poses`，见
[`scripts/prepare_kitti.sh`](scripts/prepare_kitti.sh)。依赖见
[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)。

## 当前数据模型

- `Frame::pose_cs`：子地图局部系到相机系的位姿 `T_cs`。
- `MapPoint::pos_s`：子地图局部坐标；正式关键帧观测使用 `Observation`，不使用旧的
  `observed_count` 语义。
- 全局世界到相机位姿按 `T_cw = T_cs * T_ws.inverse()` 组合；TUM 轨迹输出使用
  相机在世界系中的 `T_wc`。
- `Map` 的 Observation API 是正式关键帧观测的唯一写入口；普通跟踪帧只保留临时关联。

## 验证边界

Phase 0/1 的回环写回确定性、Observation 一致性和并发锁语义已收口；本地验证命令为
上面的构建与 CTest。2026-08-07 的 KITTI 00 / EuRoC V1_01_easy 双轮完整基准见
[`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md) §3.23；数据集指标依赖机器、配置、
词典和后端调度，不由默认单测承诺。

## 文档入口

从 [`docs/README.md`](docs/README.md) 开始。它区分当前教程/规范、历史设计稿、追加式
开发日志与待实施计划，避免把早期 Phase 2 初稿误读为当前接口。
