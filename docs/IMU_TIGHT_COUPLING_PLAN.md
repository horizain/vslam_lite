# IMU 紧耦合（VINS-Mono 风格）落地方案

> 定位：与 `PRODUCTION_LOCALIZATION_PLAN.md` 的 M6（松耦合 ESKF）互补。M6 面向
> "机器人长期定位 + map/odom 分离"，本方案面向 **前端精确度**：用陀螺仪直接测旋转，
> 解开单目/双目 VO 的旋转-平移歧义（`DEVELOPMENT_LOG.md §3.3` 的本质病态），并压制
> 累计漂移。两者共享 IMU 数据模型与同步层，优化器可并存（紧耦合管前端窗口，ESKF 管
> 输出/预测）。

---

## 1. 现状与缺口（gap analysis）

### 1.1 当前 IMU 支持状态

| 层 | 现状 | 证据 |
|---|---|---|
| 数据模型 | 无 `imu.h`，无 IMU 样本/配置/偏置类型 | `include/`、`src/` 全局 grep 仅 `camera.h:29,71` 注释 |
| 数据集 | EuRoC 仅解析 `cam0/data.csv` 图像流；`imu0/data.csv` 未读取 | `src/dataset.cpp:186` `loadEUROCImageList` |
| 帧状态 | `Frame` 无速度、bias、预积分量 | `include/vslam/frame.h` |
| 配置 | config 无 IMU 段（噪声/重力/时延） | `config/*.yaml` grep 无 `imu` |
| 估计器 | 纯视觉 PnP/对极 + g2o 局部 BA + 回环 | `vo.cpp`、`optimizer.cpp` |
| 紧耦合 | 路线图文字，零代码 | `DEVELOPMENT_LOG.md:268`（探索方向） |
| 松耦合 | M6 计划 15 维 ESKF，未实现 | `PRODUCTION_LOCALIZATION_PLAN.md §10` |

### 1.2 现有纯视觉的病态与漂移（本方案要消除的症状）

- **旋转-平移歧义**：纯旋转/旋转主导时对极几何退化（E≈0），`recoverPose` 的 t 方向任意，
  会组合出假平移。现有对策是启发式（旋转主导检测 + 只更新朝向，`§3.2`）+ 跳变保护
  （`>30m` 拒绝），只能缓解不能根治。
- **累计漂移**：KITTI 00 实测 ATE worst 55.3 m；旋转误差被 PnP/对极数值放大后随距离累积，
  单目尺度不可观测进一步放大。

### 1.3 为什么"陀螺仪直接测旋转"能根治

1. **旋转可观**：IMU 陀螺仪以 100~400 Hz 直接积分出相对旋转 ΔR，完全独立于视觉几何。
   对极退化时（纯旋转），E 的信息量为零，但 IMU 仍给出精确旋转 → 旋转-平移解耦，
   不再有"转 90° 却平移 5 m"的假位移。
2. **平移病态缓解**：已知帧间旋转后，对极约束 `E = t^× R` 中只剩平移方向 t̂ 未知，
   病态维数从 6 降到 3；双目（当前 KITTI）本身尺度已知，平移完全可观。单目则由
   VINS 初始化（重力-尺度对齐）固定尺度，IMU 的重力方向观测持续锚定，不再随距离缩放。
3. **漂移机制改变**：旋转分量由陀螺仪预积分提供短期高精度先验，视觉 BA 专注结构与
   平移；旋转误差不再经视觉数值放大累积。bias 随机游走导致的长期 yaw 漂移由因子图中
   **在线估计 bias**（预积分雅可比耦合进 BA）承接，回环继续做全局收口。

---

## 2. 总体架构

```
┌────────────────────────── 前端（主线程）──────────────────────────┐
│  Camera + ImuStream ─► TimeSynchronizer（插值到图像时刻）           │
│         │                                                          │
│         ▼                                                          │
│  ViInitializer：陀螺仪 bias + 重力/速度/尺度对齐（首 2~3 s）          │
│         │                                                          │
│         ▼                                                          │
│  ImuPreintegrator：帧间 ΔR Δv Δp + bias 雅可比 + 协方差传播            │
│         │                                                          │
│         ▼                                                          │
│  FrontendTracker：IMU 推进预测位姿 → 作为 PnP 初值 → 跟踪/重定位      │
│         │（纯旋转段直接采信 IMU 旋转，只更新朝向/结构）                │
└───────────────┬────────────────────────────────────────────────────┘
                │ KF 入队
        ┌───────▼──────── 后端（异步，BackendScheduler 复用）────────┐
        │  SlidingWindow 滑动窗口因子图：                              │
        │    Σ 视觉重投影因子 + Σ IMU 预积分因子 + 先验因子（边缘化）     │
        │  WindowBA：手写高斯牛顿 + Schur 消元（先无边缘化版，后补）     │
        │  bias（b_a, b_g）在窗口中在线估计，更新当前预积分                │
        └────────────────────────────────────────────────────────────┘
```

- 复用现有 `BackendScheduler`（单后台线程 + 覆盖式单任务槽，`§5.4`）与
  `BackendCommitter` 事务式写回；窗口 BA 结果只提交 ACTIVE 窗口帧。
- 与 M6 ESKF 的关系见 §8：紧耦合先落地在前端/窗口，M6 的 200 Hz `T_ob` 输出可在其上层叠加。

---

## 3. 里程碑拆分与文件清单

| 里程碑 | 交付物 | 依赖 | 状态 |
|---|---|---|---|
| IMU.0 数据模型与同步 | `imu.h`、`time_synchronizer.{h,cpp}`、EuRoC `imu0/data.csv` 解析、合成 IMU 生成器、config IMU 段 | — | 未开始 |
| IMU.1 预积分 | `preintegration.{h,cpp}`（中点积分 + bias 雅可比 + 协方差） | IMU.0 | 未开始 |
| IMU.2 初始化 | `vi_initializer.{h,cpp}`（陀螺仪 bias + 重力/速度/尺度对齐） | IMU.1 | 未开始 |
| IMU.3 前端接入 | `FrontendTracker` IMU 预测 → PnP 初值；纯旋转采信 IMU 旋转 | IMU.2 | 未开始 |
| IMU.4 滑动窗口 BA | `sliding_window.{h,cpp}` + `vi_optimizer.{h,cpp}`（视觉+IMU 因子，边缘化） | IMU.1/IMU.2/IMU.3 | 未开始 |
| IMU.5 基准验收 | KITTI 合成 IMU + EuRoC V1/V2 实测、旋转歧义 A/B 报告 | IMU.4 | 未开始 |

新增文件（全部 `namespace vslam`、中文注释、4 空格）：

```
include/vslam/imu.h                 # ImuSample/ImuConfig/ImuBias/ImuStream
include/vslam/time_synchronizer.h   # 图像时刻 IMU 插值（bracket 线性插值）
include/vslam/preintegration.h      # ImuPreintegrator
include/vslam/vi_initializer.h      # 初始化
include/vslam/sliding_window.h      # 滑动窗口因子图 + 边缘化
include/vslam/vi_optimizer.h        # 窗口 BA（复用或扩展）
src/imu.cpp  src/time_synchronizer.cpp  src/preintegration.cpp
src/vi_initializer.cpp  src/sliding_window.cpp
utils/synthetic_imu.cpp             # KITTI：GT 位姿微分生成 IMU（仅测试）
test/test_preintegration.cpp  test/test_vi_initializer.cpp  test/test_sliding_window.cpp
```

修改：

- `dataset.{h,cpp}`：EuRoC 解析 `imu0/data.csv`（含 CRLF 处理，仿 `loadEUROCImageList`）；
  提供 `ImuStream imuStream()`。
- `frame.h`：预留 `velocity`、`bias`（可选字段，不破坏现有构造）。
- `frontend_tracker.{h,cpp}`：`predictWithImu()` 入口 + 纯旋转分支。
- `localization_types.h`：`PoseEstimate` 可选增 `velocity`/`bias`（先不加，保持契约最小）。
- `config/*.yaml`：新增 `IMU:` 段（见 §5）。
- `CMakeLists.txt`：新增源文件与 CTest。

单提交约束沿用 `PRODUCTION_LOCALIZATION_PLAN.md §13`：≤8 个自有文件、≤500 行、
一个可独立验收行为、至少一个直接测试。

---

## 4. 数学设计（VINS-Mono 风格）

### 4.1 IMU 测量模型

```
ω̃ = ω + b_g + n_g          ã = a + b_a + n_a + Rᵀg
```

`b` 为 bias，`n` 为高斯白噪声（噪声密度 `gyro_noise_density`/`accel_noise_density`）；
bias 本身用随机游走建模（`gyro_random_walk`/`accel_random_walk`）。重力方向不可分离，
必须由初始化估计（§4.3）。

### 4.2 预积分（帧 i → 帧 j，中点积分）

```
ΔR_ij = ∏_{k<i,j} Exp((ω̃_k - b_g)·dt_k)                     （离散时间累积）
Δv_ij = Σ_k ΔR_ik · (ã_k - b_a)·dt_k
Δp_ij = Σ_k [ Δv_ik·dt_k + ½·ΔR_ik·(ã_k - b_a)·dt_k² ]
```

- **bias 一阶近似**（避免每帧重积分）：

```
ΔR(b_g) ≈ ΔR · Exp(J_ΔR·δb_g)
Δv(b_g,b_a) ≈ Δv + J_Δv·δb_g + J_ΔV·δb_a
Δp(b_g,b_a) ≈ Δp + J_Δp·δb_g + J_ΔP·δb_a
```

- **协方差传播**：`P ← F·P·Fᵀ + Q`，`F = ∂(状态增量)/∂(状态增量)`，`Q` 由噪声密度确定。
- **实现**：先积累 `(δt, δR, δv)` 闭式雅可比（VINS-Mono 论文式 17~24），再二分修正。
  数值雅可比测试对齐。

### 4.3 初始化（视觉-惯性对齐，窗口前 2~3 s）

1. **陀螺仪 bias**（旋转约束，闭合解或少量 GN 步）：
   `argmin Σ ‖ Log( ΔR_ijᵀ · R_wiᵀ·R_wj ) ‖²`，bias 一阶项从 `J_ΔR` 分离。
2. **重力/速度/尺度**（线性系统 `A x = b`，状态 `x = [v0..vn, g, s]`，s 仅单目）：
   用预积分速度/位移残差建立超定方程，`SVD/最小二乘` 解出重力方向与单目尺度 s。
3. 若双目（当前主战场），s 固定为 1，方程降维。
4. **失败降级**：静止窗口 2 s（gyro RMS `<0.05 rad/s`、accel 模长 std `<0.15 m/s²`，
   复用 M6 §10.2 判据）先定 bias；不满足则用标定 bias + 协方差乘 10 启动（状态 Degraded）。

### 4.4 滑动窗口因子图残差

窗口 `{c_0..c_n}`，状态 `x = {R_wk, p_wk, v_wk, b_ak, b_gk}`，因子：

- **IMU 预积分因子**（帧 k, k+1）：

```
r_R = Log( ΔR_kᵀ · R_wkᵀ·R_w(k+1) )
r_v = R_wkᵀ(v_w(k+1) − v_wk + g·Δt) − Δv_k
r_p = R_wkᵀ(p_w(k+1) − p_wk − v_wk·Δt + ½g·Δt²) − Δp_k
r_ba = b_a(k+1) − b_ak      r_bg = b_g(k+1) − b_gk
```

- **视觉重投影因子**：复用现有 PnP/三角化模型，误差 = 观测 − 重投影。
- **先验因子**：滑出窗口的状态用一阶边缘化（Schur 补）生成 `H·δx = −b` 信息矩阵先验。

### 4.5 边缘化（一致性关键）

- **v1（本阶段先做）**：固定滞后窗，滑出的旧帧直接丢弃，不生成先验——简单、稳定，
  代价是窗口边缘一致性损失（短期可接受，作基线）。
- **v2（后续）**：对滑出的 `(旧帧位姿, 视觉路标)` 做 Schur 补边缘化生成先验因子，
  对齐 VINS-Mono 的完整方案。
- 后端提交仍走 `BackendCommitter` 事务：只写 ACTIVE 窗口帧，FROZEN 帧只读（复用
  `THREADING_DESIGN.md §13` 三层模型，窗口=ACTIVE，滑出=FROZEN）。

### 4.6 求解器选型

- 推荐 **手写高斯牛顿 + Schur 消元**（窗口 10~20 帧、数千视觉残差，规模可控，且边缘化
  与 `(point, KF)` 消元一体实现，最贴近 VINS-Mono）。
- g2o 已 vendor，可做备选：需自定义 `EdgeSE3Imu`（g2o 无现成 IMU 因子），且边缘化需
  自行管理先验——成本高于手写，故不作为首选。
- 与现有 `local BA`（g2o）关系：`IMU.4` 落地后，窗口内 BA 切换到滑动窗口求解器；
  纯视觉 g2o 路径保留作 `enable_local_ba=false` A/B 对照。

---

## 5. 配置（config/*.yaml 新增）

```yaml
IMU:
  rate_hz: 200                    # 期望采样率（用于同步窗口/插值）
  gyro_noise_density: 1.7e-04     # rad/s/√Hz（必须来自标定 YAML，禁硬编码 EuRoC 值）
  accel_noise_density: 2.0e-03    # m/s²/√Hz
  gyro_random_walk: 4.0e-05       # rad/s²/√Hz
  accel_random_walk: 4.0e-05      # m/s³/√Hz
  gravity_mps2: 9.80665
  time_offset_s: 0.0              # 相机−IMU 时延，离线标定；首版不在线估计
  init_static_window_s: 2.0
  static_gyro_rms_radps: 0.05
  static_accel_std_mps2: 0.15
```

---

## 6. 前端接入点（IMU.3 细节）

- **正常跟踪**：`FrontendTracker` 用 `ref_frame + ImuPreintegrator(ΔR, Δv, Δp)` 推进出
  `curr` 的 IMU 预测位姿 → 作为 PnP 初值。视觉退化（特征少）时直接输出 IMU 传播位姿
  （`prediction_only=true`，与 `PoseEstimate` 契约一致）。
- **纯旋转分支**（替代现有启发式，`vo.cpp` 旋转主导检测可退役）：当 IMU 旋转可观而视觉
  平移不可分时，位姿旋转取自 IMU，平移保持（不再产生假位移），同时更新结构的几何验证。
- **短期续传**：视觉中断 ≤0.5 s 用 IMU 预积分续传，状态标 Degraded；超限按现有失败门限
  转 Relocalizing（复用 `tracking_state_machine`）。
- 回环校正仍只改全局 `T_wo`/`T_wb`，不动前端 IMU 积分链（与 M6 §10.4 一致）。

---

## 7. 量化验收

| 里程碑 | 验收 |
|---|---|
| IMU.1 | 预积分数值雅可比 vs 解析雅可比差 `<1e-6`；恒定 ω/a 闭合解一致到 `1e-9`；协方差 PSD 非负 |
| IMU.2 | 合成数据（已知 bias/重力/尺度）初始化误差：bias `<5%`、重力方向 `<0.5°`、尺度误差 `<1%` |
| IMU.3 | 纯旋转序列（合成 + 摄像头实测）：旋转段**假平移消失**（位移误差从米级 → `cm` 级）；与 `§3.2` 启发式结果 A/B |
| IMU.4 | 窗口 BA 数值收敛、残差下降单调；边缘化（v2）信息矩阵 PSD |
| IMU.5 | **KITTI 00 合成 IMU**（GT 位姿微分）：全程无 LOST、ATE 明显低于纯视觉基线（历史 55.3 m worst / 133.6 m 单目）；纯旋转段位移误差定量 A/B |
| 数据真值 | **EuRoC V1_01/V2_01/MH_01** 每序列 5 轮：有效率 `≥99%`，V1_01 ATE worst `≤0.30 m`（对齐 M6 §10.5 门限） |
| 一致项 | 确定性验收（1000 帧逐位一致）保持；双轮/多轮 mean/std/worst 基准门不回归 |

---

## 8. 与 M6 ESKF 及既有计划的衔接

- **共享层**：`imu.h`、`time_synchronizer`、config IMU 段、EuRoC IMU 解析被 M6 直接复用，
  不重复开发（M6.1 `imu.h`/`time_synchronizer` 清单与本方案 IMU.0 合并）。
- **分工**：紧耦合管**估计**（前端 + 窗口 BA 精度、旋转歧义），M6 ESKF 管**发布**
  （200 Hz `T_ob` 预测、map/odom 分离、轮速）。窗口 BA 产出的 15 维状态可直接给 ESKF 做
  更新，避免两套积分链互相打架。
- **实施顺序**：本方案 IMU.0→IMU.5 独立成链，与 M1~M5（Atlas/持久化/纯定位）并行；
  与 M6 的交汇点是 M6.1（IMU 输入），建议 IMU.0 先行落地以解锁双方。

---

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| EuRoC 数据缺失（仓库只备了 KITTI） | KITTI 用 GT 位姿微分合成 IMU 走通全链，EuRoC 作为数据真值验收 |
| g2o 无现成 IMU 因子 | 手写求解器作为首选；g2o 仅保留纯视觉路径做 A/B |
| bias 在线估计引入不稳定 | 初始化必须先锁定 bias；窗口 BA 首次迭代 bias 步长限幅；失败回退纯视觉 |
| 边缘化实现复杂导致 v1 延期 | v1 固定滞后窗无先验作基线，先拿到"旋转歧义消失"的核心收益，v2 再补一致性 |
| 与异步后端写回交错（旧问题，`§3.15`） | 窗口 BA 只写 ACTIVE 帧 + `BackendCommitter` 事务，FROZEN 帧只读；沿用 generation/rebase 协议 |
