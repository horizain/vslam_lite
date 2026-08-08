# 从教学 SLAM 到机器人长期定位组件：实施规格

> 状态：M0.1 类型契约已落地，M0.2 状态机进行中（其余尚未实施）
> 基线：`7a22edd`，2026-08-07
> 适用范围：把当前单目/双目 VO + Local BA + DBoW3 回环 + Atlas 原型，演进为
> 可供机器人连续运行 8～24 小时的核心定位组件。
> 历史算法实验与旧基准仍保留在 `IMPROVEMENT_PLAN.md` 和
> `DEVELOPMENT_LOG.md`；本文件规定后续产品化工作的目标架构、算法方向、任务顺序和
> 量化验收，不能用历史计划覆盖本文件。

---

## 0. 如何执行本计划

本计划面向能够按明确规格修改代码、但不适合自行做大范围架构判断的执行者。必须遵守：

1. 一次只实现一个带编号的任务，例如只做 `M1.1`，不得同时开始 `M1.2`。
2. 每个任务开始前记录 `git status`、CTest 和任务要求的基准 JSON。
3. 先增加失败测试，再实现功能；纯重构任务必须保持轨迹和状态转换不变。
4. 不得通过降低内点数、放宽跳变门限、关闭回环或扩大测试容差使测试通过。
5. 行为修改和大规模文件移动不得放在同一提交。
6. Map 的正式观测只通过 `Map` Observation API 修改；优化写回只通过现有
   `BackendCommitter` 或同等级的事务提交器完成。
7. 后端任务必须绑定 `Map::Ptr`、submap id、map generation、topology revision 和
   geometry revision；任一身份不符都返回 stale。
8. 出现 NaN/Inf、ASan/UBSan 报错、死锁、非物理位姿跳变或关键指标退化超过 5%时停止，
   不得继续叠加补丁。
9. 每个任务更新本文件的状态、`DEVELOPMENT_LOG.md` 和相关教程。
10. 完成报告必须列出修改文件、测试命令、结果、指标变化和未验证边界。

### 0.1 数字的两种性质

- **产品硬门槛**：例如不发布非法位姿、队列有界、24 小时无崩溃。未达到就不能进入下一
  发布阶段，不得按数据集修改。
- **首版默认参数**：例如队列容量 3、跟踪 PnP 重投影门限 3 px。这些是实现起点，后续只
  能根据不少于 3 类场景的机器人实录数据统一标定，不能为了单个 KITTI/EuRoC 序列改动。

参数变更必须用 A/B 配置各运行至少 5 次，报告 mean/std/worst，而不是只报告最好一轮。

---

## 1. 产品边界与硬件假设

### 1.1 默认机器人

- 地面轮式机器人，正常速度不超过 3 m/s、角速度不超过 2 rad/s。
- 双目相机 10～20 Hz；后续加入 100～400 Hz IMU，轮速计可选。
- CPU 平台，无独立 GPU；基准平台至少 8 个物理核、16 GiB 内存。
- 单进程连续运行 8～24 小时；相机可能短暂断流、模糊、过曝或重启。
- 控制器需要连续局部位姿，规划器需要能被回环修正的全局位姿。

若实际机器人超出上述速度，必须修改 `RobotLimits` 并用实录数据重新验证；不得在代码中
硬编码 KITTI 车辆速度。

### 1.2 明确不做

- M0～M5 不实现紧耦合视觉惯性 BA；M6 先落地松耦合 ESKF。
- 生产地图模式要求双目或 VIO 提供公制尺度；纯单目继续用于教学/诊断，不执行跨会话地图
  点融合。
- 不实现稠密重建、语义导航、路径规划或控制器。
- 不把 Viewer、数据集读取器或 ROS 绑定进定位核心库。
- 不在日常机器人运行中默认无限在线建图。

### 1.3 运行模式

```cpp
enum class LocalizationMode {
    OdometryOnly,       // 仅连续局部里程计，不加载全局地图
    Mapping,            // 新建地图，仅用于建图任务
    LocalizationOnly,   // 地图只读；机器人日常运行默认模式
    MapMaintenance      // 对 staging 地图做受控更新，不直接覆盖生产地图
};
```

---

## 2. 坐标系与输出契约

沿用当前 `T_ab` 表示“把 b 系坐标变换到 a 系”。不得引入相反约定。

| 名称 | 含义 | 是否允许回环跳变 |
|---|---|---|
| `T_cs` | 子地图 s → 相机 c，当前 `Frame::pose_cs` | 子地图内部优化可更新 |
| `T_ws` | 子地图 s → 全局地图 w | 允许事务式更新 |
| `T_wc` | 相机 c → 全局地图 w，`T_ws * T_cs.inverse()` | 允许全局修正 |
| `T_bc` | 相机 c → 机器人基座 b，固定标定 | 不允许 |
| `T_ob` | 机器人基座 b → 连续 odom 系 o | **不允许因回环跳变** |
| `T_wo` | odom 系 o → 全局地图 w | 回环只更新这里 |
| `T_wb` | 机器人基座 b → 全局地图 w，`T_wo * T_ob` | 允许全局修正 |

视觉单独工作且尚未接入 ESKF 时：

```text
T_wc = T_ws * inverse(T_cs)
T_wb = T_wc * inverse(T_bc)
```

M6 后控制器消费 `T_ob`，规划器消费 `T_wb`。`T_wo` 更新时必须发布
`GlobalCorrectionEvent`，不对 `T_ob` 做插值或回写。

---

## 3. 目标架构和系统不变量

```text
Camera / IMU / Wheel
          │
          ▼
InputValidator + TimeSynchronizer ──► Bounded Input Queue
          │
          ▼
FrontendTracker ─────────────────────► continuous T_ob
          │                                  │
          ├── Keyframe proposal              ├── PoseEstimate + covariance
          ▼                                  └── HealthStatus
LocalMapper + Local BA
          │
          ▼
Relocalizer / Loop Detector ─► Transactional Backend ─► T_wo / Atlas / MapStore
```

以下是不随算法参数变化的产品硬不变量：

1. 不发布 NaN、Inf、非单位四元数、非正定协方差或时间戳倒退的位姿。
2. 失败时发布 `pose_valid=false`；外推只能标为 `prediction_only=true`，最多持续 0.5 s。
3. 后端 invalid/stale/验收失败时，实时地图、Atlas 和持久轨迹逐项不变。
4. 输入队列、后台队列、KF、地图点、描述子、快照和日志缓存全部有硬上限。
5. 跟踪线程不等待 BA、回环、地图保存、磁盘日志或 Viewer。
6. 回环只改变 `T_wo` 或事务式地图版本，不使控制位姿 `T_ob` 跳变。
7. `LocalizationOnly` 不修改持久地图，只允许有界临时跟踪缓存。
8. 地图加载失败完整回滚并降级 `OdometryOnly`，禁止部分加载。
9. 标定、词典或地图格式不匹配时拒绝加载。
10. 每次状态变化、丢帧、拒绝位姿、回环提交和地图切换都有结构化原因码与计数。

---

## 4. M0：定位服务 API、状态机与质量契约

### 4.1 固定框架方向

采用 **Facade + 显式结果对象**：新增 `Localizer` 包装现有 `VisualOdometry`，第一阶段不
修改算法。调用方不再直接读取 `Frame`、`Map` 或 `VisualOdometry` 内部状态。

新增文件：

```text
include/vslam/localization_types.h
include/vslam/localizer.h
src/localizer.cpp
test/test_localizer_contract.cpp
config/robot.yaml
```

核心类型固定为：

```cpp
enum class TrackingState {
    Initializing, Tracking, Degraded, Relocalizing, Lost, Stopped
};

enum class FailureReason {
    None, InvalidInput, TimestampRollback, StereoUnsynchronized,
    InsufficientFeatures, GeometricRejection, MotionDiscontinuity,
    RelocalizationTimeout, BackendOverloaded, MapIncompatible, InternalError
};

struct PoseEstimate {
    uint64_t sequence = 0;
    double timestamp = 0.0;
    SE3 T_ob;
    SE3 T_wb;
    Eigen::Matrix<double, 6, 6> covariance;
    TrackingState state = TrackingState::Initializing;
    FailureReason reason = FailureReason::None;
    bool pose_valid = false;
    bool prediction_only = false;
    uint64_t map_generation = 0;
};
```

### 4.2 状态算法

状态转换采用确定性有限状态机，不使用隐含布尔组合：

| 当前状态 | 条件 | 下一状态 |
|---|---|---|
| Initializing | 连续 3 帧完整几何验收通过 | Tracking |
| Tracking | 连续 2 帧仅达到弱质量门槛 | Degraded |
| Degraded | 连续 3 帧完整质量恢复 | Tracking |
| Tracking/Degraded | 连续 `max_tracking_failures=5` 帧失败 | Relocalizing |
| Relocalizing | 重定位几何验收和运动连续性均通过 | Tracking |
| Relocalizing | 超过 20 帧或 2.0 s | Lost |
| Lost | 收到有效全局重定位 | Tracking |
| 任意 | `stop()` | Stopped |

弱质量帧可以发布 `pose_valid=true`，但协方差至少放大 4 倍；几何失败帧只能发布预测值且
`prediction_only=true`。预测超过 0.5 s 后必须 `pose_valid=false`。

### 4.3 输入硬检查

- 时间戳严格递增；倒退或相等直接拒绝，不调用 VO。
- 左右图均非空、尺寸和类型一致；双目时间差初值上限 1 ms。
- 图像尺寸必须与标定一致；不允许静默 resize。
- `T_bc` 四元数归一化误差 `<1e-6`，平移有限。

### 4.4 M0 量化验收

- 旧 CTest 全过；新增契约测试至少 12 项。
- 空图、时间倒退、重复 stop、析构、左右尺寸错误均不崩溃且 Map revision 不变。
- 同一有效输入的旧 `run_slam` 与 `Localizer` 包装输出平移差 `<1e-12 m`、旋转差
  `<1e-12 rad`。
- M0 不增加线程，处理开销 p95 `<0.1 ms/frame`。

---

## 5. M1：拆分 VisualOdometry，算法行为保持不变

### 5.1 固定框架方向

采用 **Strangler Fig（逐块替换）**，保留 `VisualOdometry` 作为兼容 Facade。禁止一次性
重写 `vo.cpp`。拆分顺序固定如下，每一步独立提交：

1. `M1.1 PoseGate`：纯几何质量与运动连续性。
2. `M1.2 Relocalizer`：候选和几何验证，只返回结果。
3. `M1.3 BackendScheduler`：后台任务生命周期与优先级。
4. `M1.4 FrontendTracker`：初始化、ORB/LK、PnP/3D-3D。
5. `M1.5 LocalMapper`：关键帧、建点、cull、Local BA 快照。

### 5.2 M1.1 PoseGate

移动现有 `pnpReprojectionRmse`、`acceptPose`、`acceptPoseCandidate`、
`checkMotionContinuity`，不改公式和默认值。API 只接收值对象：

```cpp
PoseDecision PoseGate::evaluate(const PoseCandidate&, const MotionPrediction&,
                                const TrackingQuality&, double dt) const;
```

不得访问 Map、Atlas、Viewer、日志全局状态或线程。

### 5.3 M1.2 Relocalizer

固定采用 **DBoW3 候选 + ORB 2D-3D PnP + PoseGate**：

- DBoW3 只负责召回，不负责接受。
- 候选按 BoW 分数排序，逐个执行几何验证。
- `Relocalizer` 返回 `RelocalizationResult`，不得直接切换 Atlas、写 Map 或写轨迹。
- 提交方再次检查 expected Map、submap、KF 对象身份和 generation。

### 5.4 M1.3 BackendScheduler

沿用当前 **单后台线程 + 覆盖式单任务槽**，不恢复历史双队列方案：

- 同类 Local BA 新任务覆盖旧任务。
- LoopClosure 优先于 Local BA；正在执行的优化不强制取消，结果由 stale gate 丢弃。
- 任务槽容量固定 1；等待中的任务最大内存由 M2 预算控制。
- `stop()` 设置标志、notify、join；不得 detach。

### 5.5 M1.4/M1.5 前端与局部建图

- `FrontendTracker` 只输出 `TrackingResult` 和 `KeyframeProposal`，不执行 BA/回环。
- `LocalMapper` 消费提议并通过 Map API 插入 KF、点和 Observation。
- Local BA 仍使用不可变 `OptimizationSnapshot`，锁外求解、事务提交。

### 5.6 M1 确定性验收

新增 `config/deterministic.yaml`：OpenCV 单线程、固定 `cv::setRNGSeed(0x5A17)`、
异步关闭。新增 `scripts/compare_trajectories.py`。

每个子任务前后用同一构建跑 KITTI 00 前 1000 帧：

- 有效帧 ID、状态转换、KF ID 和 Map revision 序列完全相同。
- 最大平移差 `<1e-6 m`，最大旋转差 `<1e-8 rad`。
- 地图点/KF/Observation 数完全相同。
- FPS 退化不超过 3%，RSS 退化不超过 2%。

不满足即回退当前子任务，不进入下一拆分。

---

## 6. M2：实时调度、资源硬预算与可观察性

### 6.1 固定调度框架

采用 **一个跟踪 worker + 当前单后台 worker + 可选 Viewer**：

- 传感器回调只校验并入队，不运行 ORB/PnP。
- 输入队列使用固定容量 ring buffer，默认容量 3。
- 队列满时丢最旧帧、保最新帧；禁止阻塞传感器线程和无限积压。
- Local BA/Loop 仍由一个后台 worker 执行，避免 g2o 与前端形成多任务 CPU 争抢。
- 所有 worker 空闲时阻塞在 condition variable，不自旋。

### 6.2 首版参数

```yaml
Runtime:
  input_queue_capacity: 3
  tracking_deadline_ms: 80       # 10 Hz 地面机器人；20 Hz 平台改为 40
  prediction_timeout_ms: 500
  max_backend_task_age_ms: 500
  shutdown_timeout_ms: 2000

MapBudget:
  max_active_keyframes: 1200
  max_active_points: 120000
  max_descriptor_mb: 256
  max_snapshot_mb: 256
  max_total_estimated_mb: 900
```

### 6.3 地图预算算法

预算触发时按固定顺序回收：

1. 删除 0 个正式 Observation 的点。
2. 删除 `observationCount < 2` 且超过 30 个 KF 未被跟踪命中的点。
3. 卸载非活动 KF 的原图和灰度图，仅保留关键点、描述子、位姿和 Observation。
4. 对共视重叠率 `>0.9`、相邻位姿差 `<0.15 m/3 deg`、且不是回环/子地图锚点的 KF
   做冗余剔除。
5. 冻结超过 2 个非活动子地图并卸载图像缓存；M4 完成后才能把完整子地图换出到磁盘，
   M4 前不得假装已经具备可靠的磁盘换入/换出。
6. 仍超预算时停止增加地图，进入 `Degraded + BackendOverloaded`，不得随机删除锚点。

地图点的“最近命中 KF”需新增显式字段或旁路统计，禁止复用 `observed_count` 旧语义。

### 6.4 结构化指标

新增 `MetricsSnapshot` 并输出 JSON/CSV，至少包含：

- frame latency p50/p95/p99/max 和 deadline miss。
- input received/processed/dropped、queue high-water mark。
- features、grid occupancy、PnP inliers/ratio/RMSE/condition number。
- pose accepted/rejected/prediction-only、每类 FailureReason。
- backend queued/committed/stale/invalid、task age。
- loop queried/candidates/geometric verified/committed。
- KF、MapPoint、Observation、descriptor/snapshot/estimated RSS bytes。
- LOST 次数、持续时间和 relocalization latency。

禁止继续只靠正则解析自然语言日志作为发布验收。

### 6.5 M2 量化验收

- KITTI 00 循环重放 2 小时：RSS 峰值 `<1 GiB`，后 1 小时线性增长 `<5 MiB/h`。
- 输入队列 high-water mark `<=3`，后台等待槽 `<=1`。
- 10 Hz 配置跟踪 latency p99 `<80 ms`；20 Hz 配置 p99 `<40 ms`。
- deadline miss `<1%`；主线程不得出现由 BA/回环造成的 `>500 ms` 阻塞。
- Ctrl-C、输入 EOF、构造失败三种路径均在 2 s 内 join 所有线程。
- 同一输入循环 10 次无崩溃、死锁和队列增长。

---

## 7. M3：视觉前端鲁棒性、退化检测和位姿协方差

### 7.1 固定算法方向

保留当前 ORB、双目 LK、PnP 和 3D-3D 路径；不在本阶段更换为深度网络。新增四层质量门：

1. 输入图像质量。
2. 特征空间分布和双目深度质量。
3. 几何估计与数值可观性。
4. 基于 `dt` 和机器人运动上限的连续性。

只有四层均通过才发布完整质量视觉位姿。

### 7.2 图像和特征质量

首版参数放在 `Quality` 配置段：

```yaml
Quality:
  grid_cols: 8
  grid_rows: 6
  min_features_tracking: 300
  min_occupied_cells: 12
  hard_reject_blur_variance: 10.0
  degraded_blur_variance: 60.0
  max_dark_ratio: 0.80          # gray <= 5
  max_bright_ratio: 0.80        # gray >= 250
```

- Laplacian 方差 `<10` 或暗/亮比例 `>0.8`：输入硬拒绝。
- Laplacian 方差 `[10,60)`：允许尝试跟踪，但成功后状态至少为 Degraded，协方差乘 4。
- 特征少于 300 或 8×6 网格占用少于 12 格：不得只凭总内点接受高置信位姿。
- 图像阈值只适用于 8-bit 灰度；其他类型必须先显式转换并测试。

### 7.3 双目和跟踪几何

延续当前深度范围 `0.5～35 m`。首版几何参数：

- 左右 LK forward-backward error `<=1.0 px`。
- 极线垂直误差 `<=1.5 px`。
- 双目有效点至少 40；PnP 跟踪最终内点至少 20（普通配置）/30（robot 配置）。
- PnP RANSAC 200 次、confidence 0.999、重投影门限 3 px；之后用内点做 iterative refinement。
- 正深度比例 `>=0.9`。
- PnP 内点网格至少覆盖 8 个格子，且 x/y 方向各跨越不少于图像尺寸 35%。
- 完整质量：inlier ratio `>=0.5` 且 RMSE `<=2.5 px`。
- 弱质量：ratio `[0.35,0.5)` 或 RMSE `(2.5,3.5] px`，只能进入 Degraded。
- 低于弱质量直接拒绝。

3D-3D 保留当前 0.25 m RANSAC、15 内点、0.5 ratio 和 0.25 m Kabsch RMSE；新增点集
协方差最小/最大特征值比 `>=1e-3`，否则判定退化。

### 7.4 运动连续性

不能继续只按“每帧最大位移”判断。使用时间归一化：

```yaml
RobotLimits:
  max_linear_speed_mps: 3.0
  max_angular_speed_rps: 2.0
  max_linear_accel_mps2: 5.0
  max_angular_accel_rps2: 4.0
  discontinuity_margin: 1.5
```

候选超过 `limit * dt * margin` 或相对上一速度的加速度门限时拒绝。重定位允许位置不连续，
但必须与全局候选一致，并产生明确的 `GlobalRelocalizationEvent`；不得伪装成普通跟踪帧。

### 7.5 协方差算法

首版采用最终 PnP 内点的 **中心有限差分 Jacobian**，避免写错解析导数：

1. 位姿扰动顺序固定 `[tx, ty, tz, rx, ry, rz]`；采用左扰动
   `T'_cw = Exp(δξ_c) * T_cw`，扰动在相机系表达。
2. 平移扰动 `1e-4 m`，旋转扰动 `1e-6 rad`。
3. 使用最终内点的 2D 重投影残差构造 `J`。
4. `sigma² = max(0.25, SSE / max(1, 2N-6))`，单位 px²。
5. `H = JᵀJ`，特征值小于 `1e-9` 截断；`condition(H) > 1e8` 直接判退化。
6. `cov_c = sigma² * H⁻¹`，再检查有限、对称和正定；通过 SE3 adjoint 和固定 `T_bc`
   变换为机器人基座切空间，最终 `PoseEstimate::covariance` 固定表达为 `T_ob` 左扰动
   `T'_ob = Exp(δξ_o) * T_ob` 在 odom 系中的 6×6 协方差。
7. Degraded 协方差乘 4；prediction-only 按时间每 100 ms 至少乘 2，500 ms 后无效。

这只是视觉测量协方差，不得宣称为完整系统状态协方差；M6 由 ESKF 传播融合。

### 7.6 故障注入与验收

新增黑帧、重复帧、左右错位、时间倒退、跳过 1/5/20 帧、模糊、遮挡、相机断流、
100 m 假位姿测试。每种故障要求：

- 0 崩溃、0 非有限输出、0 高置信假位姿。
- Map revision 在硬拒绝帧保持不变。
- 故障消失后 2 s 内恢复 Tracking，或明确进入 Lost。
- 连续有效帧非物理跳变为 0。

数据集阶段门槛：KITTI 00 五轮有效轨迹率每轮 `>=99%`，ATE worst `<=40 m`、std
`<=8 m`；这是 M3 阶段门，不是最终产品精度门槛。

---

## 8. M4：版本化地图、原子保存和纯定位模式

### 8.1 固定存储框架

采用 **不可变 generation 目录 + 原子 CURRENT 指针 + manifest + 固定小端二进制分片**，
不序列化裸 C++ 对象、不依赖
`sizeof(struct)`，不使用 OpenCV FileStorage 保存大地图。

```text
map_root/
  CURRENT                    # 文本，仅含当前 generation 目录名
  gen-000001/
    manifest.yaml
    submaps.bin
    keyframes.bin
    landmarks.bin
    observations.bin
    descriptors.bin
```

每个二进制文件头固定包含 magic、format version、record count、payload bytes、CRC32。
所有整型使用固定宽度类型，浮点固定 IEEE754 little-endian double/float，并由显式读写函数编码。

### 8.2 manifest 必需字段

- `format_version: 1`。
- 创建项目版本和 git commit。
- 创建时间、地图 UUID、generation。
- 相机内参、畸变、图像尺寸、baseline、`T_bc`。
- 标定 SHA-256 和 DBoW3 vocabulary SHA-256。
- 坐标系约定和单位。
- 每个分片的记录数、字节数、CRC32。
- KF/点/Observation/submap/constraint 总数。

SHA-256 用于身份指纹，CRC32 用于快速损坏检查；不得用文件名代替词典/标定指纹。
实现固定使用 OpenSSL EVP 计算 SHA-256、ZLIB `crc32()` 计算 CRC32；CMake 显式
`find_package(OpenSSL REQUIRED COMPONENTS Crypto)` 和 `find_package(ZLIB REQUIRED)`，
不得自行手写未经验证的哈希实现。

### 8.3 原子保存算法

1. 在短锁内生成 Map/Atlas 不可变快照，锁外编码。
2. 写入同一 `map_root` 下未使用的 `gen-XXXXXX.tmp.<pid>`。
3. flush + fsync 每个分片和 manifest。
4. 用正式加载器重新加载临时地图并运行完整一致性检查。
5. fsync 临时目录，将它 rename 为最终不可变 `gen-XXXXXX`，再 fsync `map_root`。
6. 写 `CURRENT.tmp.<pid>`，内容只有 `gen-XXXXXX\n`；flush、fsync 后原子 rename 覆盖
   `CURRENT`，再 fsync `map_root`。CURRENT 切换是唯一发布点。
7. 至少保留当前和上一个 generation；更旧 generation 由单独 GC 任务删除，不能在保存
   事务中删除。
8. 任一步失败清理未发布临时项；原 CURRENT 和旧 generation 保持不变。

保存任务不得阻塞跟踪；同一时刻最多一个保存任务，新请求覆盖旧请求。

### 8.4 加载验收

必须检查：magic/version、长度、CRC、SHA、ID 唯一、引用存在、四元数、有限值、Observation
双向一致性、Atlas constraint 端点、计数上限。加载到临时对象，全部通过后一次交换为 live map。

默认安全上限：KF 200000、点 5000000、Observation 30000000、单文件 8 GiB；超限拒绝，
防止损坏文件导致内存爆炸。

### 8.5 LocalizationOnly 算法边界

- 持久 Map、Atlas、描述子和 Observation 全只读。
- 允许最多 20 个临时帧、5000 个临时点的 tracking cache。
- 只做当前位姿/临时窗口 motion-only 优化，不回写持久 KF/点。
- 回环只作为全局重定位，不能改变生产地图。
- 退出时丢弃临时 cache，不自动保存。

### 8.6 M4 量化验收

- 同一地图 save→load 后所有 ID、位姿、点、Observation 和约束逐项一致；位姿/点误差
  `<1e-12`。
- 100 次保存/加载循环无对象数量变化。
- 截断每个分片、翻转随机字节、错误标定/词典/版本均 100% 拒绝。
- 保存 2 GiB 目标地图时跟踪 latency p99 退化 `<5%`。
- 进程重启并从已知区域启动，90% 录制启动点在 2 s 内重定位成功。

---

## 9. M5：长期重定位、回环确认和 Atlas 融合

### 9.1 固定算法方向

生产路径采用 **DBoW3 多候选召回 → ORB 双向匹配 → PnP 几何验证 → 重复观测确认 →
SE3 子地图图优化 → 可选点融合**。双目/VIO 使用 SE3；纯单目不做持久跨会话融合。

### 9.2 候选召回

- DBoW3 Top-20，跳过最近 30 KF。
- ORB KNN ratio 初值 0.8，并要求 mutual check。
- 每个 query 最多对 5 个候选做完整 PnP，限制最坏延迟。
- 位置先验只追加候选，不能单独接受回环。
- 记录 annotated loop ground truth 下的 candidate recall@20。

### 9.3 回环几何和确认

普通强候选必须满足：

- PnP 内点 `>=50`、ratio `>=0.70`、RMSE `<=2.5 px`。
- 正深度比例 `>=0.95`，内点至少覆盖 12/48 网格。
- 回环相对位姿与两侧各 3 条里程计边不存在超过 5σ 的冲突。
- 同一历史区域在后续 5 个新 KF 内再次命中；两个历史候选 KF id 距离 `<=20`。

若单次候选 `>=100` 内点、ratio `>=0.90`、RMSE `<=1.5 px`，可作为 strong loop
跳过第二帧确认。所有阈值放入配置并记录采用了普通还是 strong 路径。

### 9.4 Atlas 约束图

先只优化每个子地图 `T_ws`：

- 顶点：Submap。
- 边：TrackingBridge、Relocalization、LoopClosure。
- 第一子地图固定。
- TrackingBridge/Relocalization/Loop 权重初值分别 1/5/20。
- Loop 边先按信息矩阵白化，再使用 Huber；6 自由度 95% χ² 为 12.592，故首版
  `delta=sqrt(12.592)=3.55`。
- 优化前回环残差预检；优化后 robust chi² 不得增加超过 1%。
- 任一子地图平移校正 `>20 m`、旋转 `>30 deg` 或产生非有限值时整笔拒绝。

提交只更新 `T_ws`，不得遍历搬运子地图内部点/KF。

### 9.5 重复点融合

只有 Atlas 对齐成功后才尝试。两个点满足全部条件才合并：

- 世界距离 `< clamp(0.01 * depth, 0.05, 0.20) m`。
- ORB Hamming `<=40`。
- 合并前 Observation 无同 KF feature 冲突。
- 合并后每个观测重投影误差 `<=2.5 px`，正深度比例 100%。
- 合并后 Observation 至少 3 个；否则保留为两个点。

按批次事务融合，每批最多 500 点；任一一致性检查失败则回滚整批。

### 9.6 M5 量化验收

- 标注回环集上 candidate recall@20 `>=90%`、最终提交 precision `=100%`、recall
  `>=80%`。
- kidnapped robot 被移动后不得输出连续假轨迹；回到已知区后 p95 2 s 内恢复。
- 20 次跨 session 对齐成功率 `>=95%`，错误地图对齐 0 次。
- 每次回环提交后连续 `T_ob` 跳变 0；`T_wb` 修正必须有事件。
- 点融合后 Observation 一致性 100%，地图点数量不会因重复往返单调无界增长。

---

## 10. M6：IMU/轮速松耦合 ESKF 和 map/odom 分离

### 10.1 固定算法方向

先采用 **15 维误差状态 ESKF**，不在本阶段实现紧耦合视觉惯性因子图：

- 名义状态：`p_ob, v_ob, q_ob, b_a, b_g`。
- 误差状态：`δp, δv, δθ, δb_a, δb_g`，共 15 维。
- IMU 中点积分传播。
- 视觉 `T_ob`/相对位姿作为 6 维更新，使用 M3 协方差。
- 轮速作为前向速度更新，非完整约束作为可配置伪观测。
- 回环只更新 `T_wo`，不向 ESKF 注入全局跳变。

### 10.2 时间和初始化

- IMU 样本必须严格递增；重复或倒退样本丢弃并计数。
- 每个图像时刻必须由两侧 IMU 样本 bracket，线性插值到精确时间。
- 相机—IMU 固定时间偏移来自离线标定；首版不在线估计。
- 静止初始化窗口 2 s，至少 200 个 IMU 样本。
- 静止判据初值：gyro 均方根 `<0.05 rad/s`，加速度模长标准差 `<0.15 m/s²`。
- 10 s 无法静止初始化时允许使用标定 bias 启动，但状态为 Degraded、协方差乘 10。

### 10.3 噪声与门控

噪声必须来自 IMU 标定 YAML，不得使用 EuRoC 数值作为机器人默认值。配置字段：

```yaml
IMU:
  gyro_noise_density: ...
  accel_noise_density: ...
  gyro_random_walk: ...
  accel_random_walk: ...
  gravity_mps2: 9.80665
  time_offset_s: ...
```

- 视觉 6D 更新使用 Mahalanobis χ² gate，99% 门限 `16.812`。
- 连续 3 个视觉更新被 gate 拒绝时进入 Degraded；达到现有失败门限后 Relocalizing。
- 轮速 1D 更新使用 99% χ² 门限 `6.635`。
- IMU 断流 `>100 ms` 标记 Degraded，`>500 ms` 且视觉也无效时 pose invalid。

### 10.4 输出频率与全局校正

- `T_ob` 按 IMU 频率传播，按配置最高 200 Hz 发布。
- `T_wb = T_wo * T_ob` 与最新全局 transform 组合。
- 回环更新 `T_wo` 时原子发布新 transform 和 `GlobalCorrectionEvent{old,new}`。
- 不平滑 `T_ob`，不把 PGO 结果写回 ESKF 历史状态；控制器始终消费 odom。

### 10.5 M6 量化验收

- 合成静止 10 分钟：位置漂移 `<0.2 m`、姿态漂移 `<1 deg`（有零速/轮速约束时）。
- 恒速/恒转合成轨迹最终位置误差 `<1%`、姿态误差 `<1 deg`。
- EuRoC V1_01/V2_01/MH_01 每序列 5 轮：有效率 `>=99%`；V1_01 ATE worst
  `<=0.30 m`，其余序列按首次完整基线制定但不得低于当前视觉可用率。
- 视觉遮挡 0.5 s：`T_ob` 连续且协方差单调增大；超过预测期限正确 invalid。
- 回环产生 10 m 全局校正时 `T_ob` 单步变化仅来自 IMU 积分，`T_wb` 产生一次带事件的
  全局变化。

---

## 11. M7：验证、发布、运维与机器人灰度

### 11.1 CI 分层

**每次提交（目标 10 min 内）**：

- GCC/Clang Debug 构建。
- CTest、1000 帧 deterministic regression。
- Map save/load round-trip。
- ASan+UBSan 合成/短序列。
- 配置和地图解析故障测试。

**Nightly**：

- Release KITTI 00～10、EuRoC MH/V1/V2。
- 关键序列每个配置 5 轮，保存逐轮 JSON。
- 同步/异步 A/B、故障注入 30 min、RSS slope。
- 输出 candidate recall、loop precision、最差 ATE、std、P99 latency。

**Weekly/Release**：

- 目标机器人 8 h/24 h rosbag 或原始传感器回放。
- ASan/UBSan 长运行；在支持环境执行 TSan/LSan。
- 地图损坏 fuzz、相机重启、磁盘写满、CPU 降频和系统时间变化测试。

### 11.2 API 和部署框架

- 核心库保持无 ROS 依赖；新增可选 `adapter/ros2/`，只做消息转换和参数加载。
- 核心 API 不抛异常跨组件边界，使用 `Status/Result` 显式返回错误。
- 日志带 sequence、timestamp、map UUID、generation、submap 和 reason code。
- 提供 watchdog heartbeat；500 ms 无 heartbeat 由外部 supervisor 重启进程。
- 生产地图只读挂载；新地图写 staging，离线验收后原子切换版本。

### 11.3 灰度顺序

1. **Shadow**：旧定位控制机器人，新组件只记录，不输出控制。
2. **Observe**：新组件输出给规划监控，但不驱动控制。
3. **Limited**：低速、限定区域，旧系统随时 fallback。
4. **Default**：新组件主用，保留一键 OdometryOnly/旧版本回滚。

每阶段至少累计 20 h 和 10 次冷启动；出现一次高置信错误位姿立即退回上一阶段。

### 11.4 最终产品硬门槛

| 指标 | 发布门槛 |
|---|---:|
| 连续运行 | 24 h 无 crash/deadlock |
| 位姿可用率 | `>=99.9%`（排除明确传感器断流） |
| 高置信错误位姿 | 0 |
| 连续有效帧非物理跳变 | 0 |
| LOST 恢复 | p95 `<2 s` |
| 跟踪延迟 | p99 `<图像周期×0.8` |
| deadline miss | `<1%` |
| RSS 峰值 | `<1 GiB`（基准平台） |
| 稳态 RSS 增长 | `<1 MiB/h` |
| 输入/后台队列 | 永不超过配置上限 |
| 回环提交 precision | 100% |
| 损坏/不兼容地图拒绝率 | 100% |
| 关闭耗时 | `<2 s` |

机器人精度门槛不能只用 ATE。正式报告必须同时包含：覆盖率、ATE、RPE 平移/旋转、路径长度
比、连续有效帧跳变、LOST、恢复时延、回环 precision/recall、P99 latency、RSS 和最差一轮。

---

## 12. 文件级实施清单

弱执行者开始任务时，默认只允许修改本行列出的文件以及 `CMakeLists.txt`、对应测试和本
文档状态；需要扩大范围必须先停止并说明原因。

| 任务 | 新增文件 | 允许修改的现有实现 |
|---|---|---|
| M0.1 类型契约 | `localization_types.h`、`test_localization_types.cpp` | `common.h` 仅可补通用矩阵别名 |
| M0.2 状态机 | `tracking_state_machine.{h,cpp}`、对应测试 | 不修改 VO |
| M0.3 Facade | `localizer.{h,cpp}`、`robot.yaml`、契约测试 | `app/run_slam.cpp` 只增加可选入口 |
| M1.1 PoseGate | `pose_gate.{h,cpp}`、`test_pose_gate.cpp` | `vo.{h,cpp}` 只删除已迁移实现和转调 |
| M1.2 Relocalizer | `relocalizer.{h,cpp}`、`test_relocalizer.cpp` | `vo.{h,cpp}`、`loop_closure.{h,cpp}` |
| M1.3 Scheduler | `backend_scheduler.{h,cpp}`、对应测试 | `vo.{h,cpp}`、`backend_committer.*` 仅接口适配 |
| M1.4 Tracker | `frontend_tracker.{h,cpp}`、对应测试 | `vo.{h,cpp}`、`feature.*` |
| M1.5 Mapper | `local_mapper.{h,cpp}`、对应测试 | `vo.{h,cpp}`、`map.*`、`optimizer.*` |
| M2.1 输入队列 | `sensor_packet.h`、`bounded_queue.h`、队列测试 | `localizer.*` |
| M2.2 资源预算 | `resource_budget.{h,cpp}`、预算测试 | `map.*`、`frame.h`、`local_mapper.*` |
| M2.3 指标 | `metrics.{h,cpp}`、`soak_test.py` | `localizer.*`、`backend_scheduler.*`、app 入口 |
| M3.1 质量门 | `tracking_quality.{h,cpp}`、故障注入测试 | `frontend_tracker.*`、`pose_gate.*`、配置 |
| M3.2 协方差 | `pose_covariance.{h,cpp}`、数值 Jacobian 测试 | `frontend_tracker.*`、`localization_types.h` |
| M3.3 故障回放 | `fault_injection.py`、故障样本生成器 | 不修改算法门限 |
| M4.1 编解码 | `map_format.h`、`map_store.{h,cpp}`、`checksum.{h,cpp}` | `atlas.*`、`map.*` 只增加只读快照/装载入口 |
| M4.2 原子发布 | `map_generation_store.{h,cpp}`、崩溃恢复测试 | `localizer.*` |
| M4.3 纯定位 | `localization_cache.{h,cpp}`、模式测试 | `localizer.*`、`local_mapper.*`、`relocalizer.*` |
| M5.1 Atlas PGO | `atlas_optimizer.{h,cpp}`、合成子地图图测试 | `atlas.*`、`optimizer.*`、`backend_committer.*` |
| M5.2 点融合 | `landmark_fusion.{h,cpp}`、融合/回滚测试 | `map.*`、`mappoint.*` |
| M5.3 长期重定位 | 多会话 benchmark/标注脚本 | `relocalizer.*`、`loop_closure.*` |
| M6.1 IMU 输入 | `imu.h`、`time_synchronizer.{h,cpp}`、同步测试 | `localizer.*`、数据集/adapter |
| M6.2 ESKF | `eskf.{h,cpp}`、合成传播/更新测试 | `localization_types.h` |
| M6.3 轮速/输出 | `wheel_odometry.h`、`frame_transforms.{h,cpp}` | `eskf.*`、`localizer.*` |
| M7.1 CI | `.github/workflows/ci.yml`、nightly 脚本 | 构建/测试脚本 |
| M7.2 适配器 | `adapter/ros2/`（可选独立包） | 不向核心库引入 ROS include |
| M7.3 运维 | `docs/OPERATIONS.md`、watchdog/灰度脚本 | app 入口和结构化日志 |

所有新 C++ 模块放入 `namespace vslam`，保持 4 空格缩进、中文注释和 `[[nodiscard]]`
纯函数约定。测试继续使用 assert 风格，但每个新测试文件注册为独立 CTest，避免继续把所有
产品化测试堆入已有 2200 行 `test_vo.cpp`。

---

## 13. 任务依赖和建议提交规模

```text
M0
 └─ M1.1 → M1.2 → M1.3 → M1.4 → M1.5
                         └─ M2
                              ├─ M3
                              └─ M4
                                  └─ M5
                         M3 + M4 ──└─ M6
                              M2～M6 ──► M7
```

建议拆成 30～50 个提交。单提交控制在：

- 新增/修改不超过 8 个自有源文件；
- 自有代码净变化不超过约 500 行；
- 只含一个可独立验收的行为；
- 至少一个直接覆盖该行为的测试。

第一个实施任务固定为 **M0 定位结果契约与状态机**。在 M0/M1 未完成前，不开始 ESKF、
地图格式或跨子地图点融合。

---

## 14. 里程碑状态表

| 里程碑 | 状态 | 当前证据/下一步 |
|---|---|---|
| M0 API/状态机 | 进行中 | M0.1 类型契约已落地（`localization_types.h` + 19 项契约测试）；下一步 M0.2 状态机 |
| M1 模块拆分 | 未开始 | 先提取 PoseGate |
| M2 实时/资源/指标 | 未开始 | 现有异步后端可复用，但缺输入队列和硬预算 |
| M3 前端鲁棒/协方差 | 未开始 | 现有质量 gate 为基础 |
| M4 地图持久化/纯定位 | 未开始 | 当前无版本化 MapStore |
| M5 多会话/Atlas 融合 | 未开始 | 当前 Atlas 只连坐标，不融合点 |
| M6 ESKF/双坐标系 | 未开始 | 当前无 IMU 数据模型 |
| M7 发布/运维 | 未开始 | 当前只有本地 CTest 和有限数据集双轮基准 |
