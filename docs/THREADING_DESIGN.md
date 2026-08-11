# VSLAM 线程架构设计（低资源占用）

> 日期：2026-08-04（初稿）
> 更新：2026-08-06（§十二 落地修正、§十三 三层状态模型，对应 DEVELOPMENT_LOG §3.16）
> 背景：KITTI 00 实测主线程帧耗时 46ms（含关键帧处理），单帧尖峰可达 422ms
> （回环全局 BA / CPU 降频）；画面卡滞来自主线程被关键帧 Local BA 与回环
> Pose Graph + 全局 BA 阻塞。帧内并行已到顶（OpenCV TBB 已并行 ORB 提取与
> 双目 LK），进一步提速必须在**任务级**并行。

> ⚠️ 本文 2026-08-04 初稿的部分设计（§六 双队列、§八 `Threading` 配置段）在
> §3.16 落地时被**单任务覆盖式队列**取代（实现更简单且净收益为正）；请以
> §十二/§十三 为准。

## 一、设计目标

1. **消除主线程卡顿**：所有昂贵的后台工作（关键帧建点/Local BA/回环校正）
   移出跟踪主线程，主线程帧耗时稳定在 ~23ms（extract 4.2 + stereo 14.3 + track 4.2）。
2. **资源占用最少**：只新增 **1 个后台线程**；空闲线程阻塞在条件变量上（0 CPU）；
   队列有界（内存有界）；无自旋、无忙轮询。
3. **跟踪永不等待后台**：主线程推入队列即返回；后台按 FIFO/优先级尽力处理。
4. **可退化**：`threading.backend_threads=0` 时完全退回现有单线程路径（便于 A/B）。

## 二、架构

```
┌────────────── 主线程（跟踪） ──────────────────────────────┐
│ extract → stereo_depth → track(PnP) → 发布位姿             │
│      │                     │                               │
│      └─ 需要新关键帧 ──────┘  每 detection_interval 检测回环 │
│         入关键帧队列（有界，满载短暂阻塞=背压）  入回环队列(cap=1,覆盖旧)│
└───────────────┬──────────────────────────────┬─────────────┘
                ▼                              ▼
        KeyframeQueue(cap=K)             LoopQueue(cap=1)
                ▼                              ▼
        ┌────── 后台线程（单 worker） ─────────────────────────────┐
        │  drain 队列（关键帧优先，回环次之）                       │
        │  ├─ 关键帧：createMapPointsFromStereo / 三角化 /          │
        │  │          Local BA / cull / 冻结里程计边                │
        │  └─ 回环：DBoW3 query / verifyLoop PnP / PoseGraph /      │
        │             全局 BA / 轨迹插值同步                         │
        │  两队列皆空时阻塞在条件变量（0 CPU）                       │
        └──────────────────────────────────────────────────────────┘
```

仅 3 个线程：主线程 + 后台线程 + Viewer 线程（已存在，渲染独立）。

## 三、线程划分与职责

| 线程 | 职责 | 必须实时？ | 同步方式 |
|------|------|-----------|----------|
| 主线程（跟踪） | ORB 提取、双目深度、PnP 位姿估计、发布轨迹 | 是（RT） | 入队不阻塞（回环队列 cap=1 覆盖） |
| 后台线程 | 关键帧建点/三角化/Local BA/剔除 + 回环检测/校正 | 否（尽力） | 关键帧队列 cap=3，满载背压 |
| Viewer 线程 | 渲染轨迹/视频（已存在） | 否 | 独立 data_mutex_ |

## 四、关键帧所有权交接（正确性核心）

跟踪线程计算位姿后，若判定该帧为新关键帧：

1. 跟踪侧**只做轻量冻结**：记录 `odometry_edge`（用当前位姿）、更新 `last_kf_frame_id_`；
2. 把 `Frame::Ptr` **移动**（非拷贝）进关键帧队列，后台接管；
3. 后台在拿到该帧后：`createMapPointsFromStereo`（用跟踪给的位姿建点）、
   与上一关键帧三角化、`selectLocalWindow` + Local BA、`cullMapPoints`。

跟踪侧不等待建点完成即可把 `ref_frame_` 切换为新帧：新帧已通过 PnP 内点
关联继承了参考帧的地图点（`curr_frame_->map_points[trainIdx]=mp`），后台补充的
立体点/三角化点供后续帧异步使用。这保证跟踪所需的 2D-3D 对应始终存在。

## 五、数据共享与锁（最小化）

- **一把 `map_mutex_`** 保护 Map（关键帧表 + 地图点表），粗粒度但锁持有时间短：
  - 跟踪：PnP 前短锁拷贝 ref 帧点位置到局部向量（微秒级）；
  - 后台 Local BA：短锁窗口（10~30 点）；
  - 后台回环校正：**锁内快照 → 锁外优化 → 锁内写回**。
- **回环全局 BA 快照策略**：全局 BA 是唯一长任务（读全部 ~34k 点）。后台线程在
  `map_mutex_` 下拷贝全部关键帧位姿 + 地图点坐标（~0.8MB，微秒级），锁外运行
  g2o 优化，再短锁写回校正量。避免全局 BA 长时间占锁阻塞跟踪。
- **轨迹（pose_trajectory_）**：后台写（回环插值同步）、主线程写（每帧追加）、
  Viewer 读 → 用现有 `data_mutex_`（Viewer）或独立 `traj_mutex_` 保护。
- 地图点坐标在非校正期间**不可变**（append-only），跟踪读 ref 点无需锁（配合
  回环写回时的短暂全局锁保证一致性）。

## 六、队列设计（有界、阻塞、移动语义）

```cpp
template <typename T>
class BoundedQueue {
    std::mutex m_; std::condition_variable cv_;
    std::deque<T> q_; size_t cap_;
public:
    bool push(T v);      // cap>0: 满则阻塞（背压）；cap=0: 非阻塞
    bool pop(T& out);    // 空则阻塞
    void try_push(T v);  // 满则丢弃（回环队列用，cap=1）
};
```

- 关键帧队列：cap=3（默认）。后台处理 ~15ms/KF，关键帧间隔 ~2 帧/46ms，
  实际永不填满；填满时主线程短暂阻塞是合理的背压，避免内存无限增长。
- 回环队列：cap=1，`try_push` 覆盖旧请求——只有最新回环检测有意义。
- 全部 `shared_ptr` 移动，零深拷贝。

## 七、资源控制

- **线程数**：+1（后台）。Viewer 线程已存在。
- **栈**：后台线程显式 `std::thread(..., std::ref, stack_size)` 设小栈（如 2MB），
  控制虚拟内存占用。
- **CPU**：空闲线程 `wait` 在条件变量 → 0% CPU；无自旋锁、无轮询。
- **内存**：队列有界 + 快照只在回环时临时分配 → 稳态内存不增。
- **TBB 竞争**：跟踪用 TBB（提取/双目），后台 g2o 为串行求解，天然互补；
  全局 BA 快照在锁外算，不与跟踪争锁。

## 八、配置项

```yaml
Threading:
  backend_threads: 1      # 0 = 退化为单线程（现状），1 = 后台线程
  keyframe_queue_capacity: 3   # 有界关键帧队列
  loop_queue_capacity: 1       # 回环请求只保留最新
```

## 九、终止与健壮性

- 停止流程：置 `stop_` 标志 → `cv_.notify_all()` → `join()` 所有线程。
- 队列 push/pop 抛异常时通知调用方回退（跟踪线程 catch 后走单线程路径）。
- 后台线程不持有 `map_mutex_` 跨函数调用，避免死锁（锁粒度为单个操作）。

## 十、预期收益（基于实测数据）

| 指标 | 现状 | 方案后（估计） |
|------|------|----------------|
| 主线程帧耗时 | 46ms（含关键帧 15ms 抖动） | ~23ms 稳定 |
| 单帧最大尖峰 | 422ms（回环全局 BA） | ≤30ms（无后台阻塞） |
| FPS（KITTI 00） | 18.8 | ~30-40（受移动 CPU 降频制约） |
| 线程数 | 2（主+Viewer） | 3（+后台） |
| 空闲 CPU | — | 0%（后台阻塞在 CV） |

## 十一、实施步骤

1. `include/vslam/threading.h`：`BoundedQueue` + 后台线程封装（生命周期、停止）。
2. `VisualOdometry` 重构：`insertKeyFrame` 拆分为「跟踪侧轻量判定/入队」与
   「后台处理」（建点/三角化/Local BA/剔除/里程计边冻结）。
3. 回环钩子改为入队：`detectLoop`/`verifyLoop`/`handleLoopCorrection` 迁到后台。
4. Map 加 `map_mutex_`（或 VO 内独立锁），跟踪/后台/回环按第五节加锁。
5. （初稿计划）`run_vo`/`run_slam` 启动/停止后台线程；当前以后端配置和现有
   `async_backend` 路径为准，不要新增未实现的 `Threading` 段。
6. 验证：25 项单测 + KITTI 00 全程（FPS、有效位姿、轨迹长度、perf 报告对比）。

---

## 十二、§3.16 落地修正（2026-08-06，已实现）

初稿 §六/§八 的双队列（关键帧队列 cap=3 + 回环队列 cap=1）在实现时被**单任务覆盖式
队列**取代。实测与原因：

1. **覆盖式队列**（`submitBackendTask`）：滑动窗口 Local BA 的新任务包含旧任务全部
   KF → 入队时丢弃积压的旧 LocalBA、压队首优先执行；超限（cap=4）丢最旧任务。
   关键帧处理（建点）仍留在前端同步完成（下一帧 PnP 需要新点），后台只做 BA/回环。
2. **跳过活动参考帧写回**（`runBackendLocalBA`）：窗口最新 KF = 前端 `ref_frame_`
   位姿不写回，留到下一窗口再 BA。这是异步 LOST 从 74→8 的决定性修复（§3.16 实测）。
3. **锁结构**：`map_mutex_` 改 `std::shared_mutex`（前端读共享、后端写独占）；
   地图计数原子化；`snapshotFrame(with_descriptors=false)` 省全图描述子拷贝。

**为什么"双队列 + 前端只判定 + 后台建点"被简化**：初稿假设建点可异步，但本项目每帧
重新提取 ORB（非 LK），下一帧 PnP 依赖本帧 KF 的点——建点必须同步才能保跟踪连续。
异步的真正收益在**BA/回环**，建点保留在前端即可。

**A/B 结果（§3.16）**：异步默认开启后，KITTI 00 全程 FPS 27.9→34.4（+23%）、
LOST 292→8（-97%）、闭环 1→3、子地图重建 13→3。

## 十三、三层状态模型（推荐的下阶段架构，参考 ORB-SLAM3 / VINS-Mono）

滑动窗口 + 覆盖式队列解决了"数据新鲜度"，但回环校正（`handleLoopCorrection`）仍持
独占锁做全图快照 + 全量点写回，是仅剩的大临界区。参考两个开源系统后给出的完整设计：

### 13.1 开源参考要点

| 系统 | 线程模型 | 关键设计 | 对本项目的启示 |
|------|----------|----------|----------------|
| ORB-SLAM3 | Tracking / LocalMapping / LoopClosing / Viewer 四线程 | Tracking **只读**地图；LocalMapping 独占写（含窗口 BA，持 map mutex 但窗口小故块短）；`UpdateLocalMap()` 每帧重建局部共视窗口 → 前端恒跟踪"已提交"数据；MapPoint 存 `mnFirstKFid`（锚定 KF） | 所有权分离：前端读、后端写；局部窗口每帧重建保证新鲜 |
| VINS-Mono | 估计器线程（前/后端合一） | **滑动窗口全 BA + 边缘化**（Schur 补）——滑出窗口的帧不丢弃而是变先验因子，解决"窗口滑出一致性"；特征按 `(point_id, pixel)` 索引式存储（缓存友好） | 滑出窗口的 KF 应有"冻结/先验"语义，不能简单丢弃 |

### 13.2 三层状态模型

```
┌────────────────────── 跟踪线程（主线程，无锁读）──────────────────────┐
│  每帧开头: C = atomic_load(&C_)                     （帧级一致快照）  │
│  读 ACTIVE/FROZEN 点: p_w = C(anchor) ∘ X_wc⁰(anchor) ∘ p_l  （批量组合） │
│  KF 插入时 → 入队给 Local Mapping                         （不等待）   │
└───────────────┬──────────────────────────────────────────────┬────────┘
                │ KF 队列（覆盖式，窗口 BA 优先）                │
        ┌───────▼──────── Local Mapping 线程 ───────────┐  ┌────▼─── 回环线程（低频）───┐
        │ 三角化 + 窗口 BA（world 系快照, g2o 单线程）     │  │ detect → verify           │
        │ └▶ release 发布 + 原子写回（仅窗口, 短锁）      │  │ pose graph on 全量快照     │
        │ 滑出 → Freeze（只读 SoA 批）                   │  │ └▶ 生成 C → COW + 原子交换 │
        └────────────────────────────────────────────────┘  └────────────────────────────┘
```

- **局部层（滑动窗口，热）**：当前帧 + 最近 N 个 ACTIVE KF 及其点批，仅此可变
  （Local Mapping 独占写、窗口 BA）。
- **全局层（冻结，冷）**：滑出窗口的 KF 显式 **FROZEN**（位姿/点只读，批量 SoA），
  窗口 BA 永不触碰；只有全局校正场 C 能移动它 → 前端读到稳定值。
- **校正场 C（稀疏、copy-on-write、原子指针交换）**：回环位姿图结果只落成
  `C: KF_id → SE3`（约 #KF × 56B ≈ 112KB），不重写 20 万点；点坐标读时组合
  `p_w = C(anchor) ∘ X_wc⁰(anchor) ∘ p_l`（锚定局部坐标）。前端每帧 `atomic_load`
  一次 C 指针，整帧用同一份 → 帧内全局一致、零锁。

### 13.3 与当前实现的映射

| 三层模型组件 | 当前状态（§3.16） | 落地步骤 |
|--------------|-------------------|----------|
| 前端参考不被后端挪动 | ✅ 跳过活动参考帧写回 | — |
| 覆盖式队列 / BA 优先 | ✅ 已实现 | — |
| 描述子免拷贝快照 | ✅ `with_descriptors=false` | — |
| FROZEN 冻结协议 | ⬜ 隐式（滑出即不被选中） | `selectLocalWindow` 显式标记 FROZEN，前端参考限 ACTIVE/FROZEN |
| 点锚定局部坐标 | ✅ `MapPoint::pos_s` + `Submap::T_ws` | 继续验证跨子地图/大规模场景 |
| 校正场 C | ⬜ 回环全量写回 | `handleLoopCorrection` 只生成 C + 轨迹副本原子交换 |

g2o 保持单线程：窗口 BA 与回环位姿图规模已被窗口/点截断/位姿采样控制住，
多核收益应投向可并行热点（ORB/LK），而非 BA。

## 十四、版本化事务写回与子地图坐标管理（2026-08-06）

§十三的 ACTIVE/FROZEN/C 三层模型给出了低锁读路径，但还缺少两个正确性前提：

1. 优化快照必须绑定明确的地图版本，过期结果不得按 ID 覆盖实时对象；
2. 子地图必须真正拥有局部坐标，全局位姿只能由子地图变换和局部状态派生。

在这两个前提完成前直接增加校正场 C，只会产生第四份可变位姿，不能解决一致性问题。

### 14.1 强制不变量

1. Optimizer 永远不接收可写的实时 `Map::Ptr`，只读不可变快照；
2. 每个任务携带 `submap_id + topology_revision + geometry_revision`；
3. Local BA/PGO/GBA 只返回候选增量和质量指标，不能直接修改 Frame/MapPoint；
4. 只有一个 `BackendCommitter` 有权发布几何新版本；
5. 地图、约束、轨迹锚点必须在同一提交中切换版本；
6. Tracking 每帧开始读取一次只读快照，整帧不得跨版本读数据；
7. 优化失败、结果过期或验收失败时，实时状态必须逐项保持不变。

### 14.2 优化接口与版本模型

```cpp
struct OptimizationSnapshot {
    SubmapId submap_id;
    uint64_t topology_revision;  // KF/点/观测集合变化
    uint64_t geometry_revision;  // pose/point 坐标变化
    std::vector<KeyframeState> keyframes;
    std::vector<LandmarkState> landmarks;
    std::vector<Constraint> constraints;
};

struct OptimizationResult {
    SubmapId submap_id;
    uint64_t base_topology_revision;
    uint64_t base_geometry_revision;
    std::vector<PoseUpdate> poses;
    std::vector<PointUpdate> points;
    OptimizationMetrics metrics;  // chi2、最大校正、连续性、有限值
};
```

Optimizer API 改为纯计算：

```cpp
OptimizationResult solveLocalBA(const OptimizationSnapshot&);
OptimizationResult solvePoseGraph(const OptimizationSnapshot&);
CommitStatus BackendCommitter::commit(const OptimizationResult&);
```

提交器依次检查目标子地图、版本、对象存活和优化质量，然后在一次状态临界区中应用全部
pose/point/constraint 更新并增加 revision。Local BA 结果过期时直接丢弃并调度最新窗口；
PGO 第一阶段也严格丢弃。只有在“同一子地图、拓扑未变、仅尾部追加 KF”的协议和测试完成后，
才允许显式 rebase，禁止隐式猜测。

### 14.3 子地图局部坐标

定义 `T_ws` 为子地图坐标到世界坐标，`T_cs` 为子地图坐标到相机坐标，地图点存 `p_s`：

```text
T_wc = T_ws · T_sc
p_w  = T_ws · p_s
```

```cpp
enum class SubmapState { ACTIVE, FROZEN, RETIRED };
enum class ConnectionState { UNCONNECTED, TENTATIVE, CONNECTED };

struct Submap {
    SubmapId id;
    SE3 T_ws;
    uint64_t topology_revision;
    uint64_t geometry_revision;
    SubmapState state;
    ConnectionState connection;
    Map::Ptr local_map;
};
```

任意时刻只能有一个 ACTIVE 子地图。LOST 后匀速外推只能产生低置信 TrackingBridge 约束，
新图状态为 TENTATIVE，不能直接冒充可靠全局连接。更新子地图全局位置时只修改 `T_ws`，
不再遍历所有 KF 和地图点。

### 14.4 锚定轨迹与地图点

普通帧不再重复保存全局 `T_cw`，而是记录相对参考关键帧的固定局部运动：

```cpp
struct FramePoseRecord {
    FrameId frame_id;
    double timestamp;
    SubmapId submap_id;
    KeyframeId anchor_kf;
    SE3 T_c_anchor;
    bool valid;
};
```

导出/Viewer 按 `T_cw(frame) = T_c_anchor * T_anchor_w` 派生全局位姿。回环只更新锚点/子地图
校正，普通帧自动跟随，不再按 KF id 对校正量做区间插值。地图点同样保存
`anchor_kf + p_anchor`；Local BA 提交点结果时转换回锚定坐标。锚点 KF 被删除前必须先执行
显式 re-anchor，不能留下悬空引用。

### 14.5 Atlas 子地图约束图

Atlas 节点是 Submap，边分三类：

- `TrackingBridge`：跟丢后运动外推，低权重、可被后续约束修正；
- `Relocalization`：跨子地图重定位产生的 SE3/Sim3 约束；
- `LoopClosure`：完整几何验证后的高置信鲁棒约束。

跨子地图重定位不再立即 `activate + 覆盖 curr_frame pose`，而是先生成约束、优化 `T_ws`、
原子提交，再决定活动图切换。坐标连接与地图融合拆开：M5 只需统一世界坐标即可输出连续
轨迹，地图点/KF 去重融合留给独立里程碑，避免一次改动同时承担两种风险。

### 14.6 校正场 C 的发布

完成版本提交、局部坐标和锚定数据后，PGO 结果可发布不可变稀疏校正场：

```cpp
using CorrectionField = std::unordered_map<KeyframeId, SE3>;
std::atomic<std::shared_ptr<const CorrectionField>> corrections;
```

回环线程构造完整新 C，验收后原子交换；Tracking 每帧只 load 一次。该机制替代全量点写回，
但不替代 revision/Committer：C 本身也必须记录 base revision 并接受 stale 检查。

### 14.7 渐进落地与验收

| 里程碑 | 改动 | 关键验收 |
|--------|------|----------|
| M0 ✅ | 正常跟踪/重定位统一位姿验收，PGO 防爆 | 恶性边/大跳拒绝后状态不变，连续帧 >10m=0 |
| M1 ✅ | Optimizer Result + Map/Submap revision | stale Local BA/PGO 可重复拒绝 |
| M2 ✅ | BackendCommitter + 只读 TrackingSnapshot | 一帧只观察一个 revision，无部分提交 |
| M3 ✅ | `T_cs/p_s/T_ws` 局部坐标迁移 | 改 `T_ws` 时 KF/点/轨迹一致移动 |
| M4 ✅ | 锚定普通帧和地图点 | 删除全量轨迹插值与全量点搬运 |
| M5 ✅ | Atlas 约束图 | 跨子地图连接不跳变，融合可暂不实现 |
| M6 ◐ | revision rebase + 异步默认恢复已落地；COW 校正场未实现 | TSan/压力测试与 COW 发布仍待完成 |

> 2026-08-06 落地记录（§3.20）：M0 统一验收 + 同步 BA 跳过活动参考写回；
> M1/M2 快照/Result/Committer/帧级快照；M3 子地图局部坐标（对齐只改
> `T_ws`）；M4 锚定轨迹（删除全量插值，修正"回环校正世界系共轭"与
> "KF 帧锚定陈旧 ref"两个回归）；M5 Atlas 约束图（TrackingBridge +
> Relocalization 事务式约束，失败回滚）；M6 Committer 追加 rebase 协议
> （几何版本唯一硬判据）+ 异步默认恢复。
>
> M4/M6 的 `sampled-GBA`（每 3 个 KF 采样）与锚定轨迹不兼容（三周期
> 锯齿，§3.18 预测），已默认关闭（`global_ba_iterations: 0`）；全量
> GBA 留给后续异步后端（分钟级任务）恢复。回环验证召回率（DBoW 候选
> 的地图点被 cull 后 3D-2D 对应不足）是 ATE ≤30m 目标的剩余瓶颈。

### 14.8 回环前缀快照、Essential Graph 与尾段重基（2026-08-11）

同子图回环不再从验证一直持有 `map_mutex_` 独占锁到 PGO 结束。新事务分三段：

1. 短锁内重验回环、深拷贝当前 KF/点/观测/边，记录冻结前缀端点与每个
   live 对象的身份；
2. 释放 Map 锁，把逐 KF 图压缩为有界 Essential Anchor Graph（首尾、最新回环端点、
   均匀锚点），在锁外执行 g2o，再把锚点校正渐进传播到冻结前缀；
3. 短独占锁重验 Map/Submap、geometry revision 和全部前缀对象身份。求解期间仅新增的
   尾段 KF 按冻结端点校正重基，新点跟随最早存活观测 KF，然后与前缀一次原子提交。

若前缀对象被删除/替换、几何版本变化或活动子图切换，结果整笔 stale；不允许跳过缺失对象后
部分提交。跨子图回环本来就以 Submap 为图顶点，继续走 Atlas Submap Graph。运动输出同时显式
分为 `T_ob` 和 `T_wo`：前端只积分 `T_ob`，回环/重定位的全局差异留在 `T_wo`，并保持
`T_wb = T_wo * T_ob`。

KITTI 00 第一阶段门槛：时间戳一一关联覆盖率 ≥99%、连续有效帧 >10m 跳变为 0、
路径长度比 0.9~1.2、子地图重建 ≤3、SE3 ATE ≤30m。RPE 只统计真正连续的有效帧，
跨 LOST 空洞的位姿对必须单独统计，不能伪装成“一帧 RPE”。

### 14.9 有界回环索引、维护任务与验证级联（2026-08-11）

移动档使用 `flat_dbow3`：词表节点是加载后只读的连续 SoA（descriptor、weight、
first-child/count、word-id 分列），量化和 TF-IDF/L1 分数语义与官方 DBoW3 保持一致。
每个关键帧的稀疏 BoW、地点假设与索引元数据仍由 `LoopClosure::mutex_` 保护；任何线程都
不得在持有该锁时回调 Map/Atlas。索引上限为 256 个关键帧，首帧固定保留，其余槽位用
确定性 reservoir-style 采样覆盖全程，而不是 FIFO 只留下尾段。

资源预算删除关键帧时，Map 临界区只合并待清理 id。后台 `LoopMaintenance` 在 Map 锁外
批量释放 `Frame::Ptr`、BoW 和地点假设；每个后台任务开始前也先 drain，因此维护任务被
更高优先级回环覆盖时清理不会丢失。锁序保持为：任务槽锁释放后，分别获取 LoopClosure
或 Map 锁；禁止同时嵌套两者。

检索只生成候选并更新有限地点假设。mobile 的 `mature_verification_limit: 4` 令昂贵 PnP
只验证连续至少两次命中的最多 4 个成熟 BoW 地点，另允许 1 个独立位置先验兜底；默认值
0 保留桌面/高精度档的全部候选验证。这个级联只减少工作量，不改变 50 内点、0.70
内点率、RMSE、正深度和网格覆盖等最终接受门。若候选尚未成熟，本轮可以不做 PnP；前端
不能等待它成熟，也不能在回环线程中持 Map 锁轮询。
