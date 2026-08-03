# VSLAM 线程架构设计（低资源占用）

> 日期：2026-08-04
> 背景：KITTI 00 实测主线程帧耗时 46ms（含关键帧处理），单帧尖峰可达 422ms
> （回环全局 BA / CPU 降频）；画面卡滞来自主线程被关键帧 Local BA 与回环
> Pose Graph + 全局 BA 阻塞。帧内并行已到顶（OpenCV TBB 已并行 ORB 提取与
> 双目 LK），进一步提速必须在**任务级**并行。

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
5. `run_vo`/`run_slam` 启动/停止后台线程；`config` 加 `Threading` 段。
6. 验证：25 项单测 + KITTI 00 全程（FPS、有效位姿、轨迹长度、perf 报告对比）。
