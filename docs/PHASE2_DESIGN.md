# VSLAM Phase 2 设计文档：完整 SLAM 闭环

> 创建日期: 2026-08-03
> 状态: 设计定稿（待实施）
> 关联: `docs/DEVELOPMENT_LOG.md` §3.2、`docs/TUTORIAL.md` §9
> 决策记录: 范围 = 完整 SLAM 闭环；校正模型 = Sim3（7DOF 含尺度）

---

## 1. 目标与验收标准

**目标**：把当前的 VO（无回环）升级为完整 SLAM——当相机"回到老地方"时自动检测回环、验证、校正累积漂移，显著降低轨迹误差。

```
前端 VO（已有）→ 关键帧 → 词袋回环检测(DBoW3) → 几何验证(PnP) → Sim3 校正
            → 位姿图优化 → 全局 BA → 地图/轨迹更新 → KITTI 00 ATE 评估
```

**验收标准（量化）**：

| # | 验收项 | 标准 |
|---|--------|------|
| 1 | KITTI 00 全程运行 | 无 LOST，FPS ≥ 20（与 Phase 1 同量级） |
| 2 | 回环正确性 | 检测出的回环中 ≥ 90% 通过几何验证且为真实回环（KITTI 00 已知 0→末端闭合） |
| 3 | ATE 提升 | 全量 ATE RMSE 从 Phase 1 的 ~133.6m 显著下降（目标 ≤ 50m，Sim3 同时校正尺度漂移） |
| 4 | 回归 | 全部既有单元测试仍通过；新增 Sim3/回环/位姿图测试通过 |
| 5 | 教学可读 | 每个新模块 ≤ 200 行，注释中文，可逐文件读懂 |

---

## 2. 现状与差距分析

| 组件 | 现状 | Phase 2 差距 |
|------|------|--------------|
| DBoW3 依赖 | `install_deps.sh` 已支持安装；CMake `find_package(DBoW3)` + `HAS_DBOW3` 已接入 | 本机实际安装；需要预训练词典文件（ORBvoc） |
| `loop_closure.h/cpp` | 接口已留：`loadVocabulary` / `addKeyFrame` / `detectLoop` / `verifyLoop`，实现全 TODO | 完整实现（词袋加载、数据库增查、候选过滤、几何验证、Sim3 输出） |
| `optimizer.cpp` | `localBundleAdjustment` ✅、`globalBundleAdjustment` ✅（转发 Local BA）、`poseGraphOptimization` ❌ TODO | 实现位姿图优化（相邻边 + 回环边）；全局 BA 增加回环后触发路径 |
| `common.h` | 只有 `SE3` | 新增轻量 `Sim3`（含 Umeyama 求解） |
| `vo.h/cpp` | 纯前端，关键帧插入后无回环钩子 | 增加回环开关/参数；关键帧插入后喂给 LoopClosure；回环校正后更新地图 |
| `app/run_slam.cpp` | CMakeLists 中已注释预留 | 新建入口并启用 CMake 目标；`run_vo` 保持纯 VO 不变（便于 A/B 对比） |
| `config/default.yaml` | `LoopClosure` 段已有：`vocab_path` / `min_score` / `pnp_inlier_ratio` | 补充开关、检测频率、时间窗口、最小内点数、Sim3 迭代数 |
| 测试 | 12 项断言（SE3/投影/ORB/初始化/BA/长稳/MiniAtlas） | 新增 Sim3、回环几何验证、位姿图优化测试 |

---

## 3. 总体架构与数据流

```
                          ┌───────────────────────────────┐
  KITTI/TUM/EuRoC 图像 ──► │  run_slam (app/run_slam.cpp) │
                          └──────────────┬────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
                     │  VisualOdometry (src/vo.cpp)          │
                     │  addFrame → 跟踪/初始化/LOST 状态机    │
                     │         └─ 插入关键帧 kf              │
                     │              │                        │
                     │              ▼                        │
                     │  LoopClosure (src/loop_closure.cpp)   │
                     │  addKeyFrame(kf) ──► 词袋入数据库      │
                     │  每 N 个关键帧 detectLoop(kf)          │
                     │      │ 候选帧 + 分数过滤                │
                     │      ▼                                │
                     │  verifyLoop(kf, kf_loop, sim3)        │
                     │      │  PnP 几何验证 + Umeyama Sim3    │
                     │      ▼                                │
                     │  校正成功：Sim3 传播 + 地图点更新        │
                     │      ▼                                │
                     │  Optimizer::poseGraphOptimization(map) │
                     │  Optimizer::globalBundleAdjustment(...)│
                     └──────────────────┬────────────────────┘
                                        ▼
                          Viewer 可视化 + TUM 轨迹输出（校正后）
```

**新增/修改文件清单**：
- 新增 `app/run_slam.cpp`、`docs/PHASE2_DESIGN.md`（本文档）
- 修改 `include/vslam/common.h`（+Sim3）、`include/vslam/loop_closure.h`（接口扩展）、`include/vslam/optimizer.h`（位姿图参数）、`include/vslam/vo.h`（回环配置与钩子）
- 修改 `src/loop_closure.cpp`（核心）、`src/optimizer.cpp`（位姿图）、`src/vo.cpp`（集成）、`CMakeLists.txt`（启用 run_slam）、`config/default.yaml`（配置）
- 修改 `test/test_vo.cpp`（新增测试）、`scripts/`（词典下载辅助脚本）

---

## 4. 详细设计

### 4.1 轻量 Sim3（common.h）

```cpp
/// 相似变换：p' = s · (R·p) + t（7 自由度，单目回环校正的核心）
struct Sim3 {
    double s = 1.0;                     // 尺度（两子地图尺度比）
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Vec3 t = Vec3::Zero();

    [[nodiscard]] Vec3 operator*(const Vec3& p) const;   // 变换点
    [[nodiscard]] Mat44 matrix() const;                  // [sR t; 0 1]
    [[nodiscard]] Sim3 inverse() const;                  // s'=1/s, q'=q⁻¹, t'=-s⁻¹q⁻¹t
    [[nodiscard]] SE3  toSE3() const { return SE3(q, t); } // 丢弃尺度（全局 BA 前用）
    static Sim3 fromMatrix(const Mat44& T);
    /// 用 Eigen::umeyama(src→dst, with_scaling=true) 求解：dst ≈ s·R·src + t
    static bool estimate(const std::vector<Vec3>& src, const std::vector<Vec3>& dst, Sim3& out);
};
```

设计要点：
- 用 Eigen 现成 `umeyama`（内部 SVD，教学清晰），不再造轮子；g2o 侧另有 `g2o::Sim3`，两处通过 `matrix()`/构造互转。
- 单目回环的尺度来自两个子地图的尺度比，由 Umeyama 的 `s` 直接给出——这正是 SE3 方案做不到的（已决策用 Sim3）。

### 4.2 LoopClosure 完整实现（src/loop_closure.cpp）

**Impl 内部状态**（`HAS_DBOW3` 守卫，头文件只留前向声明）：
- `std::unique_ptr<DBoW3::Vocabulary> vocab_`（词典，加载后只读）
- `std::unique_ptr<DBoW3::Database> db_`（数据库，随关键帧增长）
- `std::unordered_map<unsigned long, DBoW3::EntryId> kf_db_id_`（关键帧 id → 数据库条目，用于回查）
- `std::unordered_map<unsigned long, DBoW3::BowVector> kf_bow_`（关键帧 id → 词袋向量，避免重复计算）
- 配置：`min_score_`、`temporal_window_`、`min_loop_inliers_`、`pnp_inlier_ratio_`、`ransac_threshold_`、`camera_`

**接口扩展**（决策：修改而非新增，使"检测→验证"职责清晰）：

```cpp
bool loadVocabulary(const std::string& vocab_path);       // 加载 + 建库（DBoW3 Database::setVocabulary）
void addKeyFrame(Frame::Ptr kf);                          // 计算 bow/feat 向量，db_->add，缓存 id
Frame::Ptr detectLoop(Frame::Ptr kf);                     // 返回候选关键帧；nullptr 表示无回环
bool verifyLoop(Frame::Ptr kf_curr, Frame::Ptr kf_loop, Sim3& sim3_loop_to_curr); // 验证 + 输出 Sim3
```

**detectLoop 候选过滤三步**：
1. `db_->query(bow, max_results=5)` 取 Top-5；
2. **时间过滤**：候选帧 id 与当前帧 id 差 < `temporal_window`（默认 30）→ 跳过（刚走过的路不算回环）；
3. **分数过滤**：DBoW3 归一化分数 ≥ `min_score`（默认 0.3）→ 取最高分候选返回。

**verifyLoop（几何验证 + Sim3 求解，教学简化版 ORB-SLAM 流程）**：
1. ORB 描述子匹配 kf_curr ↔ kf_loop（复用 `FeatureMatcher` 的 knn+ratio，不用基础矩阵 RANSAC）；
2. 收集 3D-2D 对应：匹配对中 kf_loop 侧已关联地图点 `mp_loop`（回环尺度）↔ kf_curr 侧特征点像素；
3. `solvePnPRansac` 求 kf_curr 在回环尺度下的位姿 `T_cw_curr'`；
4. **内点判定**：内点数 ≥ `min_loop_inliers`（默认 30）且内点比例 ≥ `pnp_inlier_ratio`（默认 0.7）→ 验证通过；
5. **Sim3 求解**：对内点中 kf_loop 关联的地图点，构造 3D-3D 对应
   `p_loop_c = T_cw_loop · mp_loop_w` ↔ `p_curr_c = T_cw_curr' · mp_loop_w`，
   `Sim3::estimate(p_loop_c, p_curr_c)` → `sim3_loop_to_curr`（含尺度比 s）；
6. 返回 true。**不通过则返回 false**（候选被拒，不影响后续）。

> 教学注记：真实 ORB-SLAM 用 Sim3Solver 做 RANSAC + 图优化迭代；本设计用"PnP 初值 → Umeyama 精化"，数学等价、实现短、可读性强，精度对教学级足够。

### 4.3 回环校正与地图更新（vo.cpp 内新增私有方法）

`VisualOdometry` 新增成员：`std::unique_ptr<LoopClosure> loop_closure_;` + 回环配置。

关键帧插入 `insertKeyFrame()` 成功后：
1. `loop_closure_->addKeyFrame(kf)`（词袋入库）；
2. 每 `loop_detection_interval`（默认 10）个关键帧触发一次检测：
   - `cand = loop_closure_->detectLoop(kf)`；无候选 → 跳过；
   - `Sim3 sim3; ok = loop_closure_->verifyLoop(kf, cand, sim3)`；失败 → 跳过；
   - 成功 → `handleLoopCorrection(sim3, kf, cand)`。

`handleLoopCorrection`（新私有方法）：
1. **Sim3 传播**：回环帧及其后续关键帧的位姿应用校正
   `T_wc_i' = S_global · T_wc_i`，其中 `S_global` 由 `sim3_loop_to_curr` 与当前关键帧位姿合成（把当前子地图拉回回环子地图的全局坐标系）；关键帧 `pose_cw` 取逆回写；
2. **地图点更新**：所有受影响地图点 `pos_w' = S_global · pos_w`；重建观测帧的 `map_points` 引用（仅位姿/坐标更新，引用关系不变）；
3. **位姿图优化**：`Optimizer::poseGraphOptimization(map_, loop_edges)`；
4. **全局 BA**：`Optimizer::globalBundleAdjustment(camera_, map_)`（全部关键帧+地图点，迭代数取 `global_ba_iterations`）；
5. 更新 `trajectory_` 中受影响的历史轨迹点（与 Step 1 同一变换），保证输出的 TUM 轨迹与校正后地图一致；
6. 状态栏/日志输出：`Loop closed! kf# -> kf#  scale=... inliers=...`。

### 4.4 位姿图优化（optimizer.cpp，g2o）

**顶点**：所有关键帧（`g2o::VertexSE3Expmap`，T_wc 语义——沿用 localBA 的取逆约定）。

**边**：
- **相邻边**（里程计约束）：`kf[i-1] ↔ kf[i]` 的相对位姿 `T_rel = T_cw[i] · T_cw[i-1]⁻¹`（组合为 T_wc 后再取相对），信息矩阵按两帧共视地图点数加权（共视多 → 置信高）；
- **回环边**（Sim3 约束）：`kf_loop ↔ kf_curr`，由 `sim3_loop_to_curr` 转 `g2o::EdgeSim3`（如 apt g2o 可用）或转 SE3 边（`sim3.toSE3()`，丢尺度但位姿图只优化位姿，尺度已由 Sim3 传播时吸收——**决策：回环边用 SE3 表示，尺度校正完全交给 Sim3 传播 + 全局 BA**，避免依赖 g2o Sim3 边，降低版本兼容风险）。

**固定**：第一个关键帧 `setFixed(true)` 锚定坐标系。

**回写**：优化后所有关键帧 `pose_cw` 取逆写回；地图点坐标随最近关键帧的相对关系重投影更新（简化：地图点不动，仅位姿更新——全局 BA 紧接着做精细修正）。

接口签名调整（`optimizer.h`）：

```cpp
/// (Phase 2) 位姿图优化：相邻边 + 回环边校正全局漂移
/// @param loop_edges  回环边列表：{kf_id_a, kf_id_b, 相对 SE3, 权重}
static void poseGraphOptimization(Map::Ptr map,
    const std::vector<LoopEdge>& loop_edges = {});
```

其中 `struct LoopEdge { unsigned long a, b; SE3 T_rel; double weight; };` 放在 optimizer.h 或 vo.h。

### 4.5 全局 BA（已有，补充触发路径）

`globalBundleAdjustment` 已实现（转发 localBA 全量窗口）。Phase 2 仅在 `handleLoopCorrection` 步骤 4 中调用，无需改实现；若因规模变慢（几千关键帧），保持 localBA 的"最早帧锚定"逻辑即可，教学规模不引入分块策略。

### 4.6 run_slam 入口与 VO 集成

**`app/run_slam.cpp`**（仿照 run_vo.cpp 结构）：
- 参数：`<dataset_path|camera_index> [config.yaml] [trajectory.txt] [--tum|--euroc]`（与 run_vo 相同）；
- 差异：`VOConfig` 读取 yaml 时 `LoopClosure.enable_loop_closure=true`（run_vo 默认 false）；
- 主循环：读帧 → `addFrame` → Viewer → FPS；结束后保存 TUM 轨迹（**必须是校正后的轨迹**，即 vo.getTrajectory() 在 handleLoopCorrection 后已更新）。

**CMakeLists.txt**：取消 `run_slam` 注释块；`VOConfig` 增加字段（见 4.7）。`run_vo` 保持纯 VO（`enable_loop_closure=false`），用于 A/B 评估对比。

### 4.7 配置项（config/default.yaml + VOConfig）

```yaml
LoopClosure:
  enable_loop_closure: true   # run_slam 默认开，run_vo 默认关
  vocab_path: ""              # 词典路径（见 §7 数据准备）
  min_score: 0.3              # 词袋候选最低分
  temporal_window: 30         # 跳过最近 N 个关键帧（时间过滤）
  detection_interval: 10      # 每 N 个关键帧检测一次
  pnp_inlier_ratio: 0.7       # 几何验证最小内点比例
  min_loop_inliers: 30        # 几何验证最小内点数
  sim3_iterations: 2          # Sim3 精化迭代次数（教学版可为 1）
```

`VOConfig` 对应新增字段（默认值如上，`fromYaml` 解析）。

---

## 5. 实施计划（里程碑 + 验证）

| 里程碑 | 内容 | 验证方式 |
|--------|------|----------|
| M0 依赖 | 安装 DBoW3（`bash scripts/install_deps.sh`）；获取 ORBvoc 词典（新增 `scripts/fetch_vocab.sh`，下载 ORB-SLAM2 的 `ORBvoc.txt.tar.gz` 并解压到 `config/`）；确认 CMake 输出 `DBoW3 found` | `cmake -S . -B build` 日志含 `HAS_DBOW3` |
| M1 Sim3 | `common.h` 加 `Sim3` + `Sim3::estimate`（Umeyama） | 单元测试：合成点集（s=1.37, R, t）恢复误差 < 1e-6；逆/矩阵互转测试 |
| M2 回环检测 | `loop_closure.cpp` 完整实现（词袋加载/入库/候选/验证/Sim3） | 单元测试：合成"先走远再回起点"场景，detectLoop 命中、verifyLoop 通过且尺度比 ≈ 1 |
| M3 校正与优化 | `poseGraphOptimization`；`handleLoopCorrection`（Sim3 传播 + 全局 BA 触发） | 单元测试：合成漂移轨迹 + 回环约束 → 优化后位姿误差下降 |
| M4 集成与评估 | `run_slam.cpp` + CMake；VO 钩子；KITTI 00 全程评估 | 验收标准表（§1）全部满足，写回 DEVELOPMENT_LOG |

---

## 6. 测试计划（test/test_vo.cpp 新增）

1. **Sim3 代数**：正变换、逆变换、`matrix()`/`fromMatrix` 互转、`toSE3` 丢尺度；
2. **Sim3::estimate**：随机 s/R/t 下 Umeyama 恢复精度；退化输入（点数 < 3 / 共线）返回 false；
3. **LoopClosure 合成回环**：构造两段轨迹（出发 → 绕行 → 回到起点附近），插入关键帧；验证 `detectLoop` 命中历史关键帧、`verifyLoop` 返回 true 且 `sim3.s ≈ 1.0`；负例（无回环）返回 nullptr/false；
4. **位姿图优化**：合成带漂移的关键帧链 + 末端回环边 → 优化后累计漂移下降（量化断言）；
5. **全局 BA 回归**：既有 `test_local_ba` 保持通过（改动不得破坏位姿语义）；
6. **长期稳定性回归**：既有 `test_long_run_stability` 保持通过（LoopClosure 数据增长须有界——数据库只增关键帧词袋，数量与关键帧数线性，无额外泄漏）。

---

## 7. 数据准备与评估方案

**词典**（前置条件，网络依赖）：
- 新增 `scripts/fetch_vocab.sh`：下载 ORB-SLAM2 的 `ORBvoc.txt.tar.gz`（约 30MB，DBoW2 格式与 DBoW3 兼容）解压到 `config/ORBvoc.txt`；若网络不可用，退回用 DBoW3 自带 `create_vocab` 工具在 KITTI 00 前 1000 帧上自训练（教学可用，回环召回略降）。
- 用户已有自行下载 KITTI 数据的先例，词典同理由用户触发下载。

**评估**：
```
bash scripts/prepare_kitti.sh        # 已有
./build/bin/run_vo  datasets/sequences/00 config/default.yaml vo_traj.txt          # 基线（回环关）
./build/bin/run_slam datasets/sequences/00 config/default.yaml slam_traj.txt        # 回环开
evo_ape tum vo_traj.txt   kitti_gt.txt -a    # 对比基线
evo_ape tum slam_traj.txt kitti_gt.txt -a    # 对比回环版
```
- 记录：回环检测次数 / 验证通过次数 / 全程 LOST 数 / FPS / 全量 ATE RMSE；
- 成功标准：§1 验收表（ATE 从 ~133.6m 显著下降）。

---

## 8. 风险与决策记录

| 风险/决策 | 说明 | 处置 |
|-----------|------|------|
| 词典获取依赖网络 | ORBvoc 约 30MB 外部下载 | `fetch_vocab.sh` + 自训练兜底 |
| apt g2o 的 Sim3 边可用性 | `g2o::EdgeSim3` 在部分版本缺失/语义差异 | 回环边用 SE3（决策，§4.4）；Sim3 只用于传播，不用于 g2o 边 |
| 单目回环的全局尺度跳变 | Sim3 传播改变全局尺度，轨迹文件必须同步 | `handleLoopCorrection` 统一更新 trajectory_（§4.3-5） |
| 回环检测频率与实时性 | DBoW3 查询在关键帧数量大时变慢 | 每 N=10 个关键帧才检测一次；数据库只存词袋向量 |
| 误回环 | 词袋误报会引入错误约束 | 三重过滤：分数 + 时间窗 + PnP 几何验证（内点比例 ≥ 0.7） |
| run_vo / run_slam 行为分叉 | 回环开/关影响轨迹输出 | run_vo 固定关闭、run_slam 固定开启，yaml 开关仍可覆盖 |
| 与既有 MiniAtlas 共存 | LOST 重建子地图后回环检测跨子地图 | LoopClosure 数据库全局唯一（跨子地图累积），Sim3 校正天然处理尺度比；不做子地图融合 |

---

## 9. 附：与既有约定的兼容性

- **位姿语义**：全程保持 `pose_cw = T_cw`（世界→相机）；g2o 侧沿用"喂 T_wc、回写取逆"的既有约定（§4.4）；
- **时间过滤阈值**（temporal_window=30）与关键帧冷却（min_keyframe_interval=10）互不干扰——回环跳过的是"刚走过的路"，关键帧冷却防的是"连续插帧"；
- **教学定位**：所有新代码坚持"≤200 行/模块 + 中文注释 + 可对照 TUTORIAL §3 数学基础"。
