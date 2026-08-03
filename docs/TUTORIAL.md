# VSLAM 从零入门教程

> 配套项目：教学级单目视觉里程计（Visual Odometry）
> 语言：C++23 ｜ 依赖：OpenCV / Eigen / Pangolin / yaml-cpp / g2o
> 阅读对象：学过 C++ 基础、线性代数、刚接触 SLAM 的初学者
> 建议阅读方式：**边读边打开源码对照**，按第 9 节的顺序逐文件阅读

---

## 目录

1. [这个项目在做什么](#1-这个项目在做什么)
2. [整体架构与数据流](#2-整体架构与数据流)
3. [数学基础（第一性原理）](#3-数学基础第一性原理)
4. [核心算法讲解](#4-核心算法讲解)
5. [代码模块逐文件导读](#5-代码模块逐文件导读)
6. [本项目用到的 C++23 语法](#6-本项目用到的-c23-语法)
7. [学习路径与动手实验](#7-学习路径与动手实验)
8. [常见坑与调试技巧](#8-常见坑与调试技巧)

---

## 1. 这个项目在做什么

**视觉里程计（Visual Odometry, VO）**：只用一台普通相机，看着眼前的场景一帧一帧移动，实时算出相机自己"走了多远、转了多少"。

```
相机连续拍照 ──► 提取特征点 ──► 匹配相邻帧 ──► 恢复运动(位姿) ──► 三角化建点 ──► 输出轨迹
```

**本项目刻意做成"教学级"**：只保留 SLAM 最核心的逻辑，砍掉工程化复杂度（词袋、回环、优化加速等），让每个文件都能独立读懂。

### 1.1 怎么运行

```bash
# 构建
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# 跑摄像头（实时）
./bin/run_vo 0

# 跑图片序列（KITTI 格式目录）
./bin/run_vo /path/to/image_dir config/default.yaml

# 跑 TUM / EuRoC 数据集
./bin/run_vo /path/to/tum_dataset --tum
./bin/run_vo /path/to/euroc_dataset --euroc

# 单元测试
cmake -DBUILD_TESTS=ON .. && make test_vo && ./test_vo
```

### 1.2 一帧图像里发生了什么（30 秒版）

1. `addFrame()` 收进一张图 → 转灰度 → CLAHE 增强 → 提取 ORB 特征
2. 如果是第一帧：存起来当"参考帧"
3. 如果是第二帧：与参考帧做特征匹配 → **对极几何**解出两帧相对运动 → 初始化成功
4. 之后每帧：拿地图里的 3D 点匹配当前 2D 特征 → **PnP** 求当前位姿
5. 走得够远 → 插入新关键帧 → 三角化更多地图点 → **局部 BA** 微调
6. 完全跟丢 → 进入 LOST → 遍历关键帧尝试**重定位**

---

## 2. 整体架构与数据流

### 2.1 模块关系

```
                    ┌──────────────┐
                    │  Dataset     │  读摄像头/数据集图片 + 时间戳
                    └──────┬───────┘
                           │ image, timestamp
                           ▼
┌──────────────────────────────────────────┐
│  VisualOdometry (src/vo.cpp)             │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Feature  │→ │ tryInit  │→ │ track  │ │
│  │ Matcher  │  │ /track   │  │ Frame  │ │
│  └──────────┘  └────┬─────┘  └───┬────┘ │
│                     │ pose_cw    │       │
│                     ▼            ▼       │
│              ┌────────────┐ ┌──────────┐│
│              │ Map        │ │ Optimizer││
│              │ 关键帧+地图点│ │ g2o Local││
│              └────────────┘ │ BA       ││
│                             └──────────┘│
└──────────────┬───────────────────────────┘
               │ pose / trajectory
               ▼
        ┌──────────────┐
        │  Viewer      │  左: 2D 轨迹图  右: 视频流+绿色特征点
        └──────────────┘
```

### 2.2 三类核心数据

| 数据结构 | 含义 | 关键字段 |
|---|---|---|
| `Frame` | 一帧图像 + 它的特征 + 它的位姿 | `keypoints`、`descriptors`、`map_points`、`pose_cw` |
| `MapPoint` | 三维世界中的一个点 | `pos_w`（世界坐标）、`observed_count` |
| `Map` | 全局容器 | `keyframes_`（关键帧）、`map_points_`（地图点） |

**最重要的概念：`map_points` 数组**

每个 Frame 里有一个与 `keypoints` 一一对应的 `map_points` 数组：
```
keypoints[i]  ──── 这个像素特征 ────►  map_points[i]（对应的 3D 点，没有则为 nullptr）
```
这个"像素 ↔ 3D点"的对应关系，正是 PnP 和三角化能工作的基础。

---

## 3. 数学基础（第一性原理）

### 3.1 坐标系与位姿语义（本项目的核心约定！）

SLAM 里有三个坐标系：
- **世界系 w**：固定的，原点在初始化第一帧的相机位置
- **相机系 c**：随相机移动，原点在相机光心
- **像素系**：图像上的 (u, v)

位姿 `T_cw` 读作"**从世界系到相机系的变换**"：

```
p_c = T_cw · p_w        （把世界点变成相机系下的点）
```

写成矩阵：

```
T_cw = [ R  t ]   其中 R: 3×3 旋转，t: 3×1 平移
       [ 0  1 ]
```

> ⚠️ **这是本项目踩过的最大的坑**。很多资料里 `T_cw` 的意思是"相机在世界系中的位姿"（即 T_wc），两种约定会写出互为逆矩阵的代码，跑起来轨迹"看起来对"，但三角化全是错的。
>
> 本项目**统一约定**：`pose_cw` = T_cw = 世界→相机。证据链：
> 1. `solvePnP` 的返回值就是 T_cw（世界→相机）
> 2. 三角化投影矩阵 `P = K·T_cw` 需要 T_cw
> 3. 深度检查 `(T_cw · p_w).z > 0` 需要 T_cw
>
> 记住判断口诀：**凡是把世界点变成相机点的变换，都是 T_cw**。

### 3.2 针孔相机投影模型

相机把 3D 点变成 2D 像素：

```
p_c = (x, y, z)                    相机系坐标
u = fx · x / z + cx                fx, fy: 焦距，cx, cy: 光心
v = fy · y / z + cy
```

写成矩阵：`z · p_uv = K · p_c`，其中内参矩阵

```
K = [ fx  0  cx ]
    [  0 fy  cy ]
    [  0  0   1 ]
```

**反投影**（像素 → 归一化相机坐标）：`p_c = K⁻¹ · (u, v, 1) = ((u-cx)/fx, (v-cy)/fy, 1)`

### 3.3 对极几何（两帧之间恢复运动）

已知两帧的匹配像素对，想恢复相对运动 `T`（旋转 R + 平移 t，尺度未知）。

**本质矩阵 E**：描述两帧匹配点之间的约束

```
p₂ᵀ · E · p₁ = 0          E = [t]× · R
```

- 用 `cv::findEssentialMat(p1, p2, K)` 求 E（内部是五点法 + RANSAC 剔外点）
- 用 `cv::recoverPose(E, p1, p2, K, R, t)` 从 E 分解出 R 和 t

**⚠️ recoverPose 的返回值语义（本项目实测验证过）：**

OpenCV 的 `recoverPose` 返回的 R, t 满足

```
p_c2 = R · p_c1 + t
```

即"把第一帧的点变换到第二帧"——**它就是 T_cw2**（当第一帧是原点时）。所以初始化代码里直接：

```cpp
curr_frame_->pose_cw = matToSE3(R, t);   // 不要再取逆！
```

> 很多教程会写 `T_12.inverse()`，那是把 R,t 当成了"1→2 的逆"——用合成点对一测就露馅：取逆方向会让所有三角化点跑到相机**后方**（深度为负）。项目 `test_vo.cpp` 的 `[Pose Semantics]` 测试专门守护这个语义。

### 3.4 PnP（已知 3D 点，求当前帧位姿）

跟踪阶段我们已经有了地图（3D 点），匹配后得到若干 **3D-2D 对应**：

```
3D 点 (世界坐标)  ────匹配────►  当前帧的 2D 像素
```

`cv::solvePnPRansac(pts3d, pts2d, K, ..., rvec, tvec)` 直接解出当前帧的 **T_cw**（世界→相机）。这也是为什么我们统一用 T_cw 语义——PnP 输出拿来就能用。

### 3.5 三角化（两个视角重建 3D 点）

有了两个视角的位姿 T1、T2 和同一 3D 点的两个像素 u1、u2，可以反推出 3D 坐标：

```
投影矩阵 P = K · [R|t]           （世界点 → 像素）
cv::triangulatePoints(P1, P2, u1, u2) → 4D 齐次坐标 → 除以 w → 3D
```

`mappoint.cpp` 的 `MapPoint::create()` 就是干这个的。三角化后一定要做**深度检查**：`(T_cw · p_w).z > 0`，深度为负的点是歧义解，直接丢弃。

### 3.6 Bundle Adjustment（后端优化）

**问题**：前面每一步都有噪声，轨迹和地图点会越走越歪。BA 就是"把所有关键帧位姿和地图点一起调整，让所有重投影误差之和最小"：

```
min  Σ  ‖ K·T_cw·P_w 投影出来的像素  −  实际观测像素 ‖²
    (T,P)

用 g2o 构建图优化：
  ● 顶点(Vertex)     = 关键帧位姿(6DOF) + 地图点(3D)
  ● 边(Edge)         = 一个观测（某关键帧看到某地图点）
  ● 误差(Error)      = 重投影残差
  ● 鲁棒核(Huber)    = 抑制外点对优化的破坏
```

> g2o 里 `VertexSE3Expmap` 的位姿语义是 **T_wc**（相机在世界系）！与我们的 T_cw 相反，所以喂给 g2o 之前要 `pose_cw.inverse()`，优化完再转回来。`optimizer.cpp` 里两处注释都标明了。

---

## 4. 核心算法讲解

### 4.1 状态机（src/vo.cpp 的骨架）

```
        ┌──────────────┐
        ▼              │
   INITIALIZING ──成功──► TRACKING
        │                │  匹配退化
        │                ▼
        │           RECOVERING
        │                │  连续失败
        │                ▼
        │              LOST
        │          ╱       ╲
        │     重定位成功   长期失败
        │        │           │
        └────────┘      新建锚定子地图
```

| 状态 | 触发条件 | 做什么 |
|---|---|---|
| `INITIALIZING` | 开始 | 存第一帧；第二帧匹配≥阈值且视差≥0.1 → 对极几何求 T_cw2，三角化初始地图，插入两个关键帧 |
| `TRACKING` | 初始化成功 | PnP 求位姿；关键帧策略满足 → 插入关键帧 + 三角化 + Local BA |
| `RECOVERING` | 单帧跟踪退化 | 最多容忍 `max_tracking_failures` 帧，保留上一有效位姿并尝试重定位 |
| `LOST` | 连续失败超过宽限期 | 在当前及历史子地图中匹配关键帧；内点≥20 恢复，长期失败后创建锚定子地图 |

### 4.2 跟踪的"三级防线"（trackFrame）

```
1. PnP（首选）    有 ≥6 个 3D-2D 对应 → solvePnPRansac → 内点≥10 成功
2. 对极回退        匹配≥20 但 PnP 失败 → 2D-2D 相对运动 → 组合到 ref 位姿
3. 判 Recovering  匹配不足或跳变 → 保持上一有效位姿，进入 RECOVERING
```

> 为什么有第二级？因为地图点有时会暂时不够（比如刚初始化、视野快速变化），这时退化成"纯 2D 的里程计"（对极几何）也能算出相对运动。

### 4.3 精简 MiniAtlas

`Atlas` 只负责保存多个局部 `Submap`，不负责子地图融合和位姿图优化。每个子地图记录一个
`origin_Twc` 锚点，但帧位姿和地图点仍统一使用全局世界坐标：

```text
短时跟踪失败 → RECOVERING → LOST
                         ├─ 重定位成功：激活对应子地图
                         └─ 长期失败：冻结旧地图，在最后有效位姿附近创建新子地图
```

创建新子地图时不清空全局轨迹，也不把第一帧设为全局原点。双目直接用当前帧深度建点；
单目使用锚定的两帧初始化。这样 KITTI 中的重新建图不会在轨迹文件中产生单位位姿跳变，
同时保留历史子地图供后续重定位使用。`Status` 中的 `submap_id` 和 `lost_frames` 可用于 Viewer
状态栏和运行诊断。

### 4.4 关键帧策略（needNewKeyFrame）

不是每帧都插入关键帧（否则地图爆炸、BA 太慢）。三个条件**满足任一**即插入：

```
1. 相机在世界系位移 > keyframe_translation (0.5m)   ← 注意用 T_wc 的平移！
2. 相对旋转角   > keyframe_rotation    (0.2 rad)
3. 跟踪内点数   < keyframe_min_inliers (15)          ← 匹配衰减，强制补帧
```

> **坑**：条件 1 若直接比较两个 `T_cw` 的平移向量，物理意义是错的（每帧的 t 是"自己的坐标系里"的位移方向）。必须先取逆成 T_wc 再比较。

### 4.5 滑动窗口 Local BA（insertKeyFrame → selectLocalWindow）

```
新关键帧 ──► 统计它与每个历史关键帧的共视地图点数 ──► 取 Top N 帧
         ──► 加上当前帧组成窗口 ──► 窗口按 id 排序（最早帧锚定）
         ──► 交给 g2o 做一次局部 BA
```

- **共视**：两个关键帧看到同一个地图点。
- **最早帧固定**：`setFixed(true)` 锚定坐标系，防止整体漂移出原点。

### 4.6 LK 光流模式（feature_method=1）

ORB 每帧提取+匹配很慢。LK 光流思路：**上一帧的特征点，这一帧我直接在周围小窗口里找它跑到哪去了**，不重新提取。

```
普通帧：LK 光流跟踪（关键点位置继承，map_points 索引对齐）
        → 用索引对齐的 3D-2D 直接 PnP
关键帧：重新做一次干净 ORB 提取
        → 与上一个关键帧 ORB 匹配 → 三角化
```

> **坑**：LK 出来的关键点没有方向（`angle=0`），直接给它算 ORB 描述子会和正常 ORB 特征对不上（实测匹配率 17/1000）。所以关键帧要"换血"成干净 ORB 特征。

### 4.7 单目尺度之谜

对极几何和单目三角化恢复的运动**没有真实尺度**（一米和一厘米在图像上可能一样）。所以：

- 初始化后轨迹是"归一化尺度"
- 与真值（如 KITTI ground truth）比较前，需要先做 Sim3 尺度对齐（EVO 工具自动处理）

---

## 5. 代码模块逐文件导读

> 建议按这个顺序读（依赖顺序 = 阅读顺序）：

### 5.1 `include/vslam/common.h` —— 一切的基石

```cpp
struct SE3 {
    Eigen::Quaterniond q;   // 旋转（四元数，4 个数，无万向锁）
    Vec3                t;  // 平移（3 个数）
};
```

只放 5 个方法：`fromMatrix`、`matrix`、`inverse`、组合乘法、点变换。**看完这个文件你就知道位姿怎么算**。所有方法标了 `[[nodiscard]]`（C++17 起，C++23 常用）——返回值不能被忽略，防止写错。

### 5.2 `camera.h/cpp` —— 相机模型

针孔投影的两个函数，是后面所有几何的地基：

```cpp
Vec2 world2pixel(const Vec3& p_w, const SE3& T_cw);   // 3D → 像素
Vec3 pixel2camera(const Vec2& pixel, double depth);   // 像素 → 归一化坐标
```

### 5.3 `frame.h / mappoint.h / map.h` —— 数据三件套

- `Frame`：看图 5.3-1，重点理解 `map_points[i]` 与 `keypoints[i]` 的配对
- `MapPoint`：`pos_w` 世界坐标 + `create()` 三角化工厂
- `Map`：两个 `std::map` 容器 + `nextMapPointId()` 全局唯一 id

### 5.4 `feature.h/cpp` —— 眼睛

```cpp
extract(frame)      // ORB 提取：关键点 + 描述子
match(f1, f2, ...)  // 三步：knn 匹配 → 比率测试(0.7) → RANSAC 基础矩阵剔外点
trackLK(prev, curr) // 光流跟踪（LK 模式用）
```

**比率测试**：最近邻距离 < 0.7 × 次近邻距离才接受——消除模糊匹配的经典技巧。

### 5.5 `vo.h/cpp` —— 大脑（最值得精读）

`addFrame()` 是唯一入口，依次做 4 件事（对照 4.1 状态机）：

```
1. 建 Frame + CLAHE 增强 + 提取特征
2. 按状态分发：tryInitialize / trackFrame(LK) / tryRelocalize
3. 关键帧判决 + 插入 + Local BA
4. 记录轨迹，返回 pose_cw
```

`VOConfig` 结构体把**所有可调参数**集中起来（对应 config/default.yaml），构造函数传入——改参数不用改代码。

### 5.6 `optimizer.cpp` —— 后端微调

六步走（源码注释里有编号）：建求解器 → 加相机参数 → 加位姿顶点（取逆！）→ 加地图点顶点 → 加重投影边（Huber 核）→ 优化并回写（再取逆）。

### 5.7 `viewer.cpp` —— 可视化

Pangolin 双窗口：**左 = 世界系 2D 轨迹图**（x-z 俯视，红色箭头表示相机朝向），
**右 = 彩色视频流 + 绿色特征点**。双目左右图上下排列，避免超宽画面被压缩；
独立渲染线程 + `std::mutex` 保护共享数据——这是多线程编程的最小示范。

### 5.8 `dataset.cpp` —— 数据输入

一个抽象，四种来源：KITTI（图片目录）、TUM（`rgb.txt` 时间戳）、EuRoC（`cam0/data.csv`）、摄像头（`VideoCapture`）。时间戳数组 `timestamps_` 是 TUM 轨迹评估的关键。

### 5.9 `run_vo.cpp` —— 组装一切

主循环就一句话：`读帧 → vo.addFrame() → viewer.updateFrame()`。结束后把轨迹写成 **TUM 格式**（`time tx ty tz qx qy qz qw`，T_wc 语义）——这是 EVO 评估工具的标准输入。

---

## 6. 本项目用到的 C++23 语法

> CMakeLists.txt 已设 `CMAKE_CXX_STANDARD 23`。下面是项目里"真的在用"的 C++23/20 特性，每个都对应一个具体场景：

### 6.1 `std::ranges::sort` + 投影（vo.cpp / dataset.cpp）

```cpp
// 旧写法（C++98 风格）：
std::sort(cands.begin(), cands.end(),
          [](const Candidate& a, const Candidate& b) { return a.cov > b.cov; });

// C++23 ranges 写法：直接对容器排序，第三参数是"投影"（取要比较的字段）：
std::ranges::sort(cands, std::greater<>{}, &Candidate::cov);
```

- 不用写 `begin()/end()`
- 投影 `&Candidate::cov` 让"按哪个字段排"一目了然
- 教学点：**ranges 的核心思想是"描述意图，而不是描述循环"**

### 6.2 `std::views::enumerate`（vo.cpp trackFrame）

```cpp
// 旧写法：
for (size_t k = 0; k < matches.size(); k++) { const auto& m = matches[k]; ... }

// C++23 写法：索引和元素一起给出来
for (auto [k, m] : matches | std::views::enumerate) { ... }
```

结构化绑定 + 视图管道，消除了经典的"索引变量 + 越界风险"样板代码。

### 6.3 `std::views::transform` + `std::ranges::max`（viewer.cpp）

```cpp
auto xs = trajectory_ | std::views::transform([](const Vec3& p) { return std::abs(p.x()); });
range = std::max({std::ranges::max(xs), std::ranges::max(zs), 5.0}) * 1.5;
```

"把每个元素映射一下，再取最大值"——用管道表达数据处理流水线。

### 6.4 `std::format`（run_vo.cpp）

```cpp
std::string status = std::format("{} | Matches:{} Parallax:{:.2f}", state_str, st.matches, st.parallax);
```

类型安全的格式化（替代 `snprintf`），`{}` 占位符，`{:.2f}` 控制小数位，写错类型编译期就报错。

### 6.5 `[[nodiscard]]`（common.h）

```cpp
[[nodiscard]] SE3 inverse() const;   // 返回值不能丢
```

如果写了 `pose.inverse();`（忘记用返回值），编译器会警告——防呆设计。

### 6.6 其他顺带用到的现代语法

- **结构化绑定** `auto [k, m] = ...`（C++17）
- **范围 for** `for (auto& mp : kf->map_points)`（C++11）
- **智能指针** `std::shared_ptr`、`std::make_shared`（C++11）
- **`auto` 推导**（C++11/14）

---

## 7. 学习路径与动手实验

### 7.1 建议阅读顺序（预计 2-3 小时）

```
第 1 步  读完本文第 3 节（数学）——不用全懂，留个印象
第 2 步  common.h → camera → frame/mappoint/map（30 分钟）
第 3 步  feature.cpp（40 分钟）——匹配的三种技巧
第 4 步  vo.cpp 的 addFrame → tryInitialize → trackFrame（60 分钟）⭐核心
第 5 步  optimizer.cpp（30 分钟）——BA 的六步模板
第 6 步  viewer/dataset/run_vo（30 分钟）——把它们串起来
第 7 步  跑起来：./test_vo 和 ./bin/run_vo 0
```

### 7.2 动手实验（由易到难）

**实验 1：改一个参数看效果**
`config/default.yaml` 里把 `Feature.num_features` 改成 500 / 2000，观察：
- 特征点密度、匹配数量、FPS 的变化
- 这就是"参数调优"的入门

**实验 2：改一个颜色**
`viewer.cpp` 里绿色特征点 `cv::Scalar(0, 255, 0)` → 改成红色 `cv::Scalar(0, 0, 255)`，观察 RGB 顺序（OpenCV 是 BGR！）

**实验 3：切换跟踪模式**
`config/default.yaml` 里 `VO.method` 改成 1，对比 ORB 匹配 vs LK 光流的速度和稳定性。

**实验 4：打破一个假设（推荐做）**
把 `needNewKeyFrame()` 的 `keyframe_translation` 改成 10.0（永远不插关键帧），观察：
- 地图点不再增长 → PnP 内点越来越少 → 最终 LOST
- 这让你直观理解"关键帧 = VO 的生命线"

**实验 5：加功能——打印每帧跟踪耗时**
在 `run_vo.cpp` 主循环里用 `std::chrono` 统计 `addFrame` 耗时，打印最慢的 3 帧。
（C++23 的 `std::chrono::system_clock` 精度足够）

**实验 6：真正跑通 KITTI 并评估**
下载 KITTI → `scripts/prepare_kitti.sh` → `run_vo` 输出轨迹 → `evo_ape tum` 对比真值。

### 7.3 评估工具 EVO 速查

```bash
pip install evo

# 估计轨迹 vs 真值（TUM 格式）
evo_ape tum trajectory.txt groundtruth.txt -a -v
evo_rpe tum trajectory.txt groundtruth.txt -a -v

# 画图
evo_traj tum trajectory.txt groundtruth.txt -p
```

`-a` = 自动 Sim3 对齐（单目尺度未知，必须加）。

---

## 8. 常见坑与调试技巧

### 8.1 本项目实战踩过的坑（真实记录）

| 坑 | 症状 | 原因 | 教训 |
|---|---|---|---|
| 位姿取反 | 轨迹"看起来在动"，但三角化 0% 深度正确 | recoverPose 语义误解 | 先写合成点对测试再写业务代码 |
| 四元数点积求角差 | 转半圈后关键帧误判 | `q` 与 `-q` 等价，`dot` 会变负 | 取绝对值或算相对旋转四元数 |
| LK 描述子不匹配 | LK 模式三角化失败 | LK 关键点 `angle=0`，ORB 描述子对不上 | 关键帧重建干净 ORB 特征 |
| `-march=native` 对齐 | 外部链接程序莫名段错误 | 库用 AVX 编译，调用方栈对齐不一致 | 链接方保持相同编译选项 |
| 关键帧平移比较 | 旋转时狂插关键帧 | 直接用 T_cw 的 t 比较 | 用 T_wc 的平移差 |
| Viewer 轨迹画圆 | 原地旋转却显示成大圆弧 | 把 `T_cw.t` 当成相机世界位置 | 使用 `C_w = -R_cw^T t_cw`（`camera_position()`） |
| RGB 视频逐行倾斜 | 单目、双目画面呈斜纹错位 | OpenGL 纹理上传仍按默认 4 字节行对齐 | RGB 上传前设置 `GL_UNPACK_ALIGNMENT=1` |
| **无界增长** | **跑几分钟后越来越卡，最后卡死** | 关键帧风暴 / 地图点从不清理 / 渲染全量轨迹 | 关键帧加冷却、定期 `cullMapPoints`、显示降采样（本项目已修复并有稳定性测试守护） |

> 诊断"卡死"口诀：先看 `keyframes` / `map_points` 数字是否还在涨——**任何不随运行收敛的数据结构都是性能定时炸弹**。

### 8.2 调试三板斧

1. **看状态**：日志里有 `Init OK! parallax=... inliers=...`、`Tracking lost! matches=...`——先看状态机走到哪了
2. **看数字**：`inliers`、`map_points`、`keyframes` 的趋势。内点骤降 → 地图没跟上；地图点不涨 → 关键帧没插
3. **最小复现**：`test_vo.cpp` 里用**合成点对**（不用图像）验证纯几何逻辑，比如 `[Pose Semantics]` 测试——图像/ORB 的不确定性先排除掉

### 8.3 单目 VO 的天然局限（坦然接受）

- 无尺度（需要真值对齐评估）
- 纯旋转初始化会失败（没有视差，`parallax < 0.1`）
- 快速运动/遮挡会丢（靠重定位救）
- 长时间运行会漂移（Phase 2 的回环检测就是干这个的）

---

## 9. 下一步：从 VO 到 SLAM（Phase 2 预告）

当前项目完成的是 **VO（无回环）**。真正的 SLAM 还要加三块，本项目已在 `loop_closure.h` 留好接口：

1. **回环检测**：用词袋（DBoW3）判断"我是不是回到了老地方"→ 检测到回环说明漂移可纠正
2. **位姿图优化**：只优化关键帧位姿（加回环边约束），速度快于全局 BA
3. **全局 BA**：回环后把全部关键帧+地图点重新优化一遍

每块的数学原理在本文第 3 节都能找到对应基础——对极几何（回环验证）、BA（全局优化）。学完 VO，你已经掌握了 SLAM 的 70%。
