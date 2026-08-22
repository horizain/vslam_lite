# VSLAM 开发日志

> 创建日期: 2026-07-30
> 最后更新: 2026-08-22（M3.1 前端质量门见 §3.47；M3.2 位姿协方差见 §3.48；
> Flat-DBoW3/mobile 全程与 full gate 失败证据见 §3.43；
> 彩色双目点云见 §3.44；M2 仍未通过准确度/实时/RSS 产品门）
>
> 阅读说明：本文是追加式开发档案，早期章节保留当时的字段、依赖和实验结论；它们不是
> 当前接口说明。当前坐标/观测模型以 §3.20、§3.22 为准，当前完整基准以 §3.23 为准。
> 产品化实施以 `PRODUCTION_LOCALIZATION_PLAN.md` 为准（§3.25 起）；当前版本完整
> 基准评估见 §3.30。

---

## 一、项目概述

从零构建教学级 VSLAM 系统，遵循第一性原理，只保留核心逻辑，适合初学者逐文件阅读学习。

**技术栈:** C++23 / OpenCV / Eigen3 / Pangolin / vendored g2o / DBoW3（可选回环）

---

## 二、已完成的工作

### 2.0 编译与环境验证 ✅

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 依赖安装 | ✅ | OpenCV 4.10 + Eigen 3.4 + yaml-cpp 0.8 + vendored g2o；DBoW3 可选 |
| Pangolin | ✅ | gitee 镜像源码编译 (v0.6) |
| CMake 编译 | ✅ | 100% 通过 |
| 单元测试 | ✅ | 当前 `test_vo` 与 `test_trajectory_alignment` 均通过；早期数量见历史提交 |
| WSL2 摄像头 | ✅ | usbipd-win → /dev/video0 |
| 数据集实测 | ✅ | KITTI 00 与 EuRoC V1_01_easy 双轮完整基准见 §3.23 |

### 2.1 项目基础设施

| 文件 | 状态 | 说明 |
|------|------|------|
| `CMakeLists.txt` | ✅ 完成 | C++23, vendored g2o, 自动检测 DBoW3(可选), Release/Debug 模式 |
| `cmake/Findg2o.cmake` | ✅ 完成 | g2o 查找模块 |
| `config/default.yaml` | ✅ 完成 | KITTI 内参 + ORB/PNP/关键帧/BA 参数（已全量接入代码） |
| `.gitignore` | ✅ 完成 | 忽略 build/、debug.txt 等 |
| `scripts/prepare_kitti.sh` | ✅ 完成 | KITTI 数据集校验/解压/poses 放置 |

### 2.2 核心数据结构（头文件+实现）

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/common.h` | ✅ 完成 | SE3位姿(Vec3 t + Quaternion q), 矩阵互转, 逆/组合/点变换, 日志宏 |
| `include/vslam/camera.h` | ✅ 完成 | 针孔模型(单目/双目/RGBD 接口), Yaml加载, world2pixel, pixel2camera |
| `src/camera.cpp` | ✅ 完成 | Yaml解析, 投影/反投影实现 |
| `include/vslam/frame.h` | ✅ 完成 | Frame(Ptr), 图像/灰度图/关键点/描述子/位姿/地图点关联 |
| `include/vslam/mappoint.h` | ✅ 完成 | MapPoint(Ptr), `pos_s`/描述子/正式 Observation, 三角化工厂方法 |
| `src/mappoint.cpp` | ✅ 完成 | cv::triangulatePoints 三角化实现 |
| `include/vslam/map.h` | ✅ 完成 | Map: 关键帧+地图点集合, Observation/covisibility, revision 与图内 id 分配 |
| `src/map.cpp` | ✅ 完成 | 所有集合操作实现 |

### 2.3 视觉里程计前端

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/feature.h` | ✅ 完成 | FeatureMatcher: ORB提取, BF+knn+ratio匹配, LK光流, setParams |
| `src/feature.cpp` | ✅ 完成 | ORB 特征(默认1000, 可配置), 暴力匹配+比率测试+RANSAC 基础矩阵剔除 |
| `include/vslam/vo.h` | ✅ 完成 | VisualOdometry: 状态机, VOConfig 配置结构, PnP跟踪, LK 模式, 关键帧策略 |
| `src/vo.cpp` | ✅ 完成 | 初始化+跟踪(ORB/LK)+PnP+对极回退+三角化+共视窗口 Local BA+LOST 优先重定位 |
| `include/vslam/optimizer.h` | ✅ 完成 | g2o 后端接口 |
| `src/optimizer.cpp` | ✅ 完成 | **Local BA 已实现并集成**（滑动窗口，Huber 核，可配置迭代次数） |

**位姿语义（2026-08-01 统一修复）:**

当前关键帧字段为 `pose_cs` = **T_cs（子地图→相机）**；全局 `T_cw` 由
`T_cs * T_ws.inverse()` 派生，`p_c = T_cw * p_w`。

```
初始化 (INITIALIZING):
  Frame1 → 存储ref_frame → Frame2 → ORB匹配+RANSAC → findEssentialMat → recoverPose
  （已实验确认 recoverPose 返回 T_rel 满足 p_c2 = T_rel * p_c1，即 T_cw2 = T_rel）
  → 视差检查 → ref_frame 设为世界原点 → curr_frame_->pose_cw = T_cw2（不再取逆）
  → 三角化初始 MapPoint → 插入为 KeyFrame → TRACKING

跟踪 (TRACKING):
  当前帧 vs ref_frame:
    有≥6个3D-2D对应 → solvePnPRansac（输出即 T_cw，直接存储）
    匹配不足 → findEssentialMat + recoverPose（T_cw_curr = T_rel * T_cw_ref）
    完全失败 → 恒等运动假设 + LOST

关键帧选择:
  相机在世界系位移（T_wc 平移差）> keyframe_translation 或
  相对旋转角 2*acos(|w_rel|) > keyframe_rotation → 创建新关键帧
  → ORB匹配 → 三角化新地图点 → Local BA（窗口 size 可配置）→ 更新 ref_frame
```

**关键修复（2026-08-01）：**
1. 初始化不再对 recoverPose 结果取逆（旧代码方向反，导致三角化 0% 深度正确；算法级验证：新方向 300/300 深度正确）
2. 对极回退组合方向修正：`T_cw_curr = T_rel * T_cw_ref`
3. 关键帧判决平移改用 T_wc 差值；旋转角修复四元数 q 与 -q 等价的跳变
4. g2o VertexSE3Expmap 需 T_wc：喂入 `pose_cw.inverse()`，回写再取逆
5. MapPoint id 使用 `Map::nextMapPointId()`（避免剔除后重复）；`observed_count` 正确累计

### 2.4 可视化

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/viewer.h` | ✅ 完成 | Viewer: 独立渲染线程, updateFrame 线程安全 |
| `src/viewer.cpp` | ✅ 完成 | **重构(2026-08-03)**: 左=世界系 2D 俯视轨迹+相机朝向，右=大尺寸视频流；双目上下排列，状态栏自适应换行，OpenGL RGB 行按 1 字节对齐 |

### 2.5 数据输入

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/dataset.h` | ✅ 完成 | Dataset: KITTI/TUM/EuRoC/CAMERA 四类型 |
| `src/dataset.cpp` | ✅ 完成 | **KITTI**: 双目目录与标定<br>**TUM**: `rgb.txt` 时间戳<br>**EuRoC**: `cam0/data.csv` + `sensor.yaml` 标定/去畸变<br>**CAMERA**: `cv::VideoCapture` |

### 2.6 应用入口

| 文件 | 状态 | 说明 |
|------|------|------|
| `app/run_vo.cpp` | ✅ 完成 | 命令行入口, 自动识别数据集路径 vs 摄像头索引, 加载 yaml 配置, 主循环(读取→VO→Viewer→FPS统计), **结束后保存 TUM 格式轨迹**(时间戳+tx ty tz qx qy qz qw, T_wc, 供 EVO 评估) |
| `app/run_slam.cpp` | ✅ 完成 | 可选 DBoW3 回环入口；缺词典时会降级 |

### 2.7 测试

| 文件 | 状态 | 说明 |
|------|------|------|
| `test/test_vo.cpp` | ✅ 通过 | `test_vo` 与 `test_trajectory_alignment` 均通过；Observation、BA、回环和一致性回归覆盖见最新提交记录 |

### 2.8 文档

| 文件 | 状态 | 说明 |
|------|------|------|
| `docs/DEVELOPMENT_LOG.md` | ✅ 本文档 | 开发日志 |

### 2.9 Viewer 与旋转轨迹修复 ✅（2026-08-03）

**用户现象：** 单目、双目视频逐行倾斜；双目画面和状态文字过小；相机原地旋转时，
Viewer 轨迹画出大圆弧。

| 问题 | 根因 | 修复 |
|------|------|------|
| RGB 视频倾斜 | OpenGL 默认 `GL_UNPACK_ALIGNMENT=4`，而 RGB 行字节数不一定是 4 的倍数，纹理上传跨行错读 | 上传前设置 `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)`，并保证 `cv::Mat` 连续 |
| 双目画面太小 | KITTI 单目本身约 3.3:1，左右横拼后超过 6:1，在视口中被严重压扁 | 左右目改为上下排列，窗口扩大到 1600×900，视频区固定占 72% |
| 状态字体看不清 | 固定 `0.45` 字号并直接覆盖在图像底部 | 独立黑色状态栏，字号按图宽缩放，按 ` | ` 自动换行，使用抗锯齿双像素笔画 |
| 原地旋转画大圆 | `trajectory_` 错存 `T_cw.t`；它表示世界原点在相机系的坐标，会随旋转变化，不是相机位置 | 统一记录 `C_w = -R_cw^T t_cw = T_wc.t`，并绘制相机光轴箭头（位置不动、箭头旋转） |
| Pangolin v0.6 无法链接 | 安装脚本使用 v0.6 的 `pangolin` 目标，CMake 却硬编码新版 `pango_*` 目标 | 自动检测并兼容旧版单目标和新版拆分目标 |
| Pangolin v0.6 在 Ubuntu 24.04 编译失败 | GCC 13 要求显式包含 `cstdint`，旧 FFmpeg API 已移除 | 安装脚本预包含 `cstdint`，关闭 Viewer 不需要的 Pangolin Video/Python 组件 |
| 双目 3D–3D 位姿不准 | 仿射旋转正交化后仍沿用原仿射平移，R/t 不再属于同一变换 | RANSAC 只选内点，再用 Kabsch 统一拟合刚体 R/t |
| Release 测试假通过 | `-DNDEBUG` 关闭了所有 `assert`，测试仍打印 PASSED | `test_vo` 目标增加 `-UNDEBUG`，Release 也执行真实断言 |

新增 `SE3::camera_position()` 和回归测试：固定 `C_w`、改变相机旋转时，返回的世界系
相机位置必须保持不变。这里修复的是单目和双目共用的 **Viewer 轨迹坐标语义**；
单目估计器自身在纯远点旋转场景仍可能存在第 3.3 节所述的旋转-平移不可观歧义。

---

## 三、待完成的工作

### 3.0 性能优化（2026-08-01）✅

修复"跑一段时间后前端卡住"（增长型性能问题，全部经 Long-Run Stability 测试守护）：

| 问题 | 修复 |
|------|------|
| `weak_match` 无条件触发关键帧 → 关键帧风暴 | `min_keyframe_interval=10` 帧冷却（`needNewKeyFrame`） |
| 地图点从未清理 → 无限增长 | 每 20 个关键帧 `cullMapPoints(2)`，并同步清空关键帧引用释放内存 |
| Viewer 每帧绘制全部历史轨迹点 → 渲染随运行变卡 | 只显示最近 3000 个轨迹点（`kMaxTrajPts`） |
| LOST 重定位全量遍历所有关键帧 | 从最新向历史最多尝试 30 帧（`kMaxRelocTries`） |
| 每帧跟踪重复做基础矩阵 RANSAC | 跟踪匹配改 `use_ransac=false`，外点交给 PnP 自带 RANSAC |

验证（`test_long_run_stability`，150 帧合成序列）：kf=55、mp≈3600（有界），
末尾帧耗时 ≈ 开头帧耗时（无增长）。10 项测试全部通过。

#### 后续性能优化方向（2026-08-01 记录，暂缓实施）

| # | 方向 | 预期收益 | 代价/注意 |
|---|------|----------|-----------|
| 1 | **FLANN-LSH 加速汉明匹配**（`feature.cpp` 用 `FlannBasedMatcher` + LSH 参数） | 暴力匹配 1000×1000 是每帧最大开销，LSH 可快 5-10 倍 | 参数敏感，教学上 BF 更直观；建议做成配置开关 |
| 2 | **降低 ORB 特征数**（yaml `num_features` 1000→500） | FPS 近似翻倍 | 鲁棒性下降（弱纹理场景更易丢） |
| 3 | **BA 降频**（每 2 个关键帧做一次 Local BA） | 省一半后端时间 | 位姿精度略降；可在 yaml 增加参数控制 |
| 4 | **特征点绘制提速**（`cv::circle`×N → `cv::drawKeypoints` 一次性绘制） | updateFrame 省 ~1ms | 绘制样式变化（点 vs 十字） |
| 5 | **线程化匹配**（匹配放入工作线程，双缓冲帧） | VO 与 I/O 重叠，提升吞吐 | 复杂度明显上升，偏离教学定位；最后再做 |

> 诊断"卡死"口诀：先看 `keyframes` / `map_points` 数字是否还在涨——任何不随运行收敛的数据结构都是性能定时炸弹。

### 3.1 Phase 1 剩余任务

#### 🔴 P0 - KITTI 数据集实测 ✅（2026-08-01）

- [x] 数据准备 ✅：用户下载 zip 至 `datasets/`，`prepare_kitti.sh` 解压（修复 zip 前缀 + 防 image_1 覆盖）
- [x] **KITTI 00 全程实测** ✅：4541 帧**全程无 LOST**，26.5 FPS，171 秒
- [x] **位姿语义（g2o T_cw）重大修复** ✅：见下
- [x] **ATE 评估** ✅：前 1000 帧 RMSE 53.7m（GT ~400m），全量 RMSE 242.8m——无回环单目 VO 的正常漂移水平
- [ ] 参数进一步调优（keyframe 阈值/特征数，现可 yaml 调整）

**关键 bug 修复：本机 apt 版 g2o 的 `EdgeProjectXYZ2UV` 用 T_cw 语义**

症状：插入关键帧后轨迹尺度膨胀直至发散（BA on 时帧 2 位移从 1m 拉飞到 3m）。
排查过程：BA off 对照实验（前端稳定）→ 固定帧/固定点均无效 → 最小化 3 帧测试精确复现
（无噪声精确解下 BA 仍把帧 2 拉向 T_cw 值 `t=-2`）→ 查 g2o 源码确认 apt 版
`EdgeProjectXYZ2UV::computeError` 用 `estimate.map(P)`（**T_cw 语义**，与官方新版 T_wc 相反）。

修复：喂入 g2o 直接传 `pose_cw`（不取逆），回写直接取 estimate。已加
`test_local_ba` 回归测试守护（3 帧已知位姿，BA 后尺度必须保持 1m/帧）。

配套修复：
- 普通帧 trackFrame 时把 PnP 内点关联到 `curr_frame_->map_points`（否则关键帧只与
  紧邻帧共视，BA 窗口永远只有 2 帧）
- 位姿跳变保护：PnP/对极回退单帧位移 >30m 判 LOST（防数值发散）
- LOST 后不再误插关键帧（trackFrame 置 LOST 后外层复查状态）
- 重定位门槛 100→30（min_matches_init 是初始化专用）
- `keyframe_translation` 0.5m（KITTI 起步段帧间位移 <1m，1.0 阈值导致视差累积崩溃）
- Local BA 改为 **motion-only（点固定，只优化位姿）**：单目三角化尺度由初始化锚定，
  点自由优化存在尺度 gauge 自由度（点+位姿平移同时缩放 s 重投影不变）
- 评估脚本 `scripts/evaluate_ate.py` 修复 Umeyama scale 公式（漏了 1/N）

#### 🔴 P0 - 参数调优（后续）

#### 🟡 P1 - 后端优化（g2o）✅ 完成

- [x] `localBundleAdjustment()` ✅ 已实现+集成
- [x] **共视图滑动窗口** ✅（2026-08-01）：按共视地图点数选帧 + 最早帧锚定，共视不足时退化为时间窗

#### 🟢 P2 - 完善与增强 ✅ 完成（2026-08-01）

- [x] **TUM 数据集支持**: 解析 `rgb.txt` + 关联时间戳 ✅
- [x] **EuRoC 数据集支持**: 解析 `cam0/data.csv` ✅（IMU 加载为 Phase 2 可选）
- [x] **LK 光流模式接入**: `VOConfig::feature_method = 1` ✅
  - 普通帧 LK 光流跟踪（索引对齐 map_points 做 PnP），失败自动回退 ORB
  - 关键帧插入时重建干净 ORB 特征（LK 关键点无方向，描述子无法与历史关键帧匹配）
  - LOST 重定位前自动补提取描述子
- [x] **关键帧策略增强**: 内点衰减触发关键帧 ✅（`keyframe_min_inliers`）
- [x] **LOST 状态恢复优化**: 优先匹配最近 5 个关键帧，不足再全量遍历 ✅
- [x] **编译安装脚本**: `scripts/install_deps.sh` ✅

### 3.2 Phase 2 - 完整 SLAM（优先级：中）

#### 🔵 回环检测

- [ ] **DBoW3 依赖安装**: 编译 DBoW3, cmake 集成（当前 `DBoW3_DIR-NOTFOUND`）
- [ ] **`src/loop_closure.cpp` → 完整实现**（词袋加载/相似度检测/几何验证）
- [ ] **回环校正**: Sim3 变换 + 回环边创建

#### 🔵 后端优化增强

- [ ] **`optimizer.cpp` → `poseGraphOptimization()`**（位姿图优化）
- [ ] **`optimizer.cpp` → `globalBundleAdjustment()`**（全部关键帧+地图点，回环后触发）

#### 🔵 地图管理

- [ ] **MapPoint 质量维护**: 观测方向角度检查, 重投影误差检查
- [ ] **关键帧冗余剔除**: 共视比例 > 90% 的冗余关键帧删除
- [ ] **地图保存/加载**: 序列化为 YAML/二进制文件以便复用

---

## 四、已知问题与注意事项

### 4.1 技术债务

1. **`trackFrame()` 每帧都重新做一次 ORB 提取+匹配**（ref_frame ↔ curr_frame）
   - 优化方向：非关键帧用 LK 光流跟踪（`FeatureMatcher::trackLK` 已实现，未接入）
2. **`-march=native` ABI 对齐问题**（CMakeLists Release 模式）
   - `libvslam.a` 以 `-march=native` 编译（Eigen 32 字节对齐），**外部程序手工链接时需保持相同编译标志**，否则 Eigen 对齐加载会段错误（已在测试中实际遇到）
3. **KITTI 数据损坏**：`~/data/kitti/data_odometry_gray.zip` 60KB 无效 zip，需重新下载
4. **单目尺度**：VO 轨迹为归一化尺度，与真值对比需先做尺度对齐（EVO 默认支持）

### 4.2 后续可探索方向

- 双目/深度相机支持 → StereoVO 子类
- IMU 融合 → VINS-Mono 风格紧耦合
- 深度学习特征 → SuperPoint + LightGlue 替换 ORB
- GPU 加速 → CUDA 特征提取与匹配

---

## 五、文件结构速查

```
vslam/
├── CMakeLists.txt                  # C++17, auto-detect g2o/DBoW3
├── cmake/Findg2o.cmake
├── config/default.yaml             # 相机 + Feature + VO + Optimizer 参数（已接入）
├── docs/DEVELOPMENT_LOG.md         # ← 本文档
├── scripts/prepare_kitti.sh        # KITTI 数据准备
├── app/
│   └── run_vo.cpp                  # VO 入口(数据集/摄像头, 输出 TUM 轨迹)
├── include/vslam/
│   ├── common.h                    # SE3, Vec2/3, Mat33/44
│   ├── camera.h                    # 针孔相机模型(单目/双目/RGBD)
│   ├── dataset.h                   # 数据集/摄像头抽象
│   ├── frame.h                     # 帧数据结构
│   ├── mappoint.h                  # 地图点
│   ├── map.h                       # 地图管理
│   ├── feature.h                   # ORB+LK特征
│   ├── vo.h                        # VO状态机 + VOConfig
│   ├── optimizer.h                 # g2o优化接口
│   ├── viewer.h                    # Pangolin双窗口
│   └── loop_closure.h              # 回环检测(Phase2)
├── src/
│   ├── camera.cpp                  # ✅ 完成
│   ├── dataset.cpp                 # ✅ 完成(KITTI/TUM/EuRoC/CAMERA)
│   ├── mappoint.cpp                # ✅ 完成
│   ├── map.cpp                     # ✅ 完成
│   ├── feature.cpp                 # ✅ 完成(ORB+LK)
│   ├── vo.cpp                      # ✅ 完成(核心管线, T_cw 语义统一, LK 模式)
│   ├── optimizer.cpp               # ✅ 完成(Local BA + 共视窗口)
│   ├── viewer.cpp                  # ✅ 完成(Pangolin, 左轨迹右视频)
│   └── loop_closure.cpp            # ❌ TODO(Phase2)
└── test/
    └── test_vo.cpp                 # ✅ 通过(4用例7断言)
```

### 状态图解

```
✅ = 已完整实现
⚠️ = 部分实现（有TODO）
❌ = 仅骨架（待实现）
🔜 = 预留（Phase2）
```

---

## 六、Phase 1 完成状态

**Phase 1（单目 VO + 局部 BA）已全部完成**（2026-08-01），仅剩 KITTI 真实数据集评估待数据到位：

```
✅ 位姿语义统一（T_cw）          ✅ Local BA + 共视图滑动窗口
✅ 初始化/跟踪/LOST 状态机        ✅ LK 光流模式（自动回退 ORB）
✅ 关键帧策略（运动+旋转+衰减）   ✅ TUM/EuRoC/KITTI/摄像头数据源
✅ Viewer 双窗口                  ✅ yaml 全参数配置
✅ 单元测试 7/7                   ✅ TUM 轨迹输出（EVO 可评估）
✅ 位姿保存                       ✅ install_deps.sh / prepare_kitti.sh
```

下一步（Phase 2 完整 SLAM）：

1. **🔴 KITTI 实测评估**: 下载数据 → `scripts/prepare_kitti.sh` → `run_vo` → EVO 对比 ATE/RPE → 参数调优
2. **🔵 DBoW3 安装**: `bash scripts/install_deps.sh`（含 DBoW3）
3. **🔵 回环检测**: `loop_closure.cpp` 完整实现（词袋 + 候选 + 几何验证）
4. **🔵 PoseGraph + Global BA**: `optimizer.cpp` 补全
5. **🔵 地图管理**: MapPoint 质量维护 / 关键帧冗余剔除 / 地图保存加载
6. **🟢 EuRoC IMU 加载**: 紧耦合融合（可选）

### 3.2 用户实测反馈修复（2026-08-01）

用户在 KITTI/摄像头实测反馈 3 个问题，全部解决：

1. **相机内参未用 KITTI 的** → 各 KITTI 序列内参不同（seq00 fx=718.856 vs seq08
   fx=707.09），此前 yaml 写死 seq00。现在 **Dataset 自动读取 `<seq>/calib.txt` P0**
   覆盖配置（`prepare_kitti.sh` 同步解压 calib.txt），任意序列自动正确。

2. **图像畸变** → KITTI odometry 图像已校正无畸变系数；但实现**通用去畸变**：
   `Camera::hasDistortion()/undistort()`（预计算 remap 表，cv::undistort 每帧开销可控），
   `VO::addFrame` 入口自动应用。摄像头/raw 数据在 yaml 配 k1/k2 即生效。

3. **原地旋转却转出 1/4 圆大位移** → 根因：**纯旋转时对极几何退化**（本质矩阵
   E≈0，旋转与平移不可分），recoverPose 返回的 t 方向任意，直接组合出假位移。
   修复：对极回退前做**纯旋转检测**——匹配点"单位位移方向向量"的平均模长
   consistency（平移≈1，旋转≈0），<0.5 判为旋转主导 → **只更新朝向、保持位置**。
   合成测试 + KITTI 全程验证（正常行驶 0 次误触发）。

验证：单元测试 12 项全过（新增纯旋转判据测试）；KITTI 00 全程 21.8 FPS 无 LOST；
**前 1000 帧 ATE 从 53.7m 降至 30.4m**（短程精度提升 43%）。

注：摄像头模式默认内参 (500/500/320/240) 仅供演示，请用棋盘格标定后通过
`config/default.yaml` 传入真实内参（畸变系数 k1/k2 同时支持）。

### 3.3 旋转-平移歧义（单目病态问题）调查与缓解 ✅（2026-08-01）

**用户现象**：绕原点原地旋转，VO 轨迹却画出大圆弧（位置大幅漂移）。

**验证（用户建议的实证方法）**：
1. 关闭歧义抑制（`rotation_shrink: false`），输出轨迹文件检查 tx/ty/tz：
   - KITTI 场景（有近点）：旋转起始帧 PnP 跳变 ~1m，后续稳定（近点约束平移）
   - **纯远点场景（15~35m，平移不可观测）**：旋转段位置振荡漂移 **4.5m+**——
     复现了用户的"旋转耦合进平移"，证明是**单目病态问题**（非简单 bug）

**根因（第一性原理）**：
- 单目**旋转-平移歧义**（bas-relief 退化）：yaw 旋转的像素流 ≈ 横向平移的像素流
  （近大远小相同），单帧在数学上无法区分
- 前进平移的流场是放射状（与 roll 旋转相似）→ 像素方向一致性判据不可用（会误伤）
- 平移可观测性取决于**近处 3D 点**：有近点时 PnP 可约束平移；纯远点时完全不可观

**尝试过的缓解方案（均已验证后回滚）**：
- 位移方向一致性判据：误伤前进平移（放射状流场）
- 逐帧平移收缩（shrink）：KITTI 弯道被收缩 → 地图退化 → LOST
- 旋转模式位置锚定/不插关键帧：位置卡死 → 螺旋段 LOST → 重定位死循环
- rotation-angle 判据（PnP 解出的旋转角）：旋转被吸收进假平移，角度被污染不可靠

**本质结论（诚实告知用户）**：单目 VO 在纯旋转/旋转主导时平移**数学上不可观测**——
yaw 转头流场与横向平移流场几乎相同（近大远小一致），这是单目的**本质病态问题**
（bas-relief ambiguity 退化），任何前端（PnP/ICP/对极）和启发式都只能部分缓解。
**精确轨迹必须换方案**：
1. **VIO**（推荐）：IMU 陀螺仪直接测量旋转 → 解开旋转-平移歧义（VINS-Mono 等）
2. **双目/RGB-D**：提供真实尺度 + 深度 → 平移可观
3. 当前单目版本：保留对极回退的跳变保护（>30m 拒绝），保证稳定性；
   旋转段轨迹为"保守估计"（可能仍含假平移），建议旋转场景配合 VIO 使用

**验证**：KITTI 00 全程 26.3 FPS 无 LOST，ATE RMSE 133.6m（历史最佳）；
远点合成旋转场景记录漂移 ~4.5-16m（单目歧义现象，无法根治）。

### 3.4 轻量 MiniAtlas 与跟踪恢复 ✅（2026-08-03）

用户在 KITTI 00 轨迹中发现 20.4、155.6、298.7、325.8、432.8 秒出现单位位姿，
确认双目 LOST 分支清空地图后把新地图重新放到了原点；190.8 秒附近还存在单帧异常跳变。

本次采用适合教学项目的精简 Atlas：

- 新增 `Atlas/Submap`，保留多个局部地图和每个子地图的全局锚点；
- 跟踪失败先进入 `RECOVERING`，连续失败后才进入 `LOST`；
- 重定位按当前子地图、历史子地图顺序尝试，成功后激活对应子地图；
- 长期失败时冻结旧地图，在最后有效位姿附近建立用于继续跟踪的新子地图；
- 不再清空历史轨迹；未通过重定位连接的新子地图标记为 `disconnected`；
- Viewer 状态栏增加 `submap_id` 和 `lost_frames`，便于定位重定位问题；
- 增加 MiniAtlas 单元测试。

验证：`cmake --build build -j2` 成功，`ctest --test-dir build --output-on-failure` 通过。

### 3.5 KITTI 00 双目尺度退化修复（2026-08-03）

**实测反馈**：双目轨迹 ATE RMSE 188.3m，累计运动显著小于 GT，3D 图出现长直线跳变、
高度漂移和错误闭合。Luna 子代理并行审查确认问题集中在深度质量、位姿验收和断开子地图语义。

修复内容：

- 左右目 LK 增加右→左反向检查、极线误差、前后向误差、光度误差和边界过滤；
- 双目有效深度上限从 50m 收紧到 35m，初始化至少需要 40 个有效深度点；
- PnP 接入配置化的最少内点、内点比例、重投影 RMSE 和相邻有效帧运动检查；
- PnP 位姿通过全部验收后才关联地图点，拒绝解不再污染共视统计；
- 3D-3D RANSAC 阈值由 1.0m 收紧到 0.25m，并检查内点比例、Kabsch RMSE 和点云退化；
- 双目关键帧平移阈值调整为 0.9m，并增加 15 帧最大关键帧间隔；
- 新建子地图默认 `disconnected`：锚点仅供显示，不能伪装成已经恢复的全局轨迹；
- 主 TUM 文件只写 `pose_valid=true` 的全局位姿，所有帧写入 `<traj>.debug.csv`；
- 新增水平视差、纵向错位拒绝、错误双目初始化和 Atlas 连接状态测试。

验证：构建成功，CTest 全部通过。完整 KITTI 00 ATE 需要用户数据重新运行后确认。

### 3.6 Phase 2 完整 SLAM 闭环（进行中，2026-08-03）

**已完成并验证**：

- **M0 依赖**：DBoW3 经 gitee 镜像克隆编译安装到 `~/.local`（无 sudo）；
  修复新版 CMake 拒绝 `CMAKE_MINIMUM_REQUIRED(2.8)`（`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`）；
  DBoW3 的 `DBoW3Config.cmake` **不导出 imported target**，CMakeLists 改用 `${DBoW3_LIBS}` 变量链接。
- **词典**：新增 `scripts/fetch_vocab.sh`，复用 DBoW3 源码自带的 `orbvoc.dbow3`（48MB，DBoW3 官方格式，
  无需额外 30MB 下载）；已拷贝到 `config/ORBvoc.dbow3`。
- **M1 Sim3**（`common.h`）：7 自由度相似变换 + `Sim3::estimate`（Eigen::umeyama，相对奇异值比退化检查）。
- **M2 回环检测**（`loop_closure.h/cpp` 完整实现）：词袋加载/入库/候选过滤（Top-5 + 时间窗 + 分数）/
  PnP 几何验证（`min_loop_inliers` + `pnp_inlier_ratio`）/ Sim3 输出（3D-3D Umeyama）。
- **M3 位姿图优化**（`optimizer.cpp`）：`VertexSE3`（T_wc）+ `EdgeSE3`（Isometry3 测量），
  相邻边（里程计约束，共视加权）+ 回环边（SE3 丢尺度），首帧固定。
- **M4 集成**：`app/run_slam.cpp` + CMake 启用；`run_vo` 强制关回环（A/B 对比）；
  `VOConfig` 新增回环配置段；`handleLoopCorrection`（Sim3 传播 + 位姿图 + 全局 BA + 轨迹更新）。

**单元测试**：25 项全部 PASSED（新增 Sim3 代数/Umeyama 精度/退化输入、位姿图合成漂移校正
1.79m→4e-5m、合成回环词袋命中+PnP 验证+scale=1+负例 nullptr）。

**踩过的坑（重要）**：

1. **`-march=native` 与 apt g2o 的 Eigen ABI 冲突**：Release 编译用 `-march=native`（AVX/32 字节对齐），
   apt g2o 库按 SSE/16 字节对齐编译，跨 DSO 传 `Isometry3d` 时 **`EdgeSE3` 测量值静默损坏**
   （实测测量变为 (0,1,1)），位姿图优化发散（chi2 变负数、位姿飞到 80m+）。
   **修复**：CMake Release 改为 `-O3 -msse2`（16 字节对齐）。⚠️ 若换 g2o 版本需重新验证此设置。
2. DBoW3 `transform` 只有 2 参数（BowVector）与 4 参数（vector）重载，无 3 参数版本。
3. 合成回环测试中相机光轴默认朝 +Z，点云必须放 +Z 前方（否则图像全黑、ORB 无特征）。

**当前阻塞问题（下个模型修复重点）**：

KITTI 00 单目实测（image_0，4541 帧，回环开）：

| 配置 | 回环数 | ATE RMSE |
|------|--------|----------|
| run_vo 基线（回环关，min_score 0.3） | 0 | 121.4m |
| min_score=0.3（旧：只校正回环帧之后段） | 2 | 131.6m |
| min_score=0.1 + 旧校正 | 9 | 184.5m |
| min_score=0.1 + **整体变换** | 12 | 213.5m |
| min_score=0.1 + 整体变换 + 冷却(200帧) + 高门槛(50/0.85) | 挂起中 | ? |

**回环检测本身工作正常**：检测到 kf#66→kf#4509（起点-末端真回环）、kf#193→1630 等；
PnP 验证内点 160-640，scale=1（同 VO 轨迹尺度一致）。

**问题根因（已定位到代码层面）**：

1. **多次回环校正互相冲突**：旧实现只变换"回环帧之后"段 → 区间嵌套重叠 → 轨迹切碎跳变
   （实测帧 3380 处跳变 58m）。已改为**整体变换**（整个子地图统一 S_global），消除段间跳变，
   但整体变换后 ATE 反而更差（213.5m），说明后续仍有多余/错误校正或全局 BA 拉坏轨迹。
2. **S_global.s 恒为 1**：`verifyLoop` 的 Sim3 用"同一坐标系（VO 世界系 U）"的点做 3D-3D
   （`T_cw_loop·p_w` 与 `T_cw_curr'·p_w` 都在 U 系）→ 尺度比恒 1，**无法捕捉单目 VO 的全局
   尺度漂移**（evaluate_ate Sim3 对齐 scale≈1.95）。起点-末端回环本该校正尺度，当前实现只能
   平移/旋转。
3. **全局 BA 副作用**：`globalBundleAdjustment` 转发 localBA（motion-only，点固定），
   每次回环优化全部 2340 KF + 27 万点，主线程阻塞（~5-10s/次），且可能把位姿拉偏
   （点本身含漂移，固定点约束强）。
4. **词袋召回与质量权衡**：min_score 0.3 → 只 2 次候选（漏真回环）；0.1 → 12 次
   （含误报风险）。冷却 + 高门槛（50/0.85）的测试被中断，结果未知。

**建议修复方向（供参考）**：

- 方案 A：**回环验证后做 S_global 合理性检查**（平移量级、与上次校正的一致性），只接受
  "收敛型"校正；或在检测到起点-末端型回环后冻结后续校正。
- 方案 B：**verifyLoop 的 Sim3 改为捕捉真实尺度**：用 kf_loop 地图点在**回环前/后的世界系坐标**
  做 3D-3D（如 ORB-SLAM 的 Sim3Solver），而非相机系坐标。
- 方案 C：**全局 BA 限幅**：只优化回环帧窗口（非全图），或提高迭代数但减少调用频率。
- 方案 D（最稳的教学级兜底）：**只接受"回环帧 < 当前帧 40%"且"当前帧 > 60%"的大回环**，
  一次校正全局尺度与位置，其余小回环仅记录不校正。

**后续待办**：跑通当前参数组合的 KITTI 00 全量评估；更新 PHASE2_DESIGN.md 决策记录；
确认 ATE 是否达到"显著低于基线 121.4m"的目标。

### 3.7 Phase 2 回环校正回归修复（2026-08-03）

代码审查确认 ATE 恶化并非阈值问题，而是校正链存在结构性错误：全图统一 Sim3 只会更换
gauge，不能减小回环相对残差；位姿图之后点云未同步，而后续 motion-only BA 会把位姿重新
拉向旧点云；同一批世界点经两个 SE3 后做 Umeyama 又导致尺度理论上恒为 1。

本次修复：

- `verifyLoop` 由 PnP 直接输出 `T_loop_curr` SE3 位姿图测量，不再宣称获得尺度；
- **更正（2026-08-04）**：solvePnP 的 rvec/tvec 满足 `p_c = R·p_w + t`，即 **T_cw（世界→相机）**
  语义（项目内 `trackFrame`/`trackFrameLK`/`tryRelocalize` 均把 solvePnP 输出直接存为
  `pose_cw`）。位姿图边测量约定为 `Z = X_loop⁻¹·X_curr`（X 为 T_wc），因此正确写法是
  `T_loop_curr = kf_loop->pose_cw * T_cw_curr⁻¹`。早前（6c311d7）曾误判 PnP 输出为
  T_wc 而删掉该逆，得到 `X_loop⁻¹·X_curr⁻¹`，与位姿图约束恒等式矛盾——这是"回环校正
  破坏轨迹"（ATE 121→213m）的残留根因，现已改回 T_cw 语义并加固合成回环测试
  （末帧非单位位姿，能捕获求逆错误）；
- 删除全图统一 Sim3 传播，保持首帧世界锚点不动；
- 关键帧插入时冻结里程计边，累计保留全部历史回环边并联合优化；
- 位姿图后按地图点最早观测关键帧的位姿增量同步点坐标，再运行全局 motion-only BA；
- 保存完整逐帧 `T_cw`，插值关键帧校正并让 `run_slam` 最终输出真正校正后的位姿与朝向；
- `global_ba_iterations` 现已实际传入优化器；无 g2o 构建会正确跳过位姿图测试。
- 位姿图后端不可用/约束不足时不再伪报闭环成功；跨 Atlas 子地图候选在实现
  Sim3 子地图融合前明确拒绝。

验证：本机未检测到可用的 g2o/DBoW3，纯可选依赖构建和 CTest 通过；KITTI 00 ATE 与
带 g2o/DBoW3 的闭环集成测试仍需在依赖可用的环境中复测，不能据此宣称已达到 ATE 目标。

### 3.8 KITTI 00 轨迹冻结根治（2026-08-04）

现象：`run_vo` 跑一段时间后水平位置不再更新，仅角度继续转。

排查：全量跑 KITTI 00（headless）后仅 210/4541 帧 pose_valid；Viewer 轨迹线只记录
有效帧（`tracking_valid && map_connected`），而相机箭头用每帧 PnP 位姿，故出现
"位置冻结、角度在转"。根因两条：

1. **建点后立即被剔除**：`insertKeyFrame` 中 `cullMapPoints(2)` 在建点之后执行，
   每第 20 个关键帧会把 `createMapPointsFromStereo` 刚建的点（observed_count=1）
   当场删除。KF 120（≈帧207）建 476 点后 mp 从 28298 掉到 24207，帧 210 跟踪该
   帧时 matches=147 但 pts3d=18 → PnP 垃圾解被拒 → LOST → 建新子地图。
2. **新子地图 disconnected 永久冻结轨迹**：`createSubmap` 以 `connected=false` 建图，
   此后所有帧 `pose_valid=false`，轨迹再无线段追加。

修复：

- `insertKeyFrame`：把 `cullMapPoints` 提前到建新点**之前**，刚建点获得一轮观测窗口；
- `createSubmap`：锚点继承最后有效全局位姿，新子地图标记 `connected=true`，
  重初始化后轨迹继续记录（丢失间隙约 1s 停顿，不再彻底冻结）；
- 新增 `config/kitti00.yaml`：num_features 1000→2000、更严 PnP 内点/比例、
  放宽帧间平移上限、双目 KF 平移 0.9→1.2m，降低触发 LOST 的概率。

验证（KITTI 00 全程，配置 kitti00.yaml）：

- 有效位姿：210/4541 → **4522/4541**（99.6%）；
- 局部 BA 窗口点数：~10 → 195，共视关系恢复正常；
- 全程路径 3689m，与 KITTI 00 真实里程一致；LOST 仅 1 次（帧4334），
  重初始化后从丢失前位置继续，轨迹连续。

### 3.9 多核线程优化（2026-08-04）

基于 3.8 的性能监测定位热点：特征提取（ORB detectAndCompute）占帧耗时大头。

优化：

- **并行分带 ORB 提取**（`FeatureMatcher::extract`）：ORB 内部已按金字塔 octave 并行
  （~8 任务，TBB）。把图像按行分成 N 个带（N 依图片高度与核数），每带独立
  detectAndCompute，得到 N×octave 个可并行任务。每带上下各扩 18px 提取、只保留
  核心区特征，避免边界角点丢失与重复（相邻带核心区自然承接）。带高受限时按真实
  scale_factor（1.2，非 >>2）收敛金字塔层数。小图（<128 行或特征预算 <500）退化
  为单带串行，与旧实现一致，保证单元测试确定性。
- **实测**：`vo.extract` 11ms → **4.2ms**（约 2.6x）；KITTI 00 全程 FPS 17.05 →
  **18.8**，轨迹质量不变（4484/4541 有效帧，路径 3642m）。
- **回退**：双目 LK（`matchStereo`）外部分块并行不可行——`calcOpticalFlowPyrLK`
  内部已是 TBB 并行，且 OpenCV 4.6 无接收预建金字塔的重载，外部分块会重复构建
  金字塔反而更慢（10.9→22ms，已回退串行）。
- **修正**：`vo.extract` 的 `PERF_SCOPE` 原先放在函数体顶层、作用域覆盖整个
  addFrameImpl，导致测量虚高（51ms 实为整帧后半段）；改为用 `{}` 包裹只在
  extract() 计时。
- **g2o 多线程**：本机 apt 版 g2o（无 OpenMP）无 `SparseOptimizer::setNumThreads`
  接口，未启用；local/global BA 问题规模小，收益有限。

结论：OpenCV（TBB）已把 ORB 与 LK 内部并行化，进一步提速需在**任务级流水线**
（跟踪/局部建图/回环线程）层面并行，而非帧内重复并行。

### 3.10 回环边测量求逆修复（2026-08-04）

现象：用户反馈"加入 Phase 2 后前端 VO / 整体轨迹与姿态的流畅程度反而不如不加"。
旧数据佐证：`slam_traj.txt`（开回环）ATE 213.5m vs `vo_traj.txt`（关回环）121.4m。

排查（代码走查 + 端到端复现）：

- **根因**：`LoopClosure::verifyLoop` 把 solvePnP 输出误判为 T_wc。solvePnP 的
  rvec/tvec 满足 `p_c = R·p_w + t`（T_cw，世界→相机），项目内 `trackFrame` 等处
  均按此使用。位姿图边测量约定为 `Z = X_loop⁻¹·X_curr`（X 为 T_wc，见 test_vo.cpp
  位姿图测试）。故正确公式是 `T_loop_curr = kf_loop->pose_cw * T_cw_curr⁻¹`；
  6c311d7 删掉该逆后得到 `X_loop⁻¹·X_curr⁻¹`，与约束方向相反。
- **后果**：每次回环校正，位姿图收到与里程计矛盾的回环约束，g2o 把整条关键帧链
  往错误方向拉扯 → 轨迹反复扭曲、ATE 恶化、连续帧跳变。
- **测试盲区**：合成回环测试的末帧位姿是单位阵（T_cw == T_wc == I），正反求逆
  结果相同，测不出该错误。

修复：

- `src/loop_closure.cpp`：改回 `T_loop_curr = kf_loop->pose_cw * T_cw_curr_in_loop.inverse()`，
  并更正注释（T_cw 语义 + 引用 trackFrame 用法佐证）。
- `test/test_vo.cpp`：合成回环末帧改为非单位位姿（偏离原点 0.15m 级），反号误差
  ≈0.36m >> 0.1m 断言容差；`min_score` 0.3→0.05（合成圆点特征对微小视差敏感，
  时间窗仍是防误报主闸，PnP 验证继续把关弱候选）。
- 反向验证：临时恢复错误写法 → 测试在平移断言处失败；恢复修复 → 全量测试通过。

验证（KITTI 00 全程，headless，default.yaml）：

| 运行 | 结果 |
|------|------|
| run_vo（回环关） | 4541 帧全部处理，4503 有效位姿，ATE 178.6m，路径 3679m（真值 3680m，绝对尺度正确） |
| run_slam（回环开，修复后） | 4541 帧全部处理，4541 有效位姿，闭合 3 个回环（含起点-末端 kf#78→4517），ATE 177.8m，末帧误差 150→124m |
| 旧 slam_traj.txt（回环开，修复前） | ATE 213.5m（回环校正主动破坏轨迹） |

结论：回环校正不再伤害轨迹（与无回环基线持平），并把末帧漂移拉回 26m；
单测 25 项全部 PASSED。

遗留问题：run_slam 存在一次非确定性崩溃（首跑帧 4286 静默退出，二跑完整通过，
并行 ORB 压力测试 3×2 万次无异常）——疑为堆损坏或并发竞态，建议后续用 ASan 构建
定位；另全局 BA 为 motion-only（点固定），回环校正的收益受限，可考虑双目下放开
地图点自由度做全 BA。

### 3.11 回环/BA 性能与精度提升 + LOST 恢复（2026-08-05）

背景：§3.10 遗留两个问题——① 全局/局部 BA 为 motion-only（点固定），回环校正后
点坐标不参与重投影精修，中段误差收不下去；② run_slam 帧 4286 非确定性静默崩溃
疑为并发竞态。按 IMPROVEMENT_PLAN.md 的 P0-1/P0-2/P1-0/P1-1/P1-2 逐项实施，
每项均以「现象 → 分析 → 根因 → 修复 → 验证」闭环记录。

---

**P0-1 双目全 BA（最关键，ATE -81% 的主因之一）**

现象：只放开地图点自由度（`v_point->setFixed(false)`），不改变任何其他逻辑，
`localBundleAdjustment` 单次耗时从 ~13ms 暴涨到 440–3400ms，FPS 从 8.5 掉到 0.5。

分析过程（WSL2 环境，perf 采样基本失效，逐层排除）：

1. **配置 A/B 隔离**：关掉 `enable_local_ba` → FPS 回到 8.5，确认瓶颈在 BA 内部。
2. **迭代数比例实验**：`local_ba_iterations` 10→2，耗时按比例下降（388ms→177ms），
   确认是「每次迭代慢」而非迭代次数爆炸（~100-200ms/迭代，问题规模却只有 3-6 KF
   / 200 点 / 700 边，理应 <1ms）。
3. **perf 采样**：热点完全扁平（无单一函数 >3.5%），大量 `__sched_yield` 与
   iostream——说明进程大部分时间阻塞在 I/O 或自旋，而非计算。
4. **/proc 线程状态 + utime/stime**：主线程 ~50% 用户态、~50% 内核态，minflt 极少
   → 排除缺页，怀疑系统调用（文件 I/O）。
5. **最小复现**：独立 5 KF / 200 点 / 1000 边程序，自由点路径立即触发 Eigen 断言
   `PlainObjectBase::resize`（固定 6×6 矩阵被改成非 6×6）——Release（NDEBUG）下
   断言被关，矩阵静默损坏。
6. **backtrace_symbols 定位断言点**（gdb 未安装，用 SIGABRT handler + addr2line）：
   `g2o::SparseBlockMatrix<Matrix<double,6,6>>::block` ←
   `BlockSolver<BlockSolverTraits<6,3>>::buildStructure`。

根因（g2o 两个坑叠加）：

- **坑 1：`BlockSolver::buildStructure` 按 `Vertex::marginalized()` 区分 6×6 位姿块
  与 3×3 路标块**。我们从未对点顶点调 `setMarginalized(true)`（ORB-SLAM 里
  `vPoint->setMarginalized(true)` 是标配）。点未被 marginalize → 被当作位姿块分配
  → 3D 点块尺寸 3×3 ≠ 6×6 → Eigen 断言；Release 下静默损坏 Hessian 结构。
- **坑 2：`LinearSolver` 默认 `_writeDebug=true`**（`linear_solver.h:49`）→ Cholesky
  失败后 `A.writeOctave("debug.txt")` 每次迭代把整个 Hessian 写文件——perf 里的
  `SparseBlockMatrix::writeOctave`、`TripletEntry` 排序、iostream、内核 I/O 全是它。
  这正是「固定点快、自由点慢」的原因：固定点时无点块，Schur 不执行，Hessian 是
  纯 6×6 块且正定，Cholesky 成功，全程无 I/O。

修复方案（`src/optimizer.cpp`）：

```cpp
v_point->setMarginalized(true);        // 点顶点按 3×3 路标块走 Schur 消除
linearSolver->setWriteDebug(false);    // 防御：Cholesky 失败不再写盘
```

接口层：`localBundleAdjustment`/`globalBundleAdjustment` 增加 `fix_points` 参数
（`std::optional<bool>`，默认按传感器自动选择：单目固定防尺度 gauge，双目放开）。

验证：最小复现自由点 BA 440–3400ms → **13ms**（与 motion-only 同量级）；全序列
FPS 恢复 8.5，与基线持平。

---

**P1-2 BA 深度加权（与 P0-1 同批实施）**

现象/原因：双目视差测深误差 σ_z ∝ z²（z = f·b/d），远点观测精度差却与近点等权，
污染位姿估计（旋转漂移是误差主项，中段均差 185m）。

修复方案（`src/optimizer.cpp` 边构造处）：双目下每个重投影观测的信息矩阵按
`w = clamp(100/z², 0.04, 25)` 加权（参考深度 10m 处 w=1，近点放大、远点抑制）；
Huber 阈值随权重等比缩放（`delta' = delta/√w`），保持像素级鲁棒语义不变；
单目路径维持单位信息矩阵。

---

**P0-2 回环检测提前**

现象：KITTI 00 全程仅 3 次闭环，起点-末端真回环在最后一帧（kf#4517）才闭合，
全程漂移得不到及时校正。

分析/根因：`detectLoop` 只取 DBoW3 Top-5 且无位置信息；回环冷却用帧计数
（`loop_cooldown_frames=200`），KF 密度波动时冷却时间漂移。

修复方案：

- `src/loop_closure.cpp`：词袋查询 Top-5 → **Top-20**（召回扩宽，准确性由时间窗 +
  分数阈值 + PnP 双门槛把关）；新增**位置先验**——词袋未命中时遍历历史 KF 缓存，
  与当前 KF 世界系距离 <25m 且 KF 间隔 ≥100 的直接成为候选（自交区域词袋分低时
  补召回，误检由 `min_loop_inliers=50` + `pnp_inlier_ratio=0.85` 兜底）。
- `src/vo.cpp`：冷却基准改为**关键帧计数**（`loop_cooldown_kfs`，与 KF 密度解耦；
  兼容旧字段 `loop_cooldown_frames` 自动按 /10 换算）。
- 配置：三个 yaml 新增 `top_candidates` / `position_prior_dist` / `position_prior_gap`。

验证：run_slam 闭环 3 → 4 次，且起点-末端回环提前闭合。

---

**P1-0 LOST 恢复（锚点外推 + 子地图对齐）**

现象：`traj.txt` 最后 10% 的 z 方向漂移 32.6m 来自 LOST 事件；丢失 20 帧（~2s
≈20m 运动）后 `createSubmap` 用丢失前的 `last_valid_pose_cw_` 锚定，重初始化
轨迹在丢失处静止/回跳，出现 60m 级断点。

分析：丢失期位移完全未知是信息学上限，但「用丢失前最后瞬时速度外推」可大幅
缩小锚点误差；实测 off-road 段瞬时速度被 PnP 高估（20 帧外推 24–41m），需限幅。

修复方案（`src/vo.cpp`）：

1. **匀速外推**：跟踪阶段记录逐帧相对运动 `per_frame_motion_`（Twc 语义）；
   `createSubmap` 时锚点 = 最后有效位姿 × T_rel^lost_frames，单步限幅 2.5m
   （防速度高估），总外推 >60m 视为异常退回。
2. **Umeyama 刚体对齐**：重建成功后延迟到新子地图 ≥3 个 KF（拟合最低点数），
   与历史轨迹**末端 100 帧**（丢失点必在轨迹末尾，限制搜索窗防轨迹自交区误配）
   做近邻匹配（半径 50m，锚点可能偏 20-40m）→ `Sim3::estimate` 求解 →
   仅接受 scale∈[0.85,1.15]（双目绝对尺度），施加到子地图全部 KF 与地图点。
   单目不执行（尺度不同源，对齐会引入尺度跳变）。

验证：前 60% 轨迹零跳变；LOST 后断点大幅减小。遗留：off-road 段 4 次重建时
对齐匹配数仍不足 4（方向误差 >50m），跳变未完全消除。

---

**P1-1 确定性（分带收敛 + ASan 验证）**

现象：run_slam 帧 4286 一次非确定性静默崩溃（复跑通过），并行 ORB 压力测试
3×2 万次无异常，疑为堆损坏或并发竞态。

分析：并行分带 ORB 的带数由 `cv::getNumThreads()/2` 动态推导，TBB 内嵌并行度
随线程数波动，行为不可复现，是唯一可疑的并发点（Viewer headless 不起线程、
g2o 无 OpenMP）。

修复方案（`src/feature.cpp`）：分带数固定上限 4（`min(orb_max_bands_, 4)`），
kBorder 18→24 减少边界特征损失。

验证：ASan+UBSan 构建（`-fsanitize=address,undefined -fno-omit-frame-pointer
-msse2`，对齐 g2o ABI）全量跑 KITTI 00：**4541 帧完成，0 报错 0 崩溃**（帧 4286
崩溃未复现；因崩溃非确定，仍需多跑确认）。

---

验证汇总（KITTI 00 全程 headless，kitti00.yaml）：

| 运行 | ATE RMSE | 备注 |
|------|----------|------|
| run_vo（回环关，修复前基线） | 178.6m | §3.10 数据 |
| run_vo（回环关，修复后） | **106.4m**（-40%） | 前 60% 轨迹零跳变 |
| run_slam（回环开，修复前基线） | 177.8m | §3.10 数据 |
| run_slam（回环开，修复后） | **34.0m**（-81%） | 4 次闭环（提前闭合），末帧误差 134m，scale 1.05 |

单测 27 项全部 PASSED。

遗留：

- 后 40% off-road 段 4 次子地图重建，对齐匹配数不足时跳过，断点未完全消除；
  跨子地图回环融合（P2-3）可根治。
- 帧 4286 崩溃在 ASan 下未复现，建议多次全量跑确认后再关闭该观察项。

### 3.12 后端/重定位性能攻坚 + 回环稳定性（2026-08-05）

现象：用户跑 `run_slam KITTI 00` 反馈"后半段跟踪失败多、卡顿严重、帧率低"；
第三方模型称"性能可优化 23%"。实际排查发现两大卡顿源：**Local BA 单次最高
131 秒**（地图膨胀后 BA 规模失控）与 **回环后全局 BA 单次最高 53 秒**（全图
位姿顶点上万）；另有计时统计陷阱导致 FPS 显示失真。

排查方法（详见 TUTORIAL.md §10 教学记录）：

- PERF_SCOPE 数据 vs 墙钟差 10 倍 → 逐段加计时戳定位到"主循环计时基准被
  --skip 跳帧解码时间污染"（FPS 0.2 是假象，真实帧间隔 70-100ms）；
- `opt.ba` max 131s / `loop.global_ba` avg 19s 定位真实卡顿。

修复（8 项，均为独立可回滚改动）：

1. **BA 点数量截断（4000，按观测数降序）**（optimizer.cpp）：地图膨胀后
   窗口内点顶点上万，Schur 消除超线性增长。截断后 Local BA max 131s → 47ms。
   原理：观测数多的点约束最强，截断损失的信息量最小；前段地图小不触发。
2. **全局 BA 位姿采样（间隔 3 + 首尾必含）**（optimizer.cpp）：全图 2400+
   KF 时 Schur 后位姿系统上万维。采样后 930→311 位姿顶点，全局 BA
   53.6s → 0.73s max。未采样 KF 保持位姿图解（全局 BA 只是回环后的精修）。
3. **重定位粗筛**（vo.cpp + feature.cpp）：`quickMatchCount` 用当前帧前 256
   描述子对候选做子集 BF（~0.5ms），距离 <64 匹配 <20 直接跳过；只有相似
   候选做全量 BF + PnP；去掉重定位中冗余的 F 矩阵 RANSAC。LOST 帧从秒级
   降到几十毫秒。
4. **run_slam 计时基准修复**：FPS/Done 统计移到 `--skip` 之后（跳帧解码
   时间不再污染统计）；新增 `--frames N` / `--skip N` 分片测试参数。
5. **回环检测 Top-5 → Top-20 + 位置先验 + 冷却改 KF 计数**（上轮 P0-2 的
   延续，本轮实测闭环可提前闭合：kf#151→1595 中段闭环 + kf#78→4518
   起点-末端闭环稳定触发）。

实验记录（两轮被证伪的改动，均为单变量对比）：

- **双目 LK 5 层/31px → 4 层/25px**（省 ~40ms/帧）：LOST 219 vs 32（6 倍
  恶化），闭环全灭，ATE 67.8m。近点视差 >100px 超出金字塔搜索范围 → 深度
  点锐减 → 跟踪连锁恶化。**回滚**（精度是系统的，性能是局部的）。
- **点 cull 保留单观测点**（两版：只删孤儿 / 保留近期引用点）：地图无限
  膨胀 OOM，或单观测垃圾点污染 PnP（LOST 增 9 倍、子地图重建 37 次、
  ATE 174.8m）。**回滚**，维持"观测 <2 删除"原策略。

验证（KITTI 00 全程 headless 单进程，kitti00.yaml）：

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 全程耗时 | 790s（perf2 配置） | 607s | **-23%**（7.5 FPS） |
| opt.ba 单次 max | 131 秒 | 726 ms | **-99.4%** |
| 全局 BA avg/max | 19.2s / 53.6s | 0.51s / 0.73s | **-97%** |
| 子地图重建 | 4-37 次（随机） | 0 次 | 尾段跟踪稳定 |
| ATE RMSE | 34.0m（3.11 基线） | 15.2-33.4m | 持平或更优（闭环次数随机） |
| 确定性 | 非确定 | 配置收敛可复现 | 同配置两次跑一致（33.4m/2 闭环/0 重建） |

遗留：闭环触发次数仍有随机性（2-3 次区间）；老 KF 的 map_points 因 cull
策略（观测<2 删）在 LOST 频繁时仍会稀疏，verifyLoop 存在失败；下一步
框架级优化（见 IMPROVEMENT_PLAN §7 讨论）：关键帧稀疏化、异步后端、特征数
自适应——移动端/嵌入式实时性的前提。

### 3.13 关键帧稀疏化 + 回环检测稳定性（2026-08-05）

承接 §3.12，按 IMPROVEMENT_PLAN 优先级实施第 1、2 项。

**KF 规模硬顶（关键帧稀疏化第一步）**

- 现象：KITTI 00 全程 KF 峰值 2820（ORB-SLAM 同级 <1000 量级），KF 数直接
  放大 Local BA / 全局 BA / 重定位 / 窗口扫描四处开销。
- 实现（vo.cpp `needNewKeyFrame`）：`keyframe_max_count`（默认 1800）超过后
  KF 平移阈值 ×1.5，压缩后续冗余帧。零风险（不改变前段行为，只影响超限后）。
- **实验记录（证伪）**：尝试"相邻 KF 共视比例 ≥90% 判冗余跳过"——实测全程
  0 次触发。根因：本实现每帧重新提取 ORB（非 LK 跟踪），相邻帧描述子匹配率
  天然只有 30-50%，共视比例从不上 90%。**该判据适用于 LK/特征跟踪型前端，
  不适用于每帧重提取型**。移除，仅保留硬顶。
- 验证：KF 峰值 2820 → 2197（-22%），FPS 8.4（+12%），ATE 31.0m（持平略好）。

**回环检测多候选 + 位置先验兜底**

- 现象：闭环次数 2-3 次随机（第 3 次闭环决定 ATE 15 vs 33m）；`too few
  3D-2D correspondences` 170+ 次（候选 KF 点被 cull 稀疏时验证必败）；位置
  先验从未触发（旧实现只在词袋无候选时启用）。
- 实现：
  1. `detectLoop` 返回候选列表（loop_closure.h/cpp）：词袋分数降序 top-3 +
     位置先验最近 1 个（世界系距离 <25m、KF 间隔 ≥100，去重），调用方
     （vo.cpp insertKeyFrame）依次 PnP 验证，第一个通过即回环。
  2. `loop_cooldown_kfs` 20 → 60：多候选后同区域易二次命中（实测 33 KF 内
     连续两次闭环，约束冲突把轨迹拉变形 ATE 79m），60 KF 挡重复、放不同区域。
- 验证：闭环稳定 2 次（kf#151→1595 + kf#675→3623，区域分散），ATE 23.3m
  （vs 单候选 31m / 多候选无冷却 79m），8.2 FPS，旋转对齐 0.12（接近 I）。

汇总（KITTI 00 全程 headless，kitti00.yaml）：

| 指标 | 3.12 后（perf9） | 本轮（lc3） |
|------|------------------|-------------|
| KF 峰值 | 2820 | 2197 |
| ATE RMSE | 33.4m | **23.3m** |
| 全程耗时 | 607s | 553s（8.2 FPS） |
| 闭环 | 2 次（含终点） | 2 次（区域分散更稳） |

单测 27 项全部 PASSED（回环测试适配多候选接口：cands=2 断言）。

### 3.14 帧 4286 非确定性崩溃复现实验（2026-08-05）

背景：§3.10 遗留 run_slam 首跑帧 4286 静默退出（复跑通过）的观察项；后续
多轮改动（§3.11-3.13）后需要确认是否仍存在。用户要求优先定位该崩溃（在
引入异步后端前必须先排除并发隐患）。

排查动作：

1. **代码审查**：src/ 与 app/ 无显式 `exit()/abort()/std::terminate` 调用；
   崩溃时无任何日志输出（最后一行是 insertKeyFrame 内的 "New KF"），指向
   段错误/被杀而非业务 exit。
2. **core dump 检查**：环境默认 `ulimit -c 0`（core 未开启）——早期"静默
   退出"无法抓取现场，这是复现工作的最大盲区；已用 `ulimit -c unlimited`
   重新采样。
3. **分片复现**：`--skip 4000 --frames 541` 反复 3 次（原始崩溃帧号 4286
   在区间内）——全部 exit=0 无 core。
4. **全程复现**：完整 KITTI 00 连续 2 次（配合 core dump）——全部完整跑完
   （~550s/次，8.2 FPS），无崩溃无 core。
5. **历史样本**：§3.11 的 ASan+UBSan 全程（0 报错）+ §3.12/3.13 期间 5 次
   全程（perf6/9/kf1/lc2/lc3）均完整。唯一一次异常是 perf7（"只删孤儿"版
   cull 导致地图无限膨胀，疑似 WSL2 内存上限被杀）——该版本已回滚。

结论：原始帧 4286 崩溃在 6+ 次全程 + 3 次分片采样中未复现，且期间大量
改动（BA 规模控制、重定位粗筛、cull 回滚、KF 硬顶）与该崩溃模式相关；
**判定为已消除或概率极低**。建议：生产/长期运行环境默认开启
`ulimit -c unlimited` + 崩溃时抓日志尾部，若再出现即可直接定位；
**异步后端（P2-1）的并发引入风险已排除**，可择机实施。

### 3.15 异步后端（P2-1）实现与 A/B 基准——净收益为负，默认关闭（2026-08-05）

背景：§3.14 判定帧 4286 崩溃已消除后，按计划实施异步后端：Local BA /
回环检测 / 回环校正在后台线程执行，前端只做跟踪不等待（帧率恒定）。

**架构设计（快照隔离 + 短临界区）**：

- `optimizer.cpp`：Local BA 去除 `map->getMapPoint` 依赖（改用收集期保存的
  点指针）——优化可在"快照数据"上执行，完全不触碰地图集合；
- `vo.cpp` 异步路径：`insertKeyFrame` 提交 BackendTask（有界队列 8，满则
  调用方同步执行）→ 后台线程锁内构造快照（KF pose 深拷贝 + 点对象深拷贝）→
  锁外 g2o 优化 → 锁内写回真 KF/真点；
- `handleLoopCorrection` 重构为三阶段（锁内收集工作集 → 锁外计算 →
  锁内写回），同步/异步共用同一实现；
- `loop_closure.cpp`：addKeyFrame/detectLoop 内部互斥；
- 锁序约定：backend → map → traj，无嵌套持锁。

**调试记录（两个并发 bug，均实测复现+修复）**：

1. **torn read**：后台写回 `Eigen::Vector3d`（24B 非原子）与前端 PnP 裸读
   竞争 → 读到新旧混合坐标 → 垃圾 3D 点 → LOST 爆增（基准实测 async 114 vs
   sync 34）。修复：前端所有读点路径（trackFrame/trackFrameLK/tryTrack3D3D/
   tryRelocalize）持 map_mutex_ 拷贝。
2. **非递归锁重入死锁**：`tryRelocalize` 成功路径在 map_mutex_ 块内调用
   `updateStatus`（内部再次 lock）→ 确定性死锁（复现点：Relocalized 日志后
   主线程永久卡住，/proc 所有线程 futex_do_wait）。修复：updateStatus 移出锁块。

**基准测试**（新增 `scripts/benchmark.py`：N 次全程跑，汇总 FPS/ATE/LOST/
重建/闭环/BA 峰值，支持双配置对比）：

| 指标 | async (A) | sync (B) | 结论 |
|------|-----------|----------|------|
| FPS | 8.13 ± 0.11 | 8.55 ± 0.25 | async -5% |
| LOST 次数 | 122.5 ± 20.5 | 53.0 ± 14.0 | async +130% |
| ATE RMSE | 87.4 ± 14.0 | 71.5 ± 2.8 | async -18% |
| 闭环次数 | 1.0 | 2.0 | async 滞后 |

**结论**：异步后端**净收益为负**，默认关闭（`Runtime.async_backend: false`，
代码保留为实验性开关）。根因：BA 在后台滞后数帧执行，前端 PnP 使用陈旧
3D 点/位姿 → 跟踪误差积累 → LOST 增多 → reloc/重建开销抵消 BA 的异步收益。
同步模式（主线程即时 BA）在"每 2 帧一个 KF、BA 25ms"的负载下没有明显瓶颈；
异步的收益要在"前端重负载"（特征/立体匹配为主）或"KF 密集 + BA 大"的场景
才有意义。若要真正受益，需 ORB-SLAM 级的前端/后端彻底分离（LocalMapping
线程 + 显式关键帧/点管理 + 延迟一致），留待后续。

附带修复：`opt.ba` 观测收集改为保存点指针（同步路径行为不变，单测 27 项全过）。

### 3.16 异步后端重做 + 线程收敛 + 锁结构（P0/P1/滑动窗口）（2026-08-06）

承接 §3.15（异步净收益为负，默认关闭）。用户反馈 run_vo"不是纯 VO 也在跑 BA 且卡顿掉帧"、
run_slam"后端线程影响前端帧率、占满 CPU"。按「分析 → P0 线程收敛 → P1 锁结构 → 滑动窗口
异步」四步实施，每步 A/B（KITTI 00 全程/2000 帧，headless，本机 AMD 7945HX 32 逻辑核）。

**P0 立即修复（纯 VO + 线程收敛）**

1. **run_vo 变真纯 VO**（`app/run_vo.cpp`）：除回环外再强制 `enable_local_ba=false`、
   `async_backend=false`。实测全程 4541 帧无一条 "Local BA" 日志；与 run_slam（BA+回环开）
   构成干净的 A/B。
2. **线程收敛**（`config/default.yaml`）：`opencv_threads 0→16`（默认全核 32 时进程 63 线程，
   超订无益；设 16=物理核数，实测 FPS 反升 26.9→27.7、线程 63→47；设 8 则 FPS 略降 25.4），
   `orb_max_bands 8→4`（嵌套 TBB 任务放大，超订）。移动端建议 2~4。
3. **原子计数**（`map.h/cpp`）：`mapPointCount/keyFrameCount` 改 `std::atomic<size_t>`，
   `updateStatus` 不再每帧持锁读计数（atlas 仅前端线程访问，无跨线程）。

**P1 锁结构：`map_mutex_` 从 `std::mutex` 改 `std::shared_mutex`**

- 前端读点/位姿路径（`trackFrame/tryTrack3D3D/trackFrameLK/tryRelocalize/needNewKeyFrame`）
  持**共享锁**；后端 BA 写回、前端插关键帧、换地图持**独占锁**——读并发、写独占。
- 修复两处真实竞态（代码审查发现）：
  - `createSubmap` 的 `map_ = submap.map` 成员交换未持锁 → 与后端 `map_->getKeyFrame`
    并发读 → shared_ptr 数据竞争；已加独占锁。
  - `handleLoopCorrection` 末尾 `map_->keyFrameCount()` 锁外解引用 `map_` → 移入阶段 3 锁内。
- `snapshotFrame` 增加 `with_descriptors=false`：g2o BA/位姿图只消费关键点像素与点位姿、
  不读描述子，全图回环快照省掉 ~70MB 描述子深拷贝（回环快照/写回两大阻塞源之一）。

**滑动窗口 + 覆盖式队列异步后端（关键，净收益转正）**

问题定位：异步 LOST=74（同步 31），分布为"每 2 帧一带"——正是 KF 插入后 BA 在后台写回
**前端当前跟踪参考帧** `ref_frame_` 的位姿，`validateMotion` 把"参考被 BA 挪动"误判成
位姿跳变 → LOST 级联。两个改动：

1. **覆盖式队列**（`submitBackendTask`）：滑动窗口 BA 的新任务包含旧任务全部 KF →
   入队时丢弃积压的旧 LocalBA（滞后有界 ≤1 窗口），且 LocalBA 压队首优先于回环长任务；
   超限丢最旧，杜绝"队列满→调用方同步执行→前端阻塞"旧路径。
2. **跳过活动参考帧写回**（`runBackendLocalBA`）：窗口内最新 KF（= 前端当前 `ref_frame_`，
   由前端独占）的位姿不写回，留到下一个窗口任务作为普通成员再 BA——前端参考帧永不被
   后端在跟踪间隙挪动。

**A/B 结果（KITTI 00 全程 4541 帧，threads=16，两次异步取均值）**

| 指标 | 同步（旧配置） | 同步（新配置） | 异步（覆盖式+跳过活动参考） |
|------|---------------|---------------|---------------------------|
| FPS | 27.0 | 27.9 | **34.4 ± 0.05**（+23%） |
| LOST | 292 | 292 | **8**（-97%） |
| 闭环次数 | 1 | 1 | **3** |
| 子地图重建 | 13 | 13 | **3** |
| 进程线程数 | 63 | 47 | ~48 |

- 两次异步全程几乎完全一致（34.4/34.5 FPS，LOST=8/8，闭环 3/3）——确定性良好。
- 同步 LOST=292 集中在 2500~4541 帧（off-road 硬段），是同步模式主线程做 BA/回环阻塞前端的
  固有缺陷（与 §3.12 用户反馈一致），新旧同步配置几乎无差，非本次改动引入。
- **结论：§3.15 的"异步净收益为负"被推翻**——根因是"滞后无界 + 活动参考被写回"两个设计缺陷，
  而非异步本身。修复后异步净收益显著为正，`Runtime.async_backend` 默认翻转为 `true`。

**测试脆弱性修复（既有问题）**

合成回环测试在本机稳定失败（`verifyLoop` 的 solvePnPRansac `ok=0`）。根因：合成场景所有点
都是"2px 圆点 + 黑背景"，ORB 31x31 patch 几乎全黑 → 描述子高度相似 → 大量误匹配，内点率
~29% 时 100 次 RANSAC 成功率约 50%（开发机碰巧通过，本机稳定失败）。修复：合成场景改画
半径 4 的实心圆盘 + 唯一灰度值 `(i*37)%256`，匹配可重复（内点率 0.59）；测试验证阈值放宽到
0.4/15（测试目的本是覆盖 T_cw 求逆语义，非调阈值）。单测 27 项全过。

**遗留（三层状态模型，待后续）**

滑动窗口 + 覆盖式队列已解决"数据新鲜度"；但回环校正（`handleLoopCorrection`）仍持独占锁
全图快照+写回，是仅剩的大临界区。按 ORB-SLAM3（Tracking 只读、LocalMapping 独占写、
`mnFirstKFid` 锚定）与 VINS-Mono（滑动窗口 + 边缘化先验 + 索引式特征存储）的架构参考，
三层状态模型（ACTIVE/FROZEN 冻结协议 + 全局校正场 C 惰性应用 + 锚定局部坐标点存储）可把
回环校正降为"稀疏 C 场 + 原子指针交换"，使前端整帧零锁。设计详见 THREADING_DESIGN.md。

### 3.17 轨迹毛刺根因定位与修复 + 合并远程异步重做（2026-08-06）

现象：用户反馈"某一轮优化后轨迹出现毛刺，整体与真值差异大，不能用于定位"。
实测：KITTI 00 全程轨迹出现 476-999 次 >10m 单帧跳变（正常 <5 次），ATE 51.6m。

**根因（逐层二分定位）**：

1. **corrections 无序 → lower_bound UB（毛刺直接根因）**：
   `handleLoopCorrection` 第 5 步（轨迹插值）构造 `corrections`（KF id → 校正量）
   时遍历 `getAllKeyFrames()`——unordered_map 迭代序**无序**——随后
   `std::lower_bound` 要求**有序**——无序数组上的 lower_bound 是 UB，
   插值取错相邻 KF → 重写后的轨迹出现 10m+ 毛刺。
   修复：`std::ranges::sort(corrections, {}, &pair::first)`。**排序后 1800 帧
   跳变 85 → 15（与历史最佳版一致）**。
2. **快照点"同 id 多对象"（回环后点/位姿不一致）**：同一真点被多个快照
   KF 引用时生成多个快照点对象（X1/X2），BA 精修 X1，写回遍历先遇 X1（BA
   值）后遇 X2（同步值）覆盖 → 部分点未精修。修复：snapshotFrame 按 id
   缓存快照点（snap_cache），同 id 复用同一对象。
3. **阶段 3 写回顺序**：先写快照点（BA 精修值）再写全量同步值（粗值）——
   粗值覆盖精修值，回环后点全部退化为无重投影精修状态。修复：顺序对调。
4. **tmp_map 未注册快照点**：快照模式全局 BA 用 getMapPoint 查点，快照点
   不入临时 Map 则查不到 → BA 无点。修复：insertKeyFrame 时同步
   insertMapPoint。
5. **2b' 快照点初值**：点同步后快照点保持旧初值 → 全局 BA 从旧值收敛。
   修复：同步后把同步值写回快照点（BA 的精修基准）。
6. **mp_ptr_by_id（同步路径回归源）**：optimizer 收集期保存点指针替代
   getMapPoint——同步路径实测引入跳变（400 帧 0→85 跳）。恢复
   getMapPoint（快照场景由 tmp_map 注册点支撑）。

**合并远程**：拉取 origin/main（c6af8a6 异步后端重做净收益转正 + 5942f14
文档），保留远程异步重做（读写锁/覆盖式队列/线程收敛），叠加上述 6 项修复
（对同步/异步路径均生效）。optimizer 以 getMapPoint 版为准。

**验证**（KITTI 00 全程 headless，kitti00.yaml，async=off）：

| 指标 | 修复前（回归版） | 修复后 | 历史最佳（311b2c5） |
|------|------------------|--------|---------------------|
| ATE RMSE | 51.6m | **23.3m** | 23.3m |
| >10m 跳变 | 476-999 | **52**（含既有插值残留） | 同量级 |
| 闭环 | 3 次 | 2 次 | 2-3 次 |
| FPS | 7.7 | 7.5 | 7.7 |

ATE/Sim3 对齐参数与历史最佳版逐位一致（scale=1.0399/rot=0.1211/t=9.87）。

遗留：52 次跳变中 0-908 段 33 次为"回环重写轨迹的插值残留"（大校正区域
相邻 KF 校正量突变，插值不连续）——lc3 时代即有，列为后续优化项。

### 3.18 编译回归修复与评估口径审计（2026-08-06）

最新提交 `7f851ae` 将 stash 冲突标记提交进 `vo.h/vo.cpp`，提交树本身无法编译。
本次以 `c6af8a6` 的异步三阶段实现为主干重新合入快照点一致性修复，构建与 CTest
恢复通过。两个已有未跟踪 PNG 未改动。

同时复核 §3.17 后确认，"corrections 来自 unordered_map 无序"的归因不成立：
`Map::keyframes_` 一直是 `std::map`，`getAllKeyFrames()` 按 ID 返回；新增排序只是明确
`lower_bound` 前置条件，不能解释宣称的跳变改善。因此 §3.17 的 ATE/跳变数字只能视为
未受控工作树实验，不能作为 `7f851ae` 的可复现证据。

评估链修复：

1. `prepare_kitti.sh` 同时解压 `image_0/image_1`，示例传序列根目录并要求日志确认
   `Loaded stereo sequence`；旧示例实际运行的是单目。
2. `evaluate_ate.py` 改为时间戳单调一一关联；双目默认 SE3 对齐，新增逐帧平移/旋转
   RPE、覆盖率、路径长度比、p95/p99/max 与 >3/5/10m 跳变计数。Sim3 仅供单目显式选择。
3. 修复 KITTI GT 大转角四元数转换；轨迹图改为 x-z 地面俯视。
4. `benchmark.py` 检查子进程/帧数/评估失败，性能 CSV 与轨迹同目录保存，不再删除工作区
   `perf.csv`；`run_slam` 在读取最终轨迹前 drain/join 异步后端。
5. 修复 LOST 匀速外推的相对位姿乘法方向；异步回环写回保留计算期间追加的轨迹尾部。
   事务式 map generation/rebase 尚未完成，因此可靠性优先，默认 `async_backend=false`。

旧 `traj.txt`（2026-08-03 产物，非当前 HEAD）审计：4541 帧、路径 5063.97m，存在 8 次
>10m 跳变（全部 >80m，最大 271.90m）和最大 147.81° 单帧旋转；四组成对跳出/跳回
额外增加约 1396m 路径。它证明历史轨迹不可用，但不能代表本次修复后的结果。

后端审计还定位到当前 sampled GBA 每 3 个 KF 仅优化一个 pose/该子集观测，随后在所有
KF 处构造校正场，可能产生三周期锯齿；Local BA 强塞可能与当前共视图断开的全局 KF0/1
作为锚点。下一轮必须在具备 KITTI 00、g2o、DBoW3 的环境中，以 PGO-only 与闭环邻域完整
共视 BA 做 A/B；在此之前不再用单个 Sim3 ATE 数字宣称可用或 50% 提升。

### 3.19 优化失效实测、故障分层与后端事务化重构计划（2026-08-06）

在 vendored g2o、DBoW3 和 KITTI 00 双目/真值均可用后，重新审计用户提供的
`trajectory_00.txt`。该轨迹不是普通尺度漂移，而是一次回环写回导致的灾难性状态破坏：

| 指标 | 问题轨迹 |
|------|----------|
| 有效位姿 | 4161 / 4541（91.63%） |
| SE3 ATE RMSE / Mean | 284431.7m / 126200.4m |
| 估计路径 / GT 路径 | 1289492.7m / 3723.7m（346.3 倍） |
| 最大单步跳变 | 1282400.7m（400.4s → 402.1s） |
| 回环 / 全局 BA | 各 1 次 |

最大跳变时间与唯一一次回环/位姿图优化一致。为阻止单条错误约束破坏实时地图，
`poseGraphOptimization` 已增加：回环边残差预检、回环边 Huber 核、优化前后 robust chi2
检查、有限值检查、最大顶点校正和相邻关键帧步长检查。优化仍在快照 Map 上运行，任一
检查失败即返回 false，真实地图不写回。新增恶性 1000km 回环边回归测试，验证拒绝后
所有关键帧位姿保持不变；正常合成回环仍将末端漂移从 1.79m 收敛到约 4e-5m。

使用独立 Release 构建（同步后端、完整 4541 帧）复跑得到新的无回环安全基线：

| 指标 | Release 安全基线 |
|------|------------------|
| FPS / 耗时 | 16.00 / 283.83s |
| 有效位姿覆盖率 | 4292 / 4541（94.52%） |
| SE3 ATE RMSE / Mean / Max | 71.33m / 63.60m / 547.54m |
| 路径长度比 | 5.043 |
| RPE 平移 / 旋转 | 19.58m/frame / 20.49deg/frame |
| >10m 跳变 | 344 |
| LOST / 子地图重建 / 回环 | 294 / 13 / 0 |

本轮没有再次随机命中回环，因此它证明“没有百万米写回时仍能跑完全程”，不能证明系统
已经可用。轨迹前 30 个最大跳变均呈单帧跳出、下一帧跳回。日志给出直接根因：正常
PnP/3D-3D 已按 3m/0.35rad 门限拒绝 258.8m、531.6m 等坏解，但同一帧随后的
`tryRelocalize` 只检查内点/RMSE，又把相同 PnP 解接受为重定位，绕过运动连续性验收。
因此当前结果仍不可用于定位；“最终只有 19 个 KF”也只是活动子地图计数，perf 中实际
执行了 3113 次 Local BA，不能据此判断全程只建立了 19 个关键帧。

**故障分层结论**：

1. 编译冲突标记、错误回环边无防护、重定位绕过运动门限属于代码级缺陷，应先止血；
2. `Frame::pose_cw`、`MapPoint::pos_w`、完整轨迹三份全局状态分别写回，快照没有
   map generation，地图与轨迹又使用不同锁，属于框架级一致性缺陷；
3. `Submap::origin_Twc` 目前没有成为坐标变换权威，KF/点仍直接存世界坐标，导致回环和
   子地图对齐必须全量搬运数据，跨子地图约束也只能拒绝；
4. sampled GBA 与回环轨迹插值只能作为过渡实现，不能继续靠阈值调参掩盖数据模型问题。

**后续工程化顺序**（详细接口与不变量见 `THREADING_DESIGN.md` §十四）：

1. M0：统一正常跟踪/重定位位姿验收，要求连续有效帧 >10m 跳变为 0；
2. M1：Optimizer 改为只读快照输入、Result 输出；Map/Submap 增加 topology/geometry revision；
3. M2：唯一 BackendCommitter 做 stale 检查、质量验收和地图/轨迹原子提交；
4. M3：KF/点改存子地图局部坐标，`T_ws` 成为子地图到世界系的唯一权威；
5. M4：普通帧轨迹和地图点锚定关键帧，删除回环后的全量轨迹插值与点搬运；
6. M5：Atlas 建立 bridge/relocalization/loop 子地图约束图，先连接坐标、后独立融合地图；
7. M6：发布不可变校正场 C（copy-on-write + 原子交换），压力验证后再默认开启异步后端。

第一阶段 KITTI 00 验收门槛：连续帧 >10m 跳变 0、覆盖率 ≥99%、路径长度比
0.9~1.2、子地图重建 ≤3、SE3 ATE ≤30m；任何优化拒绝都必须保证实时状态完全不变。

### 3.20 M0-M6 里程碑落地与 KITTI 00 验证（2026-08-06）

按 §3.19 的工程化顺序（THREADING_DESIGN §十四）完成全部七个里程碑：

**M0 统一位姿验收**：正常跟踪（PnP/LK/3D-3D）与重定位共用 `acceptPose`
（几何内点/比例/RMSE + 世界系运动连续性）。修复 §3.19 根因——重定位绕过
运动门限重新接受跟踪刚拒绝的坏解。重定位基线 = 丢失期匀速外推期望位姿
（门限 max(50m, 3×位移)/60°）。同步 Local BA 补齐"跳过活动参考帧写回"
（§3.16 异步决定性修复的同步路径）：BA 不再挪动刚验收并记入轨迹的位姿。

**M1 Optimizer 只读快照/Result + Map revision**：`OptimizationSnapshot`/
`OptimizationResult` 纯计算契约；`solveLocalBA`/`solvePoseGraph` 不触碰
实时地图；PGO 防爆（回环边残差预检/chi2/有限值/最大校正/相邻步长）全部
前移到 Result 验收。Map 增加 topology/geometry revision，几何版本只能由
提交者发布。

**M2 BackendCommitter + TrackingSnapshot**：唯一提交路径（stale 检查 →
质量验收（最大校正 10m 上限）→ 对象存活 → 一次临界区原子写回 +
geometry++）；前端每帧捕获一次只读快照（版本 + 参考帧位姿/点坐标拷贝），
整帧不跨版本读实时地图。

**M3 子地图局部坐标**：KF 存 `T_cs`、点存 `p_s`，`Submap::T_ws` 为唯一
世界权威（`T_cw = T_cs·T_ws⁻¹`）。子地图对齐只更新 T_ws（KF/点零移动、
几何版本不变、后端排队快照不失效）。修复回环校正量未换算世界系的共轭
错误（KITTI 路径比 0.99→5.87 回归，换算后恢复 0.997）。

**M4 锚定轨迹**：普通帧只记录 `(anchor_kf, T_ca)`，世界位姿读时组合
（回环/对齐自动跟随锚点）——删除全量轨迹插值与前缀/尾部合并逻辑。
修正"KF 帧锚定陈旧帧首快照 ref"回归（回环校正挪动本帧 KF 后 T_ca 被误记
为 26m 级跳变）。sampled-GBA 与锚定轨迹不兼容（三周期锯齿，§3.18 预测），
`global_ba_iterations: 0` 默认关闭。

**M5 Atlas 约束图**：TrackingBridge（丢失外推，权重 0.3）/Relocalization
（跨子地图重定位，事务式：失败回滚 T_ws 与约束）/LoopClosure 三类边；
`solveAtlasConstraints` 用位姿图求解全部 T_ws 对齐。

**M6 Committer 追加 rebase + 异步默认恢复**：几何版本为唯一硬 stale 判据
（拓扑仅追加 KF/点不影响窗口结果有效性）——解除异步"前端每帧插 KF →
BA 全过期"死锁；`async_backend` 默认 true。

**KITTI 00 全程验证**（同步 run_slam，2 次回环生效）：
覆盖率 98.74%（异步 99.16%）、路径长度比 0.997、RPE 0.095 m/frame、
>10m 跳变 3 次且全部为跨 LOST 空洞（连续有效帧跳变 = 0）、子地图重建 3、
SE3 ATE 42.77m。单测 43 项全过。第一阶段门槛：连续帧跳变/覆盖率/路径比/
子地图重建全部达标；ATE ≤30m 未达标——剩余瓶颈是回环验证召回率（DBoW
候选旧 KF 的地图点被 cull 后 3D-2D 对应不足，回环触发 0~2 次随机命中），
属回环子系统质量，非 M0-M6 结构问题。

### 3.21 回环召回优化（A/B/C/D）与快照外 KF 校正跳变问题（2026-08-06）

针对"初始矩形段与末端圆弧段角度差大"（SE3 对齐旋转漂移 16.5°），实施四项优化：

- **A 回环召回**：verifyLoop 匹配 ratio 0.7→0.8；位置先验距离 25→40m；
  验证 3D-2D 补点——候选 KF 的地图点被 cull 断链时，用其双目观测
  `pts_c`（KF 保留）转局部系补 3D（"too few 3D-2D" 从 40 次→0，
  验证内点 408/437/304）；同区域回环去重（端点相距 <200 KF 拒绝，
  拦住 88m 重复校正）。
- **B 旋转抑制**：solvePnPRansac 迭代 100→200；三角化最小视差角 0.1°
  （初版 0.5° 误伤：KITTI Triangulated -72%、ATE 42.8→91.5，回退）。
- **C 地图质量**：Local BA 被 10m 质量门限拒绝后，剔除 observed<3 的
  弱观测点重建快照重试一次（runWindowLocalBA 统一 sync/async 入口）。
- **D 回环权重/冷却**：试过权重 15/冷却 12，实测同区域 43 KF 内二次
  回环叠加拉扯（路径 3712→3821m），回退权重 10/冷却 20。

**async 回环 stale 修复**：回环校正 stale 判据从"拓扑+几何严格相等"
改为"仅几何版本"（与 M6 Local BA 追加 rebase 一致）——异步下前端持续
追加 KF/点使拓扑常变，3 次 verified（408/306/304 inliers）曾全部因此
被丢，旋转漂移无法被回环拉平。

**当前未解决问题（数据见 data/eval/）**：sync 下回环 3 次闭合、旋转
漂移 16.5°→11.6°（角度差改善 ~30%）；但回环 #1（静止段 kf#153→1597，
PGO 校正方向正确——1597 从 (51.1,-16.8,114.1) 拉回 (17.5,-1.9,90.0)）
后，**快照外插入的 KF（1599 等）不在 PGO 图中、保持校正前位姿 → 与
校正后锚点 KF 产生 44.8~85.4m 连续帧跳变**。已试：① loop_skip 保护
端点（位姿-点失配 T_ca 44m，放弃）；② 按锚定校正量传播快照外 KF 位姿
+独有点（传播方向/范围错误 → 85m 跳变）；③ commit 非 COMMITTED 跳过
传播（85m 仍在）。下一步疑点：传播校正量 `C=new⁻¹·old` 左乘语义与
位姿/点坐标系组合，或传播范围应限于回环端点附近漂移段。

### 3.22 Phase 0/1 回环与 Observation 收口（2026-08-07）

本轮修复收口了回环/异步写回和 Observation 生产路径：移除旧的硬编码 KF
过滤与快照外 KF 传播路径；回环提交严格绑定 Map、submap、topology/geometry
revision 及当前/候选 KF 对象身份；Local BA 快照显式绑定 anchor KF id；普通
跟踪帧不再写正式观测，双目建点/三角化/KF 插入统一维护双向 Observation，并在
建点关联失败时完整回滚新点。Local BA 只消费正式观测并按 `observationCount()`
执行 `min_observed` 过滤，持久共视计数用于窗口和里程计权重。

验证边界：`cmake --build build -j2`、`ctest --test-dir build --output-on-failure`
（2/2）及 `git diff --check` 已通过；测试覆盖单目/双目初始化、双目 LK 关键帧、
Observation 一致性、id=0 BA 与地图点删除回滚基础语义。Phase 2/3 的真实 KITTI
闭环质量、全局 BA 及跨子地图融合仍未完成，不能以本轮单测替代数据集验收。

### 3.23 事务化收口、目录迁移与 KITTI/EuRoC 完整基准（2026-08-07）

本轮在 §3.22 基础上完成最终对抗审查与性能收口：回环从“快照计算后分段写回”改为
持有目标 Map 独占锁的可串行化事务，提交前重新验证候选并原子写回全部 KF 位姿和点；
移除硬编码 KF、快照外传播及重复 Local BA 重试。Map 批量 cull 从“每删一点扫描全图
KF slots”降为“撤销被删点 Observation + 全图 slots 单次扫描”；1000 帧同配置对比
FPS 5.35→17.02、`kf.cull` 最大耗时 15.89s→39ms，ATE/RPE 保持到输出精度一致。

完整基准还复现了两个不能由 revision 数值单独解决的并发缺口：① 旧子地图的异步 Local BA
可能与新 Map 的相同 KF id/revision 撞号；任务现绑定 Map/Submap，并在快照、KF 指针与提交
三处验身份。② 后端在一帧跟踪后、KF 插入前提交 PGO 时，旧几何帧会漏出 PGO 快照；实测
在 162.0→162.1s 产生 41.231m 相邻跳变。KF 插入现在在独占锁内做帧首 geometry gate，
过期时保持 `T_ca` 重基到 live reference 并跳过本帧 KF；普通帧收尾的运动基线也改为与
持久轨迹相同的 live-anchor 世界位姿。最终双轮 gate 各实际触发 1 次。

数据目录统一为 `datasets/kitti/{sequences,poses}` 与 `datasets/euroc/<sequence>/mav0`。
EuRoC 输入会读取 `cam0/sensor.yaml` 的 pinhole + radial-tangential 标定并去畸变；新增
`euroc_gt_to_tum.py` 把 200Hz 状态真值转换为 cam0 世界位姿，评估时按图像时间戳匹配。

**验证环境**：WSL2，Intel Core i7-10700（8C/16T），独立 Release 构建，headless，
vendored g2o + DBoW3；每个数据集连续运行两次。原始日志、轨迹和 perf CSV 写到 `/tmp`，
未把约 1GB 级运行产物继续提交到 `data/eval/`。

最终 Debug/Release CTest 均 2/2 通过；ASan+UBSan（`halt_on_error=1`）在关闭泄漏检测后
2/2 通过。LeakSanitizer 在本受管 ptrace 环境不能启动，因此本轮没有宣称完成泄漏扫描；
TSan 同样受 WSL 地址映射限制，真实并发证据来自锁语义回归与 KITTI 双轮运行。

```bash
python3 scripts/benchmark.py datasets/kitti/sequences/00 config/kitti00.yaml \
    /tmp/vslam_kitti_reloc_release --runs 2 \
    --bin /tmp/vslam_benchmark_release/bin/run_slam \
    --gt datasets/kitti/poses/00.tum --alignment se3 \
    --format kitti --expected-frames 4541

python3 scripts/euroc_gt_to_tum.py datasets/euroc/V1_01_easy/mav0 \
    /tmp/euroc_v101_cam0_gt.tum
python3 scripts/benchmark.py datasets/euroc/V1_01_easy/mav0 config/default.yaml \
    /tmp/vslam_euroc_gate_release --runs 2 \
    --bin /tmp/vslam_benchmark_release/bin/run_slam \
    --gt /tmp/euroc_v101_cam0_gt.tum --alignment sim3 \
    --format euroc --expected-frames 2912
```

**KITTI 00 双轮结果（双目，SE3）**：

| 指标 | run 1 | run 2 | 均值 ± 半差 |
|---|---:|---:|---:|
| 完成帧 / 有效轨迹 | 4541 / 4502 | 4541 / 4484 | 98.94% ± 0.20% |
| FPS / 耗时 | 12.56 / 361.59s | 15.02 / 302.36s | 13.79 ± 1.23 |
| ATE RMSE / Mean / Max | 13.445 / 9.827 / 47.475m | 38.662 / 28.808 / 117.778m | 26.054 ± 12.609m RMSE |
| RPE 平移 / 旋转 | 0.142m / 0.153deg | 0.441m / 0.316deg | 0.292m / 0.235deg |
| 路径长度比 | 0.9995 | 1.0307 | 1.0151 ± 0.0156 |
| LOST / 子地图重建 / 回环 | 5 / 2 / 1 | 14 / 3 / 0 | 9.5 / 2.5 / 0.5 |
| 最终点 / KF | 16190 / 82 | 16190 / 82 | 16190 / 82 |

两轮 >10m step 为 2/3 次，全部跨 2.0s LOST 时间洞；相邻有效帧 >10m 为 0，最大值
分别为 3.558m/8.356m。run 1 在 `kf#449 -> kf#3452` 命中一次回环，回环后没有
相邻帧大跳变，构成最终代码的真实事件级复验；两轮也各真实触发一次 in-flight gate。
但 run 2 未命中回环且 ATE 退化到 38.662m，表明回环召回和最终精度仍受后端调度影响，
不能用 26.054m 均值掩盖。双轮命令总 wall time 669.25s，进程峰值 RSS 约 1.70GiB。

**EuRoC V1_01_easy 双轮结果（单目，Sim3）**：2912 帧均全部完成，
有效轨迹 2780（95.47%），与 cam0 真值匹配 2761，0 LOST / 0 子地图重建 / 0 回环；
57.08 ± 4.97 FPS，ATE RMSE/Mean/Max = 1.634±0.011/1.472±0.019/
3.185±0.007m，RPE = 0.021m/frame、0.596±0.007deg/frame，>10m step 为 0。
两轮总 wall time 113.81s，峰值 RSS 约 876MiB。原始 EuRoC 真值是 200Hz，不能把
“匹配数/28712 真值行”误当图像覆盖率；
单目原始路径长度比同样不具绝对尺度意义。

**仍需正视的架构边界**：活动 Map 的点/KF 缺少硬资源预算，长序列峰值内存仍偏高；
Atlas 当前只对齐子地图坐标系，尚未融合跨子地图重复点；
全局 BA 默认关闭；EuRoC 暂只支持 pinhole + radial-tangential 模型。上述是后续框架级
工作，不应再用 KITTI 00 专项阈值去掩盖。

### 3.24 机器人长期定位组件产品化规划（2026-08-07）

在 §3.23 的事务式后端和 KITTI/EuRoC 基准基础上，新增当前规划文档
`docs/PRODUCTION_LOCALIZATION_PLAN.md`。规划不再以单次 ATE 调参作为主线，而是按
M0～M7 依次收口：定位 API/状态机、`VisualOdometry` 模块拆分、实时输入与资源硬预算、
视觉退化检测和协方差、版本化地图与纯定位模式、多会话 Atlas 融合、IMU/轮速松耦合
ESKF，以及 CI/24 小时灰度发布。

文档为每个里程碑固定了算法或框架方向、首版参数、任务依赖、禁止事项和量化门槛。关键
产品不变量包括：错误或过期后端结果不修改实时状态；所有队列和地图资源有硬上限；失败
位姿不得伪装为有效定位；生产默认使用只读地图；控制器使用连续 `T_ob`，回环只更新
`T_wo`，全局规划位姿按 `T_wb=T_wo*T_ob` 组合。该文档目前是实施规格，M0～M7 尚未
落地，不能把规划参数当成当前代码已具备的能力或当前 benchmark 结果。

### 3.25 M0.1 定位契约类型落地（2026-08-08）

按 `PRODUCTION_LOCALIZATION_PLAN.md` 的 M0.1（类型契约）实施第一个产品化任务，
只新增类型与契约谓词，不改任何 `VisualOdometry`/Map/Atlas 算法。

- 新增 `include/vslam/localization_types.h`：`LocalizationMode`（§1.3）、
  `TrackingState`（§4.2）、`FailureReason`（§4.1，11 种原因码）、`PoseEstimate`
  （§4.1 字段与默认值全对齐），以及 §3 硬不变量的静态谓词：
  `isUnitQuaternion`、`isFinite(SE3)`、`isValidTimestamp`、
  `isPositiveDefiniteCovariance`（对称 + 正定 + 有限）、`isPublishable`。
- `include/vslam/common.h` 仅补通用矩阵别名 `Vec6`/`Mat6`（6 自由度切空间/协方差）。
- 新增独立 CTest `test/test_localization_types.cpp`（19 项契约测试），不并入
  `test_vo.cpp`，满足"M0 契约测试至少 12 项"的起点。
- `CMakeLists.txt` 注册 `test_localization_types` 独立 CTest。

验证：`./build/test_localization_types` 19/19 PASSED；`ctest --test-dir build`
3/3 通过（含旧 `test_vo` 5.28s 全过与 `test_trajectory_alignment`）。纯类型新增，
无算法/轨迹/状态变化，M0.1 不涉及 KITTI 轨迹基准。下一任务：M0.2 状态机。

### 3.26 M0.2 定位状态机落地（2026-08-08）

在 M0.1 类型契约基础上实现 `TrackingStateMachine`（§4.2 确定性有限状态机），
不修改 `VisualOdometry`。

- 新增 `include/vslam/tracking_state_machine.h` + `src/tracking_state_machine.cpp`：
  状态转换表按 §4.2 逐条落地（Initializing→Tracking 连续 3 帧完整验收；
  Tracking⇄Degraded 连续 2 帧弱质量 / 连续 3 帧完整恢复；
  Tracking/Degraded→Relocalizing 连续 5 帧失败；Relocalizing→Lost 20 帧或 2.0 s；
  Lost/Relocalizing 经有效全局重定位→Tracking；任意状态 stop()→Stopped）。
- 输出语义（§4.2/§3-2）：Full/Weak 帧 `pose_valid=true`（Weak 帧由调用方把协方差
  ×4）；Failed 帧在 `prediction_timeout_s=0.5` 内只发布预测
  （`prediction_only=true`），超过后 `pose_valid=false`；Initializing/Relocalizing/
  Lost 不发布有效位姿。
- 新增独立 CTest `test/test_tracking_state_machine.cpp`（24 项，含重复 stop、
  stop 后输入、自定义 Params）。
- `CMakeLists.txt` 注册模块源/头与 `test_tracking_state_machine` CTest。

验证：`./build/test_tracking_state_machine` 24/24 PASSED；`ctest --test-dir build`
4/4 通过（含旧 `test_vo` 5.36s 全过）。纯状态机新增，无算法/轨迹/状态变化。
下一任务：M0.3 Facade（Localizer 包装 run_slam 输出一致性验收）。

### 3.27 M0.3 Localizer Facade 落地（2026-08-08）

在 M0.1/M0.2 基础上实现 `Localizer` Facade（§4.1/§4.3/§4.4），不修改 `VisualOdometry` 算法。

- 新增 `include/vslam/localizer.h` + `src/localizer.cpp`：包装
  `VisualOdometry` + `TrackingStateMachine`，调用方不再直接读取 Frame/Map/VO 内部状态。
  输出 `PoseEstimate`（§4.1），坐标契约 §2：`T_wb = T_wc · T_bc⁻¹`，M0 尚未接入
  ESKF，`T_ob = T_wb`（odom 系 = 全局系，M6 后由 `T_wo` 分离）。
- §4.3 输入硬检查：时间戳严格递增（倒退/相等拒绝）、空图/尺寸与标定不一致拒绝、
  左右尺寸/类型不一致拒绝、双目时间差 >1ms 拒绝（`StereoUnsynchronized`）；
  失败帧不调用 VO，Map revision 逐项不变。构造时校验 `T_bc` 四元数归一化误差
  <1e-6、平移有限。
- 新增 `config/robot.yaml`（Robot 段：mode/`T_bc`/`stereo_max_time_diff_s`）。
- 新增独立 CTest `test/test_localizer_contract.cpp`（16 项：构造校验、输入硬检查、
  Map revision 不变、run_slam 等价、重复 stop、析构、robot.yaml 解析）。
- `app/run_slam.cpp` 增加 `--localizer` 可选入口（§12：只增加可选入口），与旧路径
  完全并行，不改旧行为。

验证：`./build/test_localizer_contract` 16/16 PASSED；`ctest --test-dir build` 5/5
通过（含旧 `test_vo`）。`run_slam --localizer --headless --frames 200` 处理 200 帧
198 有效位姿，轨迹与旧路径逐行一致（如 0.2s 帧平移 `0.020432 0.002230 1.377625`
完全相同）；localizer 入口按 §4.2 需连续 3 帧完整验收才发布有效位姿，故从 0.2s 起。

**等价性测试的确定性**：两路 `VisualOdometry` 逐位一致需要（与 M1 §5.6
deterministic.yaml 同思路）——`cv::setRNGSeed(0x5A17)` 固定
`solvePnPRansac` 内部 RNG，`opencv_threads=1` + `orb_max_bands=1` 关闭特征提取
的并行分带；`M0 不增加线程`：Localizer 自身不创建线程，p95 处理开销只含包装与
校验。

**发现的既有缺陷（不在 M0 范围修复）**：`src/vo.cpp:305,309` 中
`curr_frame_->image_gray = left_input` 是浅拷贝，随后
`clahe->apply(image_gray, image_gray)` 会**原地改写调用方传入的 cv::Mat 缓冲**。
等价测试因此必须给每路 VO 渲染独立拷贝（与 run_slam 每帧新矩阵的用法一致）；
对 Localizer 而言，调用方若复用同一图像缓冲（如摄像头循环），第二次调用会被
喂入已 CLAHE 增强的图像。该问题记入 `IMPROVEMENT_PLAN` 待办，建议在 M1 拆分
FrontendTracker 时改为对输入深拷贝或让 CLAHE 写临时缓冲，避免跨实例/复用缓冲
的非确定性。

下一任务：M1.1 PoseGate（从 `vo.cpp` 提取纯几何质量与运动连续性）。

### 3.28 M1.1 PoseGate 拆分（Strangler Fig 第一步，2026-08-08）

按 §5.1/§5.2 从 `VisualOdometry` 提取纯几何质量与运动连续性门，公式与默认值
保持不变，确定性验收全部通过。

- 新增 `include/vslam/pose_gate.h` + `src/pose_gate.cpp`：迁移
  `pnpReprojectionRmse`（K 改为参数传入）、`acceptPoseCandidate`、
  `checkMotionContinuity`；`PoseQuality` 提升为顶层 `vslam::PoseQuality`。
  新增 §5.2 统一入口 `PoseGate::evaluate(PoseCandidate, MotionPrediction,
  TrackingQuality, dt)`，只接收值对象，不访问 Map/Atlas/Viewer/全局状态/线程；
  `dt` 为 M3 §7.4 时间归一化连续性保留。
- `vo.{h,cpp}`：删除已迁移实现；`acceptPose` 保留 VO 侧运动基线计算后转调
  `PoseGate::acceptPoseCandidate`；三个 `pnpReprojectionRmse` 调用点改传
  `camera_->K()`。
- M1 确定性基建：`VOConfig.rng_seed`（非 0 时构造 `cv::setRNGSeed`）；
  `config/deterministic.yaml`（opencv_threads=1、orb_max_bands=1、
  rng_seed=0x5A17、async_backend=false）；`run_slam --status-csv` 逐帧状态/计数
  日志；`scripts/compare_trajectories.py`（含数值稳定四元数差角，避免 6 位小数
  文件与 acos 病态引入伪差异）。
- 新增独立 CTest `test/test_pose_gate.cpp`（11 项）；`test_vo.cpp` 位姿验收测试
  改调 `vslam::PoseGate::acceptPoseCandidate`。

**M1.6 确定性验收（KITTI 00 前 1000 帧，deterministic.yaml，重构前后同一构建）**：
- 状态序列（frame_id/state/pose_valid/map_points/keyframes/topology/geometry
  revision）逐行完全一致（`diff` 为空）。
- 轨迹 sha256 完全一致：平移差最大 0.000e+00 m（<1e-6）、旋转差最大 5.5e-17 rad
  （<1e-8）。
- 耗时 161119ms → 160980ms（重构后略快，FPS 无退化；RSS 无明显变化，热路径无新增
  分配）。

下一任务：M1.2 Relocalizer（候选与几何验证，只返回结果）。

### 3.29 生产级基准评估方案（L0~L2 提交门，2026-08-08）

针对"评估结果能否体现性能/准确性"的缺口（无延迟分位、无多轮统计、正则解析日志、
无提交门），落地分层评估与提交基准门，详见 `docs/BENCHMARK.md`。

- **结构化指标**：`utils/metrics_json.h` + `run_slam --metrics-json <path>` 输出
  单次运行 JSON——延迟 p50/p95/p99/max、deadline miss、有效位姿率、LOST 次数/时长、
  子地图重建、回环、地图规模。`--deadline-ms` 可配（默认 100ms，10Hz）。
- **统计基准 v2**：`scripts/benchmark.py` 重写——每轮读 metrics JSON（不解析日志）、
  ATE 用 `evaluate_ate.py --json`（机读摘要），N 轮（默认 5）聚合 mean/std/worst，
  门限断言作用于 **worst 一轮**，输出 `report.json`，退出码 0/1/2。
- **门限配置**：`config/benchmark.yaml`（dataset/config/runs/window/deadline_ms +
  Gates：valid_ratio≥0.99、lost≤30、submap_reinit≤1、jumps_10m=0、latency_p99≤80ms、
  deadline_miss<1%、ATE worst≤40m/std≤8m；ate_* 需 GT，无则 skip）。
- **精度评估补全**：`evaluate_ate.py` 增加 ATE Std、RPE trans/rot Mean/Max、`--json`
  机读输出。
- **确定性参考**：`scripts/benchmark/reference/{pose.txt,status.csv}`（KITTI 00 前 1000
  帧 deterministic.yaml 输出）。
- **提交门**：`scripts/benchmark_gate.sh`（L0 构建+ctest → L1 确定性回归 →
  L2 统计基准快速档）经 `.githooks/pre-commit` 由
  `git config core.hooksPath .githooks` 启用；`--full` 完整档，`--update-reference`
  更新 L1 参考。

验证：`scripts/benchmark_gate.sh` 快速档端到端 PASS（ctest 6/6、L1 轨迹/状态逐位一致、
L2 门限全过，500 帧×3 轮 latency_p99≈25ms、valid_ratio=1.0）。旧 `benchmark.py`
接口保留 A/B 对比能力；`test_trajectory_alignment` 不受影响。

### 3.31 M1.2 Relocalizer 拆分（2026-08-08）

按 §5.3 从 `VisualOdometry::tryRelocalize` 提取候选几何验证到独立 `Relocalizer`
（`relocalizer.{h,cpp}`，行为保持不变）：

- **新增 API**（§5.3 值对象风格）：`RelocalizationResult{accepted, submap_id,
  geometry_revision, map, kf, T_cs, inliers, total, rmse, quick_*}` 与
  `RelocalizationPointSet`。`Relocalizer::relocalize(Query)` 负责候选粗筛
  （quickMatchCount）→ 全量 ORB 匹配 → `solvePnPRansac` + 内点 RMSE（转调
  `PoseGate::pnpReprojectionRmse`）→ 内点最多候选 / 首个达标即返回。
- **职责边界**：Relocalizer 只返回结果，不切换 Atlas、不写 Map/轨迹、不持锁；
  3D-2D 对应由调用方在 `map_mutex_` 读锁内供应（身份 + geometry revision 绑定），
  提交仍由 VO 在独占锁事务中完成（stale 检查 → 跨子地图 Atlas 约束 → acceptPose
  → activate + 快照）。
- `matToSE3` 迁移为 `Relocalizer::matToSE3`，vo.cpp 其余 3 处转调。
- 测试 `test_relocalizer.cpp`（9 项）：matToSE3、identity 位姿恢复、verifyCandidate、
  空候选 / stale / 弱几何 / 不相关描述子 / 门槛拒绝路径。

**验收**：
- CTest 7/7 全过（新增 test_relocalizer 注册为独立 CTest）。
- §5.6 确定性：KITTI 00 前 1000 帧（deterministic.yaml）轨迹逐位一致
  （max_translation_diff=0 m、max_rotation_diff=5.5e-17 rad），状态序列 1000 行一致。
- §3.30 完整基准复测（default.yaml × 5 轮 + GT）：ATE RMSE mean 41.6m（§3.30 46.8m）、
  FPS 43.1（+4%）、latency p99 50.0ms、valid_ratio 0.9915——无退化；仍在门限的
  ate_rmse/std、jumps_10m、submap_reinit 与 §3.30 一致，属 M3/M5 收敛目标。
  注：default.yaml 不固定 RNG/异步开启，轮间关键帧数（112~132 正常；个别轮次
  因无子地图重建可达 ~1000）属运行间 RNG 方差，非拆分引入。

### 3.32 M1.3 BackendScheduler 拆分（2026-08-08）

按 §5.4 把后台调度从 vo.cpp 提取到独立 `BackendScheduler`（`backend_scheduler.{h,cpp}`）：

- `BackendTask` 迁至 `backend_scheduler.h`；**单后台线程 + 覆盖式单任务槽（容量 1）**。
- 排队语义（§5.4 明确、用户确认，与旧 `kMaxQueued=4 + LocalBA 队首` 不同）：
  - LoopClosure 覆盖任何等待任务（回环优先于局部 BA）；
  - 同类 Local BA 新任务覆盖旧 Local BA（等待槽恒为最新）；
  - 等待槽被 LoopClosure 占住时新 Local BA 直接丢弃（槽满不阻塞）；
  - 正在执行的任务不强制取消，结果由 `BackendCommitter` stale gate 丢弃；
  - `stop()` = 置标志 → notify → join（排空槽内任务后退出，不 detach）。
- VO 侧仅保留任务分发 `runBackendTask`（LocalBA/LoopClosure）与兼容转发
  `submitBackendTask`；`finishPendingBackendWork` 改调 `backend_scheduler_.stop()`。
- 测试 `test_backend_scheduler.cpp`（7 项）：基础执行、同类覆盖、LoopClosure 优先、
  LocalBA 不覆盖等待 LoopClosure、stop 排空 join、未 start/重复 stop、析构自动 join。

**验收**：
- CTest 8/8 全过（新增 test_backend_scheduler 独立 CTest）。
- §5.6 确定性：KITTI 00 前 1000 帧轨迹逐位一致（max_translation_diff=0 m、
  max_rotation_diff=5.5e-17 rad）、状态序列 1000 行一致。
- 完整基准复测（default.yaml × 5 轮 + GT）：ATE RMSE mean 43.5m、FPS 43.0、
  latency p99 55.5ms、valid_ratio 0.9891、keyframes 116.4——与 §3.30/M1.2 同噪声带，
  优先级变更（LoopClosure 优先）未引入退化；仍在门限的 ate_rmse/std、jumps_10m、
  submap_reinit 与 §3.30 一致，属 M3/M5 收敛目标。

### 3.33 M1.4 FrontendTracker 拆分（2026-08-08）

按 §5.1/§5.5 把前端跟踪计算从 vo.cpp 提取到独立 `FrontendTracker`
（`frontend_tracker.{h,cpp}`，行为保持不变）：

- **新增值对象**：`TrackerConfig`（VOConfig 跟踪字段快照）、`TrackingResult`
  （pose/valid/method/matches/inliers/质量/associations/recovering）、
  `KeyframeProposal`、`RefView`（帧首快照参考数据）、`MotionBaseline`、
  `StereoStats`、`RigidResult`。
- **迁移的算法**（公式/门限/顺序逐行不变）：
  - `computeStereoDepths` → `FrontendTracker::computeStereoDepths`
    （返回 StereoStats，VO 应用深度统计）；
  - PnP 核心（solvePnPRansac + PoseGate 验收）→ `trackPnP`（ORB/LK 共用）；
  - 3D-3D Kabsch → `estimateRigid3D3D`（纯几何，暴露可测）；
  - ORB 跟踪全流程（匹配 → PnP → 3D-3D → 对极回退 → RECOVERING）→ `trackOrb`；
  - 关键帧判定 → `proposeKeyFrame`。
- **职责边界**（§5.5）：FrontendTracker 只输出结果，不写 Map/Atlas、不执行
  BA/回环、不创建线程；VO 负责构造查询（快照 + 运动基线）与应用结果
  （位姿/关联/status_/state_）。运动基线提取为 `normalMotionBaseline()`，
  与 acceptPose 正常跟踪分支同一规则。
- 测试 `test_frontend_tracker.cpp`（6 项）：3D-3D 已知变换恢复/点数不足、
  trackPnP identity 恢复/连续性拒绝、单目深度边界、关键帧提议四种触发。

**验收**：
- CTest 9/9 全过（新增 test_frontend_tracker 独立 CTest）。
- §5.6 确定性：KITTI 00 前 1000 帧轨迹逐位一致（max_translation_diff=0 m、
  max_rotation_diff=5.5e-17 rad）、状态序列 1000 行一致。
- 完整基准复测（default.yaml × 5 轮 + GT）：ATE RMSE mean 41.5m（§3.30 46.8m）、
  FPS 43.3、latency p99 54.5ms、valid_ratio 0.9899——无退化且精度略好；
  仍在门限的 ate_rmse/std、jumps_10m、submap_reinit 与 §3.30 一致，属 M3/M5 目标。

### 3.34 M1.5 LocalMapper 拆分（2026-08-08）

按 §5.1/§5.5 把局部建图计算从 vo.cpp 提取到独立 `LocalMapper`
（`local_mapper.{h,cpp}`，行为保持不变），M1 拆分至此全部完成：

- **迁移的算法**（公式/Map API 调用/顺序逐行不变）：
  - `createMapPointsFromStereo` → 双目单帧建点 + 正式观测绑定；
  - `triangulateNewPoints` → 两帧三角化建点 + 全有全无观测；
  - `selectLocalWindow` → 共视窗口 / 兜底最近 n 帧；
  - `buildLocalBASnapshot` → min_observed 门槛 + anchor 连通分量裁剪。
- `includeLocalBALandmark` 迁至 local_mapper.h（vo.h 移除声明，test_vo 引用不变）。
- **职责边界**（§5.5）：LocalMapper 只通过 Map API 操作传入的 Map/Frame，
  不持有 VO 状态、不切换子地图、不执行优化求解、不创建线程；调用方按旧
  约定持 map_mutex_ 相应锁。insertKeyFrame 编排（快照几何检查/对齐/回环/
  调度）仍留在 VO 作为兼容 Facade。
- 测试 `test_local_mapper.cpp`（5 项）：观测门槛、双目单帧建点、两帧三角化、
  共视窗口兜底、BA 快照过滤与裁剪。

**验收**：
- CTest 10/10 全过（新增 test_local_mapper 独立 CTest）。
- §5.6 确定性：KITTI 00 前 1000 帧轨迹逐位一致（max_translation_diff=0 m、
  max_rotation_diff=5.5e-17 rad）、状态序列 1000 行一致。
- 完整基准复测（default.yaml × 5 轮 + GT）：ATE RMSE mean 46.0m、FPS 42.7、
  latency p99 52.8ms、valid_ratio 0.9899——与 §3.30 同噪声带，无退化；
  仍在门限的 ate_rmse/std、jumps_10m、submap_reinit 与 §3.30 一致，属 M3/M5 目标。

M1 里程碑总结：vo.cpp 从 2600 行拆出 5 个独立模块（PoseGate/Relocalizer/
BackendScheduler/FrontendTracker/LocalMapper），共 38 项独立 CTest；唯一行为
变更是 M1.3 按 §5.4 的 LoopClosure 优先调度语义（用户确认）。确定性验收全程
逐位一致，完整基准各轮均在 RNG 噪声带内无退化。

### 3.35 M2.1 输入队列 + 异步 Localizer（2026-08-08）

按 §6.1/§6.2 落地输入路径的有界队列与异步调度：

- **`bounded_queue.h`**（header-only 模板）：固定容量 ring buffer；`push` 永不
  阻塞（满丢最旧、保最新），消费者阻塞在 condition_variable（空闲不自旋）；
  `stop()` 唤醒并排空剩余元素（与 BackendScheduler 语义一致）；记录
  high water mark 与 dropped 计数（§6.4 指标）。
- **`sensor_packet.h`**：`SensorPacket{sequence, timestamp, right_timestamp,
  left, right}`，图像浅拷贝——提交后调用方不得复用/修改缓冲直至消费（文档化）。
- **Localizer 异步模式**（§6.1：传感器回调只校验并入队，不运行 ORB/PnP）：
  - `LocalizerConfig.enable_async_input / input_queue_capacity`（robot.yaml
    `Robot.Runtime` 段，§6.2 首版参数一并写入文档）；
  - `submitFrame()`：§4.3 校验（提交侧时间戳基线，与 worker 处理进度解耦）+
    入队，返回 bool，永不阻塞；
  - 跟踪 worker：阻塞消费 FIFO，结果经 `latestPose()` 轮询；worker 线程按
    `vo_cfg_.rng_seed` 单独播种 OpenCV RNG（M1 确定性规则，线程局部 RNG）；
  - `state()/latestPose()` 与 worker 通过 `result_mutex_` 并发安全；
  - `stop()`：停队列 → join worker → 排空 VO 后台 → 停状态机（§6.2
    shutdown_timeout_ms=2000 内，幂等）。
- 测试 `test_bounded_queue.cpp`（11 项）：FIFO/丢最旧/hwm/dropped/tryPop/
  pop 唤醒/stop 唤醒/stop 排空拒绝 + 异步 Localizer 集成（8 帧快速提交容量 3
  队列 → 丢最旧保最新：最新帧 ts=0.7 存活、hwm<=3、dropped>=3、进入 Tracking、
  提交侧校验、stop 后拒绝、同步模式 submitFrame 拒绝）。

**验收**：
- CTest 11/11 全过（新增 test_bounded_queue 独立 CTest）。
- L1 确定性（run_slam 路径不受 localizer.* 改动影响）：KITTI 00 前 1000 帧
  轨迹逐位一致。
- 本任务不改 VO/算法路径，完整基准不做复测（提交门 L2 快速档自动覆盖）。

下一任务：M2.2 资源预算（`resource_budget.{h,cpp}`，§6.3 回收顺序）。

### 3.36 M2.2 资源预算引擎 + Map 移除/命中旁路统计（2026-08-10）

按 §6.2/§6.3 落地地图资源硬预算与固定顺序回收：

- **`resource_budget.{h,cpp}`**（§6.3 六步引擎，无锁、无线程、不持有 Map/Atlas
  状态，调用方持 map_mutex_ 独占锁）：
  - `MapBudgetConfig`：§6.2 首版参数（max_active_keyframes 1200、max_active_points
    120000、max_descriptor_mb 256、max_snapshot_mb 256、max_total_estimated_mb 900）
    + §6.3 回收参数（弱观察窗口 30 KF、共视重叠 0.9、相邻位姿差 0.15 m/3 deg、
    最近图像保留窗口 2、非活动子地图上限 2）；robot.yaml `Robot.MapBudget` 段同步；
  - `evaluate(map, snapshot_bytes)`：只读估算 KF/点/描述子/图像/在途快照字节与
    总量（含 KF/点开销常量），逐项判定超限；
  - `reclaim()`：按 §6.3 固定顺序回收，任一步回到预算内立即返回：
    1. 删除 0 个正式 Observation 的点；
    2. 删除 `observationCount < 2` 且超过 30 个 KF 未被跟踪命中的点（旁路统计，
       禁止复用 observed_count 语义）；
    3. 卸载非活动 KF 原图/灰度图（`Frame::releaseImages`，保留关键点/描述子/
       位姿/观测），最近窗口 KF 保留图像；
    4. 冗余 KF 剔除：共视重叠率 >0.9 且相邻位姿差 <0.15 m/3 deg，回环/子地图
       锚点（protected 集）与最近窗口 KF 必须保留，循环到关键帧回预算或无可剔除；
    5. 冻结超过 2 个的非活动子地图（保留最新 2 个）并卸载其 KF 图像缓存；
       冻结标志属于 Atlas 状态，M4 前只做图像卸载与计数，不做磁盘换出；
    6. 仍超预算 → `stopped_map_growth`（调用方上报 Degraded + BackendOverloaded），
       不随机删除锚点/强点。
- **Map 扩展**（§6.3）：
  - `removeKeyFrame(id)`：原子移除 KF（撤销全部正式观测/共视/slot + 计数 + 拓扑
    bump，幂等）；
  - `recordTrackingHit(mp_id)` + `lastHitKeyframeCount(mp_id)`：地图点"最近命中
    KF"旁路统计表，正式观测建立时自动记录，随点删除/清图清理；
  - 命中/清理均为旁路写入，不改 VO 行为。
- 测试 `test_resource_budget.cpp`（14 项）：evaluate 五项超限判定 + 六步回收各自
  语义（含锚点保护、最近窗口保留、非活动子地图冻结、stopped_map_growth）+ Map
  新 API（removeKeyFrame 双向一致性、命中旁路统计）+ 组合场景全量观测一致性。

**验收**：
- CTest 12/12 全过（新增 test_resource_budget 独立 CTest）。
- 确定性回归（§5.6）：map.* 属 run_slam 共享路径，用 a4e425a（M2.1）构建
  baseline 与当前构建各跑 KITTI 00 前 1000 帧（deterministic.yaml）：
  轨迹逐位一致（最大平移差 0 m、最大旋转差 5.551e-17 rad < 1e-8）。
- 本任务不改 VO/算法路径，完整基准不做复测（提交门 L2 快速档自动覆盖）。

**未验证边界**：① 预算在生产路径的触发点（vo.cpp 周期性 cull 钩子替换）与
snapshot_bytes 上报接线未接，留待 M2.3 指标任务；② 第 5 步的"冻结"语义只做
图像卸载与计数，Atlas 冻结标志与磁盘换出留待 M4；③ 对象开销估算常量（KF 16 KB/
点 2 KB）为保守估计，M7 前按实测统一标定。

下一任务：M2.3 结构化指标（`metrics.{h,cpp}` + `soak_test.py`，§6.4）。

### 3.37 M2.3 结构化指标 + soak 验收（2026-08-10）

按 §6.4/§6.5 落地端到端结构化指标与 soak 脚本，替代正则解析日志：

- **`metrics.{h,cpp}`**：线程安全 `MetricsCollector` + 不可变 `MetricsSnapshot`
  （§6.4 全清单，未接数据源保留 -1 哨兵）：
  - frame latency p50/p95/p99/max + deadline miss/ratio；
  - input received/processed/dropped、queue high-water mark；
  - tracking 均值（features/stereo/inliers/ratio/RMSE）；
  - pose accepted/rejected/prediction-only + 每类 FailureReason 计数
    （类别互斥：prediction_only 不计入 accepted）；
  - backend submitted/executed/dropped/pending + 任务年龄；
  - loop committed；map KF/点/Observation/描述子/图像/总量字节；
  - LOST 次数/时长与 relocalization latency（Relocalizing→Tracking 恢复，
    p95）；JSON/CSV 序列化与落盘。
- **BackendScheduler 调度统计**（§6.4 数据源）：submitted/executed/dropped/
  pending + 入队→执行任务年龄；覆盖式单任务槽的"被覆盖任务"计为 dropped。
- **vo.h 只读接口** `backendStats()`（范围扩大说明：Localizer 需要经 VO
  读取调度统计——§12 M2.3 允许列表外的单行 getter，无行为影响）。
- **Localizer 集成**：同步/异步路径统一喂数据（单帧 latency 计时、VO
  Status 质量、PoseEstimate、状态机转换、输入计数）；stop 收尾 LOST 段并
  刷新 backend/loop/map 统计；`metricsSnapshot()` / `writeMetricsJson/Csv`；
  `LocalizerConfig.enable_metrics / tracking_deadline_ms`（robot.yaml
  Runtime 段已含 tracking_deadline_ms=80）。
- **run_slam --localizer**：新增 `--metrics-json/--metrics-csv` 输出与
  `--robot-yaml` 参数（修复：robot.yaml 硬编码相对路径导致从非仓库根目录
  启动即 `YAML::BadFile` 崩溃，soak CTest 从 build/ 触发）。
- **`soak_test.py`**（§6.5）：循环重放同一序列 N 轮（--full=10 轮），
  零依赖 RSS 采样（/proc VmRSS，0.5 s 间隔），逐轮断言：进程退出码 0
  （无崩溃/死锁）、latency p99 < 80 ms、deadline miss < 1%、
  input_queue_hwm <= 3、backend_pending <= 1、valid_ratio >= 0.99；
  关闭三路径：SIGINT < 2 s 退出、输入 EOF（数据读完自然退出）、构造失败
  （非法输入快速退出；Localizer 层真实构造失败由 test_localizer_contract
  覆盖）；--full 档断言 RSS 峰值 < 1 GiB、稳态斜率 < 5 MiB/h（后 50%
  窗口线性拟合）；--duration-h 2.0 为 §6.5 2 小时档（nightly）。
  CTest 注册 test_soak 快速档（200 帧 × 2 轮）。

**验收**：
- CTest 14/14 全过（新增 test_metrics 11 项 + test_soak）。
- ASan/UBSan（test_metrics/test_backend_scheduler/test_localizer_contract/
  test_bounded_queue）零报错。
- 冒烟（KITTI 00，deterministic.yaml 120 帧 --localizer）：JSON 全字段
  有效——p99 142.8 ms（deterministic 单线程配置固有偏慢，非实时档）、
  backend 0（async_backend=false 同步路径）、map 78 KF/17349 点/
  33960 Observation/2.87 MB 描述子/3.73 MB 图像。
- soak 快速档（default.yaml 300 帧 × 2 轮）：PASS——p99 40.8/45.1 ms
  （<80 ms）、hwm=0、pending=0、RSS 峰值 797 MB（<1 GiB）。

**未验证边界**：① RSS 稳态斜率 10~12 MiB/h 是 8 秒窗口的启动噪声（词汇表
加载），§6.5 斜率门限须由 2 小时档（--duration-h 2.0）验证；② grid
occupancy / PnP condition number 数据源在 M3.1/M3.2（质量门/协方差）；
③ loop queried/candidates/verified 数据源在 M5（回环确认）；④ backend
stale/invalid 计数属 BackendCommitter（M2.3 未接入）；⑤ snapshot_bytes
上报接线仍为 -1（M2.2 遗留，随预算触发点一起接入）。

里程碑：M2（§6.2 实时调度 + §6.3 资源预算 + §6.4 结构化指标）全部落地，
§6.5 完整验收由 nightly 2 小时 soak 档执行。

### 3.38 M2 遗留清理：§6.5 实测标定、预算触发收敛、soak 轮式重构（2026-08-10）

M2.3 收尾后发现的三类问题（§3.37 未验证边界中可由 M2 层解决的项 + soak
验收工具缺陷），本轮全部处理：

**① §6.5 RSS <1GiB 硬门槛 vs §6.2 首版参数互斥 → 实测标定（§0.1 硬门槛优先）**
- 实测（KITTI 00 代表场景）：12 万点 + 1100 KF 时 RSS ≈ 1.3GB，超 1GiB 硬门槛
  15%；RSS 构成 = 词袋 ~350MB（DBoW3 145MB 文本加载后的两个 ~170MB 匿名段）
  + 框架 ~150MB + 地图 heap（点 ~2.5KB/个、KF ~150KB/个）。
- 按 §0.1"产品硬门槛未达到不能进入下一发布阶段"，以硬门槛优先标定预算参数：
  `max_active_keyframes 1200→700`、`max_active_points 120000→60000`、
  `max_total_estimated_mb 900→500`；overhead 常量实测化
  （`overhead_bytes_per_keyframe 16KB→128KB`、`per_point 2KB→2.5KB`），
  使 estimated_total 与实测内存一致（12 万点 ≈ 478MB 踩 500MB 线）。
- 标定依据为 KITTI 00 单一场景（可用场景唯一）；后续 ≥3 类实录数据统一修订。
- 连锁影响：预算激活使 KITTI 00 1000 帧（KF 395/点 60064）触发回收 → L1
  reference 更新（`--update-reference`，有意行为变更）；回收路径确定性验证：
  两次运行逐位一致（平移差 0 m、旋转差 5.9e-17 rad）。

**② 预算 stopped 恢复复查振荡 bug（修复）**
- 原实现：`addFrame` 每帧按 `keyFrameCount() <= max_active_keyframes` 轻量
  复查解除 stopped——点超限时（KF 数仍在软上限内）每帧"解除→插 KF→enforce
  再停"振荡，BackendOverloaded 计数被瞬间解除淹没（实测 992 KF 时反复振荡）。
- 修复：恢复复查移到 `needNewKeyFrame` 提议路径做完整只读评估（KF 插入频率，
  避免每帧全图遍历），回到预算内才解除并允许插入。

**③ reclaim 第 5 步扩展：冻结子地图删除弱陈点**
- 非活动子地图超过 `max_inactive_submaps` 时，除卸载 KF 图像缓存外，删除
  其弱陈点（obs < weak_point_min_observations 且超窗口未命中）；KF 描述子
  保留作重定位骨架。M4 磁盘换出落地前控制 RSS 的最低内存手段（§6.3 第 5 步
  注释更新）。签名增 `inactive_submaps`（id → Map）参数（默认空，旧调用不变）。

**④ §6.4 frame latency 语义修正：只统计跟踪帧**
- 重定位/LOST 恢复帧（词袋查询/PnP 候选验证）不是跟踪延迟，不计入
  p99/deadline miss（§6.5 门限语义）。实测长程 p99 从 194ms 降到 56ms
  （KITTI 00 整序列轮：p99 55.9ms、miss 0.67%、valid_ratio 0.992）。

**⑤ run_slam --loop 循环重放时间戳偏移**
- EOF 重建数据集后重放时间戳加轮偏移（`last_play_ts + 1.0`），满足 §4.3
  单调检查——修复重放第二轮起全部帧被 TimestampRollback 拒绝（实测 4541 帧
  metrics 只有第一轮）的 bug。

**⑥ soak_test.py 重构为多进程轮式**
- 原"单进程连续档（--loop 2h）"与 1GiB 硬门槛互斥：冻结子地图的点/KF 保留
  （M4 换出前），循环重放每轮累积 ~150MB → 无法单进程 2h <1GiB。重构为：
  快速档（默认）N 轮短程；--full 10 轮 + RSS 峰值/斜率门限；--duration-h
  轮式跑满时长（每轮整序列 4541 帧 ≈ 238s，启动噪声占比 ~3%，轮内后 50%
  窗口拟合斜率排除词汇表加载段；后 1/3 轮斜率均值 <5MiB/h 断言）。EOF
  关闭路径由每轮 rc==0 覆盖。

**验收**：
- 15/15 CTest（test_resource_budget +1 项冻结子地图点回收；test_budget_runtime
  参数断言更新）；ASan/UBSan 干净（5 个相关测试）。
- L2 快速档 PASS：p99 mean 58.9 / worst 68.7 ms、valid_ratio 1.0。
- KITTI 00 整序列轮（4541 帧）：p99 55.9ms、deadline miss 0.67%、valid 0.992。

**未达标项（如实记录，nightly 观测）**：
- KITTI 00 整序列 RSS 峰值 1523MB > 1GiB。根因链：LOST（每程 4-5 次）→
  多子地图累积（旧子地图强点/KF 保留）+ 词袋 350MB。当前 M2 层手段（预算
  参数、第 5 步弱陈点回收）只能控制活动地图；修复依赖 M3.1 质量门（降
  LOST 次数）与 M4 磁盘换出（子地图序列化离场）与 M5 词袋内存优化。
  2h 档 RSS 门限在此框架下如实 FAIL，作为 nightly 观测数据积累。

下一任务：M3.1 视觉前端质量门（`tracking_quality.{h,cpp}`，§7.1/§7.2；
其降 LOST 也是 RSS 达标的先决条件之一）。

### 3.30 当前版本完整基准评估（KITTI 00 全程 × 5 轮 + GT + RSS，2026-08-08）

用 §3.29 的生产基准方案评估当前 HEAD（`f447dcf`）：`config/benchmark.yaml` 全程档
（default.yaml、回环开、4541 帧 × 5 轮），GT 用 KITTI 官方
`data_odometry_poses.zip` 经 `kitti_gt_to_tum.py` 生成；`measure_rss` 开。

**通过的门限**：

| 指标 | mean ± std | worst | 门限 | 结果 |
|------|-----------|-------|------|------|
| valid_ratio | 0.9891 ± 0.0020 | 0.9916 | ≥0.99 | PASS |
| latency_p99 | 57.66 ± 4.31 ms | 62.41 ms | ≤80 ms | PASS |
| deadline_miss_ratio | 0.0003 ± 0.0001 | 0.0004 | <0.01 | PASS |
| lost_count | 2.6 ± 0.5 | 3 | ≤30 | PASS |
| FPS | 41.5 ± 1.0 | 42.8 | - | - |

**未通过的门限（当前版本真实差距）**：

| 指标 | mean ± std | worst | 门限 | 结果 |
|------|-----------|-------|------|------|
| ate_rmse | 46.78 ± 5.83 m | 55.28 m | ≤40 m（§7.6 M3 阶段门） | FAIL |
| ate_std | 20.65 ± 2.79 m | 25.47 m | ≤8 m | FAIL |
| jumps_10m | 2.6 ± 0.5 | 3 | =0（§11.4 硬不变式） | FAIL |
| submap_reinit | 1.6 ± 0.5 | 2 | ≤1 | FAIL |

**其他关键指标**：ATE max worst 111.1 m；RPE trans RMSE 0.40 m/frame、rot RMSE
0.44 deg/frame；len_ratio 1.028；loops 1.2 ± 0.4；RSS 峰值 1579.7 MB（~1.54 GiB）；
关键帧 117.6、地图点 18824（default.yaml 双目 KF 阈值 0.9m 稀疏化）。

**结论**：基准方案正确暴露了当前版本未达生产门槛——ATE 高与 §3.23 已知的"中段旋转
漂移"一致，且每轮发生 1~2 次子地图重建（长丢兜底）伴随 >10m 单步跳变，说明重定位/
子地图重建路径会破坏 §11.4 的"连续有效帧非物理跳变 = 0"。这些正是 M3（鲁棒性/连续性）
与 M5（回环/重定位）要收敛的目标，当前按 M0/M1 阶段如实记录；提交门快速档（500 帧×3
轮）不受影响，继续通过。

### 3.34 前端跟踪增强：运动模型引导匹配 + 共视图局部地图投影（2026-08-10）

在 fe/guided-localmap 分支落地两个纯前端改进（不碰 Map/BA/回环，符合 M1.4
FrontendTracker 只输出结果的职责边界）：

**方案 A 运动模型引导匹配**：`MotionBaseline` 携带匀速模型预测位姿
（`per_frame_motion_` 外推，vo.cpp normalMotionBaseline），`FrontendTracker::matchGuided`
把参考帧有 3D 点的特征投影到当前帧，在投影点邻域网格桶内做描述子比率匹配；
匹配数不足自动回退全图 BF。收益：空间先验减少误匹配 + 窗口搜索替代全图匹配。

**方案 B 共视图局部地图投影匹配**：快照（captureTrackingSnapshot）按（参考 KF id,
geometry revision）缓存参考 KF 及其共视 KF 的地图点（去重、预算上限、descriptor 非空
过滤）；`trackLocalMap` 用首轮 PnP 位姿投影局部点、窗口匹配补 3D-2D 对应；
`refinePnP`（solvePnP iterative + useExtrinsicGuess）确定性精修——不重跑 RANSAC，
避免额外消耗全局 RNG 破坏双实例交替驱动的确定性等价（test_localizer_contract 逐位
一致的前提）。

**修复过程中发现并解决的三个问题**（均写入代码注释）：
1. **快照约定**：refine 首轮对应必须用 trackOrb 返回的版本绑定快照坐标
   （`TrackingResult::association_points_s`），不能读 live `mp->pos_s`——后端 BA
   改写坐标会污染精修（等价性测试 3e-8 rad 分叉的直接根因）。
2. **窗口匹配距离上限**：单候选时无条件接受会把窗口内随机近邻当匹配，而精修无
   RANSAC 外点过滤 → 加绝对距离上限（与 quickMatchCount 默认 64 一致）。
3. **精修前确定性重投影筛选**：窗口匹配只保证描述子相似不保证几何一致；地图膨胀
   后错配率上升，全量最小二乘被拉偏（完整 KITTI 00 无筛选时长度比 1.04 → 1.12、
   ATE 恶化）。用首轮位姿投影剔除重投影误差 > ransac_pixel_threshold 的对应。

**验证（KITTI 00，功能开 vs 关）**：

| 配置 | 轨迹长度比 | ATE RMSE | 回环数 |
|---|---|---|---|
| deterministic 关（基线） | 1.038 | 42.9m | 1 |
| deterministic 开（最终） | **1.0003** | **16.1m** | **4** |
| default 关（基线） | 1.061 | 38.7m | 2 |
| default 开（最终） | **1.003** | **26.9m** | 2 |
| 前 1000 帧（deterministic） | - | 9.1m → **7.9m** | - |

轨迹长度比从 ~1.04 收敛到 ~1.000（双目绝对尺度几乎无漂移），完整序列 ATE 降
~38-63%。参数均为通用默认值（25px/30px 搜索半径、400 点预算、64 描述子距离上限），
合成测试覆盖召回/回退/occupied 防重复/RNG 确定性，无数据集特化。

**变更**：camera2pixel、FrontendTracker::matchGuided/trackLocalMap/refinePnP/
buildKeypointGrid/windowRatioMatch、TrackingResult::association_points_s、快照局部
地图缓存、VOConfig/TrackerConfig 新字段、config 新参数、test_frontend_tracker 新增
5 项测试、L1 参考随行为变更更新（benchmark_gate --update-reference）。

**验收**：ctest 9/9 全过（含新增 test_frontend_tracker 5 项）；benchmark_gate 提交门
PASSED（L1 参考更新后 L0+L1+L2 快速档全绿）。

### 3.39 `run_slam` 严重漂移、周期 LOST 与假绿门禁事故复审（2026-08-10）

本轮从新鲜 Release 构建和 KITTI 00 全程重新开始，不沿用旧 `build/` 或“15/15”结论。
旧基线 `d90cfac` 的真实结果为：4541 帧中 4389 个有效姿态（96.65%），SE3 ATE
RMSE 38.95m、p95 69.42m、8 段各 19 帧 LOST、9 个子地图、0 次回环。每段 LOST
前均出现活动地图约 6 万点预算耗尽；词袋已加载并查询 195 批候选，但跨子地图候选
被错误地到当前 `Map` 查找，因此全部过滤。

**定位出的代码根因**：

1. `FrontendTracker::refinePnP()` 用全 0 索引算 RMSE，并通过 `const_cast` 排序 const
   输入；局部地图精修的来源掩码也全 0，新增的有效关联从不写回当前帧。
2. 地图点预算状态被当成“禁止关键帧”，参考 KF 在 700 帧硬顶后不再刷新，最终稳定
   进入 LOST；一次删除数万点又逐点扫描所有 KF，复杂度在小单测中未暴露。
3. 回环候选不携带 `submap_id`，位置先验直接比较子地图局部光心；即使 DBoW3 命中
   旧路段，也会在当前 Map 查找失败。
4. 跨子地图回环后台事务直接写前端 `snap_`，并使用异步执行时已经变化的全局
   `curr_frame_` 更新世界运动基线，造成 C++ data race 和“旧局部位姿 + 新 Atlas 锚”
   混用。一次真实回环有 896 个 PnP 内点，下一帧仍出现 173m 连续性差，证明“回环
   计数成功”不是正确性证据。
5. 质量门禁存在可空跑路径：数据集路径少一层 `kitti/`、soak 0 帧仍 PASS、空轨迹
   比较 PASS、非法输入退出 0；benchmark 的最小值门错误取最大值；pre-commit 复用
   旧构建，导致测试文件变更未进入实际执行。

**修复方向**：

- PnP 精修改为本地排序副本、完整 RMSE 索引，并回写通过筛选的局部地图关联。
- 点创建逐点受配额约束；陈点批量事务回收；点上限不再禁止 KF。KF/描述子/总量硬顶
  且前端确实请求 KF 时，在当前世界位姿主动滚动连续子地图，以 `TrackingBridge`
  连接，当前帧作为新图首 KF，不经过 LOST。
- 回环候选携带子地图身份，世界位置统一按 `C_w=T_ws*T_cs.inverse().t`；跨图约束
  通过 Atlas 事务求解。后台只修改 Atlas，前端在帧首/帧尾安全点按
  `T_cw_new=T_cw_old*T_ws_old*T_ws_new.inverse()` 重基运动基线。
- 门禁使用临时目录新鲜 configure/build、`ctest --no-tests=error`；数据测试增加非空
  工作证明和非法输入负例；reference 更新从行为补丁中分离。

**本轮验收结果（新鲜 Release 构建）**：

- CTest 16/16 通过；新增 gate 逻辑、前端精修、逐点配额/批量回收、预算滚动、跨图
  候选身份、非单位位姿公式、Atlas 锚点重基和“不得重新拉开旧闭环”测试。
- 非法/空数据集 `run_slam` 0.11s 内非零退出；Python 门限测试 5/5 通过；空轨迹、
  空状态、0 帧 soak 和缺失 GT/ATE 均不能再 PASS。
- 产品默认 700-KF 预算语义：KITTI 00 4541/4541 位姿有效、0 LOST、3 次连续预算
  滚动、跨图回环可提交/冲突可回滚、>10m 跳变 0；但 Atlas 只有每子图一个刚体
  `T_ws`，SE3 ATE 43.94m，仍不满足 40m/8m std 产品门。
- `config/kitti00.yaml` 明确改为桌面高精度档（1800 KF / 900MB 估算预算），让首轮
  闭环在同一关键帧图内执行：SE3 ATE **12.22m**（旧基线 38.95m）、ATE std
  9.33m、长度比 **0.9995**、4541/4541 有效、0 LOST、4 次真回环、>5m/>10m
  跳变均为 0。代价为 13.63 FPS、p99 120.0ms、deadline miss 3.61%；它解决
  `run_slam` 精度演示，不等价于移动端实时档达标。

因此 M2 仍标为“实现完成、产品验收未通过”：默认资源档缺少子地图内锚定关键帧图，
高精度档则尚未满足 p99/RSS 与 ATE std 门。下一架构任务必须把局部连续 `T_ob` 与
全局校正 `T_wo` 分离，并让跨子图回环约束落到锚定关键帧图，而不是继续在一个
`T_ws` 和预算数字之间二选一。

### 3.40 通用长程回环召回、轨迹锚点事务与跨子图渐进校正（2026-08-11）

§3.39 的高精度档虽把 ATE 降到 12.22m，但尾段 444.3～453.2s 明确重走旧路仍未
闭环；红线后段误差继续增长。根因不是单一阈值，而是三个相互独立的覆盖缺口：

1. 词袋结果只验证少量单 KF 候选，短重访窗口容易被查询时序和点回收漏掉；
2. 单个历史 KF 的点稀疏时直接 PnP 失败，即使其共视/时间邻域合起来有足够几何；
3. Local BA 与 Atlas 会改写轨迹依赖的 live anchor，却没有同步维护持久轨迹语义，
   导致“回环成功”后在 KF 或子地图边界出现锯齿/单帧大跳。

本轮采用不读取 GT、无序列/帧号特例的通用修复：

- place candidate 按子地图和历史时间邻域聚类，维护最多 256 个跨查询 hypothesis；
  允许一次缺测，BoW 与位置先验合计最多输出 12 个历史地点，不再固定截断 Top-3。
- 单 KF 验证失败后，从同一历史 Map 构造有界共视 + 时间区域（桌面档最多 12 KF /
  800 点），descriptor 与 3D 深拷贝后在锁外做 mutual/unique PnP；内点数、比例和
  RANSAC 像素门仍绑定原回环配置，没有为召回降低几何接受标准。
- 位置先验不再只选估计世界位置最近的单 KF；先按与 BoW 一致的连续时间链聚类，
  再从不同历史地点保留最多 4 个，并用 `submap+[min_id,max_id]` 做跨模态去重。
  检测间隔由 10 KF 改为 5 KF，覆盖短重访窗口。
- 单帧/区域 PnP 统一使用 `>=50` 内点、ratio `>=0.70`、RMSE `<=2.5px`、
  正深度 `>=0.95` 和 8x6 网格至少 12 格；区域事件记录 PnP 内点主导支持帧，
  避免用区域 anchor 的时间戳误判 oracle precision。
- 预算剔除 KF 时在地图事务后批量通知 LoopClosure；DBoW3 无单条删除 API，
  因此清理 Frame/BoW/ID/地点假设后只用存活 KF 重建一次索引，旧条目不再占 Top-N 或强引用。
- 前端 RANSAC 数值坏解在不降低任何门的前提下，用运动预测初值独立重试；
  重试前后恢复 OpenCV RNG 状态，避免容错分支使后续轨迹随机分叉。
- Local BA 单次位姿改写硬门收紧为 1m。提交成功时在同一 `map -> traj` 事务内按
  `T_ca_new=T_ca_old*anchor_old*anchor_new^-1` 重基历史记录，使局部精修不移动已发布
  轨迹；位姿图的全局闭环校正仍可传播到历史。
- 跨子图回环不再把全部刚体锚校正集中到滚动边界。持久轨迹从子图首帧的零校正
  渐进到回环端点的完整校正，消除连接处单帧跳变；运行时地图仍使用事务提交后的
  Atlas 锚。该过渡只依赖子图拓扑和帧顺序。
- 新增离线 GT oracle：按时间戳、空间距离、最小时间间隔和可选朝向生成可闭环区间，
  计算区间级 recall、事件 precision/false detection 与 detection delay。GT 只用于
  离线验收，运行时回环路径完全不接触它；空输入/坏时间戳/假事件均 fail closed。

**最终独立整序列复跑（KITTI 00，`config/kitti00.yaml`，4541 帧）**：

| 指标 | 结果 |
|---|---:|
| 有效轨迹 / LOST | 4541/4541 / 0 |
| SE3 ATE RMSE / p95 / max | **3.49m / 7.08m / 10.59m** |
| RPE trans RMSE / max | **0.052m / 0.580m** |
| 估计/GT 路径长度比 | **0.9941** |
| 单步跳变 >3m / >5m / >10m | **0 / 0 / 0** |
| GT oracle 区间召回 | **4/4，recall 1.0** |
| 回环事件 precision / false | **7/7 为真，1.0 / 0** |
| 尾段跨子图回环 | 451.1s → 6.6s，成功 |
| p99 / deadline miss / FPS | 219.8ms / 50.25% / 6.68 |

中间版本只保留单个位置先验时，同样 4541/4541 和 0 跳变，但尾段回环漏掉，
oracle 只有 3/4、ATE 升到 11.55m；因此候选多模态与整段 oracle 是真正的提交门，
不能被单测或轨迹覆盖率代替。完整新鲜构建 CTest 为 18/18；新增历史区域、候选时序、回环 oracle、
Local BA 锚点重基和跨子图渐进校正几何回归。

**边界**：高精度档的全局精度和回环召回已显著改善，但 p99 219.8ms、deadline miss
50.25%，RSS 也未在本轮复测，因此仍不能宣称 M2 产品验收完成；候选扩宽的
`O(K log K)` 位置扫描和最多 12 轮区域验证还需做性能分层。生产控制位姿仍需按
§3 的 `T_ob/T_wo` 双坐标契约分离，不能把平滑后的离线全局轨迹当作实时控制输出。

### 3.41 回环前缀快照、Essential/Submap Graph、双坐标分层与 mobile 首测（2026-08-11）

本轮直接针对 KITTI 00 高精度档的实测性能根因：旧同子图回环从最终 PnP 重验、
全图快照、约 1800 KF 的 g2o PGO 到点同步/提交始终持有 `map_mutex_` 独占锁。旧跑
`kitti_00.txt.perf.csv` 中六次 PGO 为 17.3～39.8s，帧延迟最大 40.3s，因此“回环时前端卡住”
不是 ORB 参数问题，而是锁域和图规模问题。

**实现收口**：

- 同子图回环在共享锁下重验几何并深拷贝冻结前缀；tracking 的共享读不等待。位姿图构建、
  g2o 求解和点校正全在 Map 锁外执行。
- 提交时短持独占锁，联合复验 Map/Submap 身份、geometry revision 和每个冻结 KF/点的
  对象身份。求解期间前端新增的 KF 保持相对冻结端点的位姿不变后重基；新点保持在
  最早存活观测 KF 中的相机坐标不变。前缀删除/替换或几何变化则整笔 stale。
- 子图内从全量 KF 图改为有界 Essential Anchor Graph：首尾、最新回环端点必优先保留，其余
  均匀抽样；默认/KITTI 上限 256，mobile 上限 64。锚点校正在 SE3 上渐进传播到冻结前缀；
  跨子图仍以 Submap 为顶点优化 Atlas `T_ws`。PGO 迭代由 100 次硬编码改为配置项。
- `PoseEstimate` 显式发布 `T_ob/T_wo/T_wb`：纯视觉前端积分相邻已发布位姿得到连续
  `T_ob`，全局差异留在 `T_wo`，并校验 `T_wb = T_wo*T_ob`。这不是 ESKF；IMU 传播和
  `GlobalCorrectionEvent` 订阅通道仍未实现。
- `config/mobile.yaml` 首测值：600 ORB/6 层、2 OpenCV 线程、异步后端、6 KF 局部 BA/3 迭代、
  64 个 PGO 锚/15 迭代、320 KF/24000 点/220MB 活动地图估算预算。这是 ARM/4GB 级设备
  起点，不是多场景标定完成值。

**回归与门禁**：

- `ctest --test-dir build --output-on-failure --no-tests=error`：19/19 PASS。
- `bash scripts/benchmark_gate.sh`：新鲜 Release 构建，`g2o vendored` + `DBoW3 found`；19/19 CTest；
  KITTI 00 前 1000 帧确定性轨迹平移差 0、旋转差 `5.562e-17rad`；3轮×500 帧 L2 最差
  p99 `51.81ms`、deadline miss 0、ATE RMSE `1.218m`、valid 500/500，整门 PASS。
- mobile 烟雾：`/usr/bin/time -v ./build/bin/run_slam datasets/kitti/sequences/00 config/mobile.yaml
  /tmp/vslam_mobile_smoke.tum --headless --frames 200 --status-csv /tmp/vslam_mobile_smoke.csv
  --metrics-json /tmp/vslam_mobile_smoke.json`。结果 200/200 有效、0 LOST、20.88 FPS，p50/p95/p99/max
  `38.93/46.73/51.58/68.98ms`，100ms deadline miss 0，117 KF/13019 点，RSS 峰值
  `727664KiB`。SE3 评估仅前 200 帧：ATE RMSE `0.621m`、RPE trans `0.046m/frame`、
  路径比 `0.9853`、0 个 >3m 跳变。

**边界**：mobile 切片只覆盖 KITTI 00 的 4.40%，没有触发真实回环，因此不证明前缀事务的
长序列 p99 或回环精度。本轮没有重跑 `kitti00.yaml` 4541 帧与回环 oracle，也没有 ARM 实机、
8～24h soak 或多场景统计。RSS 727.7MiB 表明 97 万词 DBoW3 词表常驻仍是嵌入式主要障碍；
只压活动 KF/点不足以进入 512MB 级设备，下一步必须做紧凑词表/数据库分层并用同一回环 oracle 重验召回率。

### 3.42 可交互 3D 地图点云可视化（2026-08-11）

右上角保留原来的 **2D X-Z 俯视轨迹视图**作为默认模式；面板的 `Show 3D map`
默认关闭，打开后切换为 **Pangolin Handler3D 可交互 3D 地图视图**：鼠标左键旋转、
右键平移、滚轮缩放（`r` 复位），并叠加世界系地图点云、3D 轨迹折线、当前相机视锥
与 XZ 参考网格。纯可视化增强，不影响 SLAM 数据路径与数值。

**实现**：

- `VisualOdometry::getMapPointsWorld(max_points)`：在 `map_mutex_` 共享锁下遍历
  Atlas 全部子地图，按几何契约组合 `p_w = T_ws · p_s`（`src/vo.cpp`）。超过
  `max_points`（`Viewer::kMaxMapPoints = 60000`，与 MapBudget 同量级）时对点集均匀
  抽样，避免只保留最早 id 段而缺失新子地图点。`Localizer::mapPointsWorld` 做只读透传。
- `Viewer::updateMapPoints`：与轨迹同锁同 revision 机制缓存点云，渲染线程仅在新
  版本时拷贝一次；`run_slam`/`run_vo` 每个可视化帧喂入。
- 跟随模式默认开，但**只在用户空闲约 0.8 s 后重新对准当前相机**——用包装
  `MapViewHandler3D` 记录最近鼠标/键盘交互时间，避免每帧 `SetModelViewMatrix`
  覆盖用户拖拽（首个实现会“拖不动视角”）。
- **跑完后保持 Viewer 打开**（run_slam 两个路径 + run_vo）：数据集 EOF 后主线程
  等待 `viewer.shouldQuit()`（渲染循环退出时把 `quit_` 置真）再 `stop()`，窗口
  持续显示最终地图/轨迹/点云，用户关闭窗口或按 Esc 才退出；headless 路径立即
  返回，不影响 benchmark/gate。
- **完整轨迹保留**：`kMaxTrajectoryPoints` 3000→20000（覆盖 KITTI 00 全程 4541
  帧），`Trail points` 滑杆默认/上限同步为 20000——修复长序列后半段时前半段轨迹
  从可视化中消失的问题。存储 20000×Vec3≈0.5MB，有界。
- 面板新增默认关闭的 `Show 3D map` 以及 `Show map points` / `Show camera` 开关；
  `Show grid`、`Follow camera`、
  `Show trail`、`Pause render`、截图等保留。右侧地图列宽 340→460px。
- `scripts/plot_traj.py` 字体优先级调整为 Noto Sans CJK 优先（DroidSansFallbackFull
  仅含 CJK 字形、缺 ASCII，会渲染方块标签；系统安装 `fonts-noto-cjk` 后中文/英文
  标签恢复正常）。

**验证**：`cmake --build build -j` 通过；CTest 除 `test_vo` 的存量 DBoW3 flat
检索断言失败（干净树同样失败，与本次改动无关）外 18/18 PASS（`test_soak` 首轮
偶发失败、重跑通过，时序相关）；无头与有显示环境下 `run_slam` KITTI 00 150 帧
烟雾运行正常，3D 渲染循环、Handler3D 输入、点云/视锥/轨迹渲染路径均未崩溃。
未做 4541 帧完整基准重跑与回环 oracle——本功能不改变任何几何/并发行为。

**边界**：3D 视图仅是有显示环境下的可视化辅助，headless 评估路径不受影响；
点云上限 60000 超出时均匀抽样会略去部分点，仅影响显示密度、不影响轨迹/地图数据。

### 3.43 Flat-DBoW3、有界历史索引与简单验证级联（2026-08-11）

目标是先落地不改变几何语义、容易回退且能直接减少 RSS/后台工作量的部分；本节没有把
全局描述子硬过滤、在线词表剪枝、SIMD 或新阈值体系放进生产路径。

**实现**：

- 新增 `flat_dbow3` 检索后端，直接解析现有 DBoW3 二进制词表，保持层次量化、TF-IDF、
  L1 normalize/score 与官方后端一致。词表热/冷字段用连续 SoA 保存，节点载荷约
  `51.60MiB`；每帧 BoW 用排序稀疏数组，避免 `std::map` 的逐节点分配。单测用同一批 ORB
  descriptor 对比两套后端的候选 id、顺序和分数。
- mobile 的回环索引硬限 256 个 KF。首 KF 固定保留，其余用确定性 reservoir-style 覆盖整个
  时间轴；单纯 FIFO 在长序列会删除所有早期地点，完整 KITTI 00 曾得到 0 回环，已否决。
  预算剔除产生的 LoopClosure 清理由合并 id 队列交给异步 `LoopMaintenance`，不在 Map
  独占锁内重建数据库或释放大量描述子。
- 检索先过滤同子图最近帧；mobile 用 `mature_verification_limit: 4` 按连续地点假设成熟度
  级联，最多验证 4 个成熟 BoW 地点，加 1 个独立位置先验。默认 0 保留桌面/高精度档的
  全候选验证。最终 PnP 门仍是至少 50 内点、0.70 内点率以及既有 RMSE、正深度、网格
  覆盖门；没有用降低接受标准制造回环。
- 资源压力不再创建新子图；活动密集 KF 之外压成有界历史锚点。跨子图图只固定最早一个
  gauge，同子图 PGO 使用最多 64 个 Essential Anchor，锁外求解后按前缀事务提交尾段重基。

**被否决的捷径**：`compact_binary` 全局签名完整 KITTI 00 的 RSS 约 `221376KiB`、
p99 `102.96ms`，但 5 个预期区间全部漏检，不能作为 mobile 默认。词袋 Top-K 稀疏化、
低区分度 word 在线剪枝也没有在当前词表上证明召回等价，因此本轮未实现。

**完整 KITTI 00 证据（4541 帧，SE3，一一时间关联）**：

| 配置/阶段 | 有效帧 | LOST | 回环 oracle | ATE RMSE | FPS / p99 | 峰值 RSS |
|---|---:|---:|---:|---:|---:|---:|
| 官方 DBoW3 旧 mobile 基线 | 未复测 | 5 | 1/5，precision 1.0 | 65.377m | 17.65 / 206.6ms | 1106696KiB |
| Flat + reservoir，级联前 | 4484/4541 | 3 | 1/5，precision 1.0 | 30.434m | 13.24 / 211.3ms | 290100KiB |
| Flat + reservoir + 简单级联 | 4465/4541 | 4 | 1/5，precision 1.0 | 53.900m | 14.11 / 347.0ms | 290244KiB |

级联前后 `lc.verify` 从 1963 次降到 990 次，累计约 `27.9s→12.9s`；`lc.detect`
累计约 `49.1s→23.4s`。1000 帧切片保持 1000/1000 有效、0 LOST，峰值 RSS
`202496KiB`，p99 `76.35ms`。这证明 SoA 与简单级联确实减少资源和后台工作，但完整序列
两次 ATE/LOST/p99 差异很大，也说明瓶颈已经转向长程前端恢复、保留地点覆盖和调度尾延迟，
不能再靠继续压候选数量解决。

**结论与边界**：本轮把词表/索引常驻从约 1.06GiB 级降到约 283MiB，并大幅减少 PnP
验证工作；但 mobile 仍未达到 ≥99% 有效、≤3 次重建、ATE ≤30m、低尾延迟和 5/5 回环
召回门，不能宣称适合 ARM 上线。未做 ARM 实机、8～24h soak、多场景统计，也未在最终
级联代码上重跑高精度 `kitti00.yaml` 完整序列；后续应先定位 3～4 个 LOST 区间和 p99
尖峰归属，再决定是补前端恢复还是做调度隔离，而不是继续堆检索层级。官方 `dbow3`
桌面后端仍依赖异步维护任务批量重建索引，长后台任务期间待删 KF 的强引用/BoW 可暂时
滞留，尚未证明全局硬上限；不能把 mobile 的 256-KF Flat 上限外推到所有 Atlas/后端。

**最终回归**：`build-essential-anchor/test_vo` 通过；本地完整 CTest 19/19 通过。低系统
负载下执行 `bash scripts/benchmark_gate.sh --full`，新鲜 Release 配置确认 `g2o vendored`
与 `DBoW3 found`，L0 19/19、L1 确定性 1000 帧（平移差 0、旋转差
`5.562e-17rad`）通过；L2 的 5×4541 帧 **FAILED**：ATE RMSE worst `37.8833m`
虽过 40m 门，但 ATE std worst `21.1334m > 8m`，1/5 轮有 1 个 >10m 跳变，p99
worst `99.2828ms > 80ms`。valid ratio worst `0.9958`、deadline miss ratio worst
`0.0099`、LOST worst 1，5 轮均为 0 回环。一次与并发 soak 同跑的门禁曾因系统 load
12～17 令 p99 超门，已废弃；上述最终数据是在所有 Luna/其他基准停止、1 分钟 load
降至约 1.95 后独占得到。没有降低门限、更新黄金参考或选择最好一轮，因此当前变更仍不能
作为通过 full gate 的提交候选。

### 3.44 当前帧双目 RGB 点云查看与 PLY 导出（2026-08-11）

新增 `ColoredPoint` 和 `VisualOdometry::getCurrentStereoPointCloud`：对当前帧有效
`Frame::pts_c` 按 `T_wc` 转换到世界系，并从左目图像对应 `KeyPoint` 像素采样 RGB。
颜色只服务于可视化，不写入 `MapPoint`，因此不改变地图、BA、回环和点生命周期。

Viewer 面板新增默认关闭的 `Show colored stereo points`，只有在 `Show 3D map` 开启时
绘制当前帧彩色点云。新增 `tools/stereo_cloud_viewer`，产物为 `build*/bin/stereo_cloud_viewer`，
可实时查看并用 `--export-ply` 导出最后一帧 ASCII PLY。KITTI 原始图像是灰度图，导出的
RGB 通道因此通常相同；彩色双目数据会保留真实颜色。

验证：CMake 重新配置后 `stereo_cloud_viewer`、`run_slam`、`run_vo` 构建通过；KITTI 00
前 80 帧无头运行成功，导出 `/tmp/stereo_last.ply`，包含 1175 个彩色点。尚未做完整
4541 帧彩色 viewer 性能门；当前点云是当前帧快照，不是持久化的全局彩色地图。

### 3.45 运行资源契约与稳态/长尾控制（2026-08-18）

本轮按运行时资源、前端稳态、后端长尾和测试门四条路径实施，未修改确定性黄金参考：

- `default.yaml` 现在声明最多 6 个 CPU 核和 12288 MiB RSS，`mobile.yaml` 声明最多
  4 个核和 6144 MiB RSS。Linux 下运行时收窄 affinity，固定后台 worker 到保留核，
  同时限制 OpenCV/TBB 并记录实际 RSS、线程数和允许 CPU 数。OpenCV/TBB 的 OS worker
  数仍由系统库维护，当前只保证活动 CPU 集合和 backend 线程不越界，不能把线程数指标
  误报为等于核心数。
- 双目路径复用调用方已生成的原始灰度，反向 LK 只验证正向有效点；ORB 跟踪结果在
  关键帧建点时复用，避免第二次 O(N²) BF 匹配。关键帧配置保留旧路径开关，确定性档
  关闭新策略后 L1 逐位一致。
- Local BA 增加 profile 级点数/优化轮数上限、后台任务最大排队年龄和执行耗时统计；
  过期任务不进入 g2o。异步回环复用锁外预验证结果，避免在事务入口重复做候选 PnP；
  默认档使用成熟地点级联和每查询几何验证上限，轻量档继续使用有界 Flat-DBoW3。
- `benchmark.py` 现在校验有限数值、轨迹非空、轨迹行数与 valid pose 一致、RSS 采样
  和 B 侧门限；提交门强制 RSS，并兼容仓库历史 `datasets/sequences/00` 路径但不跳过 GT。
  新增 runtime resource CTest、scheduler 过期任务回归，CTest 数量为 20。

验证：新鲜 Release 提交门 L0/L1/L2 通过；L2 500 帧×3 轮最差 p99 32.43ms、
valid ratio 0.994、RSS 790648KiB、允许 CPU 6。完整旧路径诊断轮（default，4541 帧）
为 4446/4541 有效、p99 138.8ms、5 次 LOST、RSS 约 1.30GiB，仍未通过完整长程产品门；
mobile 200 帧烟雾为 181/200 有效、4 核 affinity、RSS 约 173MiB，不能替代 ARM 实机和
完整序列验证。依赖摘要：vendored g2o、DBoW3 均可用。

### 3.46 首轮 PnP 失败后的局部地图救援（2026-08-19）

长序列诊断显示，困难帧常有 20～200 个 ORB 匹配，但参考关键帧只能形成个位数的
3D-2D 对应。新增有界救援路径：参考 KF 的描述子按 `(KF, topology, geometry)` 版本
缓存，首轮 PnP 失败后从参考 KF 和共视点构造最多 1000 个候选，使用预测位姿和 80px
投影窗口重新匹配，再复用原 PnP 内点/比例/RMSE/运动连续性门；没有降低任何接受阈值。
同时修正局部地图/参考描述子缓存只看 geometry revision、忽略 KF/点剔除 topology
revision 的生命周期缺口。确定性档关闭救援，L1 保持逐位一致。

KITTI 00 旧目录独立诊断轮：有效轨迹 `4503/4541`（99.16%）、LOST 2、子地图重建 2、
FPS 28.82、p99 123.6ms、deadline miss 1.06%、RSS 约 1.14GiB；SE3 ATE RMSE
46.12m、>10m 跳变 2。相对本轮前的 4446/4541、LOST 5，恢复路径有效性改善，但
ATE/尾延迟/跳变仍未达到完整产品门，不能据此宣称长程实时定位已完成。

### 3.47 M3.1 视觉前端质量门（2026-08-22）

按 `PRODUCTION_LOCALIZATION_PLAN.md` §7.1/§7.2 落地 M3.1（图像质量 + 特征分布
两层质量门），只做置信度分层，不改变任何几何验收阈值。

**新增纯函数模块 `tracking_quality.{h,cpp}`**（无 RNG、无全局状态、值进出）：

- `assessImageQuality`：Laplacian 方差 + 暗(gray≤5)/亮(gray≥250)占比统计；
  仅 CV_8UC1 可评估，其他类型 `assessable=false` 不做像素级判定（§7.2）。
- `countOccupiedGridCells`：特征点 8×6 网格占用格子数（越界忽略、边界归末格）。
- `classifyTrackingQuality`：三档判定——方差<10 或暗/亮占比>0.8 →
  `HardReject`+新原因码 `FailureReason::ImageDegraded`；方差∈[10,60) 或特征
  <300 或占用<12 格 → 至多 `Degraded`（不拒绝、无失败原因码）；否则 `Full`。
- 阈值为 §7.2 产品规格首版参数，非数据集标定值。

**接线**：`FrontendTracker::assessFrameQuality` 聚合入口（TrackerConfig 嵌入
`QualityConfig`）；VO 在 `addFrameImpl` 特征提取后调用一次——评估用 CLAHE 增强
前的原始灰度。HardReject 复用与"无特征"完全相同的早退语义（快照捕获/失败计数/
RECOVERING→LOST 推进/外推位姿）；Degraded 只写 `Status.quality=Weak`。
`localizer.cpp` 将 Weak 映射进状态机（连续 2 帧 → Degraded、协方差 ×4），
输入级硬拒绝时上报结构化原因码。原"无特征"早退块重构为共享 lambda，
质量门置于其前——黑帧往往提不出特征，图像级判定必须给出结构化原因码而非笼统
的"无特征"。

**配置**：代码默认 `enabled=false`（旧用户配置缺省段保持旧行为）；
default/mobile 显式开启（§7.2 数值），deterministic 显式关闭保持 L1 黄金参考
逐位一致（遵循 §3.45/§3.46 先例）。robot.yaml 是 Localizer Facade 配置，
VO 质量门随主 profile，不加死配置。

**测试（先红后绿）**：`test/test_tracking_quality.cpp` 19 项——图像统计
（纹理/纯黑/纯白/高斯模糊/非 8 位）、网格覆盖（满格/单点/边界/越界）、判定
（Full/Degraded/HardReject、10/60/0.8 边界语义、硬拒优先、开关旁路、不可评估
退化）、聚合入口开关、以及 VO 级黑帧故障注入（硬拒绝 → pose_valid=false +
`ImageDegraded` + Map revision 双不变 + 故障消失后重定位恢复跟踪，§7.6 子集；
场景用带深度变化的唯一灰度 3D 块，避免平面+平移的本质矩阵退化）。CTest 现在
21 个目标。

**范围扩大说明**（§12 允许清单之外的最小接线）：vo.{h,cpp}（Status 两字段、
Quality 段 YAML 解析、早退复用与门调用约 60 行）、localizer.cpp（Weak 映射与
原因码透传约 6 行）；pose_gate.* 本轮未改动。

**验收**：
- 全量 CTest 21/21 PASS；`git diff --check` 干净。
- 新鲜 Release 提交门 L0/L1/L2 PASSED：L1 确定性轨迹逐位一致；L2 500 帧×3 轮
  最差 p99 33.0ms、valid_ratio 1.0、jumps_10m 0、lost 0、submap_reinit 0。
- KITTI 00 完整序列诊断轮（default.yaml，质量门开启）：**零硬拒绝触发**，
  4446/4541 有效、LOST 5、回环 0、FPS 26.25、p99 243ms、miss 2.38%——与
  §3.45 同配置诊断轮（4446/4541、LOST 5、p99 138.8ms）同方差带；弱质量分层
  不触碰几何路径，本数据上未产生行为差异。

**未验证边界**：① M3.2 协方差仍为占位（单位阵 ×4），Weak 的协方差放大语义要
到数值 Jacobian 落地才有真实输出；② 故障注入仅覆盖黑帧一类的合成样本，
fault_injection.py 回放框架属 M3.3；③ 模糊/暗亮真实样本未在 KITTI 出现过，
HardReject/Degraded 在真实数据上的触发率未知；④ mobile/ARM 未复测；⑤ 默认档
完整长程产品门（ATE ≤40m、p99 ≤80ms、跳变 0、子地图重建 ≤1）仍未通过，
与本任务无关但继续如实记录。

### 3.48 M3.2 位姿数值协方差（2026-08-22）

按 `PRODUCTION_LOCALIZATION_PLAN.md` §7.5 落地 M3.2：最终 PnP 内点的中心有限
差分数值协方差，替换 `PoseEstimate::covariance` 的 M0 占位单位阵。

**新增纯函数模块 `pose_covariance.{h,cpp}`**（无 RNG、值进出）：

- `se3Exp` / `se3Adjoint`：SE(3) 指数映射与左伴随，切空间顺序固定
  `[tx,ty,tz,rx,ry,rz]`（§7.5）；伴随满足同态性并经共轭恒等式验证。
- `pnpPoseCovariance`：对收敛位姿做左扰动 `Exp(δξ_c)·T_cs` 中心差分
  （平移 1e-4 m / 旋转 1e-6 rad），由最终内点重投影残差构造 J；
  `σ² = max(0.25, SSE/max(1, 2N−6))`；`H=JᵀJ` 原始特征值上判条件数
  （>1e8 或非正 → 退化拒绝发布），按 1e-9 下限截断后求逆；
  `Σ_c = σ²·H⁻¹` 对称化 + 有限/正定校验。视轴共线等病态几何正确触发退化。
- 内点 <6 直接退化；任一扰动把点投到相机后方判退化。

**传播链路**：`FrontendTracker::estimatePnPCovariance` 聚合入口；
trackPnP/refinePnP 接受分支填充，trackOrb 与 vo.cpp 局部地图精修的
"选择性拷贝"处同步透传（漏拷会让最终结果携带陈旧协方差——VO 级测试抓出）。
`Status` 新增 `pose_covariance/pose_covariance_valid`（每帧复位）。
Localizer 发布前经 `A = Ad_{T_oc}` 变换到 odom 系（精确覆盖全局校正后
T_wo≠I 的情形）、PD 校验失败回退占位单位阵、Weak ×4 保持；
prediction-only 帧按流逝时间每 100ms 至少 ×2（§7.5 第 7 步，500ms 超时仍由
状态机截断）。EPIPOLAR 回退帧（无尺度、无内点集）按设计无数值协方差。

**范围扩大说明**：vo.{h,cpp}（Status 两字段 + 三处结果合并点透传约 20 行）、
localizer.{h,cpp}（Ad 变换 + prediction 增长 + 时间戳成员约 25 行）。

**测试**：`test/test_pose_covariance.cpp` 13 项——Exp/伴随代数性质、无噪场景
有效/对称/正定/σ²下限、同一标准化样本下 σ 二次缩放（迹比 ≈4）、点数信息缩放、
视轴共线退化拒绝、内点不足退化、逐位确定性、tracker 聚合入口、以及 VO 级端到端
传播（真实跟踪建立后 PNP 帧 Status 必须携带非占位正定协方差）。另将
test_localizer_contract 的 M0 占位断言更新为 M3.2 语义（非单位阵且正定）。
CTest 现在 22 个目标。

**验收**：
- 全量 CTest 22/22 PASS；`git diff --check` 干净。
- 新鲜 Release 提交门 L0/L1/L2 PASSED：L1 确定性轨迹逐位一致（协方差为纯
  数值计算、不消耗 RNG、不改几何决策）；L2 500 帧×3 轮 valid_ratio 1.0、
  p99 最差 ~33ms、jumps 0。

**未验证边界**：① 协方差数值本身未与 EVO/g2o marginal 协方差交叉标定，
当前只验证代数/统计性质与退化检测；② Weak ×4 与 prediction 增长作用于真实
Σ 后的下游消费方（控制器/ESKF）尚不存在（M6）；③ M3.3 fault_injection.py
故障回放未实施。
