# VSLAM 开发日志

> 创建日期: 2026-07-30
> 最后更新: 2026-08-03 (修复 Viewer 图像倾斜、显示尺寸和 T_cw 轨迹语义)

---

## 一、项目概述

从零构建教学级 VSLAM 系统，遵循第一性原理，只保留核心逻辑，适合初学者逐文件阅读学习。

**技术栈:** C++23 / OpenCV / Eigen3 / Pangolin / g2o / DBoW3(Phase2)

---

## 二、已完成的工作

### 2.0 编译与环境验证 ✅

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 依赖安装 | ✅ | OpenCV 4.10 + Eigen 3.4 + yaml-cpp 0.8 + g2o (apt) |
| Pangolin | ✅ | gitee 镜像源码编译 (v0.6) |
| CMake 编译 | ✅ | 100% 通过 |
| 单元测试 | ✅ | 4 个用例 7 项断言全部 PASSED（2026-08-01 修复编译后） |
| WSL2 摄像头 | ✅ | usbipd-win → /dev/video0 |
| KITTI 实测 | ⚠️ | 数据下载受阻（见 4.2），已提供 `scripts/prepare_kitti.sh` 待用户自行下载 |

### 2.1 项目基础设施

| 文件 | 状态 | 说明 |
|------|------|------|
| `CMakeLists.txt` | ✅ 完成 | C++17, 自动检测 g2o/DBoW3(可选), Release/Debug 模式 |
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
| `include/vslam/mappoint.h` | ✅ 完成 | MapPoint(Ptr), 3D位置/描述子/观测计数, 三角化工厂方法 |
| `src/mappoint.cpp` | ✅ 完成 | cv::triangulatePoints 三角化实现 |
| `include/vslam/map.h` | ✅ 完成 | Map: 关键帧+地图点集合, 增删查, cullMapPoints, 全局 id 分配 |
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

统一约定 `pose_cw` 为 **T_cw（世界→相机）**：`p_c = pose_cw * p_w`

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
| `src/dataset.cpp` | ✅ 完成 | **KITTI**: ✅ 完成(彩色读取→VO 内转灰度)<br>**TUM**: ✅ 完成(解析 `rgb.txt` 时间戳 + 路径)<br>**EUROC**: ✅ 完成(解析 `cam0/data.csv`)<br>**CAMERA**: ✅ 完成(cv::VideoCapture) |

### 2.6 应用入口

| 文件 | 状态 | 说明 |
|------|------|------|
| `app/run_vo.cpp` | ✅ 完成 | 命令行入口, 自动识别数据集路径 vs 摄像头索引, 加载 yaml 配置, 主循环(读取→VO→Viewer→FPS统计), **结束后保存 TUM 格式轨迹**(时间戳+tx ty tz qx qy qz qw, T_wc, 供 EVO 评估) |
| `app/run_slam.cpp` | 🔜 Phase 2 | 已注释在 CMakeLists.txt 中 |

### 2.7 测试

| 文件 | 状态 | 说明 |
|------|------|------|
| `test/test_vo.cpp` | ✅ 通过 | 4 个用例 7 项断言全部 PASSED: SE3运算(3), Camera投影(2), ORB提取(1), VO初始化(1, 合成场景23地图点)。**2026-08-01 修复编译**(TEST宏语法/vslam::Eigen/imgproc缺失/字段名) |

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
