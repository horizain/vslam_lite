# VSLAM 开发日志

> 创建日期: 2026-07-30
> 最后更新: 2026-08-01 (位姿语义统一修复 + 测试基线 + Viewer 重构 + 配置接入)

---

## 一、项目概述

从零构建教学级 VSLAM 系统，遵循第一性原理，只保留核心逻辑，适合初学者逐文件阅读学习。

**技术栈:** C++17 / OpenCV / Eigen3 / Pangolin / g2o / DBoW3(Phase2)

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
| `include/vslam/vo.h` | ✅ 完成 | VisualOdometry: 状态机, VOConfig 配置结构, PnP跟踪, 关键帧策略 |
| `src/vo.cpp` | ✅ 完成 | 初始化+跟踪+PnP+对极回退+三角化+关键帧判决+Local BA 集成 |
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
| `src/viewer.cpp` | ✅ 完成 | **重构(2026-08-01)**: 左=2D 俯视轨迹图(x-z 平面, 青色轨迹+红色当前位置, 自动缩放)，右=彩色视频流+绿色特征点(优先彩色, 灰度自动转BGR), 宽高比自适应图像尺寸 |

### 2.5 数据输入

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/dataset.h` | ✅ 完成 | Dataset: KITTI/TUM/EuRoC/CAMERA 四类型 |
| `src/dataset.cpp` | ⚠️ 部分 | **KITTI**: ✅ 完成(彩色读取→VO 内转灰度, Viewer 显示彩色)<br>**CAMERA**: ✅ 完成(cv::VideoCapture)<br>**TUM**: ❌ TODO - 需解析rgb.txt关联时间戳<br>**EUROC**: ❌ TODO - 需解析cam0/data.csv |

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

---

## 三、待完成的工作

### 3.1 Phase 1 剩余任务（优先级：高）

#### 🔴 P0 - KITTI 数据集实测（被数据下载阻塞）

- [x] 安装依赖并编译 ✅
- [x] 编译错误修复 ✅
- [ ] **KITTI 数据集实测**：当前 `/home/ruijianding/data/kitti/data_odometry_gray.zip` 为损坏文件(60KB, BadZipFile)。
  使用 `scripts/prepare_kitti.sh` 准备数据后运行：
  `./build/bin/run_vo ~/data/kitti/sequences/00/image_0 config/default.yaml trajectory_00.txt`
- [ ] **结果评估**：与 KITTI ground truth(`~/data/kitti/poses/00.txt`) 用 EVO 对比 ATE/RPE
- [ ] **参数调优**：keyframe 阈值、ORB 特征数 vs 速度、RANSAC 阈值（现全部可在 yaml 调整）

#### 🟡 P1 - 后端优化（g2o）

- [x] `src/optimizer.cpp` → `localBundleAdjustment()` ✅ 已实现+集成（滑动窗口）
- [ ] 共视图滑动窗口增强：按共视地图点数选帧（当前按时间窗取最近 N 帧）

#### 🟢 P2 - 完善与增强

- [ ] **TUM 数据集支持**: 解析 `rgb.txt` + 关联时间戳，单目模式(无深度)
- [ ] **EuRoC 数据集支持**: 解析 `cam0/data.csv`，加载 IMU 数据(可选)
- [ ] **LK 光流模式接入**: `VOConfig::feature_method = 1`（接口已就绪，trackFrame 尚未实现 LK 路径）
- [ ] **关键帧策略增强**：匹配点数衰减触发关键帧
- [ ] **LOST 状态恢复优化**：重定位目前遍历全部关键帧全图匹配，慢（Phase 2 词袋后解决）
- [ ] **编译安装脚本**: `scripts/install_deps.sh` 自动化安装依赖

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
4. **Viewer 状态栏**：图像底部叠加状态文字，KITTI 图像较窄(1241×376)时文字可能溢出
5. **单目尺度**：VO 轨迹为归一化尺度，与真值对比需先做尺度对齐（EVO 默认支持）

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
│   ├── dataset.cpp                 # ⚠️ KITTI+CAMERA完成, TUM/EuROC TODO
│   ├── mappoint.cpp                # ✅ 完成
│   ├── map.cpp                     # ✅ 完成
│   ├── feature.cpp                 # ✅ 完成
│   ├── vo.cpp                      # ✅ 完成(核心管线, T_cw 语义统一)
│   ├── optimizer.cpp               # ✅ 完成(Local BA)
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

## 六、下次开发入口清单

按优先级排列：

1. **🔴 KITTI 实测**: 下载数据 → `scripts/prepare_kitti.sh` → `run_vo` → EVO 评估 ATE/RPE
2. **🟡 LK 光流接入**: `trackFrame()` 支持 `feature_method=1`（接口已就绪），非关键帧提速
3. **🟡 共视图滑动窗口**: Local BA 窗口按共视关系选帧（替代时间窗）
4. **🟢 TUM/EuRoC 数据集支持**: 解析时间戳文件
5. **🔵 Phase 2**: DBoW3 安装 → 回环检测 + PoseGraph + 全局 BA
