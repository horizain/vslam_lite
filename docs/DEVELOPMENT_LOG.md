# VSLAM 开发日志

> 创建日期: 2026-07-30
> 最后更新: 2026-07-30 (编译通过 + 摄像头实时测试成功)

---

## 一、项目概述

从零构建教学级 VSLAM 系统，遵循第一性原理，只保留核心逻辑，适合初学者逐文件阅读学习。

**技术栈:** C++17 / OpenCV / Eigen3 / Pangolin / g2o(可选) / DBoW3(Phase2)

---

## 二、已完成的工作（截至 2026-07-30）

### 2.0 编译与环境验证 ✅ (2026-07-30)

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 依赖安装 | ✅ | OpenCV 4.10 + Eigen 3.4 + yaml-cpp 0.8 + g2o (apt) |
| Pangolin | ✅ | gitee 镜像源码编译 (v0.6) |
| CMake 编译 | ✅ | 100% 通过 (修复 OpenCV/Pangolin target 名 + cv::Mat→Eigen 转换) |
| 合成图像测试 | ✅ | 100帧 @ 32FPS, 490地图点, 4关键帧, 轨迹(x,z)连续正确 |
| WSL2 摄像头 | ✅ | usbipd-win → /dev/video0 (HP True Vision HD Camera) |
| 摄像头 VO 实测 | ✅ | 实时初始化(34匹配/50特征) + PnP跟踪 + 关键帧创建 |

### 2.1 项目基础设施

| 文件 | 状态 | 说明 |
|------|------|------|
| `CMakeLists.txt` | ✅ 完成 | C++17, 自动检测 g2o/DBoW3(可选), Release/Debug 模式 |
| `cmake/Findg2o.cmake` | ✅ 完成 | g2o 查找模块 |
| `config/default.yaml` | ✅ 完成 | 默认配置(KITTI 内参, ORB/PNP/关键帧参数) |

### 2.2 核心数据结构（头文件+实现）

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/common.h` | ✅ 完成 | SE3位姿(Vec3 t + Quaternion q), 矩阵互转, 逆/组合/点变换, 日志宏 |
| `include/vslam/camera.h` | ✅ 完成 | 针孔模型, Yaml加载, world2pixel, pixel2camera |
| `src/camera.cpp` | ✅ 完成 | Yaml解析, 投影/反投影实现 |
| `include/vslam/frame.h` | ✅ 完成 | Frame(Ptr), 图像/灰度图/关键点/描述子/位姿/地图点关联 |
| `include/vslam/mappoint.h` | ✅ 完成 | MapPoint(Ptr), 3D位置/描述子/观测计数, 三角化工厂方法 |
| `src/mappoint.cpp` | ✅ 完成 | cv::triangulatePoints 三角化实现(深度正向检查) |
| `include/vslam/map.h` | ✅ 完成 | Map: 关键帧+地图点集合, 增删查, cullMapPoints(剔除不可靠点) |
| `src/map.cpp` | ✅ 完成 | 所有集合操作实现 |

### 2.3 视觉里程计前端

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/feature.h` | ✅ 完成 | FeatureMatcher: ORB提取, BF+knn+ratio匹配, LK光流, 匹配点提取 |
| `src/feature.cpp` | ✅ 完成 | **完整实现**: ORB 1000特征, 暴力匹配+比率测试(0.7)+RANSAC基础矩阵外点剔除 |
| `include/vslam/vo.h` | ✅ 完成 | VisualOdometry: 状态机(INITIALIZING/TRACKING/LOST), PnP跟踪, 关键帧策略 |
| `src/vo.cpp` | ✅ 完成 | **完整实现**: 初始化(对极几何+视差检查)+跟踪(PnP优先,对极回退)+三角化+关键帧判决 |

**VO 管线已实现:**

```
初始化 (INITIALIZING):
  Frame1 → 存储ref_frame → Frame2 → ORB匹配+RANSAC → findEssentialMat
  → recoverPose → 视差检查(>0.1) → ref_frame设为世界原点
  → 三角化初始MapPoint → 插入为KeyFrame → 切换到 TRACKING

跟踪 (TRACKING):
  当前帧 vs ref_frame:
    有>=6个3D-2D对应 → solvePnPRansac(PnP, 内点阈值4px)
    匹配不足 → findEssentialMat + recoverPose(2D-2D回退)
    完全失败 → 恒等运动假设 + LOST

关键帧选择:
  当前帧 vs 上一关键帧 平移距离 > 0.5m → 创建新关键帧
  → ORB匹配 → 三角化新地图点 → 更新ref_frame
```

### 2.4 可视化

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/viewer.h` | ✅ 完成 | Viewer: 独立渲染线程, updateFrame线程安全 |
| `src/viewer.cpp` | ✅ 完成 | **完整实现**: Pangolin双窗口(左侧:OpenCV图像+绿色特征点叠画, 右侧:2D俯视轨迹x-z平面+青色线条+红色当前位置), 自动缩放视野 |

### 2.5 数据输入

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/vslam/dataset.h` | ✅ 完成 | Dataset: KITTI/TUM/EuRoC/CAMERA 四类型 |
| `src/dataset.cpp` | ⚠️ 部分 | **KITTI**: ✅ 完成(cv::glob图片列表+按序读取+假10fps时间戳)<br>**CAMERA**: ✅ 完成(cv::VideoCapture)<br>**TUM**: ❌ TODO - 需解析rgb.txt关联时间戳<br>**EUROC**: ❌ TODO - 需解析cam0/data.csv |

### 2.6 应用入口

| 文件 | 状态 | 说明 |
|------|------|------|
| `app/run_vo.cpp` | ✅ 完成 | 命令行入口, 自动识别数据集路径 vs 摄像头索引, 主循环(数据集读取→VO→Viewer更新→FPS统计) |
| `app/run_slam.cpp` | 🔜 Phase 2 | 已注释在 CMakeLists.txt 中 |

### 2.7 测试

| 文件 | 状态 | 说明 |
|------|------|------|
| `test/test_vo.cpp` | ✅ 骨架 | 4个测试用例: SE3运算, Camera投影, ORB提取(合成图像), VO初始化(合成图像) |

### 2.8 文档

| 文件 | 状态 | 说明 |
|------|------|------|
| `docs/DEVELOPMENT_LOG.md` | ✅ 本文档 | 开发日志 |

---

## 三、待完成的工作

### 3.1 Phase 1 剩余任务（优先级：高）

#### 🔴 P0 - 编译验证与数据集测试

- [ ] **安装依赖并编译**：安装 OpenCV/Eigen/Pangolin/yaml-cpp，执行 `cmake .. && make`
- [ ] **编译错误修复**：检查是否有遗漏的头文件、类型不匹配等问题
- [ ] **KITTI 数据集实测**：下载 KITTI odometry 序列，验证 VO 初始化+跟踪+可视化
- [ ] **结果评估**：与 KITTI ground truth 对比，评估 ATE/RPE
- [ ] **参数调优**：
  - `keyframe_translation` 阈值(当前0.5m可能太大/太小)
  - ORB 特征数量 vs 速度的权衡
  - RANSAC 阈值

#### 🟡 P1 - 后端优化（g2o 局部 BA）

- [ ] **`src/optimizer.cpp` → 实现 `localBundleAdjustment()`**
  - 文件中已有详细步骤注释（6步）
  - 参考: g2o `tutorial_sba2d` 示例
  - 核心: VertexSE3Expmap(位姿) + VertexSBAPointXYZ(3D点) + EdgeSE3ProjectXYZ(重投影)
  - 需要在 `vo.cpp` 中集成调用：每插入关键帧后对滑动窗口执行局部BA
- [ ] **滑动窗口策略**：维护最近N个关键帧+共视地图点

#### 🟢 P2 - 完善与增强

- [ ] **TUM 数据集支持**: 解析 `rgb.txt` + 关联时间戳，单目模式(无深度)
- [ ] **EuRoC 数据集支持**: 解析 `cam0/data.csv`，加载 IMU 数据(可选)
- [ ] **LK 光流模式测试**: `feature_method_ = 1`，对比 ORB 匹配的效果/速度
- [ ] **关键帧策略增强**：增加旋转阈值 + 匹配点数衰减触发
- [ ] **LOST 状态恢复**：重新初始化或回环检测恢复(需 Phase 2)
- [ ] **配置文件加载优化**: 从 yaml 读取所有 VO/Feature 参数
- [ ] **位姿保存**: 输出 KITTI/TUM 格式位姿文本文件，便于评估工具(EVO)对比
- [ ] **编译安装脚本**: `scripts/install_deps.sh` 自动化安装依赖

### 3.2 Phase 2 - 完整 SLAM（优先级：中）

#### 🔵 回环检测

- [ ] **DBoW3 依赖安装**: 编译 DBoW3, cmake 集成
- [ ] **`src/loop_closure.cpp` → 完整实现**
  - 加载 ORB 词袋词典(需生成或下载预训练)
  - 关键帧插入时提取 BowVector
  - 相似度检测 + 候选帧筛选(排除时间邻近帧)
  - 几何一致性验证(3D-2D PnP)
- [ ] **回环校正**: 计算回环帧间的 Sim3 变换，校正当前帧位姿
- [ ] **回环边创建**: 为 Pose Graph Optimization 添加回环约束

#### 🔵 后端优化增强

- [ ] **`optimizer.cpp` → 实现 `poseGraphOptimization()`**
  - 仅优化关键帧位姿(不优化地图点)
  - 帧间相对位姿约束 + 回环约束
- [ ] **`optimizer.cpp` → 实现 `globalBundleAdjustment()`**
  - 全部关键帧 + 全部地图点参与优化
  - 仅在回环后触发(非实时)

#### 🔵 地图管理

- [ ] **MapPoint 质量维护**: 观测方向角度检查, 重投影误差检查
- [ ] **关键帧冗余剔除**: 共视比例 > 90% 的冗余关键帧删除
- [ ] **地图保存/加载**: 序列化为 YAML/二进制文件以便复用

---

## 四、已知问题与注意事项

### 4.1 技术债务

1. **`vo.cpp` 的 `trackFrame()`** 每次都重新做一次特征匹配(ref_frame ↔ curr_frame)
   - 优化方向：对非关键帧，用 LK 光流跟踪代替 ORB 重新提取+匹配
   - 当前: 每帧都 extract + match → 速度瓶颈

2. **SE3 位姿表示**（`common.h`）使用 `Quaterniond + Vec3`，但内部 `cv::Mat R` 转换是 RowMajor → Eigen 默认 ColMajor
   - 当前代码使用 `Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>` 处理，需验证正确性

3. **`MapPoint::create` 三角化** 使用 `cv::triangulatePoints` 返回的是齐次坐标(4D)，除以 w 后单位是归一化的
   - 需要确认位姿 T_cw 的语义：`p_w = T_cw * p_c` 还是 `p_c = T_cw * p_w`

4. **Viewer Pangolin 纹理** 动态 Reinitialise 每次尺寸变化可能有效率问题
   - 摄像头分辨率固定时没问题，数据集加载时仅在首帧触发一次

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
├── config/default.yaml             # KITTI 默认参数
├── docs/
│   └── DEVELOPMENT_LOG.md          # ← 本文档
├── app/
│   └── run_vo.cpp                  # VO 入口(数据集/摄像头)
├── include/vslam/
│   ├── common.h                    # SE3, Vec2/3, Mat33/44
│   ├── camera.h                    # 针孔相机模型
│   ├── dataset.h                   # 数据集/摄像头抽象
│   ├── frame.h                     # 帧数据结构
│   ├── mappoint.h                  # 地图点
│   ├── map.h                       # 地图管理
│   ├── feature.h                   # ORB+LK特征
│   ├── vo.h                        # VO状态机
│   ├── optimizer.h                 # g2o优化接口
│   ├── viewer.h                    # Pangolin双窗口
│   └── loop_closure.h              # 回环检测(Phase2)
├── src/
│   ├── camera.cpp                  # ✅ 完成
│   ├── dataset.cpp                 # ⚠️ KITTI+CAMERA完成, TUM/EuROC TODO
│   ├── mappoint.cpp                # ✅ 完成
│   ├── map.cpp                     # ✅ 完成
│   ├── feature.cpp                 # ✅ 完成
│   ├── vo.cpp                      # ✅ 完成(核心管线)
│   ├── optimizer.cpp               # ❌ TODO(g2o BA)
│   ├── viewer.cpp                  # ✅ 完成(Pangolin)
│   └── loop_closure.cpp            # ❌ TODO(Phase2)
└── test/
    └── test_vo.cpp                 # ✅ 4个测试骨架
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

1. **🔴 立即**: 安装依赖 → cmake 编译 → 修复编译错误 → 跑通 KITTI
2. **🟡 然后**: 实现 g2o 局部 BA (`optimizer.cpp`)，集成到 VO 跟踪
3. **🟢 接着**: TUM/EuRoC 数据集支持，参数调优，性能测试
4. **🔵 最后**: Phase 2 回环检测+全局优化+地图管理
