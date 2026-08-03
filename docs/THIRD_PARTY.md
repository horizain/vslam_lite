# 第三方依赖说明

本仓库将需要源码编译的第三方库 vendor 在 `thirdparty/` 下，以便离线可复现构建。
系统级 apt 依赖（OpenCV / Eigen3 / yaml-cpp / g2o / GL 系列）不纳入仓库，版本记录见下表。

## 一、系统级依赖（apt 安装）

| 依赖 | 版本（本机验证） | 安装方式 |
|------|------------------|----------|
| OpenCV | 4.6.0 | `apt install libopencv-dev` |
| Eigen3 | 3.4.0 | `apt install libeigen3-dev` |
| yaml-cpp | 0.8.0 | `apt install libyaml-cpp-dev` |
| g2o | apt 版 | `apt install libg2o-dev` |
| GL 系列 | - | `apt install libgl1-mesa-dev libglew-dev libglfw3-dev ...` |

## 二、源码编译依赖（vendor 在 thirdparty/）

| 依赖 | 版本 | 上游 commit | 仓库路径 |
|------|------|-------------|----------|
| Pangolin | v0.6 | `dd801d244db3a8e27b7fe8020cd751404aa818fd` | `thirdparty/Pangolin/` |
| DBoW3 | master | `1cc587bbfc08c0adb0ea2d00d81ad6cd4103ce39` | `thirdparty/DBoW3/` |

> 上游地址:
> - Pangolin: https://github.com/stevenlovegrove/Pangolin (tag v0.6)
> - DBoW3: https://github.com/rmsalinas/DBow3
>
> 如需更新，请在对应仓库 `git clone` 后切换到上述 commit，再重新同步 `thirdparty/`。

## 三、构建方式

统一由 `scripts/install_deps.sh` 从 `thirdparty/` 本地源码编译安装：

```bash
bash scripts/install_deps.sh            # 完整安装（含 DBoW3）
bash scripts/install_deps.sh --skip-dbow3  # 跳过 DBoW3
```

## 四、注意事项

- `thirdparty/` 仅保留源码，不包含构建产物与 DBoW3 预生成词袋二进制
  （`orbvoc.dbow3`，约 48MB，属于运行时产物，由用户自行生成或下载）。
- Pangolin 构建时关闭 Video 模块并预包含 `<cstdint>`，以兼容 GCC 13 / Ubuntu 24.04。
- DBoW3 为可选依赖（`-DHAS_DBOW3`），Phase 2 回环检测需要时才编译安装。
