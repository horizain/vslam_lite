#!/usr/bin/env bash
# ============================================================
# install_deps.sh - 安装 vslam 全部依赖
#
# 目标系统: Ubuntu 22.04 / 24.04 (含 WSL2)
# 安装内容:
#   1. apt 包: build-essential / cmake / git / OpenCV / Eigen3 / yaml-cpp / g2o / GL 依赖
#   2. Pangolin v0.6 (源码编译, 可视化, 源码 vendor 在 thirdparty/)
#   3. DBoW3 (可选, Phase 2 回环检测, 源码 vendor 在 thirdparty/)
#
# 依赖源码已 vendor 在仓库 thirdparty/ 下，版本与 commit 号见 docs/THIRD_PARTY.md，
# 本脚本不再联网 clone，可离线复现构建。
#
# 用法: bash scripts/install_deps.sh [--skip-dbow3]
# ============================================================
set -euo pipefail

SKIP_DBOW3=0
[ "${1:-}" = "--skip-dbow3" ] && SKIP_DBOW3=1

# 定位仓库根目录（脚本位于 scripts/ 下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PANGOLIN_SRC="$ROOT_DIR/thirdparty/Pangolin"
DBOW3_SRC="$ROOT_DIR/thirdparty/DBoW3"

echo "==== 1/3 安装 apt 依赖 ===="
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libopencv-dev \
    libeigen3-dev \
    libyaml-cpp-dev \
    libg2o-dev \
    libgl1-mesa-dev libglew-dev \
    libx11-dev libxkbcommon-dev \
    libwayland-dev libegl1-mesa-dev libglfw3-dev \
    libpng-dev libjpeg-dev libtiff-dev

echo "==== 2/3 编译安装 Pangolin (v0.6) ===="
if pkg-config --exists pangolin 2>/dev/null; then
    echo "Pangolin 已安装，跳过"
else
    [ -d "$PANGOLIN_SRC" ] || { echo "错误: 未找到 $PANGOLIN_SRC，请确认 thirdparty/ 已同步"; exit 1; }
    cd "$PANGOLIN_SRC"
    mkdir -p build && cd build
    # v0.6 在 GCC 13 下有若干头文件漏显式包含 <cstdint>，并且其旧 FFmpeg
    # 接口无法兼容 Ubuntu 24.04 的新版 FFmpeg。Viewer 不依赖 Pangolin Video，
    # 因此关闭该组件，并通过编译器统一预包含 cstdint。
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_PANGOLIN_PYTHON=OFF \
          -DBUILD_PANGOLIN_VIDEO=OFF \
          -DCMAKE_CXX_FLAGS="-include cstdint" \
          ..
    make -j"$(nproc)"
    sudo make install
    sudo ldconfig
    echo "Pangolin 安装完成"
fi

if [ "$SKIP_DBOW3" = "0" ]; then
    echo "==== 3/3 编译安装 DBoW3 (Phase 2 回环检测) ===="
    if pkg-config --exists DBoW3 2>/dev/null; then
        echo "DBoW3 已安装，跳过"
    else
        [ -d "$DBOW3_SRC" ] || { echo "错误: 未找到 $DBOW3_SRC，请确认 thirdparty/ 已同步"; exit 1; }
        cd "$DBOW3_SRC"
        mkdir -p build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j"$(nproc)"
        sudo make install
        sudo ldconfig
        echo "DBoW3 安装完成"
    fi
else
    echo "==== 3/3 跳过 DBoW3 (Phase 2 需要时再装) ===="
fi

echo ""
echo "全部依赖安装完成。现在可以构建 vslam:"
echo "  cmake -S . -B build && cmake --build build -j\$(nproc)"
