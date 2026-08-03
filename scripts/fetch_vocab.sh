#!/usr/bin/env bash
# ============================================================
# fetch_vocab.sh - 获取 ORB 词袋词典（Phase 2 回环检测用）
#
# 词典来源（按优先级）：
#   1. 本地 DBoW3 源码自带 orbvoc.dbow3（/tmp/DBow3，DBoW3 官方
#      二进制格式，约 48MB，无需下载）
#   2. gitee 镜像克隆 DBoW3 后取 orbvoc.dbow3
#   3. GitHub 克隆 DBoW3 后取 orbvoc.dbow3
#   4. 下载 ORB-SLAM2 的 ORBvoc.txt.tar.gz（DBoW2 文本格式，
#      约 30MB，DBoW3 可直接加载）
#
# 用法: bash scripts/fetch_vocab.sh [目标目录]（默认 config/）
# ============================================================
set -euo pipefail

TARGET_DIR="${1:-config}"
TARGET_FILE="$TARGET_DIR/ORBvoc.dbow3"
mkdir -p "$TARGET_DIR"

# 已存在则直接复用（幂等）
if [ -f "$TARGET_FILE" ]; then
    echo "词典已存在: $TARGET_FILE"
    exit 0
fi

# ---- 1. 本地 DBoW3 源码 ----
if [ -f /tmp/DBow3/orbvoc.dbow3 ]; then
    echo "使用本地 DBoW3 源码自带词典 /tmp/DBow3/orbvoc.dbow3"
    cp /tmp/DBow3/orbvoc.dbow3 "$TARGET_FILE"
    echo "词典已复制到: $TARGET_FILE"
    exit 0
fi

# ---- 2/3. 克隆 DBoW3（先 gitee 后 github）----
DBOW3_SRC=/tmp/DBow3-fetch
rm -rf "$DBOW3_SRC"
for REPO in \
    "https://gitee.com/mirrors/DBow3.git" \
    "https://github.com/rmsalinas/DBow3.git"; do
    if git clone --depth 1 "$REPO" "$DBOW3_SRC" 2>/dev/null; then
        if [ -f "$DBOW3_SRC/orbvoc.dbow3" ]; then
            cp "$DBOW3_SRC/orbvoc.dbow3" "$TARGET_FILE"
            echo "词典已复制到: $TARGET_FILE (来自 $REPO)"
            exit 0
        fi
    fi
done

# ---- 4. 兜底：下载 ORB-SLAM2 的 ORBvoc.txt.tar.gz（DBoW2 文本格式）----
echo "DBoW3 克隆失败，尝试下载 ORB-SLAM2 的 ORBvoc.txt.tar.gz ..."
TMP_TAR=$(mktemp /tmp/ORBvoc.XXXXXX.tar.gz)
for URL in \
    "https://raw.githubusercontent.com/raulmur/ORB_SLAM2/master/Vocabulary/ORBvoc.txt.tar.gz"; do
    if curl -fL --connect-timeout 30 -o "$TMP_TAR" "$URL"; then
        tar -xzf "$TMP_TAR" -C "$TARGET_DIR"
        # 解压得到 ORBvoc.txt → 重命名，DBoW3 按扩展名识别格式
        if [ -f "$TARGET_DIR/ORBvoc.txt" ]; then
            mv "$TARGET_DIR/ORBvoc.txt" "$TARGET_FILE"
            echo "词典已解压到: $TARGET_FILE (来自 $URL)"
            exit 0
        fi
    fi
done

echo "ERROR: 所有词典获取方式均失败，请检查网络后重试" >&2
exit 1
