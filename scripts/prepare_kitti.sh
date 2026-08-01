#!/usr/bin/env bash
# ============================================================
# prepare_kitti.sh - 准备 KITTI odometry 数据集
#
# 用法：
#   scripts/prepare_kitti.sh [数据目录] [序列列表]
#
# 默认数据目录: ~/data/kitti
# 默认序列:     00 01 02 03 04 05 06 07 08 09 10
#
# 前提：先自行下载以下文件并放入数据目录（注意本机代理可能阻断 S3，
#       可尝试 curl --noproxy '*' 或配置代理白名单）：
#   https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip  (~8GB, 灰度图像)
#   https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip (~100KB, 真值)
#
# 效果：解压出
#   <dir>/sequences/XX/image_0/000000.png ...   (灰度图像序列)
#   <dir>/poses/XX.txt                          (00-10 真值位姿, 3x4 矩阵每行)
# ============================================================
set -euo pipefail

DATA_DIR="${1:-$HOME/data/kitti}"
SEQUENCES="${2:-00 01 02 03 04 05 06 07 08 09 10}"
GRAY_ZIP="$DATA_DIR/data_odometry_gray.zip"
POSES_ZIP="$DATA_DIR/data_odometry_poses.zip"

mkdir -p "$DATA_DIR"

echo "==> 数据目录: $DATA_DIR"

# ---- 校验 zip 文件 ----
check_zip() {
    local f="$1" name="$2"
    if [ ! -f "$f" ]; then
        echo "[错误] 缺少 $name: $f"
        echo "       请先下载 https://s3.eu-central-1.amazonaws.com/avg-kitti/$name 到 $DATA_DIR"
        exit 1
    fi
    # 用 python 校验 zip 完整性（unzip 可能未安装）
    if ! python3 -c "import zipfile,sys; zipfile.ZipFile('$f').testzip() is None or sys.exit(1)" 2>/dev/null; then
        echo "[错误] $name 不是有效的 zip 文件（可能下载不完整，请重新下载）: $f"
        exit 1
    fi
    echo "==> $name 校验通过 ($(du -h "$f" | cut -f1))"
}

check_zip "$GRAY_ZIP" "data_odometry_gray.zip"
check_zip "$POSES_ZIP" "data_odometry_poses.zip"

# ---- 解压图像序列（只解压需要的序列，避免全部 8GB 展开）----
cd "$DATA_DIR"
for seq in $SEQUENCES; do
    target="sequences/$seq/image_0"
    if [ -d "$target" ] && [ "$(ls "$target" | wc -l)" -gt 100 ]; then
        echo "==> 序列 $seq 已存在，跳过"
        continue
    fi
    echo "==> 解压序列 $seq ..."
    python3 - "$GRAY_ZIP" "$seq" << 'PYEOF'
import sys, zipfile, os
zip_path, seq = sys.argv[1], sys.argv[2]
# 只解压左目 image_0；精确到 image_0/ 避免右目 image_1 同名文件覆盖
# （兼容两种 zip 内部前缀：官方 "dataset/sequences/..." 或镜像 "data_odometry_gray/..."）
prefixes = [f"dataset/sequences/{seq}/image_0/", f"data_odometry_gray/dataset/sequences/{seq}/image_0/"]
os.makedirs(f"sequences/{seq}/image_0", exist_ok=True)
z = zipfile.ZipFile(zip_path)
count = 0
for name in z.namelist():
    if not name.endswith(".png"):
        continue
    if not any(name.startswith(p) for p in prefixes):
        continue
    base = os.path.basename(name)
    with open(f"sequences/{seq}/image_0/{base}", "wb") as out:
        out.write(z.read(name))
    count += 1
print(f"    {seq}: {count} 张图像")
PYEOF
done

# ---- 解压真值位姿 ----
if [ ! -d "poses" ]; then
    echo "==> 解压真值位姿 ..."
    python3 -c "
import zipfile, os
z = zipfile.ZipFile('$POSES_ZIP')
os.makedirs('poses', exist_ok=True)
for name in z.namelist():
    if name.endswith('.txt'):
        with open(os.path.join('poses', os.path.basename(name)), 'wb') as out:
            out.write(z.read(name))
print('    poses 解压完成:', len([n for n in z.namelist() if n.endswith('.txt')]), '个序列')
"
else
    echo "==> poses 已存在，跳过"
fi

echo ""
echo "==> 完成。运行示例："
echo "  ./build/bin/run_vo $DATA_DIR/sequences/00/image_0 config/default.yaml trajectory_00.txt"
echo "  # 用 EVO 评估："
echo "  evo_ape tum trajectory_00.txt <(python3 -c 'import sys; [print(f\"{i/10:.6f}\", *map(float, l.split()[:12])) for i,l in enumerate(open(\"$DATA_DIR/poses/00.txt\"))]')"
