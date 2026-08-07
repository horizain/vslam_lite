#!/usr/bin/env bash
# ============================================================
# prepare_kitti.sh - 准备 KITTI odometry 数据集
#
# 用法：
#   scripts/prepare_kitti.sh [数据目录] [序列列表]
#
# 默认数据目录: datasets/kitti（输出 datasets/kitti/sequences 与 datasets/kitti/poses）
# 默认序列:     00 01 02 03 04 05 06 07 08 09 10
#
# 前提：先自行下载以下文件并放入数据目录（注意本机代理可能阻断 S3，
#       可尝试 curl --noproxy '*' 或配置代理白名单）：
#   https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip  (~8GB, 灰度图像)
#   https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip (~100KB, 真值)
#
# 效果：解压出
#   <dir>/sequences/XX/image_0/000000.png ...   (左目灰度图像)
#   <dir>/sequences/XX/image_1/000000.png ...   (右目灰度图像)
#   <dir>/poses/XX.txt                          (00-10 真值位姿, 3x4 矩阵每行)
# ============================================================
set -euo pipefail

DATA_DIR="${1:-datasets/kitti}"
SEQUENCES="${2:-00 01 02 03 04 05 06 07 08 09 10}"

mkdir -p "$DATA_DIR"
# 后续会进入 DATA_DIR；先规范成绝对路径，避免默认相对路径被重复拼接。
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
GRAY_ZIP="$DATA_DIR/data_odometry_gray.zip"
POSES_ZIP="$DATA_DIR/data_odometry_poses.zip"

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

# ---- 解压双目图像序列（只解压需要的序列，避免全部 8GB 展开）----
pushd "$DATA_DIR" >/dev/null
for seq in $SEQUENCES; do
    target_left="sequences/$seq/image_0"
    target_right="sequences/$seq/image_1"
    if [ -d "$target_left" ] && [ -d "$target_right" ] \
       && [ "$(ls "$target_left" | wc -l)" -gt 100 ] \
       && [ "$(ls "$target_left" | wc -l)" -eq "$(ls "$target_right" | wc -l)" ]; then
        echo "==> 序列 $seq 已存在，跳过"
        continue
    fi
    echo "==> 解压序列 $seq ..."
    python3 - "$GRAY_ZIP" "$seq" << 'PYEOF'
import sys, zipfile, os
zip_path, seq = sys.argv[1], sys.argv[2]
# 同时解压 image_0/image_1；双目评估必须把序列根目录传给 run_vo/run_slam。
# 兼容官方和镜像 zip 的两种内部前缀。
roots = [f"dataset/sequences/{seq}", f"data_odometry_gray/dataset/sequences/{seq}"]
for camera in ("image_0", "image_1"):
    os.makedirs(f"sequences/{seq}/{camera}", exist_ok=True)
z = zipfile.ZipFile(zip_path)
counts = {"image_0": 0, "image_1": 0}
for name in z.namelist():
    if not name.endswith(".png"):
        continue
    for camera in counts:
        if any(name.startswith(f"{root}/{camera}/") for root in roots):
            base = os.path.basename(name)
            with open(f"sequences/{seq}/{camera}/{base}", "wb") as out:
                out.write(z.read(name))
            counts[camera] += 1
            break
if counts["image_0"] == 0 or counts["image_0"] != counts["image_1"]:
    raise SystemExit(f"{seq}: 双目帧数无效 image_0={counts['image_0']} image_1={counts['image_1']}")
print(f"    {seq}: 双目 {counts['image_0']} 对图像")
# 顺带解压标定文件（内参自动加载用）
for calib_name in [f"dataset/sequences/{seq}/calib.txt",
                   f"data_odometry_gray/dataset/sequences/{seq}/calib.txt"]:
    if calib_name in z.namelist():
        with open(f"sequences/{seq}/calib.txt", "wb") as out:
            out.write(z.read(calib_name))
        print(f"    {seq}: calib.txt")
        break
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

popd >/dev/null

echo ""
echo "==> 完成。运行示例："
echo "  ./build/bin/run_slam $DATA_DIR/sequences/00 config/kitti00.yaml trajectory_00.txt --headless"
echo "  # 日志必须出现 Loaded stereo sequence；传 image_0 会退化为单目。"
echo "  # 用 EVO 评估："
echo "  python3 scripts/kitti_gt_to_tum.py $DATA_DIR/poses/00.txt $DATA_DIR/poses/00.tum"
echo "  evo_ape tum $DATA_DIR/poses/00.tum trajectory_00.txt -a   # 双目只做 SE3 对齐，不加 -s"
