#!/usr/bin/env python3
import sys, zipfile, os

zip_path, seq = sys.argv[1], sys.argv[2]
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
print(f"{seq}: 双目 {counts['image_0']} 对图像")
for calib_name in [f"dataset/sequences/{seq}/calib.txt",
                   f"data_odometry_gray/dataset/sequences/{seq}/calib.txt"]:
    if calib_name in z.namelist():
        with open(f"sequences/{seq}/calib.txt", "wb") as out:
            out.write(z.read(calib_name))
        print(f"{seq}: calib.txt")
        break
