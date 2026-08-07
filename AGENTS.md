# Repository Guidelines

Contributor guide for **vslam**, a C++23 visual odometry / SLAM system: VO front-end, optional g2o local BA, Pangolin viewer, and DBoW3 loop-closure hooks (Phase 2).

## Project Structure & Module Organization

- `include/vslam/` — public headers, one per module (`camera`, `dataset`, `frame`, `feature`, `map`, `mappoint`, `vo`, `optimizer`, `viewer`, `loop_closure`). Shared type aliases (`Vec3`, `Mat33`, …) and the minimal `SE3` live in `common.h`.
- `src/` — implementation, one `.cpp` mirroring each header.
- `app/run_vo.cpp` — the `run_vo` demo executable.
- `test/` — `test_vo.cpp` unit tests.
- `config/default.yaml` — runtime parameters (camera intrinsics, feature/VO/optimizer settings).
- `scripts/` — dependency installer, KITTI preparation, ATE evaluation helpers.
- `docs/` — `TUTORIAL.md`, `DEVELOPMENT_LOG.md`.
- `datasets/` — `kitti/sequences` + `kitti/poses` and other datasets (git-ignored; fetched via `prepare_kitti.sh`).

## Build, Test, and Development Commands

```bash
bash scripts/install_deps.sh        # Install OpenCV, Eigen3, Pangolin, yaml-cpp, g2o, DBoW3
cmake -S . -B build                 # Configure (Release + C++23 by default)
cmake --build build -j              # Build the vslam library and run_vo
./build/bin/run_vo datasets/kitti/sequences/00 config/default.yaml traj.txt   # Run VO on KITTI 00

# Tests
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --target test_vo
./build/test_vo                     # or: ctest --test-dir build
```

## Coding Style & Naming Conventions

- C++23, 4-space indentation, `#pragma once` headers.
- Everything in `namespace vslam`. Types `CamelCase`, functions/variables `snake_case`, private members end with `_` (e.g., `camera_`, `cfg_`).
- Include order: project headers, then third-party, then std.
- Comments are written in Chinese (project convention); use `[[nodiscard]]` on pure helpers; log via `LOG_INFO`/`LOG_WARN`.
- No formatter/linter is configured — match the surrounding style.

## Testing Guidelines

- Plain `assert`-based tests with local `TEST`/`TEST_PASS`/`TEST_FAIL` macros in `test/test_vo.cpp`; no framework.
- Cover SE3 geometry, camera projection, ORB/LK matching, two-frame initialization, and pose-semantics regressions (e.g., `recoverPose` → `T_cw`).
- Build with `-DBUILD_TESTS=ON`, then run `./build/test_vo` or `ctest --test-dir build`.

## Commit & Pull Request Guidelines

- Messages are descriptive Chinese sentences; use a lowercase type prefix when scoped, e.g. `chore:`, `docs:`, `run_vo:`.
- State what and why, and quantify results where relevant (e.g., “KITTI 00 全程无 LOST, ATE 133.6m”).
- Work on `main` (no feature branches used). For PRs, describe the change, link the related issue, and report dataset validation (sequence, ATE, LOST frames) as evidence.

## Architecture Overview

Pipeline: `Camera → Frame → FeatureMatcher → VisualOdometry → Map/MapPoint` with an optional g2o local BA (`optimizer`) and a Pangolin `Viewer`. `run_vo` parses `<dataset_path|camera_index> [config.yaml] [trajectory.txt] [--tum|--euroc]`; g2o/DBoW3 are compile-time optional (`-DHAS_G2O`, `-DHAS_DBOW3`).
