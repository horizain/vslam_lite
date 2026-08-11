# Repository Guidelines

Contributor and agent guide for **vslam_lite**, a C++23 visual odometry / SLAM
system with a stereo/monocular frontend, g2o local BA and pose-graph optimization,
DBoW3 loop closure, Atlas submaps, bounded asynchronous backend work, localization
facades, structured metrics, and an optional Pangolin viewer.

The detailed product specification is
`docs/PRODUCTION_LOCALIZATION_PLAN.md`. This file is the short, mandatory execution
contract. When the two disagree, preserve the mathematical/safety invariants in the
product plan and update both documents in the same change.

## Project Structure

- `include/vslam/` and `src/` — public interfaces and matching implementations.
  Important modules include `vo`, `frontend_tracker`, `local_mapper`, `optimizer`,
  `backend_committer`, `atlas`, `loop_closure`, `loop_region`, `resource_budget`,
  `localizer`, `metrics`, and `viewer`.
- `app/run_slam.cpp` — primary SLAM executable with loop closure and structured
  status/metrics output. `app/run_vo.cpp` is the VO-only A/B path.
- `test/` — assert-based C++ regression executables registered with CTest. Tests are
  split by module; `test_vo.cpp` is no longer the only test target.
- `config/default.yaml`, `mobile.yaml`, `robot.yaml` — product/runtime profiles.
  `deterministic.yaml` and `benchmark.yaml` are validation profiles;
  `kitti00.yaml` is an explicitly dataset-specific high-accuracy experiment profile.
- `scripts/benchmark_gate.sh` — fresh-build L0-L2 submission gate.
  `benchmark.py`, `evaluate_loop_recall.py`, `compare_trajectories.py`, and the
  accompanying Python tests form the evaluation/oracle chain.
- `scripts/benchmark/reference/` — deterministic golden trajectory/status. Golden
  files are evidence, not a mechanism for making a failing change green.
- `docs/` — tutorial, append-only development evidence, and the M0-M7 production
  plan. Material behavior or product-boundary changes must update the docs.
- `datasets/` and generated trajectories/images/reports are user data and are not
  source files. Preserve them unless the task explicitly targets them.

## Build, Run, and Test

```bash
bash scripts/install_deps.sh
cmake -S . -B build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure --no-tests=error

# Primary SLAM run; use SE3 evaluation for stereo.
./build/bin/run_slam datasets/kitti/sequences/00 config/default.yaml traj.txt \
    --headless --status-csv status.csv --metrics-json metrics.json

# VO-only A/B path.
./build/bin/run_vo datasets/kitti/sequences/00 config/default.yaml vo_traj.txt \
    --headless

# Mandatory fast submission gate. It configures a fresh temporary Release build,
# runs all CTest/Python tests, deterministic replay, and three GT-backed runs.
bash scripts/benchmark_gate.sh

# Long/dataset-sensitive changes additionally require the full statistical gate.
bash scripts/benchmark_gate.sh --full
```

Do not treat an old `build/`, a successful compile, or CTest alone as proof of SLAM
correctness. Loop closure, Atlas correction, async commit, long-sequence accuracy,
RSS, and p99 latency require their corresponding integration/benchmark evidence.

### Build Notes and Troubleshooting

- Run CMake and executables from the repository root. Several configs, vocabulary
  paths, robot profiles, and benchmark paths are repository-relative.
- Use a new build directory after changing the compiler, Eigen/OpenCV/Pangolin,
  DBoW3, g2o, or important CMake flags. Do not delete an existing user build merely
  to make it clean; configure another directory instead:

  ```bash
  cmake -S . -B build-release -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build-release -j
  ctest --test-dir build-release --output-on-failure --no-tests=error
  ```

- g2o is vendored under `thirdparty/g2o` and must produce the configure message
  `g2o vendored ... backend optimization enabled`. DBoW3 is optional at compile time,
  but a loop-closure result is invalid unless configure prints
  `DBoW3 found - loop closure enabled`. The installer normally places DBoW3 under
  `$HOME/.local`; if discovery fails, pass its CMake package directory explicitly:

  ```bash
  cmake -S . -B build-release -DBUILD_TESTS=ON \
      -DDBoW3_DIR="$HOME/.local/lib/cmake/DBoW3"
  ```

- Pangolin is currently a required build dependency even for `--headless`. Headless
  suppresses the runtime window; it does not remove the link-time dependency.
- Do not add `-march=native`. The repository deliberately uses SSE-compatible Eigen
  alignment to match vendored g2o across library boundaries; changing the ABI can
  cause crashes that look like optimizer corruption.
- Release is the accuracy/performance benchmark profile. Debug/ASan builds are useful
  for diagnosis, but their latency/FPS must not be compared with Release gates.
- If a parallel build is killed by the OOM killer, reduce parallelism (`-j2` or
  `-j1`) instead of changing optimization flags or disabling modules.
- CMake success with `DBoW3 not found` is a valid VO-only build, not a valid SLAM
  loop-closure validation. Always retain the configure summary in evidence.
- `run_slam` forces loop closure on; `run_vo` forces it off. Use the same data,
  configuration, build, frame window, and headless setting for an A/B comparison.
- Dataset calibration has priority over camera values in YAML when the dataset exports
  calibration. Before attributing a trajectory change to YAML intrinsics, confirm
  which calibration source the executable logged.
- Use `--frames N` for a smoke test and `--skip N --frames M` only for performance
  slicing. A slice is not a replacement for a continuous full-sequence accuracy run.
- Useful diagnosis pattern:

  ```bash
  ./build-release/bin/run_slam datasets/kitti/sequences/00 \
      config/default.yaml /tmp/slam_smoke.tum --headless --frames 200 \
      --status-csv /tmp/slam_smoke.csv --metrics-json /tmp/slam_smoke.json
  ```

  Check the process exit code, nonempty trajectory/status files, processed frame
  count, `pose_valid`, LOST intervals, keyframes, loops, and metrics JSON together.
  A generated file with zero valid poses is a failure.

## Coding Style

- C++23, 4-space indentation, `#pragma once`, everything in `namespace vslam`.
- Types use `CamelCase`; functions and variables use `snake_case`; private members
  end in `_`.
- Include project headers first, then third-party headers, then the standard library.
- Comments follow the existing Chinese project convention. Log through
  `LOG_INFO`/`LOG_WARN`; use `[[nodiscard]]` on pure helpers where useful.
- Match surrounding style; there is no repository-wide formatter. Run
  `git diff --check` before staging.

## Non-Negotiable Geometry Contracts

- `T_ab` transforms coordinates from frame `b` to frame `a`. Never introduce the
  opposite convention implicitly.
- `Frame::pose_cs` is `T_cs` (submap to camera), not a global pose.
  `Submap::T_ws` is submap to world. Therefore:

  ```text
  T_cw = T_cs * inverse(T_ws)
  T_wc = T_ws * inverse(T_cs)
  p_w  = T_ws * p_s
  ```

- `SE3::cameraPosition()` is valid for a world-to-camera transform and returns the
  camera center. Do not compare the translation component of `T_cw` as if it were
  a world position.
- Stereo trajectory evaluation uses SE3 alignment. Sim3 is allowed only for a
  deliberately scale-ambiguous monocular evaluation, and timestamps must be matched
  one-to-one.
- Every pose-direction fix needs a non-identity rotation and translation regression
  test. Identity-only tests do not detect composition-order errors.
- Global correction changes Atlas anchors or optimized local poses; it must not
  directly overwrite historical output poses. `FramePoseRecord` stays relative to a
  live anchor, is re-anchored before keyframe culling, and composes world pose at read
  time. Missing anchors must never silently produce identity.

## Trajectory Evaluation and Plotting

`run_slam` and `run_vo` write TUM rows in the following order, with pose semantics
`T_wc` (camera to world):

```text
timestamp tx ty tz qx qy qz qw
```

For KITTI, convert the official 3x4 pose file only if the matching TUM GT does not
already exist:

```bash
python3 scripts/kitti_gt_to_tum.py \
    datasets/kitti/poses/00.txt datasets/kitti/poses/00.tum 10
```

Evaluate first, then plot. Stereo/KITTI must use SE3 so scale drift remains visible:

```bash
python3 scripts/evaluate_ate.py trajectory.tum datasets/kitti/poses/00.tum \
    --alignment se3

python3 scripts/plot_traj.py trajectory.tum datasets/kitti/poses/00.tum \
    --out trajectory_vs_gt.png
```

Plotting and interpretation rules:

- `scripts/plot_traj.py` currently performs **Sim3** alignment internally and draws
  raw estimate, Sim3-aligned estimate, GT, 3D trajectory, and per-frame position
  error. For stereo it is a visualization aid only; its printed Sim3 ATE is not the
  release metric. Quote the SE3 result from `evaluate_ate.py`.
- The plotting script accepts timestamps within 0.05 s; the formal evaluator defaults
  to 0.02 s. Never compare the two metrics without matching association tolerance.
- KITTI uses `y` as the vertical camera axis, so the meaningful ground-plane view is
  **x-z**, not x-y. Keep equal axis scaling (`axis("equal")`); otherwise a compressed
  axis can make drift look smaller.
- Inspect both raw and aligned curves. Alignment is for comparing shape and residual
  drift; the raw curve exposes origin, orientation, and scale mistakes. Never crop the
  tail merely because its error grows.
- Check `scale`, matched-frame count, coverage, path-length ratio, and >3/5/10 m jumps
  before trusting a visually close overlay. Sim3 scale far from 1.0 is a stereo
  failure even when the red aligned curve looks good.
- ATE error must use the exact matched timestamp indices. Do not truncate two arrays
  to the same length, reuse one GT frame for multiple estimates, or plot errors against
  an unrelated full timestamp vector.
- Quaternions in TUM are `(qx,qy,qz,qw)`, not `(qw,qx,qy,qz)`. Normalize them before
  rotation calculations; `q` and `-q` represent the same rotation.
- The plotting backend is noninteractive (`Agg`), so it writes PNG files without a
  display server. Missing Chinese fonts may render square labels but do not affect the
  numbers; install a Noto/Droid CJK font or use English labels when publishing. In a
  restricted environment, point Matplotlib at a writable cache to avoid rebuilding
  the font cache on every run:

  ```bash
  mkdir -p /tmp/vslam-matplotlib-cache
  MPLCONFIGDIR=/tmp/vslam-matplotlib-cache \
      python3 scripts/plot_traj.py trajectory.tum datasets/kitti/poses/00.tum \
      --out trajectory_vs_gt.png
  ```
- Use explicit, unique output paths for A/B runs, for example
  `baseline_vs_gt.png` and `candidate_vs_gt.png`. Do not overwrite the only raw
  trajectory/report, and do not stage generated PNG/TUM/CSV/JSON files unless the
  task explicitly requests those artifacts.
- A publication-quality comparison should state dataset/sequence, mono or stereo,
  config, commit, alignment mode, matched/total frames, and whether the displayed
  curve is raw or aligned. Color alone is not sufficient; label GT and estimate in
  the legend and mark start/end.

For deterministic code regression, do not use a plotted image or ATE. Compare the
trajectory and state sequence directly:

```bash
python3 scripts/compare_trajectories.py candidate.tum reference.tum \
    --status candidate.csv reference.csv
```

For missed-loop analysis, additionally run the interval oracle rather than inferring
loop success from the overlay:

```bash
python3 scripts/evaluate_loop_recall.py --help
```

Use the exact event/status inputs produced by the run and record expected/recalled
intervals plus accepted true/false loops. A visually corrected tail can still contain
a false loop, and a low ATE can still hide a missed required interval.

## Loop Closure and Global-Correction Contracts

- Generic code under `src/` and `include/` must not contain KITTI frame IDs, sequence
  branches, experiment filenames, or thresholds selected only to repair one dataset.
  Dataset-specific values belong only in an explicitly named config profile.
- Loop retrieval is place-based, not raw-keyframe-based. The current contract is a
  hard maximum of 12 distinct places per query, including at most 4 nearest position
  priors. BoW and position candidates must use the same submap-aware, continuous
  temporal-cluster deduplication.
- Single-keyframe and bounded-region PnP use the same acceptance gates. Do not lower
  the region fallback thresholds to create apparent recall. The product defaults are
  at least 50 inliers, 0.70 inlier ratio, RMSE/positive-depth checks, and spatial-grid
  coverage.
- Region construction is bounded (`region_max_keyframes` and `region_max_points`) and
  deduplicates map points. Adding more candidates must not create an unbounded PnP or
  map-copy path.
- Resource-budget keyframe removal must, in the same lifecycle, re-anchor trajectory
  records and remove/rebuild the corresponding LoopClosure cache/DBoW entries. A
  deleted keyframe may not retain a strong frame reference or occupy Top-N retrieval.
- PGO acceptance uses robust residual/finite/correction/jump checks and is atomic on
  rejection. Cross-submap correction updates `T_ws`; do not traverse and move every
  historical point/keyframe as a substitute.
- Randomized OpenCV/RANSAC code must preserve caller-visible RNG state when the
  deterministic profile requires it. Tests must not depend on execution order.
- Recall claims require the interval oracle in `scripts/evaluate_loop_recall.py`:
  report expected intervals, recalled intervals, accepted true/false loops, precision,
  coverage, ATE/RPE, path-length ratio, and >3/5/10 m jumps. A nonzero loop count alone
  is not evidence of correct loop closure.

## Async, Ownership, and Resource Contracts

- Backend workers optimize immutable snapshots. They must not mutate live
  `Frame`, `Map`, `MapPoint`, `Submap`, or frontend state directly.
- `BackendCommitter` commits under the owning Map's exclusive lock only after jointly
  revalidating Map/Submap identity, topology revision, geometry revision, and every
  referenced object identity. Missing objects or changed topology are stale work, not
  partial-success conditions.
- Lock order must remain explicit. Do not call Map/Atlas code while holding a module
  mutex unless the established order documents it; never add a callback that reverses
  Map, trajectory, LoopClosure, and viewer lock order.
- Queues, active keyframes, map points, descriptors, snapshots, and per-query work need
  hard bounds plus production-scale tests. A bound on returned candidates is not a
  bound on an O(K log K) scan performed under a mutex.
- Preserve the continuous odometry/control contract: global loop correction must not
  create a same-frame invalid pose, discontinuous control pose, NaN/Inf, or an invalid
  quaternion. Rejection leaves live state unchanged.

## Required Development Workflow

1. Inspect `git status`, `git diff`, recent commits, and the relevant product-plan
   section before editing. Existing untracked outputs and unrelated modifications
   belong to the user.
2. Work on one numbered behavior objective at a time. Write the failing regression or
   oracle case first, then implement the smallest generic fix.
3. State the algorithm direction, initial parameters, allowed files, forbidden
   shortcuts, and quantitative gates before delegating implementation to a smaller
   model. A prose roadmap without these constraints is not an executable task.
4. Run the narrow test while iterating, then a fresh full CTest. Geometry, concurrency,
   loop, resource, or benchmark changes must also run `scripts/benchmark_gate.sh`.
5. For loop/Atlas/trajectory changes, run the complete representative sequence and the
   loop-recall oracle. Compare mean/std/worst over repeated runs when randomness or
   scheduling can affect results; never report only the best run.
6. Independently inspect worker-produced diffs and raw reports. A worker saying that
   tests passed is not review evidence. Severity-rank real findings and do not invent
   issues merely to make a review look adversarial.
7. Do not make a red test green by weakening gates, disabling features, swallowing
   errors, accepting empty work, or updating reference data. Negative-path tests must
   prove bad paths/configs/empty trajectories fail nonzero.
8. If behavior legitimately changes deterministic output, first obtain passing A/B
   and dataset evidence. Commit behavior separately from
   `scripts/benchmark/reference/{pose.txt,status.csv}`; the golden-only update receives
   its own review/commit.
9. Update `docs/DEVELOPMENT_LOG.md` and relevant tutorial/plan text for visible or
   architectural changes. Separate implemented facts from planned work and record
   exact commands, dependency availability, sequence, metrics, and limitations.
10. Stage only scoped files. Before commit, run `git diff --cached --check`; after push,
    verify `HEAD`, `origin/main`, and remaining worktree files explicitly.

## Validation and Product Claims

- A credible SLAM result reports: frames processed, valid-pose ratio/coverage, LOST
  count and duration, ATE RMSE/mean/p95/max, RPE, path-length ratio, jump counts, loop
  true/false/recall/precision, latency p50/p95/p99, deadline misses, FPS, and resource
  peaks when measured.
- Current verified reference evidence is recorded in `docs/DEVELOPMENT_LOG.md`, not in
  commit-message folklore. Re-run it when behavior, compiler, dependencies, machine,
  or dataset changes.
- As of 2026-08-11, accuracy/loop correctness has a passing full KITTI 00 run, but the
  high-accuracy profile's full-run p99/deadline rate is not product-real-time, and the
  inactive Atlas/global DBoW memory lifetime still lacks a demonstrated global hard
  bound. Do not describe the system as production-ready until those L3/resource gates
  pass on multiple scene classes.

## Commit and Pull Request Guidelines

- Work on `main` unless the user requests another workflow. Use descriptive Chinese
  commit subjects with a lowercase scope prefix where helpful, for example
  `slam: 修复回环候选生命周期` or `test: 增加跨子图轨迹回归`.
- Explain what changed and why, and quantify the actual validation. Include dependency
  availability (`HAS_G2O`, `HAS_DBOW3`) and clearly list validation that was not run.
- Preserve repository SSH configuration. Never reset, delete, or overwrite user data
  to obtain a clean tree, and never stage generated datasets, trajectories, images, or
  reports unless explicitly requested.
