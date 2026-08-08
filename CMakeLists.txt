cmake_minimum_required(VERSION 3.10)
project(vslam VERSION 0.1.0 LANGUAGES CXX)

# ============================================================
# 编译选项
# ============================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译优化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# 性能监测（utils/perf_monitor.h）
#   ON  : PERF_SCOPE 计时生效，run_vo/run_slam 退出时输出 perf 报告
#   OFF : 全部展开为空实现，零运行时开销
# ============================================================
option(VSLAM_ENABLE_PERF "Enable performance instrumentation (PERF_SCOPE)" ON)
if(VSLAM_ENABLE_PERF)
    add_compile_definitions(VSLAM_ENABLE_PERF)
    message(STATUS "Performance instrumentation ENABLED (VSLAM_ENABLE_PERF=ON)")
endif()
# 注意：不能用 -march=native（AVX/32字节对齐）——g2o 按 SSE/16 字节
# 对齐编译（含 thirdparty/g2o 源码，子目录继承本 flags），跨 DSO 传递
# Eigen Isometry3d 等固定大小类型时若对齐假设不一致，会导致 EdgeSE3
# 测量值静默损坏、位姿图优化发散（实测）。用 SSE2 保持 16 字节对齐，
# 与 g2o 一致（Eigen 向量化收益几乎不受影响）。
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DDEBUG")

# ============================================================
# 查找依赖
# ============================================================
# --- OpenCV (必需) ---
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc features2d calib3d highgui videoio imgcodecs video)

# --- Eigen3 (必需) ---
find_package(Eigen3 REQUIRED)

# --- Pangolin (必需，可视化) ---
find_package(Pangolin REQUIRED)
if(TARGET pango_core)
    # Pangolin 新版使用拆分组件目标。
    set(VSLAM_PANGOLIN_LIBS pango_core pango_display pango_opengl pango_plot)
elseif(TARGET pangolin)
    # scripts/install_deps.sh 安装的 Pangolin v0.6 只导出单一目标。
    set(VSLAM_PANGOLIN_LIBS pangolin)
elseif(Pangolin_LIBRARIES)
    # 兼容仅导出旧式变量的 Pangolin 包。
    set(VSLAM_PANGOLIN_LIBS ${Pangolin_LIBRARIES})
else()
    message(FATAL_ERROR "Pangolin found, but no supported library target was exported")
endif()

# --- yaml-cpp (必需，配置) ---
find_package(yaml-cpp REQUIRED)

# --- g2o (Phase 1 后端优化，源码 vendored) ---
# g2o 源码位于 thirdparty/g2o（tag 20241228_git，commit
# eec325a1da1273e87bc97887d49e70570f28570c），随仓库管理、add_subdirectory
# 直接集成，不依赖系统 apt 的 libg2o-dev，保证各机器离线可复现构建。
# 只构建本项目需要的核心库，关闭 apps/examples/unit tests/benchmarks、OpenGL
# （g2o_viewer 不需要）与 spdlog/fmt 日志（GCC 13 + C++23 下与系统 fmt 9
# 存在 constexpr 兼容性问题，且本项目用 LOG_INFO 自管日志）。
# CACHE+FORCE 是为了覆盖旧 build/CMakeCache 中 g2o 默认的 ON；普通 set()
# 会被 option() 的既有缓存值反向覆盖，导致 logging/无关 types 悄悄重新启用。
set(G2O_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
set(G2O_USE_OPENGL OFF CACHE BOOL "" FORCE)
set(G2O_USE_LOGGING OFF CACHE BOOL "" FORCE)
set(G2O_USE_CHOLMOD OFF CACHE BOOL "" FORCE)
set(G2O_USE_CSPARSE OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM2D_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM2D_ADDON_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_DATA_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SCLAM2D_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM3D_TYPES ON CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM3D_ADDON_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SBA_TYPES ON CACHE BOOL "" FORCE)
set(G2O_BUILD_ICP_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SIM3_TYPES OFF CACHE BOOL "" FORCE)
# g2o 根工程会声明其他 solver 目标；EXCLUDE_FROM_ALL 保证默认构建只沿下方
# 五个链接目标的依赖闭包编译，不额外构建 pcg/dense/structure_only 等库。
add_subdirectory(thirdparty/g2o EXCLUDE_FROM_ALL)
add_definitions(-DHAS_G2O)
message(STATUS "g2o vendored in thirdparty/g2o - backend optimization enabled")

# --- DBoW3 (可选，Phase 2 回环检测) ---
find_package(DBoW3 QUIET)
# install_deps.sh / fetch_vocab.sh 将 DBoW3 安装到 ~/.local（无需 sudo），
# 此处补一次用户前缀查找，避免手工传 -DDBoW3_DIR。
if(NOT DBoW3_FOUND AND DEFINED ENV{HOME})
    set(DBoW3_DIR "$ENV{HOME}/.local/lib/cmake/DBoW3")
    find_package(DBoW3 QUIET)
endif()
if(DBoW3_FOUND)
    message(STATUS "DBoW3 found - loop closure enabled")
    add_definitions(-DHAS_DBOW3)
else()
    message(STATUS "DBoW3 not found - loop closure disabled (Phase 2 will need it)")
endif()

# ============================================================
# VSLAM 静态库
# ============================================================
set(VSLAM_SOURCES
    src/camera.cpp
    src/dataset.cpp
    src/mappoint.cpp
    src/map.cpp
    src/backend_committer.cpp
    src/atlas.cpp
    src/feature.cpp
    src/vo.cpp
    src/optimizer.cpp
    src/viewer.cpp
    src/loop_closure.cpp
    src/tracking_state_machine.cpp
    src/localizer.cpp
    src/pose_gate.cpp)

set(VSLAM_HEADERS
    include/vslam/common.h
    include/vslam/localization_types.h
    include/vslam/tracking_state_machine.h
    include/vslam/localizer.h
    include/vslam/pose_gate.h
    include/vslam/camera.h
    include/vslam/dataset.h
    include/vslam/frame.h
    include/vslam/observation.h
    include/vslam/mappoint.h
    include/vslam/map.h
    include/vslam/atlas.h
    include/vslam/feature.h
    include/vslam/vo.h
    include/vslam/optimizer.h
    include/vslam/viewer.h
    include/vslam/loop_closure.h)

add_library(vslam STATIC ${VSLAM_SOURCES} ${VSLAM_HEADERS})
target_include_directories(vslam
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_SOURCE_DIR}/utils       # perf_monitor.h（性能监测）
        ${Pangolin_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(vslam
    PUBLIC
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${VSLAM_PANGOLIN_LIBS}
        yaml-cpp)

# g2o 的 add_library 目标名不带前缀（core/stuff/...），
# 这里统一建别名 g2o::* 便于链接与维护。
add_library(g2o::core ALIAS core)
add_library(g2o::stuff ALIAS stuff)
add_library(g2o::solver_eigen ALIAS solver_eigen)
add_library(g2o::types_slam3d ALIAS types_slam3d)
add_library(g2o::types_sba ALIAS types_sba)
target_link_libraries(vslam PUBLIC
    g2o::core
    g2o::stuff
    g2o::solver_eigen
    g2o::types_slam3d
    g2o::types_sba)
# 头文件路径由 g2o target 的 BUILD_INTERFACE 自动传播，无需手动添加。

if(DBoW3_FOUND)
    # DBoW3 的 DBoW3Config.cmake 只导出变量（DBoW3_LIBS/DBoW3_INCLUDE_DIRS），
    # 不导出 imported target，因此用变量链接而非 DBoW3::DBoW3。
    target_link_libraries(vslam PUBLIC ${DBoW3_LIBS})
    target_include_directories(vslam PUBLIC ${DBoW3_INCLUDE_DIRS})
endif()

# ============================================================
# 可执行文件
# ============================================================

# --- run_vo: 视觉里程计演示（回环固定关闭，A/B 对比基线）---
add_executable(run_vo app/run_vo.cpp)
target_link_libraries(run_vo vslam)
set_target_properties(run_vo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# --- run_slam: 完整 SLAM 演示 (Phase 2，回环默认开启) ---
add_executable(run_slam app/run_slam.cpp)
target_link_libraries(run_slam vslam)
set_target_properties(run_slam PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ============================================================
# 测试
# ============================================================
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_vo test/test_vo.cpp)
    target_link_libraries(test_vo vslam)
    # 测试可能从任意构建目录启动；资源路径不得依赖进程当前工作目录。
    target_compile_definitions(test_vo PRIVATE
        VSLAM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    # 测试使用 assert；Release 全局定义了 NDEBUG，必须为测试目标显式恢复断言。
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_vo PRIVATE -UNDEBUG)
    endif()
    add_test(NAME test_vo COMMAND test_vo)

    # M0.1 定位契约类型（独立 CTest，不与 test_vo 混排）
    add_executable(test_localization_types test/test_localization_types.cpp)
    target_link_libraries(test_localization_types vslam)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_localization_types PRIVATE -UNDEBUG)
    endif()
    add_test(NAME test_localization_types COMMAND test_localization_types)

    # M0.2 定位状态机（独立 CTest）
    add_executable(test_tracking_state_machine test/test_tracking_state_machine.cpp)
    target_link_libraries(test_tracking_state_machine vslam)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_tracking_state_machine PRIVATE -UNDEBUG)
    endif()
    add_test(NAME test_tracking_state_machine COMMAND test_tracking_state_machine)

    # M0.3 Localizer Facade 契约（独立 CTest）
    add_executable(test_localizer_contract test/test_localizer_contract.cpp)
    target_link_libraries(test_localizer_contract vslam)
    target_compile_definitions(test_localizer_contract PRIVATE
        VSLAM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_localizer_contract PRIVATE -UNDEBUG)
    endif()
    add_test(NAME test_localizer_contract COMMAND test_localizer_contract)

    # M1.1 PoseGate（独立 CTest）
    add_executable(test_pose_gate test/test_pose_gate.cpp)
    target_link_libraries(test_pose_gate vslam)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_pose_gate PRIVATE -UNDEBUG)
    endif()
    add_test(NAME test_pose_gate COMMAND test_pose_gate)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME test_trajectory_alignment
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_trajectory_alignment.py)
    endif()
endif()

# ============================================================
# 安装
# ============================================================
install(TARGETS vslam
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(FILES config/default.yaml DESTINATION share/vslam)
install(FILES config/robot.yaml DESTINATION share/vslam)
install(FILES config/deterministic.yaml DESTINATION share/vslam)
