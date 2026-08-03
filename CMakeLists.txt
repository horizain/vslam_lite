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
# 注意：不能用 -march=native（AVX/32字节对齐）——apt 版 g2o 库按 SSE/16 字节
# 对齐编译，跨 DSO 传递 Eigen Isometry3d 等固定大小类型时对齐假设不一致，
# 会导致 EdgeSE3 测量值静默损坏、位姿图优化发散（实测）。用 SSE2 保持
# 16 字节对齐，与系统 g2o 一致（Eigen 向量化收益几乎不受影响）。
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

# --- g2o (可选，Phase 1 后端优化) ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(g2o QUIET)
if(g2o_FOUND)
    message(STATUS "g2o found - backend optimization enabled")
    add_definitions(-DHAS_G2O)
else()
    message(WARNING "g2o not found - backend optimization disabled (Phase 1 still works)")
endif()

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
    src/atlas.cpp
    src/feature.cpp
    src/vo.cpp
    src/optimizer.cpp
    src/viewer.cpp
    src/loop_closure.cpp)

set(VSLAM_HEADERS
    include/vslam/common.h
    include/vslam/camera.h
    include/vslam/dataset.h
    include/vslam/frame.h
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

if(g2o_FOUND)
    target_link_libraries(vslam PUBLIC ${g2o_LIBRARIES})
    target_include_directories(vslam PUBLIC ${g2o_INCLUDE_DIRS})
endif()

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
    # 测试使用 assert；Release 全局定义了 NDEBUG，必须为测试目标显式恢复断言。
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_vo PRIVATE -UNDEBUG)
    endif()
    add_test(NAME test_vo COMMAND test_vo)
endif()

# ============================================================
# 安装
# ============================================================
install(TARGETS vslam
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(FILES config/default.yaml DESTINATION share/vslam)
