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
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
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
    target_link_libraries(vslam PUBLIC DBoW3::DBoW3)
endif()

# ============================================================
# 可执行文件
# ============================================================

# --- run_vo: 视觉里程计演示 ---
add_executable(run_vo app/run_vo.cpp)
target_link_libraries(run_vo vslam)
set_target_properties(run_vo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# --- run_slam: 完整 SLAM 演示 (Phase 2) ---
# add_executable(run_slam app/run_slam.cpp)
# target_link_libraries(run_slam vslam)
# set_target_properties(run_slam PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

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
