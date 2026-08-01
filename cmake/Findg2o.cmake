# Findg2o.cmake
# 查找 g2o 图优化库
#
# 定义以下变量：
#   g2o_FOUND          - 是否找到
#   g2o_INCLUDE_DIRS   - 头文件路径
#   g2o_LIBRARIES      - 链接库列表

find_path(g2o_INCLUDE_DIR
    NAMES g2o/core/base_vertex.h
    PATHS /usr/local/include /usr/include
    PATH_SUFFIXES g2o)

find_library(g2o_CORE_LIBRARY       NAMES g2o_core       PATHS /usr/local/lib /usr/lib)
find_library(g2o_STUFF_LIBRARY      NAMES g2o_stuff      PATHS /usr/local/lib /usr/lib)
find_library(g2o_TYPES_SBA_LIBRARY  NAMES g2o_types_sba  PATHS /usr/local/lib /usr/lib)
find_library(g2o_TYPES_SLAM3D_LIBRARY NAMES g2o_types_slam3d PATHS /usr/local/lib /usr/lib)
find_library(g2o_SOLVER_EIGEN_LIBRARY NAMES g2o_solver_eigen PATHS /usr/local/lib /usr/lib)
find_library(g2o_CSPARSE_EXTENSION_LIBRARY NAMES g2o_csparse_extension PATHS /usr/local/lib /usr/lib)

set(g2o_INCLUDE_DIRS ${g2o_INCLUDE_DIR})

set(g2o_LIBRARIES
    ${g2o_CORE_LIBRARY}
    ${g2o_STUFF_LIBRARY}
    ${g2o_TYPES_SBA_LIBRARY}
    ${g2o_TYPES_SLAM3D_LIBRARY}
    ${g2o_SOLVER_EIGEN_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(g2o
    REQUIRED_VARS g2o_INCLUDE_DIR g2o_CORE_LIBRARY)
