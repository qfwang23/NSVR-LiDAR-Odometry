cmake_minimum_required(VERSION 3.16)
project(nsvr_lidar_odometry LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(catkin REQUIRED COMPONENTS rosbag roscpp sensor_msgs)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(TBB REQUIRED)
include(FetchContent)
FetchContent_Declare(sophus
  URL https://github.com/nachovizzo/Sophus/archive/refs/tags/1.22.11.tar.gz
  URL_HASH SHA256=0a8a2c39fed766ab958b274c0cf2858e7812c20f83d04b8bbf9b61b840f8d126)
FetchContent_Declare(robin_map
  URL https://github.com/Tessil/robin-map/archive/refs/tags/v1.2.1.tar.gz
  URL_HASH SHA256=2b54d2c1de2f73bea5c51d5dcbd64813a08caf1bfddcfdeee40ab74e9599e8e3)
foreach(dep sophus robin_map)
  FetchContent_GetProperties(${dep})
  if(NOT ${dep}_POPULATED)
    FetchContent_Populate(${dep})
  endif()
endforeach()

add_library(lo_core STATIC
  cpp/LO/core/Registration.cpp
  cpp/LO/core/VerticalSurface.cpp
  cpp/LO/core/VoxelHashMap.cpp)
target_include_directories(lo_core PUBLIC cpp "${sophus_SOURCE_DIR}" "${robin_map_SOURCE_DIR}/include")
target_compile_definitions(lo_core PUBLIC SOPHUS_USE_BASIC_LOGGING=1)
target_link_libraries(lo_core PUBLIC Eigen3::Eigen TBB::tbb)

add_library(lo_pipeline STATIC cpp/LO/pipeline/LO.cpp)
target_link_libraries(lo_pipeline PUBLIC lo_core)
add_executable(lo_bag app/lo_bag.cpp)
target_include_directories(lo_bag PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(lo_bag PRIVATE lo_pipeline ${catkin_LIBRARIES})
target_compile_options(lo_bag PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME adapter_help COMMAND lo_bag --help)
add_executable(vertical_gain_test tests/vertical_gain_test.cpp)
target_include_directories(vertical_gain_test PRIVATE cpp)
target_link_libraries(vertical_gain_test PRIVATE Eigen3::Eigen)
add_test(NAME vertical_gain_retention COMMAND vertical_gain_test)
add_executable(voxel_search_test tests/voxel_search_test.cpp)
target_link_libraries(voxel_search_test PRIVATE lo_core)
add_test(NAME voxel_search_and_pruning COMMAND voxel_search_test)
add_executable(centered_update_test tests/centered_update_test.cpp)
target_link_libraries(centered_update_test PRIVATE lo_core)
add_test(NAME centered_update_invariance COMMAND centered_update_test)
add_executable(vertical_geometry_test tests/vertical_geometry_test.cpp)
target_link_libraries(vertical_geometry_test PRIVATE lo_core)
add_test(NAME vertical_geometry_and_registration COMMAND vertical_geometry_test)
add_executable(vertical_surface_test tests/vertical_surface_test.cpp)
target_link_libraries(vertical_surface_test PRIVATE lo_pipeline)
add_test(NAME cross_validated_vertical_surface COMMAND vertical_surface_test)
