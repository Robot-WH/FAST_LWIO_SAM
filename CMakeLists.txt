cmake_minimum_required(VERSION 2.8.3)
project(lwio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g") 

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  pcl_conversions
  cv_bridge
  pcl_ros
  std_msgs
  pcl_msgs
  sensor_msgs
  geographic_msgs
  nmea_msgs
  nav_msgs
  message_generation
  ndt_omp
)

find_package(catkin REQUIRED )
find_package(Ceres REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(Eigen3  REQUIRED)
find_package(OpenCV 3.0 QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(glog  REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer)
find_package(yaml-cpp REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

add_service_files(FILES
  SaveMap.srv
  SaveData.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
  geographic_msgs
  nmea_msgs
)

# 如果没有这句话 那么生成的可执行文件就会到build文件下  正确的是应该到devel下
catkin_package(
  DEPENDS PCL GTSAM

  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  geographic_msgs
  sensor_msgs
  nmea_msgs
  message_runtime
  message_generation
)

include_directories(
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  # ${G2O_INCLUDE_DIR} 
  # ${G2O_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${yaml-cpp_INCLUDE_DIRS}
)

include(cmake/Macros.cmake)
add_subdirectory(src)



