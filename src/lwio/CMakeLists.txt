cmake_minimum_required(VERSION 3.0.2)
project(lwio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
# set(CMAKE_BUILD_TYPE Debug)

find_package(Boost REQUIRED COMPONENTS thread)
find_package(Boost REQUIRED COMPONENTS timer)
find_package(Ceres REQUIRED)
find_package(PCL 1.7 REQUIRED)
# find_package(G2O REQUIRED)
find_package(Eigen3  REQUIRED)
find_package(OpenCV 3.0 QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(glog  REQUIRED)


find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
    include 
    ${Boost_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${SlamLib_INCLUDE_DIR}
)
link_directories(${PCL_LIBRARY_DIRS})

message("+++++++++++++++++++++++++++++++++++++++++=PCL_INCLUDE_DIRS" ${PCL_INCLUDE_DIRS})
message("+++++++++++++++++++++++++++++++++++++++++=PCL_LIBRARY_DIRS" ${PCL_LIBRARY_DIRS})

include(cmake/g2o.cmake)
#message(FATAL_ERROR ${PROJECT_NAME})

add_library(lwio SHARED
    src/system.cpp
    # src/parameters.cpp
)

target_link_libraries(lwio PUBLIC
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${CERES_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${OpenCV_LIBS}
    slam_pointcloud
    gtsam
    yaml-cpp
)

# add_executable(lib_test src/test/test.cpp)
# target_link_libraries(lib_test
#     lwio
# )

set(LWIO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH 
    "project include location" FORCE)
