cmake_minimum_required(VERSION 3.0.2)
project(app)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")

include_directories(
  include
  # ${MSAL3D_INCLUDE_DIR}
  ${LWIO_INCLUDE_DIR}
  )

# message(FATAL_ERROR ${SLAM_INCLUDE_DIR})

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(faster_loam_node
  src/faster_loam_node.cpp
)

add_dependencies(faster_loam_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(faster_loam_node
  ${catkin_LIBRARIES}
  lwio
  /usr/local/lib/libtbb.so
  /usr/local/lib/libtbbmalloc.so
  /usr/local/lib/libtbbmalloc_proxy.so
)

add_executable(advanced_lio_node
  src/advanced_lio_node.cpp
)

add_dependencies(advanced_lio_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(advanced_lio_node
  ${catkin_LIBRARIES}
  lwio
  /usr/local/lib/libtbb.so
  /usr/local/lib/libtbbmalloc.so
  /usr/local/lib/libtbbmalloc_proxy.so
)

# add_executable(advanced_lio_sam_node
#   src/advanced_lio_sam_node.cpp
# )

# add_dependencies(advanced_lio_sam_node
#   ${catkin_EXPORTED_TARGETS}
# )

# target_link_libraries(advanced_lio_sam_node
#   ${catkin_LIBRARIES}
#   ${SLAM_LIBRARIES}
#   /usr/local/lib/libtbb.so
#   /usr/local/lib/libtbbmalloc.so
#   /usr/local/lib/libtbbmalloc_proxy.so
# )

# add_executable(LWIO_SAM_node
#   src/LWIO_SAM_node.cpp
# )

# add_dependencies(LWIO_SAM_node
#   ${catkin_EXPORTED_TARGETS}
# )

# target_link_libraries(LWIO_SAM_node
#   ${catkin_LIBRARIES}
#   ${Boost_LIBRARIES}
#   ${PCL_LIBRARY_DIRS}
#   ${SLAM_LIBRARIES}
# )

# add_executable(lio_node src/LIO/FilterLIO/simple_lio_node.cpp)
# target_link_libraries(lio_node 
#                                             ${catkin_LIBRARIES} 
#                                             ${ALL_TARGET_LIBRARIES}
#                                             ${SLAM_LIBRARIES}
# )
                                            