cmake_minimum_required(VERSION 3.12)
project(testseg)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#OpenCV
set(OpenCV_DIR "/home/nreal/software/opencv-3.4.0/build")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(testseg test_seg.cpp)

target_link_libraries(testseg 
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/libSegmentation_x86.so
  ${OpenCV_LIBS}
  libdl.so
)
