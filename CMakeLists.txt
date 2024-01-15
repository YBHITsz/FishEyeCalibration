cmake_minimum_required(VERSION 2.8.3)
project(calib)

add_compile_options(-std=c++11)
find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRECTORIES}
)

add_executable(calib
  fish_eye_calib.cpp
)

target_link_libraries(calib
  ${OpenCV_LIBRARIES}
)