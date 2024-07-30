cmake_minimum_required(VERSION 3.10)
project(DetectStripes)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(include ${OpenCV_INCLUDE_DIRS})

# 指定源文件
file(GLOB SOURCES "src/*.cpp")

# 生成可执行文件
add_executable(opencv_test1_camera ${SOURCES})

# 链接OpenCV库
target_link_libraries(opencv_test1_camera ${OpenCV_LIBS})

