cmake_minimum_required(VERSION 3.6)
project(QLZM)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Import OpenCV
find_package(OpenCV)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(SOURCE_FILES main.cpp QLZM.cpp QLZM.h)
add_executable(QLZM ${SOURCE_FILES})

target_link_libraries(QLZM ${OpenCV_LIBS})
