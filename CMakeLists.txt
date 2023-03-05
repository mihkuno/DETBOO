cmake_minimum_required(VERSION 3.0.0)
project(DETBOO VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(DETBOO main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(DETBOO ${OpenCV_LIBS} ${CMAKE_THREADS_LIBS_INIT} opencv_dnn)