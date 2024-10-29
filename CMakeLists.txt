cmake_minimum_required(VERSION 3.19)
project(OpenCL)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

add_executable(vector_add main.cpp)
target_include_directories(vector_add PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(vector_add PRIVATE ${OpenCL_LIBRARIES})
