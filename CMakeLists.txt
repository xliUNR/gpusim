cmake_minimum_required(VERSION 2.6)
project(gpusim)

# set project standards and locations
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules")
#set(CXX11_FLAGS -std=gnu++11)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread --std=c++11 -O3 -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std c++11 -Xcompiler -fopenmp -g -G")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -Wall -O3 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES") # to get rid of issues with strcpy

# find the dependencies
find_package(CUDA REQUIRED)
find_cuda_helper_libs(cusolver)

# set the includes and the src

file(GLOB_RECURSE SOURCES "src/*.h" "src/*.cpp" "src/*.cu")

# finally the exec
cuda_add_executable(
        ${PROJECT_NAME}
        ${SOURCES}
        )

