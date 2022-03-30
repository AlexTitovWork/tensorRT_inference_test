cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Use home location CUDA 11.5+CUDNN8.3.2.
#SET(CUDA_HOME /usr/local/cuda-11.6)
SET(CUDA_HOME /home/alex/lib/cuda-11.6-cudnn8.3.2)

# Use standard compiler path
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)

# Use home lib's location CUDA 11.5+CUDNN8.3.2.
#SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.6)
SET(CUDA_TOOLKIT_ROOT_DIR /home/alex/lib/cuda-11.6-cudnn8.3.2)

# Use home lib's location CUDA 11.5+CUDNN8.3.2.
#SET(CUDA_CUBLAS_LIBRARIES /home/alex/lib/cuda-11.6/lib64)
SET(CUDA_CUBLAS_LIBRARIES /home/alex/lib/cuda-11.6-cudnn8.3.2/lib64)

# Use home lib's location CUDA 11.5+CUDNN8.3.2.
#SET(CUDA_cublas_device_LIBRARY /usr/local/cuda-11.6/lib64)
SET(CUDA_cublas_device_LIBRARY /home/alex/lib/cuda-11.6-cudnn8.3.2/lib64)

# Use home include location CUDA 11.5+CUDNN8.3.2.
#SET(CUDA_INCLUDE_DIRS /usr/local/cuda-11.6/include)
SET(CUDA_INCLUDE_DIRS /home/alex/lib/cuda-11.6-cudnn8.3.2/include)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

# set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
set(OpenCV_DIR /home/alex/local/lib/cmake/opencv4)

# set(OpenCV_DIR /usr/local/lib)

project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

include_directories(${CMAKE_SOURCE_DIR})

INCLUDE_DIRECTORIES(
  /usr/src/tensorrt/samples/common
    )

include_directories(../common)



# Use new location in home dir
#find_package(CUDA REQUIRED PATHS="/usr/local/cuda-11.6")
find_package(CUDA REQUIRED PATHS="/home/alex/lib/cuda-11.6-cudnn8.3.2")

# cudnn path on RTX 3090
# cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2
# https://stackoverflow.com/questions/31326015/how-to-verify-cudnn-installation
# not operate wit 1.8.2-cu111 torch version
# Solution. need 1.10.2-cu113 torch version, for CUDA 11.6

# use 1.10.2-cu113 for cuda 11.6
list(APPEND CMAKE_PREFIX_PATH ../libtorch)
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")

# set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
# list(APPEND CMAKE_PREFIX_PATH  ./opencv4)

# libtorch with OpenCV support , with fixed Target BUG !!!!!
#find_package(OpenCV REQUIRED PATHS "/usr/local")
find_package(OpenCV REQUIRED PATHS)
# include_directories( ${OpenCV_INCLUDE_DIRS} )
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")


  
# Request that particles_new_lib be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles_new_lib will also build with -std=c++11
# target_compile_features(particles_new_lib PUBLIC cxx_std_14)
 
# We need to explicitly state that we need all CUDA files in the 
# particles_new_lib library to be built with -dc as the member functions 
# could be called by other libraries and executables
# set_target_properties( particles_new_lib
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
# add_executable(loader_torch_init_step loader_torch_init_step.cpp)
# add_executable(loader_torch_init_step loader_torch_init_step.cpp)
# add_executable(loader_torch_init_step loader_direct_cuda.cu)
add_executable(deserialize_model_to_get_engine deserialize_model_to_get_engine.cpp)

set_property(TARGET deserialize_model_to_get_engine 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(deserialize_model_to_get_engine PRIVATE 
                        ${TORCH_LIBRARIES} 
                        ${OpenCV_LIBS}
                        /usr/lib/x86_64-linux-gnu/libnvinfer.so)
# target_link_libraries(loader_torch_init_step PRIVATE ${TORCH_LIBRARIES} )

# target_link_libraries(particle_cuda_test PRIVATE particles_new_lib)

# LINK LIBRARIES

# target_link_libraries(particle_cuda_test PRIVATE "${OpenCV_LIBS}"  )

# target_link_libraries(particle_cuda_test PRIVATE "${TORCH_LIBRARIES}")

# target_link_libraries(particle_cuda_test PRIVATE particles_new_lib) # Duplicate link, use if not use ${TORCH_LIBRARIES} ${OpenCV_LIBS} libs common.
set_property(TARGET deserialize_model_to_get_engine PROPERTY CXX_STANDARD 14)