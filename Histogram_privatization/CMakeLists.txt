cmake_minimum_required(VERSION 3.18)
project(Histogram_Image_Privatized)

set(CMAKE_PREFIX_PATH "/home/praneet/libtorch")

enable_language(CUDA)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

add_executable(app main.cpp histogram_privatized.cu)
target_link_libraries(app "${TORCH_LIBRARIES}")
target_include_directories(app PRIVATE ${TORCH_INCLUDE_DIRS})
