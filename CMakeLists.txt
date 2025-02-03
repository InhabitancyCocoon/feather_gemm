cmake_minimum_required(VERSION 3.19)
project(FEATHER_GEMM LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_COMPUTE_CAPABILITY 89)

# This option demonstrates resource usage.
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/kernels)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/kernels)


file(GLOB KERNEL_SOURCES "${CMAKE_SOURCE_DIR}/src/kernels/*.cu")

# generate executable

add_executable(dummy_kernel dummy_kernel.cu)
target_link_libraries(dummy_kernel ${CUDA_LIBRARIES})


add_executable(cublas_sgemm cublas_sgemm.cu src/run_utils.cu ${KERNEL_SOURCES})
target_link_libraries(cublas_sgemm ${CUDA_LIBRARIES} cublas)


add_executable(main main.cu src/run_utils.cu ${KERNEL_SOURCES})
target_link_libraries(main ${CUDA_LIBRARIES} cublas)