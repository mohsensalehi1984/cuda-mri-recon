cmake_minimum_required(VERSION 3.18)
project(cuda_mri_recon LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(mri_recon
    src/main.cu
    src/nufft.cu
    src/utils.cu
)

target_include_directories(mri_recon PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mri_recon ${OpenCV_LIBS})

set_target_properties(mri_recon PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70;75;80;86)   # adjust for your GPU