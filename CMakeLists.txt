# cmake_minimum_required(VERSION 3.16.3)
# enable_language(CUDA)
# project(Cuda)
# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#     set(CMAKE_CUDA_STANDARD 11)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()
# set_property(TARGET cudaaa PROPERTY CUDA_ARCHITECTURES all-major)
# add_executable ( cuda123 kernelv2lab.cu)
set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

set(APP_NAME mainCudaApp)
cmake_minimum_required(VERSION 3.16)
include_directories(include) 
# add_compile_options(-g -G -Xcompiler -rdynamic -lineinfo )
add_compile_options(-g -G  )
add_link_options( -fsanitize=address -lasan)
# find_package(Thrust REQUIRED)
# thrust_create_target(Thrust)
project(CudaApp LANGUAGES CXX CUDA)
# target_include_directories(${APP_NAME} PRIVATE include)
# Locate sources and headers for this project
# include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include /usr/local/cuda/include )

# file(GLOB sources ${PROJECT_SOURCE_DIR}/*.cc ${PROJECT_SOURCE_DIR}/src/*.cc)
# file(GLOB headers ${PROJECT_SOURCE_DIR}/*.h ${PROJECT_SOURCE_DIR}/include/*.h)
# file(GLOB cuda_sources ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB cuda_sources "src/*.cu")

# Add the executable, and link 
# add_executable(myTarget ${sources} ${cuda_sources} ${headers})
add_executable(${APP_NAME} ${cuda_sources} )
# target_link_libraries(CudaApp pthread z)

# target_link_libraries(${APP_NAME} )

# Compile Info
# set(CMAKE_CUDA_ARCHITECTURES native)


set_target_properties(${APP_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_features(${APP_NAME} PUBLIC cxx_std_11)
target_compile_features(${APP_NAME} PUBLIC cuda_std_11)