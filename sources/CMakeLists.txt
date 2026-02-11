cmake_minimum_required(VERSION 3.20)

project(GPU_ECC LANGUAGES CXX CUDA)

# Visual Studio 2022+ recommended
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default build type for single-config generators (VS ignores this)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA >= 12.6 check (best-effort; requires CUDA language enabled)
if(DEFINED CMAKE_CUDA_COMPILER_VERSION)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.6")
    message(FATAL_ERROR "CUDA >= 12.6 required. Found ${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
endif()

# Build target
add_executable(GPU_ECC
  main.cpp
  routines.cpp
  utility.cpp
  kernel.cu
)

target_include_directories(GPU_ECC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(GPU_ECC PRIVATE OpenMP::OpenMP_CXX)

# MSVC runtime: /MT (Release) and /MTd (Debug)
if(MSVC)
  set_property(TARGET GPU_ECC PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  # Enable OpenMP for MSVC (CMake's OpenMP target usually adds it, but keep explicit)
  target_compile_options(GPU_ECC PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp>)

  # Useful MSVC warnings/flags
  target_compile_options(GPU_ECC PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CXX>:/DNDEBUG>
  )

  # CUDA flags for MSVC host compiler
  target_compile_options(GPU_ECC PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/DNDEBUG>
  )
else()
  # Non-MSVC (Linux) flags
  target_compile_options(GPU_ECC PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()

# CUDA architectures (edit to match your needs)
# You can also set CMAKE_CUDA_ARCHITECTURES via CMake GUI.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET GPU_ECC PROPERTY CUDA_ARCHITECTURES 75;80;86;89;90)
endif()

# Ensure separable compilation OFF unless you need it
set_property(TARGET GPU_ECC PROPERTY CUDA_SEPARABLE_COMPILATION OFF)