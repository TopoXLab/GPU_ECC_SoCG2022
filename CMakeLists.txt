cmake_minimum_required(VERSION 3.14)
set(CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler=\"-MT -O2 -Ob2\" -DNDEBUG")

project(GPU_ECC LANGUAGES CUDA CXX)

# multi-threaded
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify boost path
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
#set(Boost_INCLUDE_DIRS "D:/Tools/boost_1_77_0")
#set(Boost_LIBRARY_DIRS "D:/Tools/boost_1_77_0/lib64-msvc-14.2")
#set(OpenCV_INCLUDE_DIRS "D:/Tools/opencv/build/x64/vc14/lib")

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_package(OpenMP REQUIRED)
find_package(OpenCV COMPONENTS core REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
set(CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler=\"-MT -O2 -Ob2\" -DNDEBUG")

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_executable(GPU_ECC main.cpp routines.cpp utility.cpp kernel.cu)
set(CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler=\"-MT -O2 -Ob2\" -DNDEBUG")

target_link_libraries(GPU_ECC PUBLIC OpenMP::OpenMP_CXX ${Boost_LIBRARIES} ${OpenCV_LIBS})
set(CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler=\"-MT -O2 -Ob2\" -DNDEBUG")