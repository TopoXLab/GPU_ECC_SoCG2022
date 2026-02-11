# GPU-ECC Makefile (Linux + Windows via MSYS2/MinGW or WSL)
# Requires: CUDA (nvcc) + a C++ compiler + make
# OpenMP enabled

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++

# Output
BUILD_DIR ?= build
TARGET    ?= GPU_ECC

# Detect Windows (MSYS2/MinGW make)
ifeq ($(OS),Windows_NT)
  EXE := .exe
else
  EXE :=
endif

# Sources
CPP_SRCS := main.cpp routines.cpp utility.cpp
CU_SRCS  := kernel.cu

CPP_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CU_OBJS  := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))

INCLUDES := -I. -I$(CUDA_HOME)/include
LIBS     := -L$(CUDA_HOME)/lib64 -lcudart -lcuda

# Tune these as you like
CXXSTD   := -std=c++17
OPT      := -O3 -DNDEBUG

# OpenMP flags (works for GCC/MinGW g++)
OMP_CXX  := -fopenmp
OMP_NVCC := -Xcompiler -fopenmp

# CPU tuning (safe on Linux; you can remove on Windows if you want)
CPU_TUNE := -march=native -mtune=native

# gencode (edit to match your GPUs)
GENCODE ?= \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90

CXXFLAGS  := $(OPT) $(CXXSTD) $(OMP_CXX) $(CPU_TUNE) $(INCLUDES) -fPIC
NVCCFLAGS := $(OPT) -use_fast_math $(CXXSTD) $(OMP_NVCC) -Xcompiler -fPIC -Xcompiler "$(CPU_TUNE)" $(INCLUDES) $(GENCODE)

.PHONY: all clean

all: $(BUILD_DIR)/$(TARGET)$(EXE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile C++ sources
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources
$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link (use nvcc for simplest CUDA linking)
$(BUILD_DIR)/$(TARGET)$(EXE): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(OPT) $(CXXSTD) $(OMP_NVCC) -o $@ $^ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)