#pragma once

#include <cuda_runtime.h>
#include <vector>

extern "C" void setBasicParams_const_(
    std::vector<std::pair<int, int>>& section,
    int imageW,
    int imageH = -1
);

extern "C" cudaArray** allocate_cudaArray_array_(
    const int num,
    const int imageH,
    const int imageW, 
    const int imageD = -1
);

extern "C" cudaMemcpy3DParms setup_3DCopy_params(
    cudaArray * texSrc,
    float* input_host,
    const int imageH,
    const int imageW,
    const int imageD,
    const int offset
);

extern "C" cudaTextureObject_t* create_cudaTextureObject_array_(
    cudaArray * *cudaArray_array,
    const int num
);

extern "C" void free_cudaArray_array(
    cudaArray** cudaArray_array,
    const int num
);

extern "C" cudaEvent_t * create_cudaEvent_array(
    const int num
);

extern "C" cudaStream_t * create_cudaStream_array(
    const int num
);

extern "C" void free_cudaEvent_array(
    cudaEvent_t * events,
    const int num
);

extern "C" void free_cudaStream_array(
    cudaStream_t * streams,
    const int num
);

extern "C" void computeECC(
    int imageH,
    int imageW,
    int binNum,
    cudaTextureObject_t texSrc,
    int* VCEC_device,
    float* ascend_unique_arr_device_,
    int chunk_type_id,
    cudaStream_t * stream
);

extern "C" void computeECC_3D(
    int imageH,
    int imageW,
    int imageD,
    int binNum,
    cudaTextureObject_t texSrc,
    int* VCEC_device,
    float* ascend_unique_arr_device_,
    int chunk_type_id,
    cudaStream_t * stream
);

extern "C" void init_VCEC_device(
    int binNum,
    int* VCEC_device,
    cudaStream_t * stream
);