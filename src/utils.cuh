#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(err), __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while(0)

#define CHECK_CUFFT(call)                                                 \
    do {                                                                  \
        cufftResult_t err = call;                                         \
        if (err != CUFFT_SUCCESS) {                                       \
            fprintf(stderr, "cuFFT error %d at %s:%d\n",                  \
                    err, __FILE__, __LINE__);                             \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while(0)