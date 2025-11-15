#pragma once
#include <cuda_runtime.h>
#include <cufft.h>

void nufft_type1_radial(
    const float2* kdata,      // complex k-space samples (Nspokes * Ns)
    const float*  kx,         // x-coordinates in [-0.5,0.5)
    const float*  ky,         // y-coordinates
    int Nspokes, int Ns,      // sampling geometry
    float2* img,              // output image (N x N)
    int N,                    // image size
    float osf = 2.0f,         // oversampling factor
    int kernel_width = 6);    // Kaiser-Bessel width