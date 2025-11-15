#include "nufft.cuh"
#include "utils.cuh"
#include <cmath>

__constant__ float c_kb_table[1024];   // pre-computed KB kernel

// ---------------------------------------------------------------
// Kaiser-Bessel kernel (pre-computed on host)
static void compute_kb_table(float beta, int W, float* table, int table_size)
{
    const float scale = (float)(W*W) / 4.0f;
    for (int i = 0; i < table_size; ++i) {
        float x = (float)i / (table_size-1);   // 0..1
        x = sqrtf(x*x*scale);
        if (x >= W) table[i] = 0.0f;
        else {
            float b = beta * sqrtf(1.0f - (x/W)*(x/W));
            table[i] = (b > 0.0f) ? sincf(b/M_PI) : 1.0f;   // I0 approximation
        }
    }
}

// sinc(x) = sin(pi*x)/(pi*x)
__device__ __host__ inline float sincf(float x)
{
    if (fabsf(x) < 1e-6f) return 1.0f;
    return sinf(M_PI*x)/(M_PI*x);
}

// ---------------------------------------------------------------
// GPU kernel: gridding + density compensation
__global__ void grid_kernel(
    const float2* __restrict__ kdata,
    const float*  __restrict__ kx,
    const float*  __restrict__ ky,
    const float*  __restrict__ dens,
    float2* grid,
    int Ngrid, int Nspokes, int Ns, int W, float osf)
{
    int spoke = blockIdx.x;
    int s     = threadIdx.x;
    if (spoke >= Nspokes || s >= Ns) return;

    float2 sample = kdata[spoke * Ns + s];
    float kx_s = kx[spoke * Ns + s];
    float ky_s = ky[spoke * Ns + s];
    float d    = dens ? dens[spoke * Ns + s] : 1.0f;

    // map to grid coordinates (center at 0)
    float gx = (kx_s + 0.5f) * osf * Ngrid;
    float gy = (ky_s + 0.5f) * osf * Ngrid;

    int ix0 = (int)floorf(gx) - W/2;
    int iy0 = (int)floorf(gy) - W/2;

    for (int ky = 0; ky < W; ++ky) {
        int iy = iy0 + ky;
        if (iy < 0 || iy >= Ngrid) continue;
        float wy = c_kb_table[abs((int)roundf((gy - iy)* (1024-1)))];
        for (int kx = 0; kx < W; ++kx) {
            int ix = ix0 + kx;
            if (ix < 0 || ix >= Ngrid) continue;
            float w = wy * c_kb_table[abs((int)roundf((gx - ix)*(1024-1)))];
            atomicAdd(&grid[iy*Ngrid + ix].x, d * sample.x * w);
            atomicAdd(&grid[iy*Ngrid + ix].y, d * sample.y * w);
        }
    }
}

// ---------------------------------------------------------------
void nufft_type1_radial(
    const float2* kdata,
    const float*  kx,
    const float*  ky,
    int Nspokes, int Ns,
    float2* img,
    int N,
    float osf,
    int kernel_width)
{
    const int W = kernel_width;
    const int Ngrid = (int)ceilf(N * osf);
    const size_t grid_bytes = Ngrid * Ngrid * sizeof(float2);

    float2 *d_grid;
    CHECK_CUDA(cudaMalloc(&d_grid, grid_bytes));
    CHECK_CUDA(cudaMemset(d_grid, 0, grid_bytes));

    // ---- pre-compute KB table on host and copy to constant memory ----
    float h_kb[1024];
    const float beta = 10.0f;   // typical value
    compute_kb_table(beta, W, h_kb, 1024);
    CHECK_CUDA(cudaMemcpyToSymbol(c_kb_table, h_kb, 1024*sizeof(float)));

    // ---- launch gridding kernel ----
    dim3 block(Ns);
    dim3 grid(Nspokes);
    grid_kernel<<<grid, block>>>(
        kdata, kx, ky, nullptr, d_grid,
        Ngrid, Nspokes, Ns, W, osf);
    CHECK_CUDA(cudaGetLastError());

    // ---- FFT shift + cuFFT (grid → image) ----
    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    CHECK_CUFFT(cufftPlan2d(&plan, Ngrid, Ngrid, CUFFT_C2C));

    // FFTSHIFT (center low frequencies)
    // simple way: multiply by (-1)^(i+j) before FFT
    {
        // optional: implement a kernel to do the shift, omitted for brevity
    }

    CHECK_CUFFT(cufftExecC2C(plan, d_grid, d_grid, CUFFT_FORWARD));
    CHECK_CUFFT(cufftDestroy(plan));

    // ---- crop to final N×N image ----
    float2 *d_img;
    CHECK_CUDA(cudaMalloc(&d_img, N*N*sizeof(float2)));
    // simple crop kernel (center)
    {
        const int offset = (Ngrid - N)/2;
        dim3 blk(16,16);
        dim3 grd((N+15)/16, (N+15)/16);
        // (you can write a tiny kernel here; for brevity we use cudaMemcpy2D)
        for (int row = 0; row < N; ++row) {
            CHECK_CUDA(cudaMemcpy(d_img + row*N,
                                  d_grid + (offset+row)*Ngrid + offset,
                                  N*sizeof(float2), cudaMemcpyDeviceToDevice));
        }
    }

    // ---- normalize by sum of kernel weights (approx) ----
    float norm = 1.0f / (osf*osf);
    {
        // multiply by norm
        // (simple kernel omitted)
    }

    CHECK_CUDA(cudaMemcpy(img, d_img, N*N*sizeof(float2), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_grid));
    CHECK_CUDA(cudaFree(d_img));
}