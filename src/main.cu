#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include "nufft.cuh"
#include "utils.cuh"

int main()
{
    const int N        = 256;   // image size
    const int Ns       = 404;   // samples per spoke (Nyquist for N=256)
    const int Nspokes  = 360;   // full circle

    // -------------------------------------------------
    // 1. Load / generate phantom
    cv::Mat phantom = cv::imread("../data/phantom.png", cv::IMREAD_GRAYSCALE);
    if (phantom.empty()) {
        fprintf(stderr, "Run generate_phantom.py first!\n");
        return -1;
    }
    cv::resize(phantom, phantom, cv::Size(N,N));
    phantom.convertTo(phantom, CV_32F, 1.0/255.0);

    // -------------------------------------------------
    // 2. Simulate radial k-space (forward NUFFT on CPU for demo)
    std::vector<float2> kdata(Nspokes * Ns);
    std::vector<float>  kx(Nspokes * Ns), ky(Nspokes * Ns);

    const float kmax = 0.5f;               // normalized frequency
    const float dk   = kmax / (Ns/2);

    for (int ang = 0; ang < Nspokes; ++ang) {
        float theta = M_PI * ang / Nspokes;
        float cx = cosf(theta), sx = sinf(theta);
        for (int s = 0; s < Ns; ++s) {
            float r = -kmax + s * dk;      // from -kmax .. +kmax
            kx[ang*Ns + s] = r * cx;
            ky[ang*Ns + s] = r * sx;

            // forward projection (simple sum over image)
            float re = 0, im = 0;
            for (int y = 0; y < N; ++y) {
                for (int x = 0; x < N; ++x) {
                    float px = (x - N/2.0f)/N;   // normalized -0.5..0.5
                    float py = (y - N/2.0f)/N;
                    float phase = 2*M_PI * (kx[ang*Ns+s]*px + ky[ang*Ns+s]*py);
                    float val = phantom.at<float>(y,x);
                    re += val * cosf(phase);
                    im += val * sinf(phase);
                }
            }
            kdata[ang*Ns + s] = make_float2(re, im);
        }
    }

    // -------------------------------------------------
    // 3. Allocate GPU memory for k-space
    float2 *d_kdata;
    float  *d_kx, *d_ky;
    CHECK_CUDA(cudaMalloc(&d_kdata, kdata.size()*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&d_kx,    kx.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ky,    ky.size()*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_kdata, kdata.data(), kdata.size()*sizeof(float2), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kx,    kx.data(),    kx.size()*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ky,    ky.data(),    ky.size()*sizeof(float),  cudaMemcpyHostToDevice));

    // -------------------------------------------------
    // 4. Reconstruction
    float2 *h_img = (float2*)malloc(N*N*sizeof(float2));
    nufft_type1_radial(d_kdata, d_kx, d_ky, Nspokes, Ns, h_img, N, 1.8f, 6);

    // -------------------------------------------------
    // 5. Convert to OpenCV image & display
    cv::Mat recon(N, N, CV_32F);
    for (int i = 0; i < N*N; ++i) {
        float mag = sqrtf(h_img[i].x*h_img[i].x + h_img[i].y*h_img[i].y);
        recon.at<float>(i/N, i%N) = mag;
    }
    cv::normalize(recon, recon, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Reconstruction", recon);
    cv::imwrite("recon.png", (recon*255).astype(cv::Mat_<uchar>()));
    cv::waitKey(0);

    // -------------------------------------------------
    // cleanup
    free(h_img);
    CHECK_CUDA(cudaFree(d_kdata));
    CHECK_CUDA(cudaFree(d_kx));
    CHECK_CUDA(cudaFree(d_ky));
    return 0;
}