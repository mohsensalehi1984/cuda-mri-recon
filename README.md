# CUDA MRI Reconstruction (Radial NUFFT)

A minimal CUDA project that reconstructs an image from **radial k-space samples** using a **GPU-accelerated NUFFT**.

## Features
* Generates a Shepp-Logan phantom (CPU)
* Simulates radial sampling
* GPU NUFFT (type-1: k-space → image)
* Simple gridding + density compensation
* Visual output with OpenCV

## Requirements
* CUDA Toolkit ≥ 12.0
* CMake ≥ 3.18
* OpenCV (for display)
* cuFFT (bundled with CUDA)

## Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Run
```bash
./mri_recon
```
The reconstructed image is shown in a window and saved as `recon.png`.

## License
MIT


---


Feel free to expand the `nufft.cu` kernel, add density compensation, or plug in real raw `.dat` files from your hypothetical MRI device. Happy coding!