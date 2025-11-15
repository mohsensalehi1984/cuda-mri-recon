#!/usr/bin/env python3
import numpy as np, cv2, sys, os

def shepp_logan(N):
    img = np.zeros((N, N), dtype=np.float32)
    ellipses = [
        (1,    0.69,  0.92,  0, 0, 0),
        (-0.8, 0.6624,0.8740,0, -0.0184, 0),
        (-0.2, 0.1100,0.3100,0.22, 0, -18),
        (-0.2, 0.1600,0.4100,-0.22,0, 18),
        (0.1,  0.2100,0.2500,0, 0.35, 0),
        (0.1,  0.0460,0.0460,0, 0.1, 0),
        (0.1,  0.0460,0.0460,0, -0.1, 0),
        (0.1,  0.0460,0.0230,-0.08,-0.605,0),
        (0.1,  0.0230,0.0230,0, -0.606,0),
        (0.1,  0.0230,0.0460,0.06,-0.605,0),
    ]
    for a, b, c, x0, y0, phi in ellipses:
        a2, b2 = a*a, b*b
        phi = np.deg2rad(phi)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        x, y = np.meshgrid(np.linspace(-1,1,N), np.linspace(-1,1,N))
        x, y = x - x0, y - y0
        xrot = x*cos_p + y*sin_p
        yrot = -x*sin_p + y*cos_p
        inside = ((xrot/c)**2 + (yrot/b)**2) <= 1
        img[inside] += a
    return img

if __name__ == "__main__":
    N = 256
    img = shepp_logan(N)
    cv2.imwrite("../data/phantom.png", (img*255).astype(np.uint8))
    np.save("../data/phantom.npy", img)