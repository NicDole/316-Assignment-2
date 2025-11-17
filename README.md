# ECSE-316 ASSIGNMENT 2

**November 26th, 2025**

S. Nicolas Dolgopolyy *(261115875)*<br/>
Matthew Eiley *(261177542)*

### Overview

This project implements the Discrete Fourier Transform (DFT) and the Fast Fourier Transform (FFT) from scratch and applies them to several image processing tasks. The assignment requires building both a naive O(N²) DFT and a recursive Cooley Tukey O(N log N) FFT, then extending these algorithms to two dimensions. The 2D transforms are used to visualize frequency content, denoise images, perform frequency based compression, and compare runtimes of different algorithms.

All core Fourier operations are implemented without using NumPy’s built in FFT functions. The program accepts command line arguments to select between four modes: FFT visualization, denoising, compression, and runtime analysis.

### Python Version
Python 3.11.3 (tested successfully on Windows 11 and macOS 14.5)

### Compilation Instructions
No compilation required. The program runs directly with the Python interpreter.

### Execution
The program is executed as:

python fft.py [-m mode] [-i image]

Arguments

-m mode

1 (default) compute the 2D FFT of an image and display both the original and the log scaled Fourier magnitude

2 denoise the image by zeroing high frequency components and reconstructing it with the inverse FFT

3 compress the image using frequency thresholding and display six compression levels in a grid

4 run timing experiments comparing the naive 1D DFT against the FFT on 2D arrays of increasing size

-i image
Path to the input image file. If not provided, a default assignment image is used.

Each mode handles all required steps automatically, including padding the image to a power of two, computing FFT and inverse FFT, generating plots, displaying reconstructed images, and printing statistics such as coefficient counts or runtime summaries.
