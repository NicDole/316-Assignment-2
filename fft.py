#!/usr/bin/env python3
"""
ECSE 316 - Assignment 2
FFT implementation and applications.

Fill in all TODO blocks with your own code.
Do not use np.fft for the core algorithms.
"""

import argparse
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2  # optional, for image loading and resizing
except ImportError:
    cv2 = None


# ============================================================
# 1. Core 1D transforms
# ============================================================

def dft_naive_1d(x):
    """
    Naive O(N^2) Discrete Fourier Transform on a 1D array.

    Parameters
    ----------
    x : 1D numpy array of complex or float

    Returns
    -------
    X : 1D numpy array of complex
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    X = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        s = 0.0 + 0.0j
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            s += x[n] * np.exp(angle)
        X[k] = s

    return X


def fft_1d(x):
    """
    Cooley Tukey FFT for 1D arrays.
    Assume length is a power of 2 (you can pad before calling).

    Parameters
    ----------
    x : 1D numpy array of complex or float

    Returns
    -------
    X : 1D numpy array of complex
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    # Base case: for small N, just use the naive DFT
    if N <= 16:
        return dft_naive_1d(x)

    # Recursive Cooley Tukey FFT
    # split into even and odd indices
    X_even = fft_1d(x[0::2])
    X_odd = fft_1d(x[1::2])

    # twiddle factors
    k = np.arange(N)
    twiddle = np.exp(-2j * np.pi * k / N)

    X = np.zeros(N, dtype=np.complex128)
    half = N // 2

    X[:half] = X_even + twiddle[:half] * X_odd
    X[half:] = X_even - twiddle[:half] * X_odd

    return X


def ifft_1d(X):
    """
    Inverse FFT using the conjugate trick and fft_1d.

    Parameters
    ----------
    X : 1D numpy array of complex

    Returns
    -------
    x : 1D numpy array of complex
    """
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]

    # Conjugate, run forward FFT, conjugate again, then divide by N
    x_conj = fft_1d(np.conjugate(X))
    x = np.conjugate(x_conj) / N
    return x


# ============================================================
# 2. 2D transforms (built using the 1D transforms)
# ============================================================

def fft_2d(img):
    """
    2D FFT using fft_1d applied to rows then columns.

    Parameters
    ----------
    img : 2D numpy array

    Returns
    -------
    F : 2D numpy array of complex
    """
    img = np.asarray(img, dtype=np.complex128)
    rows, cols = img.shape

    # FFT along rows
    F = np.zeros((rows, cols), dtype=np.complex128)
    for i in range(rows):
        F[i, :] = fft_1d(img[i, :])

    # FFT along columns
    for j in range(cols):
        F[:, j] = fft_1d(F[:, j])

    return F


def ifft_2d(F):
    """
    2D inverse FFT using ifft_1d applied to rows then columns.

    Parameters
    ----------
    F : 2D numpy array of complex

    Returns
    -------
    img : 2D numpy array (complex, take real part for image)
    """
    F = np.asarray(F, dtype=np.complex128)
    rows, cols = F.shape

    # IFFT along rows
    img = np.zeros((rows, cols), dtype=np.complex128)
    for i in range(rows):
        img[i, :] = ifft_1d(F[i, :])

    # IFFT along columns
    for j in range(cols):
        img[:, j] = ifft_1d(img[:, j])

    return img

def naive_dft_2d(arr):
    """
    Naive 2D DFT using dft_naive_1d on rows then columns.
    Used only for runtime comparison in mode 4.
    """
    arr = np.asarray(arr, dtype=np.complex128)
    rows, cols = arr.shape

    # DFT along rows
    temp = np.zeros((rows, cols), dtype=np.complex128)
    for i in range(rows):
        temp[i, :] = dft_naive_1d(arr[i, :])

    # DFT along columns
    out = np.zeros((rows, cols), dtype=np.complex128)
    for j in range(cols):
        out[:, j] = dft_naive_1d(temp[:, j])

    return out


# ============================================================
# 3. Image utilities
# ============================================================

def load_image(path, default_size=None):
    """
    Load image as grayscale float64 in range [0, 1].

    If cv2 is available you can use it. Otherwise use matplotlib.
    """
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {path}")
        img = img.astype(np.float64) / 255.0
        if default_size is not None:
            img = cv2.resize(img, default_size, interpolation=cv2.INTER_AREA)
    else:
        img = plt.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {path}")
        if img.ndim == 3:
            # simple conversion to grayscale
            img = img[..., :3].mean(axis=2)
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img /= 255.0
        # resizing with matplotlib is possible but more work
    return img


def pad_to_power_of_two(arr):
    """
    Pad a 2D array with zeros so that each dimension is a power of 2.
    """
    rows, cols = arr.shape

    def next_power_of_two(n):
        return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

    new_rows = next_power_of_two(rows)
    new_cols = next_power_of_two(cols)

    padded = np.zeros((new_rows, new_cols), dtype=arr.dtype)
    padded[:rows, :cols] = arr
    return padded


def fftshift_2d(F):
    """
    Shift zero frequency to the center of the spectrum for visualization.
    """
    return np.fft.fftshift(F)


def plot_side_by_side(img1, img2, title1, title2, logscale_second=False):
    """
    Simple helper to show two images in a 1 by 2 subplot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(title1)
    axes[0].axis("off")

    if logscale_second:
        from matplotlib.colors import LogNorm
        axes[1].imshow(img2, cmap="gray", norm=LogNorm())
    else:
        axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_grid(images, titles, nrows=2, ncols=3):
    """
    Plot a grid of images, for example 2 by 3 for compression levels.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
    axes = axes.ravel()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def low_frequency_mask(shape, radius_ratio=0.2):
    """
    Create a circular low pass mask that keeps only low frequencies.

    Parameters
    ----------
    shape : (rows, cols)
    radius_ratio : fraction of half diagonal to keep

    Returns
    -------
    mask : 2D numpy array of 0 or 1
    """
    rows, cols = shape
    cy, cx = rows // 2, cols // 2

    y = np.arange(rows)[:, None]
    x = np.arange(cols)[None, :]

    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_radius = np.sqrt(cy ** 2 + cx ** 2)
    radius = radius_ratio * max_radius

    mask = (dist <= radius).astype(np.float64)
    return mask


# ============================================================
# 4. Mode handlers
# ============================================================

def run_mode1(image_path):
    """
    Mode 1:
    - load image
    - pad to power of 2
    - compute 2D FFT
    - show original image and log magnitude of FFT
    """
    # Load and pad image
    img = load_image(image_path)
    orig_rows, orig_cols = img.shape
    img_padded = pad_to_power_of_two(img)

    # Compute 2D FFT using your own implementation
    F = fft_2d(img_padded)

    # Shift for better visualization and take magnitude
    F_shifted = fftshift_2d(F)
    magnitude = np.abs(F_shifted) + 1e-8  # avoid log(0)

    # Show original (uncropped) image vs FFT magnitude (log scale)
    plot_side_by_side(
        img,
        magnitude,
        "Original image",
        "FFT magnitude",
        logscale_second=True,
    )



def run_mode2(image_path):
    """
    Mode 2:
    - denoising by zeroing high frequencies
    """
    # Load and pad image
    img = load_image(image_path)
    orig_rows, orig_cols = img.shape
    img_padded = pad_to_power_of_two(img)

    # Forward 2D FFT
    F = fft_2d(img_padded)

    # Shift to put low frequencies in the center
    F_shifted = fftshift_2d(F)

    # Design a low frequency mask
    # You can tune radius_ratio for better denoising performance
    mask = low_frequency_mask(F_shifted.shape, radius_ratio=0.2)

    # Apply mask in frequency domain
    F_filtered_shifted = F_shifted * mask

    # Count nonzero coefficients and fraction
    nnz = np.count_nonzero(F_filtered_shifted)
    total = F_filtered_shifted.size
    frac = nnz / total
    print(f"Nonzero coefficients after denoising: {nnz} ({frac:.6f} of total)")

    # Unshift back
    F_filtered = np.fft.ifftshift(F_filtered_shifted)

    # Inverse 2D FFT
    img_denoised_padded = np.real(ifft_2d(F_filtered))

    # Crop back to original size
    img_denoised = img_denoised_padded[:orig_rows, :orig_cols]

    # Plot original vs denoised
    plot_side_by_side(
        img,
        img_denoised,
        "Original image",
        "Denoised image",
        logscale_second=False,
    )



def run_mode3(image_path):
    """
    Mode 3:
    - compression with several levels
    - plot 2 by 3 grid of reconstructions
    - print number of nonzeros per level
    """
    # Load and pad image
    img = load_image(image_path)
    orig_rows, orig_cols = img.shape
    img_padded = pad_to_power_of_two(img)

    # Forward 2D FFT
    F = fft_2d(img_padded)

    # Flatten magnitudes for thresholding
    flat = F.flatten()
    mags = np.abs(flat)

    # Total number of coefficients
    total = F.size

    # Define compression levels:
    # fraction_zero is the fraction of coefficients set to zero
    # must include 0 percent and 99.9 percent
    # compression_levels = [0.0, 0.9, 0.99, 0.999, 0.9995, 0.9999]
    compression_levels = [0.0, 0.50, 0.90, 0.95, 0.99, 0.999]

    images = []
    titles = []

    # Sort magnitudes once to derive thresholds
    # indices sorted ascending by magnitude
    sorted_indices = np.argsort(mags)

    for frac_zero in compression_levels:
        # Number of coefficients to zero
        num_zero = int(round(frac_zero * total))
        num_zero = min(max(num_zero, 0), total)

        # Create a copy of F as 1D
        F_comp_flat = flat.copy()

        if num_zero > 0:
            # Zero out the smallest 'num_zero' coefficients
            zero_indices = sorted_indices[:num_zero]
            F_comp_flat[zero_indices] = 0.0

        # Count nonzero after compression
        nnz = np.count_nonzero(F_comp_flat)
        frac_keep = nnz / total

        print(
            f"Compression level: zero {frac_zero*100:.3f}% of coeffs  "
            f"kept {frac_keep*100:.3f}% ({nnz} / {total})"
        )

        # Reshape back to 2D
        F_comp = F_comp_flat.reshape(F.shape)

        # Inverse 2D FFT and crop back to original size
        img_rec_padded = np.real(ifft_2d(F_comp))
        img_rec = img_rec_padded[:orig_rows, :orig_cols]

        images.append(img_rec)
        titles.append(f"Zero {frac_zero*100:.3f}%")

    # Ensure we have exactly 6 images for a 2x3 grid
    plot_grid(images, titles, nrows=2, ncols=3)



def run_mode4():
    """
    Mode 4:
    runtime experiments for naive 2D DFT vs FFT based 2D
    print means and variances
    plot runtime vs size
    """
    sizes = [2 ** n for n in range(5, 10)]  # 32 to 512
    trials = 10

    naive_means = []
    naive_vars = []
    fft_means = []
    fft_vars = []

    for N in sizes:
        print(f"Running size N = {N}")
        naive_times = []
        fft_times = []

        for t in range(trials):
            arr = np.random.rand(N, N)

            # Time naive 2D DFT
            start = time.time()
            naive_dft_2d(arr)
            naive_times.append(time.time() - start)

            # Time FFT 2D
            start = time.time()
            fft_2d(arr)
            fft_times.append(time.time() - start)

        naive_means.append(np.mean(naive_times))
        naive_vars.append(np.var(naive_times))
        fft_means.append(np.mean(fft_times))
        fft_vars.append(np.var(fft_times))

    # Print means and variances
    print("\nNaive 2D DFT runtimes:")
    for N, mean, var in zip(sizes, naive_means, naive_vars):
        print(f"N = {N:4d}  mean = {mean:.6e} s  var = {var:.6e}")

    print("\nFFT based 2D runtimes:")
    for N, mean, var in zip(sizes, fft_means, fft_vars):
        print(f"N = {N:4d}  mean = {mean:.6e} s  var = {var:.6e}")

    # Error bars: 2 * std for a roughly 97 percent confidence interval
    naive_std = np.sqrt(naive_vars)
    fft_std = np.sqrt(fft_vars)

    plt.figure(figsize=(8, 5))
    plt.errorbar(sizes, naive_means, yerr=2 * naive_std, fmt="o-", label="Naive 2D DFT")
    plt.errorbar(sizes, fft_means, yerr=2 * fft_std, fmt="s-", label="2D FFT")
    plt.xlabel("Matrix size N (N x N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Naive 2D DFT vs 2D FFT runtime")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Argument parsing and main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ECSE 316 FFT assignment")
    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        default=1,
        help="Mode 1: FFT visualization (default)  "
             "2: denoising  3: compression  4: runtime plots",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="input.png",  # replace with provided default image name
        help="Input image file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 1:
        run_mode1(args.image)
    elif args.mode == 2:
        run_mode2(args.image)
    elif args.mode == 3:
        run_mode3(args.image)
    elif args.mode == 4:
        run_mode4()
    else:
        print(f"Unknown mode {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()