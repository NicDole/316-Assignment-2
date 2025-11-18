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

    # TODO: implement the direct DFT sum
    # for k in range(N):
    #     s = 0
    #     for n in range(N):
    #         angle = -2j * np.pi * k * n / N
    #         s += x[n] * np.exp(angle)
    #     X[k] = s

    raise NotImplementedError("dft_naive_1d is not implemented yet")
    # return X


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

    # Base case(s)
    # TODO: choose a base size (for example N <= 1 or N <= 2) and compute directly

    # Recursive case
    # TODO:
    # 1. split x into even and odd indices
    # 2. recursively compute FFT of each half
    # 3. combine using twiddle factors:
    #    W_N^k = exp(-2j * pi * k / N)
    # 4. return concatenated result

    raise NotImplementedError("fft_1d is not implemented yet")
    # return X


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

    # TODO:
    # 1. take conjugate of X
    # 2. run fft_1d on that
    # 3. take conjugate of the result
    # 4. divide by N

    raise NotImplementedError("ifft_1d is not implemented yet")
    # return x


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
    # TODO: for each row, apply fft_1d

    # FFT along columns
    # TODO: for each column, apply fft_1d

    raise NotImplementedError("fft_2d is not implemented yet")
    # return F


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
    # TODO

    # IFFT along columns
    # TODO

    raise NotImplementedError("ifft_2d is not implemented yet")
    # return img


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
    img = load_image(image_path)
    img_padded = pad_to_power_of_two(img)

    # TODO: use your fft_2d (not np.fft.fft2)
    # F = fft_2d(img_padded)

    # Optionally shift for better visualization
    # F_shifted = fftshift_2d(F)
    # magnitude = np.abs(F_shifted) + 1e-8

    raise NotImplementedError("run_mode1 is not implemented yet")
    # plot_side_by_side(img, magnitude, "Original", "FFT magnitude (log scale)", logscale_second=True)


def run_mode2(image_path):
    """
    Mode 2:
    - denoising by zeroing high frequencies
    """
    img = load_image(image_path)
    img_padded = pad_to_power_of_two(img)

    # TODO:
    # F = fft_2d(img_padded)
    # F_shifted = fftshift_2d(F)
    # mask = low_frequency_mask(F_shifted.shape, radius_ratio=0.2)  # tune this
    # F_filtered_shifted = F_shifted * mask
    # unshift back: F_filtered = np.fft.ifftshift(F_filtered_shifted)
    # img_denoised = np.real(ifft_2d(F_filtered))

    # TODO: print number of nonzeros and fraction
    # nnz = np.count_nonzero(F_filtered)
    # frac = nnz / F_filtered.size
    # print(f"Nonzero coefficients: {nnz} ({frac:.6f} of total)")

    raise NotImplementedError("run_mode2 is not implemented yet")
    # plot_side_by_side(img, img_denoised, "Original", "Denoised")


def run_mode3(image_path):
    """
    Mode 3:
    - compression with several levels
    - plot 2 by 3 grid of reconstructions
    - print number of nonzeros per level
    """
    img = load_image(image_path)
    img_padded = pad_to_power_of_two(img)

    # TODO:
    # F = fft_2d(img_padded)
    # flat = F.flatten()
    # magnitudes = np.abs(flat)
    # choose thresholds or number of coefficients to keep for each level
    # levels = [0.0, 0.9, 0.99, 0.999, ...]  # fraction of coefficients to zero, or similar

    # images = []
    # titles = []
    # for level in levels:
    #     ...
    #     images.append(reconstructed_image)
    #     titles.append(f"compression ...")
    #     print nonzeros

    raise NotImplementedError("run_mode3 is not implemented yet")
    # plot_grid(images, titles)


def run_mode4():
    """
    Mode 4:
    - runtime experiments for naive 2D DFT vs FFT based 2D
    - print means and variances
    - plot runtime vs size
    """
    sizes = [2 ** n for n in range(5, 11)]  # 32 to 1024, adjust if too slow
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
            # TODO: define a function that applies dft_naive_1d to rows then columns
            # start = time.time()
            # naive_2d(arr)
            # naive_times.append(time.time() - start)

            # Time FFT 2D
            # start = time.time()
            # fft_2d(arr)
            # fft_times.append(time.time() - start)

        # naive_means.append(np.mean(naive_times))
        # naive_vars.append(np.var(naive_times))
        # fft_means.append(np.mean(fft_times))
        # fft_vars.append(np.var(fft_times))

    # TODO: print the means and variances
    # TODO: make a plot of size vs runtime with error bars

    raise NotImplementedError("run_mode4 is not implemented yet")


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