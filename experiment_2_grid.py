#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Import your FFT functions
from fft import (
    load_image,
    pad_to_power_of_two,
    fft_2d,
    ifft_2d,
    fftshift_2d,
    low_frequency_mask,
)

def denoise_with_mask(F_shifted, radius_ratio):
    mask = low_frequency_mask(F_shifted.shape, radius_ratio=radius_ratio)
    F_filtered_shifted = F_shifted * mask
    F_filtered = np.fft.ifftshift(F_filtered_shifted)
    img_denoised = np.real(ifft_2d(F_filtered))
    return img_denoised

def run_denoise_grid(image_path):
    # Load + pad
    img = load_image(image_path)
    rows, cols = img.shape
    img_padded = pad_to_power_of_two(img)

    # Forward FFT
    F = fft_2d(img_padded)
    F_shifted = fftshift_2d(F)

    # Choose 5 different denoising strengths
    # radius_levels = [0.05, 0.25, 0.45, 0.65, 0.85]
    radius_levels = [0.80, 0.60, 0.30, 0.20, 0.05]

    images = [img]   # include original first
    titles = ["Original"]

    # Compute denoised versions
    for r in radius_levels:
        den = denoise_with_mask(F_shifted, r)
        den = den[:rows, :cols]  # crop back to original
        images.append(den)
        titles.append(f"radius={r}")

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_denoise_grid("moonlanding.png")
