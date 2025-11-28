#!/usr/bin/env python3

import numpy as np
import os
from scipy.sparse import csc_matrix, save_npz
import matplotlib.pyplot as plt

# Import your FFT-code without modifying it
import fft as myfft  


def main():

    image_path = "moonlanding.png"

    # ----------------------------------------------------------
    # 1. Load and pad image (using your existing utilities)
    # ----------------------------------------------------------
    img = myfft.load_image(image_path)
    orig_rows, orig_cols = img.shape
    img_padded = myfft.pad_to_power_of_two(img)

    # ----------------------------------------------------------
    # 2. Compute FFT once
    # ----------------------------------------------------------
    F = myfft.fft_2d(img_padded)

    flat = F.flatten()
    mags = np.abs(flat)
    total = F.size

    # Same 6 compression levels as your assignment:
    compression_levels = [0.0, 0.50, 0.90, 0.95, 0.99, 0.999]

    # Pre-sort indices by magnitude for thresholding
    sorted_indices = np.argsort(mags)

    recon_images = []
    titles = []

    print("\n=== Sparse File Sizes for Each Compression Level ===\n")

    for frac_zero in compression_levels:

        # ------------------------------
        # 3. Zero-out smallest coeffs
        # ------------------------------
        num_zero = int(round(frac_zero * total))
        F_comp_flat = flat.copy()

        if num_zero > 0:
            zero_idx = sorted_indices[:num_zero]
            F_comp_flat[zero_idx] = 0.0

        # ------------------------------
        # 4. Convert to sparse & save
        # ------------------------------
        F_sparse = csc_matrix(F_comp_flat.reshape(F.shape))

        filename = f"sparse_{int(frac_zero*1000):03d}.npz"
        save_npz(filename, F_sparse)

        size_kb = os.path.getsize(filename) / 1024
        nnz = F_sparse.count_nonzero()
        keep_frac = nnz / total

        print(f"Compression {frac_zero*100:.3f}% zeroed: "
              f"{size_kb:.2f} KB, "
              f"kept {keep_frac*100:.3f}% ({nnz}/{total})")

        # ------------------------------
        # 5. Reconstruct image for grid
        # ------------------------------
        img_rec_padded = np.real(myfft.ifft_2d(F_sparse.toarray()))
        img_rec = img_rec_padded[:orig_rows, :orig_cols]

        recon_images.append(img_rec)
        titles.append(f"{frac_zero*100:.1f}% zeroed")

    # ----------------------------------------------------------
    # 6. Display 2×3 grid (optional)
    # ----------------------------------------------------------
    myfft.plot_grid(recon_images, titles, nrows=2, ncols=3)


if __name__ == "__main__":
    main()
