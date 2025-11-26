import numpy as np
from fft import load_image, pad_to_power_of_two, fft_2d
import matplotlib.pyplot as plt

img = load_image("moonlanding.png")
img_padded = pad_to_power_of_two(img)

# Your FFT
F_custom = fft_2d(img_padded)

# NumPy FFT
F_numpy = np.fft.fft2(img_padded)

# Compare numerically
error = np.max(np.abs(F_custom - F_numpy))
print("Max difference:", error)

# Create figure with 3 subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis('off')

# Custom FFT
axes[1].imshow(np.log(np.abs(np.fft.fftshift(F_custom)) + 1e-8), cmap="gray")
axes[1].set_title("Custom FFT (log magnitude)")
axes[1].axis('off')

# NumPy FFT
axes[2].imshow(np.log(np.abs(np.fft.fftshift(F_numpy)) + 1e-8), cmap="gray")
axes[2].set_title("NumPy FFT2 (log magnitude)")
axes[2].axis('off')

plt.tight_layout()
plt.show()