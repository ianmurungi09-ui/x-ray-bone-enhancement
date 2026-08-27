import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Path Setup
input_path = "xray_input.png"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 2. Load Image (Grayscale)
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(
        f"Could not find '{input_path}'. Place your X-ray image in the project directory."
    )

# 3. Step-by-Step Algorithm Execution
denoised = cv2.bilateralFilter(
    img, d=9, sigmaColor=75, sigmaSpace=75
)  # Step A: Denoising
low_freq = cv2.GaussianBlur(
    denoised, (21, 21), 0
)  # Step B: Low-pass extraction
high_freq = cv2.subtract(
    denoised, low_freq
)  # Step C: High-pass extraction (Crack Map)

amount = 1.5
sharpened = cv2.addWeighted(
    denoised, 1.0, high_freq, amount, 0
)  # Step D: Unsharp Masking

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
final_result = clahe.apply(sharpened)  # Step E: CLAHE Normalization

# 4. Save Individual Visual Stage Outputs
cv2.imwrite(f"{output_dir}/1_denoised.png", denoised)
cv2.imwrite(f"{output_dir}/2_low_freq.png", low_freq)
cv2.imwrite(f"{output_dir}/3_crack_map.png", high_freq)
cv2.imwrite(f"{output_dir}/4_sharpened.png", sharpened)
cv2.imwrite(f"{output_dir}/5_final_result.png", final_result)

# 5. Display 5-Panel Visual Demo
titles = ["Original", "Denoised", "Sharpened", "Crack Map", "Final Result"]
images = [img, denoised, sharpened, high_freq, final_result]

plt.figure(figsize=(16, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i], cmap="bone")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/visual_demo_comparison.png", dpi=300, bbox_inches="tight"
)
plt.show()