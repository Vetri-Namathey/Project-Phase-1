import numpy as np
from PIL import Image

idx = "000"  # change this to check other frames, e.g. "001", "025"

img = Image.open(f"data/images/anomaly_{idx}.png").convert("RGB")
mask = np.load(f"data/masks/anomaly_mask_{idx}.npy")

print("Mask shape:", mask.shape, "| unique values:", np.unique(mask), "| anomaly pixel count:", mask.sum())

overlay = np.array(img)
overlay[mask == 1] = [255, 0, 0]  # paint anomaly pixels red

Image.fromarray(overlay).save(f"check_overlay_{idx}.png")
print(f"Saved check_overlay_{idx}.png")
