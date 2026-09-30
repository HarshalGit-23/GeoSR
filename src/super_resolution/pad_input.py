import rasterio
import numpy as np
import math

src_path = "data/processed/input_rgbn.tif"
dst_path = "data/processed/input_rgbn_padded.tif"

with rasterio.open(src_path) as src:
    data = src.read().astype(np.float32)

    original_height = src.height
    original_width = src.width

    # Make dimensions multiples of 128
    padded_height = max(128, math.ceil(original_height / 128) * 128)
    padded_width = max(128, math.ceil(original_width / 128) * 128)

    pad_height = padded_height - original_height
    pad_width = padded_width - original_width

    # Extend edge pixels instead of adding zeros
    padded = np.pad(
        data,
        (
            (0, 0),
            (0, pad_height),
            (0, pad_width)
        ),
        mode="edge"
    )

    profile = src.profile.copy()

    profile.update(
        width=padded_width,
        height=padded_height,
        dtype="float32"
    )

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(padded)

print("Created:", dst_path)
print(f"Original: {original_width} x {original_height}")
print(f"Padded:   {padded_width} x {padded_height}")