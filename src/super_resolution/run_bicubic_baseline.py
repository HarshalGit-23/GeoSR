"""Create a 4x bicubic GeoTIFF baseline for GeoSR comparison."""

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling


INPUT_PATH = Path("data/processed/input_rgbn.tif")
OUTPUT_PATH = Path("outputs/bicubic_2p5m.tif")
PREVIEW_PATH = Path("outputs/bicubic_rgb.png")
SCALE = 4


def write_preview(image: np.ndarray, path: Path) -> None:
    """Write a B04/B03/B02 preview with GeoSR's 2nd--98th percentile stretch."""
    rgb = np.moveaxis(image[:3], 0, -1)
    low, high = np.percentile(rgb, (2, 98))
    if high <= low:
        preview = np.zeros(rgb.shape, dtype=np.uint8)
    else:
        preview = np.clip((rgb - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="RGB").save(path)


def create_bicubic_baseline(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    preview_path: Path = PREVIEW_PATH,
) -> None:
    """Upsample all RGBN bands by 4x while retaining the input geographic extent."""
    with rasterio.open(input_path) as src:
        if src.count != 4:
            raise ValueError(f"Expected four RGBN bands, found {src.count}.")

        output_height = src.height * SCALE
        output_width = src.width * SCALE
        image = src.read(
            out_shape=(src.count, output_height, output_width),
            out_dtype="float32",
            resampling=Resampling.cubic,
        )
        profile = src.profile.copy()
        tags = src.tags()
        band_tags = {band: src.tags(band) for band in src.indexes}

    profile.update(
        count=4,
        width=output_width,
        height=output_height,
        dtype="float32",
        transform=profile["transform"] @ Affine.scale(1 / SCALE, 1 / SCALE),
        compress="deflate",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(image)
        dst.update_tags(**tags)
        for band, values in band_tags.items():
            dst.update_tags(band, **values)

    write_preview(image, preview_path)


if __name__ == "__main__":
    create_bicubic_baseline()
