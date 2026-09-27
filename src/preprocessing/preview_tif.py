"""Create an RGB PNG preview from the processed four-band GeoTIFF."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


INPUT_PATH = Path("data/processed/input_rgbn_small.tif")
OUTPUT_PATH = Path("data/processed/preview_small_rgb.png")
RGB_BAND_INDICES = (1, 2, 3)  # B04, B03, B02 (1-based raster band indices)
RGB_BAND_MAPPING = ((1, "Red", "B04"), (2, "Green", "B03"), (3, "Blue", "B02"))


def true_color_stretch(band: np.ndarray) -> np.ndarray:
    """Match preview.py: scale L2A reflectance values from 0–3000 to uint8."""
    values = band.astype(np.float32, copy=False)
    scaled = np.clip(values, 0, 3000) * (255.0 / 3000.0)
    return scaled.astype(np.uint8)


def create_preview(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    """Read B04/B03/B02 and save an RGB PNG using preview.py's true-color stretch."""
    input_path, output_path = Path(input_path), Path(output_path)
    print(f"Reading band indices {RGB_BAND_INDICES} from {input_path}")
    print("Mapping: band 1 -> Red (B04), band 2 -> Green (B03), band 3 -> Blue (B02)")
    print("Band 4 (B08/NIR) is not used for this true-color preview.")
    with rasterio.open(input_path) as source:
        if source.count < 3:
            raise ValueError(f"Expected at least 3 bands in {input_path}, found {source.count}")
        bands = [source.read(index) for index in RGB_BAND_INDICES]

    for (band_index, color, band_name), band in zip(RGB_BAND_MAPPING, bands):
        finite_values = band[np.isfinite(band)]
        if finite_values.size:
            minimum = float(finite_values.min())
            maximum = float(finite_values.max())
            mean = float(finite_values.mean())
            print(
                f"Band {band_index} ({band_name}, {color}) raw stats: "
                f"min={minimum:g}, max={maximum:g}, mean={mean:g}"
            )
        else:
            print(f"Band {band_index} ({band_name}, {color}) raw stats: no finite values")

    # Keep channel-last layout (height, width, RGB), as required by PIL.
    rgb = np.stack([true_color_stretch(band) for band in bands], axis=-1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    print(f"Saved RGB preview ({rgb.shape[1]}x{rgb.shape[0]}) to {output_path}")
    return output_path


if __name__ == "__main__":
    create_preview()
