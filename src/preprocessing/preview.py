"""Create a quick, downsampled RGB preview of a Sentinel-2 tile."""

from pathlib import Path
import argparse

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling


TILE_DIR = Path("data/raw/tile1")
OUTPUT_PATH = Path("data/processed/preview_rgb.png")
DATE_PREFIX = "T43QGD_20260922T051649"
RGB_BANDS = ("B04", "B03", "B02")  # Red, green, blue
MAX_PREVIEW_SIZE = 1000


def true_color_stretch(band: np.ndarray) -> np.ndarray:
    """Scale Sentinel-2 L2A reflectance values from 0–3000 to uint8."""
    values = band.astype(np.float32, copy=False)
    scaled = np.clip(values, 0, 3000) * (255.0 / 3000.0)
    return scaled.astype(np.uint8)


def create_preview(
    tile_dir: str | Path = TILE_DIR,
    output_path: str | Path = OUTPUT_PATH,
    date_prefix: str = DATE_PREFIX,
    max_size: int = MAX_PREVIEW_SIZE,
) -> Path:
    """Read and stretch B04/B03/B02 and save a downsampled RGB PNG."""
    tile_dir, output_path = Path(tile_dir), Path(output_path)
    first_path = tile_dir / f"{date_prefix}_{RGB_BANDS[0]}_10m.jp2"
    with rasterio.open(first_path) as src:
        scale = min(1.0, max_size / max(src.width, src.height))
        out_height = max(1, round(src.height * scale))
        out_width = max(1, round(src.width * scale))
        bands = [src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
        )]

    for band_name in RGB_BANDS[1:]:
        path = tile_dir / f"{date_prefix}_{band_name}_10m.jp2"
        with rasterio.open(path) as src:
            bands.append(src.read(
                1,
                out_shape=(out_height, out_width),
                resampling=Resampling.average,
            ))

    for band_name, band in zip(RGB_BANDS, bands):
        raw = band.astype(np.float64, copy=False)
        minimum, percentile_2, percentile_50, percentile_98, maximum = np.percentile(
            raw, (0, 2, 50, 98, 100)
        )
        mean = float(np.mean(raw))
        print(
            f"{band_name} raw stats: min={minimum:g}, max={maximum:g}, "
            f"mean={mean:g}, p2={percentile_2:g}, p50={percentile_50:g}, "
            f"p98={percentile_98:g}"
        )

    rgb = np.stack([true_color_stretch(band) for band in bands], axis=-1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    print(f"Saved RGB preview ({out_width}x{out_height}) to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a downsampled Sentinel-2 RGB preview.")
    parser.add_argument("tile_dir", nargs="?", default=str(TILE_DIR), help="Directory containing the JP2 bands")
    parser.add_argument("date_prefix", nargs="?", default=DATE_PREFIX, help="Shared filename prefix for the bands")
    args = parser.parse_args()
    create_preview(tile_dir=args.tile_dir, date_prefix=args.date_prefix)
