"""Run reliable 4x SEN2SRLite RGBN inference on a Sentinel-2 GeoTIFF.

The input band order is B04, B03, B02, B08 (RGBN), with reflectance values
stored as float32 values near [0, 1]. The output is model-derived, not a
ground-truth 2.5 m satellite observation.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from huggingface_hub import snapshot_download
from PIL import Image

INPUT_PATH = Path("data/processed/input_rgbn.tif")
OUTPUT_PATH = Path("outputs/geosr_2p5m.tif")
PREVIEW_PATH = Path("outputs/geosr_rgb.png")
TILE_SIZE = 128
OVERLAP = 32
SCALE = 4


def load_model():
    """Load the WEO-SAS SEN2SRLite RGBN x4 public model interface."""
    model_dir = snapshot_download("WEO-SAS/sen2sr")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from model import Model
    return Model(local_dir=model_dir)


def tile_starts(length: int, tile_size: int = TILE_SIZE, overlap: int = OVERLAP):
    """Return full-tile starts, with the final tile aligned to the image edge."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def blend_window(size: int = TILE_SIZE) -> np.ndarray:
    """Use non-zero Hann weights so overlapping predictions blend smoothly."""
    one_dimension = np.maximum(np.hanning(size).astype(np.float32), 0.01)
    return np.outer(one_dimension, one_dimension).astype(np.float32)


def super_resolve(image: np.ndarray, model) -> np.ndarray:
    """Run valid 128x128 model windows and blend them into an exact 4x image."""
    if image.ndim != 3 or image.shape[0] != 4:
        raise ValueError(f"Expected RGBN input shaped (4, H, W), got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Input contains NaN or infinite reflectance values.")

    _, height, width = image.shape
    padded_height = max(height, TILE_SIZE)
    padded_width = max(width, TILE_SIZE)
    padded = np.pad(
        image,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    ).astype(np.float32, copy=False)

    starts_y = tile_starts(padded_height)
    starts_x = tile_starts(padded_width)
    output = np.zeros((4, padded_height * SCALE, padded_width * SCALE), dtype=np.float32)
    weights = np.zeros((padded_height * SCALE, padded_width * SCALE), dtype=np.float32)
    window = blend_window(TILE_SIZE * SCALE)

    print(f"Inference windows: {len(starts_x)} x {len(starts_y)} (all 128x128)")
    for y0 in starts_y:
        for x0 in starts_x:
            tile = padded[:, y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
            if tile.shape != (4, TILE_SIZE, TILE_SIZE):
                raise RuntimeError(f"Invalid inference tile shape: {tile.shape}")
            prediction = np.asarray(model.predict(tile), dtype=np.float32)
            expected_shape = (4, TILE_SIZE * SCALE, TILE_SIZE * SCALE)
            if prediction.shape != expected_shape:
                raise RuntimeError(f"Model returned {prediction.shape}; expected {expected_shape}.")

            y1, x1 = (y0 + TILE_SIZE) * SCALE, (x0 + TILE_SIZE) * SCALE
            output[:, y0 * SCALE:y1, x0 * SCALE:x1] += prediction * window
            weights[y0 * SCALE:y1, x0 * SCALE:x1] += window

    output /= weights[None, :, :]
    return output[:, :height * SCALE, :width * SCALE]


def write_preview(sr: np.ndarray, path: Path) -> None:
    """Write a diagnostic B04/B03/B02 RGB PNG without altering the GeoTIFF."""
    rgb = np.moveaxis(sr[:3], 0, -1)
    low, high = np.percentile(rgb, (2, 98))
    if high <= low:
        preview = np.zeros(rgb.shape, dtype=np.uint8)
    else:
        preview = np.clip((rgb - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="RGB").save(path)


def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> None:
    print("Loading model...")
    model = load_model()
    with rasterio.open(input_path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile.copy()
        tags = src.tags()
        band_tags = {band: src.tags(band) for band in src.indexes}
        original_bounds = src.bounds

    print(f"Input: {image.shape[2]} x {image.shape[1]}, range [{image.min():.5f}, {image.max():.5f}]")
    if image.min() < -0.01 or image.max() > 1.1:
        print("Warning: input is outside the expected reflectance range near [0, 1].")
    sr = super_resolve(image, model)
    output_height, output_width = sr.shape[1:]
    profile.update(
        width=output_width, height=output_height, count=4, dtype="float32",
        transform=profile["transform"] * Affine.scale(1 / SCALE, 1 / SCALE),
        compress="deflate",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(sr)
        dst.update_tags(**tags)
        for band, values in band_tags.items():
            dst.update_tags(band, **values)
    write_preview(sr, PREVIEW_PATH)

    with rasterio.open(output_path) as dst:
        print(f"Output: {output_path} ({dst.width} x {dst.height}, {dst.crs})")
        print(f"Bounds: {dst.bounds} (input: {original_bounds})")
        print(f"Output range: [{sr.min():.5f}, {sr.max():.5f}]")
    print(f"RGB preview: {PREVIEW_PATH}")


if __name__ == "__main__":
    run()
