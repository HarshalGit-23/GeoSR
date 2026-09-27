"""Mask Sentinel-2 cloud classes and fill them with per-band medians."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject


INVALID_SCL_CLASSES = (3, 8, 9, 10, 11)
DEFAULT_TILE_DIR = Path("data/raw/tile1")
DEFAULT_OUTPUT_DIR = Path("data/processed")
# Model input order: red, green, blue, near-infrared.
RGBN_BANDS = ("B04", "B03", "B02", "B08")
DEFAULT_RGBN_INPUT = DEFAULT_OUTPUT_DIR / "input_rgbn.tif"
DEFAULT_RGBN_SMALL_OUTPUT = DEFAULT_OUTPUT_DIR / "input_rgbn_small.tif"


def crop_existing_rgbn(
    input_path: str | Path = DEFAULT_RGBN_INPUT,
    output_path: str | Path = DEFAULT_RGBN_SMALL_OUTPUT,
    size: int = 1024,
) -> Path:
    """Write a centered crop of an existing RGBN GeoTIFF, preserving its grid."""
    input_path, output_path = Path(input_path), Path(output_path)
    if size <= 0:
        raise ValueError("size must be a positive integer")

    with rasterio.open(input_path) as source:
        if source.width < size or source.height < size:
            raise ValueError(
                f"Input raster ({source.width}x{source.height}) is smaller than "
                f"the requested {size}x{size} crop"
            )
        col_off = (source.width - size) // 2
        row_off = (source.height - size) // 2
        window = Window(col_off, row_off, size, size)
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            width=size,
            height=size,
            transform=rasterio.windows.transform(window, source.transform),
            compress="LZW",
            predictor=2,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(source.read(window=window))
            for band_index, description in enumerate(source.descriptions, start=1):
                if description:
                    destination.set_band_description(band_index, description)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.2f} MB)")
    return output_path


def fill_invalid_pixels(
    rgbn: np.ndarray, scl: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Fill RGBN pixels marked invalid by an already aligned SCL array.

    Args:
        rgbn: Four-band array shaped ``(4, height, width)``.
        scl: SCL classes shaped ``(height, width)`` on the same grid.

    Returns the filled array (float32), the boolean invalid mask, and the
    median used for each band. A ValueError is raised if a band has no valid,
    finite pixels from which to calculate a median.
    """
    rgbn = np.asarray(rgbn)
    scl = np.asarray(scl)
    if rgbn.ndim != 3 or rgbn.shape[0] != 4:
        raise ValueError("rgbn must have shape (4, height, width)")
    if scl.shape != rgbn.shape[1:]:
        raise ValueError("scl must have the same height and width as rgbn")

    invalid = np.isin(scl, INVALID_SCL_CLASSES)
    filled = rgbn.astype(np.float32, copy=True)
    medians: list[float] = []
    for band_index in range(4):
        band = filled[band_index]
        valid_values = band[~invalid & np.isfinite(band)]
        if valid_values.size == 0:
            raise ValueError(f"Band {band_index + 1} has no valid finite pixels")
        median = float(np.median(valid_values))
        medians.append(median)
        band[invalid] = median

    return filled, invalid, tuple(medians)


def _resample_scl(
    scl: np.ndarray,
    scl_transform: rasterio.Affine,
    scl_crs: rasterio.crs.CRS,
    out_shape: tuple[int, int],
    out_transform: rasterio.Affine,
    out_crs: rasterio.crs.CRS,
) -> np.ndarray:
    """Warp categorical SCL labels onto the target grid using nearest."""
    destination = np.zeros(out_shape, dtype=scl.dtype)
    reproject(
        source=scl,
        destination=destination,
        src_transform=scl_transform,
        src_crs=scl_crs,
        dst_transform=out_transform,
        dst_crs=out_crs,
        resampling=Resampling.nearest,
    )
    return destination


def process_scene(
    tile_dir: str | Path = DEFAULT_TILE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_prefix: str = "T43QGD_20260922T051649",
    crop_existing: bool = False,
    input_rgbn_path: str | Path = DEFAULT_RGBN_INPUT,
) -> tuple[Path, Path] | Path:
    """Read the configured Sentinel-2 tile, process it, and write GeoTIFFs.

    Outputs ``rgbn_filled.tif`` (four bands in B04/B03/B02/B08 order) and
    ``cloud_mask.tif`` (1 for filled pixels, 0 for valid pixels).
    """
    if crop_existing:
        return crop_existing_rgbn(
            input_rgbn_path, Path(output_dir) / "input_rgbn_small.tif"
        )

    tile_dir, output_dir = Path(tile_dir), Path(output_dir)
    band_paths = [tile_dir / f"{date_prefix}_{band}_10m.jp2" for band in RGBN_BANDS]
    scl_path = tile_dir / f"{date_prefix}_SCL_20m.jp2"

    print("Reading bands...")
    with rasterio.open(band_paths[0]) as first:
        target_profile = first.profile.copy()
        target_shape = (first.height, first.width)
        target_transform, target_crs = first.transform, first.crs
    bands = []
    for path in band_paths:
        with rasterio.open(path) as dataset:
            same_shape = (dataset.height, dataset.width) == target_shape
            same_grid = dataset.transform == target_transform and dataset.crs == target_crs
            if not same_shape or not same_grid:
                raise ValueError(f"RGBN band grid does not match {band_paths[0]}: {path}")
            bands.append(dataset.read(1))

    print("Resampling SCL...")
    with rasterio.open(scl_path) as dataset:
        scl_aligned = _resample_scl(
            dataset.read(1), dataset.transform, dataset.crs,
            target_shape, target_transform, target_crs,
        )

    print("Filling masked pixels...")
    filled, invalid, medians = fill_invalid_pixels(np.stack(bands), scl_aligned)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgbn_path = output_dir / "rgbn_filled.tif"
    mask_path = output_dir / "cloud_mask.tif"

    target_profile.update(driver="GTiff", count=4, dtype="float32", compress="deflate")
    print("Writing rgbn_filled.tif...")
    with rasterio.open(rgbn_path, "w", **target_profile) as dataset:
        dataset.write(filled)
        for index, band_name in enumerate(RGBN_BANDS, start=1):
            dataset.set_band_description(index, band_name)

    mask_profile = target_profile.copy()
    mask_profile.update(count=1, dtype="uint8", nodata=None)
    print("Writing cloud_mask.tif...")
    with rasterio.open(mask_path, "w", **mask_profile) as dataset:
        dataset.write(invalid.astype(np.uint8), 1)
        dataset.set_band_description(1, "cloud_mask (1=filled, 0=valid)")

    total = invalid.size
    masked = int(invalid.sum())
    print(f"Total pixels: {total}")
    print(f"Masked: {masked} ({100.0 * masked / total:.2f}%)")
    median_summary = ", ".join(
        f"{name}={value:g}" for name, value in zip(RGBN_BANDS, medians)
    )
    print(f"Median fill values: {median_summary}")
    print("Done.")
    return rgbn_path, mask_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop-existing",
        action="store_true",
        help="crop the existing processed input_rgbn.tif instead of processing raw bands",
    )
    args = parser.parse_args()
    process_scene(crop_existing=args.crop_existing)
