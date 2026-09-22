"""Create NDVI products that compare GeoSR and bicubic RGBN outputs."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


INPUT_PATH = Path("data/processed/input_rgbn.tif")
GEOSR_PATH = Path("outputs/geosr_2p5m.tif")
BICUBIC_PATH = Path("outputs/bicubic_2p5m.tif")
OUTPUT_DIRECTORY = Path("outputs")
NODATA_VALUE = np.float32(-9999.0)
DENOMINATOR_EPSILON = 1e-6


def calculate_ndvi(rgbn: np.ndarray) -> np.ndarray:
    """Calculate NDVI from B04/B03/B02/B08 data, returning NaN when invalid."""
    if rgbn.ndim != 3 or rgbn.shape[0] != 4:
        raise ValueError(f"Expected RGBN data shaped (4, H, W), got {rgbn.shape}.")

    red = rgbn[0].astype(np.float32, copy=False)
    nir = rgbn[3].astype(np.float32, copy=False)
    denominator = nir + red
    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & (np.abs(denominator) > DENOMINATOR_EPSILON)
    )

    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    np.divide(nir - red, denominator, out=ndvi, where=valid)
    return ndvi


def write_ndvi_tif(
    ndvi: np.ndarray,
    source_profile: dict,
    source_tags: dict[str, str],
    path: Path,
) -> None:
    """Write a single-band float32 NDVI GeoTIFF with source georeferencing."""
    data = np.where(np.isfinite(ndvi), ndvi, NODATA_VALUE).astype(np.float32)
    profile = source_profile.copy()
    profile.update(count=1, dtype="float32", nodata=float(NODATA_VALUE), compress="deflate")

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(**source_tags)
        dst.update_tags(
            product="NDVI",
            formula="(B08-B04)/(B08+B04)",
            source_band_order="B04,B03,B02,B08",
        )


def write_ndvi_preview(ndvi: np.ndarray, path: Path) -> None:
    """Create an NDVI diagnostic preview: brown (-1), white (0), green (+1)."""
    values = np.nan_to_num(ndvi, nan=0.0)
    normalized = np.clip((values + 1.0) / 2.0, 0.0, 1.0)
    red = (255 * (1.0 - normalized)).astype(np.uint8)
    green = (255 * normalized).astype(np.uint8)
    blue = (80 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
    Image.fromarray(np.dstack((red, green, blue)), mode="RGB").save(path)


def write_difference_preview(difference: np.ndarray, path: Path) -> None:
    """Create a zero-centred blue/white/red difference diagnostic preview."""
    valid = difference[np.isfinite(difference)]
    limit = float(np.percentile(np.abs(valid), 98)) if valid.size else 1.0
    limit = max(limit, 1e-6)
    scaled = np.clip(difference / limit, -1.0, 1.0)
    red = (255 * np.maximum(scaled, 0.0)).astype(np.uint8)
    blue = (255 * np.maximum(-scaled, 0.0)).astype(np.uint8)
    green = (255 * (1.0 - np.abs(scaled))).astype(np.uint8)
    Image.fromarray(np.dstack((red, green, blue)), mode="RGB").save(path)


def summary(values: np.ndarray) -> dict[str, float | int]:
    """Return basic statistics using only finite pixels."""
    valid = values[np.isfinite(values)]
    if not valid.size:
        return {"valid_pixel_count": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "valid_pixel_count": int(valid.size),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
    }


def difference_summary(difference: np.ndarray) -> dict[str, float | int]:
    """Return comparison statistics for a common-grid NDVI difference."""
    valid = difference[np.isfinite(difference)]
    if not valid.size:
        return {"valid_pixel_count": 0, "mean_absolute_difference": float("nan"), "rmse": float("nan")}
    return {
        "valid_pixel_count": int(valid.size),
        "mean_absolute_difference": float(np.mean(np.abs(valid))),
        "rmse": float(np.sqrt(np.mean(np.square(valid)))),
    }


def run_ndvi_validation(
    input_path: Path = INPUT_PATH,
    geosr_path: Path = GEOSR_PATH,
    bicubic_path: Path = BICUBIC_PATH,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, dict[str, float | int]]:
    """Write NDVI products and compare GeoSR with bicubic on the 2.5 m grid."""
    output_directory.mkdir(parents=True, exist_ok=True)
    with rasterio.open(input_path) as source, rasterio.open(geosr_path) as geosr, rasterio.open(bicubic_path) as bicubic:
        original_rgbn = source.read().astype(np.float32)
        geosr_rgbn = geosr.read().astype(np.float32)
        bicubic_rgbn = bicubic.read().astype(np.float32)
        source_profile = source.profile.copy()
        geosr_profile = geosr.profile.copy()
        bicubic_profile = bicubic.profile.copy()
        source_tags = source.tags()
        geosr_tags = geosr.tags()
        bicubic_tags = bicubic.tags()

        if geosr.shape != bicubic.shape or geosr.transform != bicubic.transform or geosr.crs != bicubic.crs:
            raise ValueError("GeoSR and bicubic outputs must share the same 2.5 m grid.")

    original_ndvi = calculate_ndvi(original_rgbn)
    geosr_ndvi = calculate_ndvi(geosr_rgbn)
    bicubic_ndvi = calculate_ndvi(bicubic_rgbn)
    difference = geosr_ndvi - bicubic_ndvi
    difference[~np.isfinite(geosr_ndvi) | ~np.isfinite(bicubic_ndvi)] = np.nan

    write_ndvi_tif(original_ndvi, source_profile, source_tags, output_directory / "ndvi_original.tif")
    write_ndvi_tif(geosr_ndvi, geosr_profile, geosr_tags, output_directory / "ndvi_geosr.tif")
    write_ndvi_tif(bicubic_ndvi, bicubic_profile, bicubic_tags, output_directory / "ndvi_bicubic.tif")
    write_ndvi_tif(difference, geosr_profile, geosr_tags, output_directory / "ndvi_difference_geosr.tif")
    write_ndvi_preview(geosr_ndvi, output_directory / "ndvi_geosr.png")
    write_difference_preview(difference, output_directory / "ndvi_difference_geosr.png")

    stats = {
        "original_sentinel2_ndvi": summary(original_ndvi),
        "geosr_ndvi": summary(geosr_ndvi),
        "bicubic_resampled_ndvi": summary(bicubic_ndvi),
        "geosr_minus_bicubic_ndvi": difference_summary(difference),
    }
    for name, values in stats.items():
        print(f"{name}: {values}")
    return stats


if __name__ == "__main__":
    run_ndvi_validation()
