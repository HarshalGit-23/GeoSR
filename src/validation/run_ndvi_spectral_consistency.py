"""Validate GeoSR NDVI spectral consistency against the original 10 m observation."""

from pathlib import Path

import numpy as np
import rasterio

try:
    from src.validation.run_ndvi_validation import NODATA_VALUE, write_difference_preview
except ModuleNotFoundError:  # Allows `python src/validation/run_ndvi_spectral_consistency.py`.
    from run_ndvi_validation import NODATA_VALUE, write_difference_preview


GEOSR_NDVI_PATH = Path("outputs/ndvi_geosr.tif")
ORIGINAL_NDVI_PATH = Path("outputs/ndvi_original.tif")
OUTPUT_DIRECTORY = Path("outputs")


def read_ndvi(path: Path) -> tuple[np.ndarray, dict, dict]:
    """Read an NDVI layer and represent nodata values as NaN for calculations."""
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            values[values == nodata] = np.nan
        return values, src.profile.copy(), src.tags()


def aggregate_to_grid(values: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Average valid high-resolution pixels into exactly aligned target-grid cells."""
    height, width = values.shape
    if height % target_height or width % target_width:
        raise ValueError("Source NDVI dimensions must be integer multiples of the target grid.")

    factor_y = height // target_height
    factor_x = width // target_width
    blocks = values.reshape(target_height, factor_y, target_width, factor_x)
    valid = np.isfinite(blocks)
    sums = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float32)
    counts = valid.sum(axis=(1, 3))
    aggregated = np.full((target_height, target_width), np.nan, dtype=np.float32)
    np.divide(sums, counts, out=aggregated, where=counts > 0)
    return aggregated


def write_ndvi_layer(values: np.ndarray, profile: dict, source_tags: dict[str, str], path: Path, product: str) -> None:
    """Write a float32 single-band NDVI product retaining grid metadata and tags."""
    data = np.where(np.isfinite(values), values, NODATA_VALUE).astype(np.float32)
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="float32", nodata=float(NODATA_VALUE), compress="deflate")
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data, 1)
        dst.update_tags(**source_tags)
        dst.update_tags(product=product, nodata_meaning="invalid NDVI")


def consistency_statistics(aggregated: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    """Measure aggregate GeoSR NDVI against the original Sentinel-2 observation."""
    valid = np.isfinite(aggregated) & np.isfinite(reference)
    aggregate_valid = aggregated[valid]
    reference_valid = reference[valid]
    difference = aggregate_valid - reference_valid
    if not difference.size:
        return {
            "valid_pixel_count": 0,
            "aggregated_geosr_mean_ndvi": float("nan"),
            "original_reference_mean_ndvi": float("nan"),
            "mean_absolute_difference": float("nan"),
            "rmse": float("nan"),
            "difference_min": float("nan"),
            "difference_max": float("nan"),
        }
    return {
        "valid_pixel_count": int(difference.size),
        "aggregated_geosr_mean_ndvi": float(aggregate_valid.mean()),
        "original_reference_mean_ndvi": float(reference_valid.mean()),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "difference_min": float(difference.min()),
        "difference_max": float(difference.max()),
    }


def run_spectral_consistency(
    geosr_ndvi_path: Path = GEOSR_NDVI_PATH,
    original_ndvi_path: Path = ORIGINAL_NDVI_PATH,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, float | int]:
    """Create 10 m aggregate and difference products, then return their statistics."""
    geosr_ndvi, geosr_profile, _ = read_ndvi(geosr_ndvi_path)
    original_ndvi, original_profile, original_tags = read_ndvi(original_ndvi_path)

    with rasterio.open(geosr_ndvi_path) as geosr, rasterio.open(original_ndvi_path) as original:
        if geosr.bounds != original.bounds or geosr.crs != original.crs:
            raise ValueError("GeoSR and original NDVI must have the same CRS and geographic extent.")

    aggregated = aggregate_to_grid(geosr_ndvi, original_ndvi.shape[0], original_ndvi.shape[1])
    difference = aggregated - original_ndvi
    difference[~np.isfinite(aggregated) | ~np.isfinite(original_ndvi)] = np.nan

    output_directory.mkdir(parents=True, exist_ok=True)
    write_ndvi_layer(
        aggregated,
        original_profile,
        original_tags,
        output_directory / "ndvi_geosr_aggregated_10m.tif",
        "GeoSR NDVI aggregated to original Sentinel-2 10 m grid",
    )
    write_ndvi_layer(
        difference,
        original_profile,
        original_tags,
        output_directory / "ndvi_consistency_difference.tif",
        "Aggregated GeoSR NDVI minus original Sentinel-2 NDVI reference observation",
    )
    write_difference_preview(difference, output_directory / "ndvi_consistency_difference.png")

    stats = consistency_statistics(aggregated, original_ndvi)
    print(f"GeoSR NDVI spectral consistency: {stats}")
    return stats


if __name__ == "__main__":
    run_spectral_consistency()
