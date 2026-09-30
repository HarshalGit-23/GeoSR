"""Create relative quality / agreement evidence for the GeoSR RGBN output.

This produces scene-relative agreement indicators, not calibrated confidence or
probabilities of correctness.  The 10 m evidence layers are replicated to the
2.5 m grid so each value remains tied to its original Sentinel-2 observation.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


INPUT_PATH = Path("data/processed/input_rgbn.tif")
GEOSR_PATH = Path("outputs/geosr_2p5m.tif")
BICUBIC_PATH = Path("outputs/bicubic_2p5m.tif")
NDVI_CONSISTENCY_PATH = Path("outputs/ndvi_consistency_difference.tif")
OUTPUT_DIRECTORY = Path("outputs")
NODATA_VALUE = np.float32(-9999.0)
LAPLACIAN_KERNEL_DESCRIPTION = "4-neighbour discrete Laplacian: [[0,1,0],[1,-4,1],[0,1,0]]"


def read_array(path: Path) -> tuple[np.ndarray, dict, dict]:
    """Read a raster as float32, changing its declared NoData values to NaN."""
    with rasterio.open(path) as src:
        values = src.read().astype(np.float32)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
        return values, src.profile.copy(), src.tags()


def aggregate_rgbn_to_grid(rgbn: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Average valid GeoSR pixels into aligned original-observation cells."""
    bands, height, width = rgbn.shape
    if bands != 4 or height % target_height or width % target_width:
        raise ValueError("Expected a four-band GeoSR grid that is an integer multiple of the target grid.")
    factor_y, factor_x = height // target_height, width // target_width
    blocks = rgbn.reshape(bands, target_height, factor_y, target_width, factor_x)
    valid = np.isfinite(blocks)
    sums = np.where(valid, blocks, 0.0).sum(axis=(2, 4), dtype=np.float32)
    counts = valid.sum(axis=(2, 4))
    result = np.full((bands, target_height, target_width), np.nan, dtype=np.float32)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def replicate_to_high_resolution(values: np.ndarray, factor_y: int, factor_x: int) -> np.ndarray:
    """Replicate observation-cell evidence without interpolating its error values."""
    return np.repeat(np.repeat(values, factor_y, axis=0), factor_x, axis=1)


def spectral_closure_error(geosr_rgbn: np.ndarray, original_rgbn: np.ndarray) -> np.ndarray:
    """Mean absolute RGBN reflectance closure error on the original 10 m grid."""
    aggregated = aggregate_rgbn_to_grid(geosr_rgbn, original_rgbn.shape[1], original_rgbn.shape[2])
    valid = np.isfinite(aggregated).all(axis=0) & np.isfinite(original_rgbn).all(axis=0)
    error = np.full(original_rgbn.shape[1:], np.nan, dtype=np.float32)
    error[valid] = np.mean(np.abs(aggregated[:, valid] - original_rgbn[:, valid]), axis=0)
    return error


def detail_deviation(geosr_rgbn: np.ndarray, bicubic_rgbn: np.ndarray) -> np.ndarray:
    """RMS RGBN departure of GeoSR from the conservative bicubic baseline."""
    valid = np.isfinite(geosr_rgbn).all(axis=0) & np.isfinite(bicubic_rgbn).all(axis=0)
    result = np.full(geosr_rgbn.shape[1:], np.nan, dtype=np.float32)
    difference = geosr_rgbn - bicubic_rgbn
    result[valid] = np.sqrt(np.mean(np.square(difference[:, valid]), axis=0))
    return result


def high_frequency_residual(geosr_rgbn: np.ndarray, bicubic_rgbn: np.ndarray) -> np.ndarray:
    """RMS Laplacian of GeoSR-minus-bicubic residual, with no smoothing applied.

    The 4-neighbour kernel is [[0,1,0],[1,-4,1],[0,1,0]].  Outer-edge pixels
    are left invalid because a full kernel neighbourhood is unavailable.
    """
    residual = geosr_rgbn - bicubic_rgbn
    result = np.full(residual.shape[1:], np.nan, dtype=np.float32)
    center = residual[:, 1:-1, 1:-1]
    laplacian = (
        residual[:, :-2, 1:-1]
        + residual[:, 2:, 1:-1]
        + residual[:, 1:-1, :-2]
        + residual[:, 1:-1, 2:]
        - 4.0 * center
    )
    valid = np.isfinite(laplacian).all(axis=0)
    interior = result[1:-1, 1:-1]
    interior[valid] = np.sqrt(np.mean(np.square(laplacian[:, valid]), axis=0))
    result[1:-1, 1:-1] = interior
    return result


def empirical_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return an empirical CDF rank in (0, 1] for each finite scene value.

    Ties receive the same right-inclusive empirical rank. Invalid values remain
    NaN, so they can never contribute to relative risk or quality.
    """
    ranks = np.full(values.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(values)
    observations = values[valid]
    if not observations.size:
        return ranks
    unique, inverse, counts = np.unique(observations, return_inverse=True, return_counts=True)
    del unique
    cumulative = np.cumsum(counts, dtype=np.float64) / observations.size
    ranks[valid] = cumulative[inverse].astype(np.float32)
    return ranks


def relative_quality_from_evidence(evidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute worst-signal relative risk and the clipped Relative Quality Index."""
    if evidence.shape[0] != 4:
        raise ValueError("Expected exactly four relative quality / agreement evidence layers.")
    percentile_layers = np.stack([empirical_percentile_rank(layer) for layer in evidence])
    valid = np.isfinite(percentile_layers).all(axis=0)
    risk = np.full(evidence.shape[1:], np.nan, dtype=np.float32)
    risk[valid] = np.max(percentile_layers[:, valid], axis=0)
    quality = np.full(risk.shape, np.nan, dtype=np.float32)
    quality[valid] = np.clip(1.0 - risk[valid], 0.0, 1.0)
    return risk, quality


def write_raster(data: np.ndarray, profile: dict, source_tags: dict[str, str], path: Path, descriptions: list[str]) -> None:
    """Write float32 evidence with invalid pixels represented by a declared NoData value."""
    output = np.where(np.isfinite(data), data, NODATA_VALUE).astype(np.float32)
    output_profile = profile.copy()
    output_profile.update(count=output.shape[0], dtype="float32", nodata=float(NODATA_VALUE), compress="deflate")
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(output)
        dst.update_tags(**source_tags)
        for band, description in enumerate(descriptions, start=1):
            dst.set_band_description(band, description)
            dst.update_tags(band, evidence_description=description)


def write_quality_preview(quality: np.ndarray, path: Path) -> None:
    """Write a static red-to-green Relative Quality Index diagnostic preview."""
    values = np.nan_to_num(quality, nan=0.0)
    values = np.clip(values, 0.0, 1.0)
    red = (255 * (1.0 - values)).astype(np.uint8)
    green = (255 * values).astype(np.uint8)
    blue = (60 * (1.0 - np.abs(2.0 * values - 1.0))).astype(np.uint8)
    Image.fromarray(np.dstack((red, green, blue)), mode="RGB").save(path)


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize finite values without changing their distribution."""
    valid = values[np.isfinite(values)]
    if not valid.size:
        return {"valid_pixel_count": 0, "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "valid_pixel_count": int(valid.size),
        "median": float(np.median(valid)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
    }


def run_quality_evidence(
    input_path: Path = INPUT_PATH,
    geosr_path: Path = GEOSR_PATH,
    bicubic_path: Path = BICUBIC_PATH,
    ndvi_consistency_path: Path = NDVI_CONSISTENCY_PATH,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict:
    """Create relative quality / agreement evidence and its scene-relative index."""
    original_rgbn, original_profile, _ = read_array(input_path)
    geosr_rgbn, geosr_profile, geosr_tags = read_array(geosr_path)
    bicubic_rgbn, bicubic_profile, _ = read_array(bicubic_path)
    ndvi_difference, ndvi_profile, _ = read_array(ndvi_consistency_path)

    if original_rgbn.shape[0] != 4 or geosr_rgbn.shape[0] != 4 or bicubic_rgbn.shape[0] != 4:
        raise ValueError("Input, GeoSR, and bicubic rasters must each contain B04/B03/B02/B08.")
    if geosr_rgbn.shape != bicubic_rgbn.shape or geosr_profile["transform"] != bicubic_profile["transform"]:
        raise ValueError("GeoSR and bicubic RGBN rasters must share a common 2.5 m grid.")
    if geosr_profile["crs"] != original_profile["crs"] or geosr_profile["crs"] != ndvi_profile["crs"]:
        raise ValueError("All quality inputs must share a CRS.")
    if geosr_rgbn.shape[1] != original_rgbn.shape[1] * 4 or geosr_rgbn.shape[2] != original_rgbn.shape[2] * 4:
        raise ValueError("GeoSR dimensions must be exactly four times the original observation grid.")
    if ndvi_difference.shape != (1, original_rgbn.shape[1], original_rgbn.shape[2]):
        raise ValueError("NDVI consistency difference must use the original 10 m grid.")

    spectral_10m = spectral_closure_error(geosr_rgbn, original_rgbn)
    ndvi_10m = np.abs(ndvi_difference[0])
    spectral_2p5m = replicate_to_high_resolution(spectral_10m, 4, 4)
    ndvi_2p5m = replicate_to_high_resolution(ndvi_10m, 4, 4)
    detail_2p5m = detail_deviation(geosr_rgbn, bicubic_rgbn)
    high_frequency_2p5m = high_frequency_residual(geosr_rgbn, bicubic_rgbn)
    evidence = np.stack((spectral_2p5m, ndvi_2p5m, detail_2p5m, high_frequency_2p5m))
    risk, quality = relative_quality_from_evidence(evidence)

    output_directory.mkdir(parents=True, exist_ok=True)
    write_raster(
        evidence,
        geosr_profile,
        geosr_tags,
        output_directory / "geosr_quality_evidence.tif",
        ["spectral_closure_error", "ndvi_closure_error", "detail_deviation", "high_frequency_residual"],
    )
    write_raster(
        quality[None, :, :],
        geosr_profile,
        geosr_tags,
        output_directory / "geosr_relative_quality.tif",
        ["Relative Quality Index (scene-relative agreement evidence; not calibrated confidence)"],
    )
    write_quality_preview(quality, output_directory / "geosr_relative_quality.png")

    valid_risk = risk[np.isfinite(risk)]
    worst_decile_fraction = float(np.mean(valid_risk >= np.percentile(valid_risk, 90))) if valid_risk.size else float("nan")
    summary = {
        "terminology": "Relative quality / agreement evidence; not calibrated confidence or probability of correctness.",
        "spectral_closure_error": distribution_summary(evidence[0]),
        "ndvi_closure_error": distribution_summary(evidence[1]),
        "detail_deviation": distribution_summary(evidence[2]),
        "high_frequency_residual": distribution_summary(evidence[3]),
        "relative_quality_index": {
            **distribution_summary(quality),
            "minimum": float(np.nanmin(quality)),
            "maximum": float(np.nanmax(quality)),
            "mean": float(np.nanmean(quality)),
        },
        "worst_relative_risk_decile_fraction": worst_decile_fraction,
        "laplacian_kernel": LAPLACIAN_KERNEL_DESCRIPTION,
        "risk_logic": "maximum of the four right-inclusive empirical percentile ranks",
    }
    with open(output_directory / "geosr_quality_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_quality_evidence()
