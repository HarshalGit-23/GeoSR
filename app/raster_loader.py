"""Small reusable Rasterio GeoTIFF reader."""
from pathlib import Path
import numpy as np
import rasterio


class RasterLoadError(Exception):
    """A raster is missing or could not be read."""


def load_raster(path: str | Path):
    """Read raster bands and metadata, masking NoData, NaN, and infinity."""
    raster_path = Path(path)
    if not raster_path.is_file():
        raise RasterLoadError(f"Raster file does not exist: {raster_path}")
    try:
        with rasterio.open(raster_path) as dataset:
            data = np.ma.masked_invalid(dataset.read(masked=True).astype(float))
            metadata = {
                "filename": raster_path.name,
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count,
                "crs": str(dataset.crs) if dataset.crs else "Not defined",
                "resolution": tuple(dataset.res) if dataset.res else None,
                "bounds": tuple(dataset.bounds) if dataset.bounds else None,
            }
        return data, metadata
    except (rasterio.errors.RasterioError, OSError, ValueError) as error:
        raise RasterLoadError(f"Could not read raster {raster_path.name}: {error}") from error
