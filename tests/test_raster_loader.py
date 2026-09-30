"""Small loader tests using a temporary raster."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from app.raster_loader import RasterLoadError, load_raster


class RasterLoaderTests(unittest.TestCase):
    def test_reads_metadata_and_masks_nodata_and_non_finite(self):
        values = np.array([[[1.0, -999.0], [np.nan, np.inf]]], dtype="float32")
        with tempfile.TemporaryDirectory() as folder:
            raster_path = Path(folder) / "temporary.tif"
            with rasterio.open(
                raster_path, "w", driver="GTiff", height=2, width=2, count=1,
                dtype="float32", nodata=-999.0, crs="EPSG:4326",
                transform=Affine(1, 0, 0, 0, -1, 2),
            ) as dataset:
                dataset.write(values)
            data, metadata = load_raster(raster_path)
        self.assertEqual(data.shape, (1, 2, 2))
        self.assertEqual(metadata["bands"], 1)
        self.assertEqual(metadata["crs"], "EPSG:4326")
        mask = np.ma.getmaskarray(data)
        self.assertTrue(mask[0, 0, 1])
        self.assertTrue(mask[0, 1, 0])
        self.assertTrue(mask[0, 1, 1])

    def test_missing_path_has_clear_error(self):
        with self.assertRaisesRegex(RasterLoadError, "does not exist"):
            load_raster("a-file-that-does-not-exist.tif")


if __name__ == "__main__":
    unittest.main()
