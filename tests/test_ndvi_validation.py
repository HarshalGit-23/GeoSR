import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

from src.validation.run_ndvi_validation import calculate_ndvi, run_ndvi_validation


class NdviCalculationTests(unittest.TestCase):
    def test_calculate_ndvi_uses_red_band_one_and_nir_band_four(self):
        rgbn = np.array([[[0.2, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.6, 0.0]]], dtype=np.float32)
        ndvi = calculate_ndvi(rgbn)
        np.testing.assert_allclose(ndvi[0, 0], 0.5, atol=1e-6)
        self.assertTrue(np.isnan(ndvi[0, 1]))

    def test_outputs_preserve_grid_metadata(self):
        input_path = Path("data/processed/input_rgbn.tif")
        geosr_path = Path("outputs/geosr_2p5m.tif")
        bicubic_path = Path("outputs/bicubic_2p5m.tif")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            run_ndvi_validation(input_path, geosr_path, bicubic_path, output_directory)

            with rasterio.open(input_path) as source, rasterio.open(output_directory / "ndvi_original.tif") as original:
                self.assertEqual(original.count, 1)
                self.assertEqual(original.dtypes, ("float32",))
                self.assertEqual(original.crs, source.crs)
                self.assertEqual(original.bounds, source.bounds)

            with rasterio.open(geosr_path) as reference:
                for name in ("ndvi_geosr.tif", "ndvi_bicubic.tif", "ndvi_difference_geosr.tif"):
                    with rasterio.open(output_directory / name) as output:
                        self.assertEqual(output.width, reference.width)
                        self.assertEqual(output.height, reference.height)
                        self.assertEqual(output.crs, reference.crs)
                        self.assertEqual(output.bounds, reference.bounds)


if __name__ == "__main__":
    unittest.main()
