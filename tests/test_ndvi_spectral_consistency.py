import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

from src.validation.run_ndvi_spectral_consistency import (
    aggregate_to_grid,
    consistency_statistics,
    run_spectral_consistency,
)


class NdviSpectralConsistencyTests(unittest.TestCase):
    def test_aggregate_to_grid_averages_each_aligned_block(self):
        values = np.arange(16, dtype=np.float32).reshape(4, 4)
        aggregated = aggregate_to_grid(values, 2, 2)
        np.testing.assert_allclose(aggregated, [[2.5, 4.5], [10.5, 12.5]])

    def test_statistics_compare_aggregate_with_reference(self):
        aggregated = np.array([[0.3, 0.5]], dtype=np.float32)
        reference = np.array([[0.2, 0.4]], dtype=np.float32)
        stats = consistency_statistics(aggregated, reference)
        self.assertEqual(stats["valid_pixel_count"], 2)
        self.assertAlmostEqual(stats["mean_absolute_difference"], 0.1, places=6)
        self.assertAlmostEqual(stats["rmse"], 0.1, places=6)

    def test_outputs_use_original_ten_meter_grid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            run_spectral_consistency(output_directory=output_directory)
            with rasterio.open("outputs/ndvi_original.tif") as original:
                for name in ("ndvi_geosr_aggregated_10m.tif", "ndvi_consistency_difference.tif"):
                    with rasterio.open(output_directory / name) as output:
                        self.assertEqual((output.width, output.height), (original.width, original.height))
                        self.assertEqual(output.count, 1)
                        self.assertEqual(output.dtypes, ("float32",))
                        self.assertEqual(output.crs, original.crs)
                        self.assertEqual(output.bounds, original.bounds)


if __name__ == "__main__":
    unittest.main()
