import tempfile
import unittest
from pathlib import Path

import rasterio

from src.super_resolution.run_bicubic_baseline import create_bicubic_baseline


class BicubicBaselineTests(unittest.TestCase):
    def test_output_geometry_matches_four_times_input(self):
        input_path = Path("data/processed/input_rgbn.tif")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "bicubic.tif"
            preview_path = Path(temporary_directory) / "bicubic.png"
            create_bicubic_baseline(input_path, output_path, preview_path)

            with rasterio.open(input_path) as source, rasterio.open(output_path) as output:
                self.assertEqual(output.width, source.width * 4)
                self.assertEqual(output.height, source.height * 4)
                self.assertEqual(output.count, 4)
                self.assertEqual(output.dtypes, ("float32",) * 4)
                self.assertEqual(output.crs, source.crs)
                self.assertEqual(output.bounds, source.bounds)

            self.assertTrue(preview_path.exists())


if __name__ == "__main__":
    unittest.main()
