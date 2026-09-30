import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

from src.validation.run_quality_evidence import (
    empirical_percentile_rank,
    relative_quality_from_evidence,
    run_quality_evidence,
)


class QualityEvidenceTests(unittest.TestCase):
    def test_percentile_rank_and_worst_signal_risk(self):
        values = np.array([[1.0, 2.0, 2.0, np.nan]], dtype=np.float32)
        np.testing.assert_allclose(
    empirical_percentile_rank(values)[0, :3],
    [1 / 3, 1.0, 1.0],
)
        evidence = np.stack([np.array([[0.1, 0.9]], dtype=np.float32)] * 4)
        risk, quality = relative_quality_from_evidence(evidence)
        np.testing.assert_allclose(risk, [[0.5, 1.0]])
        np.testing.assert_allclose(quality, [[0.5, 0.0]])

    def test_invalid_evidence_stays_invalid(self):
        evidence = np.ones((4, 1, 2), dtype=np.float32)
        evidence[2, 0, 1] = np.nan
        risk, quality = relative_quality_from_evidence(evidence)
        self.assertTrue(np.isfinite(risk[0, 0]))
        self.assertTrue(np.isnan(risk[0, 1]))
        self.assertTrue(np.isnan(quality[0, 1]))

    def test_output_metadata_range_and_determinism(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            first = run_quality_evidence(output_directory=output_directory)
            with rasterio.open(output_directory / "geosr_relative_quality.tif") as quality:
                values = quality.read(1)
                valid = values != quality.nodata
                self.assertTrue(np.all((values[valid] >= 0.0) & (values[valid] <= 1.0)))
                with rasterio.open("outputs/geosr_2p5m.tif") as geosr:
                    self.assertEqual((quality.width, quality.height), (geosr.width, geosr.height))
                    self.assertEqual(quality.crs, geosr.crs)
                    self.assertEqual(quality.bounds, geosr.bounds)
            second = run_quality_evidence(output_directory=output_directory)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
