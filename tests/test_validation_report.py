import json
import tempfile
import unittest
from pathlib import Path

from src.validation.generate_validation_report import generate_validation_report


class TestValidationReport(unittest.TestCase):

    def test_generate_validation_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_directory = Path(temp_dir)

            quality_summary = {
                "relative_quality_index": {
                    "mean": 0.3142,
                    "median": 0.2720
                }
            }

            quality_file = output_directory / "geosr_quality_summary.json"

            with open(quality_file, "w") as file:
                json.dump(quality_summary, file)

            report = generate_validation_report(output_directory)

            report_file = output_directory / "geosr_validation_summary.json"

            self.assertTrue(report_file.exists())
            self.assertEqual(report["project"], "GeoSR")
            self.assertEqual(report["validation_stage"], "M4")
            self.assertIn("validation_outputs", report)
            self.assertIn("quality_evidence", report["validation_outputs"])


if __name__ == "__main__":
    unittest.main()