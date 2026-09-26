"""Generate a final GeoSR validation summary report."""

from pathlib import Path
import json


OUTPUT_DIRECTORY = Path("outputs")


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r") as file:
        return json.load(file)


def generate_validation_report(
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict:
    """Combine validation results into one report."""

    quality_summary = load_json(
        output_directory / "geosr_quality_summary.json"
    )

    report = {
        "project": "GeoSR",
        "validation_stage": "M4",
        "validation_outputs": {
            "quality_evidence": quality_summary
        },
        "notes": [
            "Relative quality is agreement evidence.",
            "It is not a probability of correctness.",
            "NDVI consistency was validated against original Sentinel-2 observations."
        ]
    }

    report_path = output_directory / "geosr_validation_summary.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=2)

    print(f"Validation report saved to: {report_path}")

    return report


if __name__ == "__main__":
    generate_validation_report()