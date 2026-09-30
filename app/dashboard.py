"""Streamlit viewer for GeoSR pipeline GeoTIFF outputs."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from raster_loader import RasterLoadError, load_raster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LAYER_CATALOG = {
    "Original Sentinel-2": ("sentinel2_original.tif", "Original Sentinel-2 RGBN input, if exported.", "rgb"),
    "Bicubic": ("bicubic_2p5m.tif", "Bicubic RGB visualization.", "rgb"),
    "GeoSR": ("geosr_2p5m.tif", "GeoSR RGB visualization.", "rgb"),
    "Original NDVI": ("ndvi_original.tif", "Original NDVI.", "index"),
    "GeoSR NDVI": ("ndvi_geosr.tif", "GeoSR NDVI.", "index"),
    "Bicubic NDVI": ("ndvi_bicubic.tif", "Bicubic NDVI.", "index"),
    "NDVI Difference": ("ndvi_difference_geosr.tif", "GeoSR NDVI difference.", "difference"),
    "NDVI Consistency Difference": ("ndvi_consistency_difference.tif", "NDVI consistency difference.", "difference"),
    "Aggregated GeoSR NDVI (10 m)": ("ndvi_geosr_aggregated_10m.tif", "GeoSR NDVI aggregated to 10 m.", "index"),
    "Relative Quality Index": ("geosr_relative_quality.tif", "Relative quality / agreement evidence.", "quality"),
    "Quality Evidence (band meaning undocumented)": ("geosr_quality_evidence.tif", "Band meanings are not documented in this repository.", "evidence"),
}


def output_path(filename: str) -> Path:
    return OUTPUTS_DIR / filename


@st.cache_data(show_spinner=False)
def cached_raster(path_text: str):
    """Cache successful reads so reruns avoid loading a raster repeatedly."""
    return load_raster(path_text)


def finite_values(array):
    values = np.asarray(array.compressed(), dtype=float)
    return values[np.isfinite(values)]


def percentile_stretch(band):
    """Create a 2nd–98th percentile display stretch without changing source data."""
    source = np.ma.masked_invalid(band.astype(float))
    values = finite_values(source)
    if not values.size:
        raise ValueError("This raster band has no valid finite pixels to display.")
    low, high = np.percentile(values, (2, 98))
    if high <= low:
        raise ValueError("Cannot stretch a constant or empty raster band.")
    return np.ma.clip((source - low) / (high - low), 0, 1)


def display_array(data, kind):
    """Prepare display arrays; source raster values are never changed."""
    if kind == "rgb":
        if data.shape[0] != 4:
            raise ValueError(f"Expected a four-band RGBN GeoTIFF, found {data.shape[0]} bands.")
        # Project layout: B04 red, B03 green, B02 blue, B08 NIR.
        rgb = np.ma.stack([data[0], data[1], data[2]])
        stretched = np.ma.stack([percentile_stretch(band) for band in rgb])
        return np.moveaxis(stretched, 0, -1), "RGB (2nd–98th percentile stretch)", None, None
    if data.shape[0] != 1:
        raise ValueError(f"Expected a single-band raster, found {data.shape[0]} bands.")
    band = np.ma.masked_invalid(data[0].astype(float))
    values = finite_values(band)
    if not values.size:
        raise ValueError("This raster band has no valid finite pixels to display.")
    if kind == "difference":
        limit = float(np.percentile(np.abs(values), 98)) or 1.0
        return band, "RdBu_r", -limit, limit
    if kind == "index":
        return band, "RdYlGn", -1.0, 1.0
    return band, "viridis", float(np.percentile(values, 2)), float(np.percentile(values, 98))


def render_layer(name, filename, description, kind):
    path = output_path(filename)
    st.subheader(name)
    st.caption(description)
    if not path.is_file():
        st.info("Output not available yet")
        return
    try:
        with st.spinner("Reading GeoTIFF..."):
            data, metadata = cached_raster(str(path))
        valid = finite_values(data)
        valid_range = (float(valid.min()), float(valid.max())) if valid.size else "No valid pixels"
        st.subheader("Raster information")
        st.write({
            "Filename": metadata["filename"], "Width": metadata["width"], "Height": metadata["height"],
            "Band count": metadata["bands"], "CRS": metadata["crs"],
            "Resolution": metadata["resolution"], "Bounds": metadata["bounds"],
            "Valid range": valid_range,
        })
        if kind == "evidence" and data.shape[0] != 1:
            st.warning("Band meanings for geosr_quality_evidence.tif are not documented in this repository; no band labels are inferred.")
            return
        image, color_map, minimum, maximum = display_array(data, kind)
        figure, axis = plt.subplots(figsize=(10, 7))
        rendered = axis.imshow(image, cmap=color_map, vmin=minimum, vmax=maximum)
        axis.set_axis_off()
        axis.set_title(name)
        if kind != "rgb":
            figure.colorbar(rendered, ax=axis, shrink=0.8)
        caption = ("RGB display uses a 2nd–98th percentile visualization stretch; source values are unchanged."
                   if kind == "rgb" else "Single-band display; source values are unchanged.")
        st.pyplot(figure, clear_figure=True)
        st.caption(caption)

    except (RasterLoadError, rasterio.errors.RasterioError, OSError, ValueError) as error:
        st.error(f"Could not display {path.name}: {error}")


def main():
    st.set_page_config(page_title="GeoSR Dashboard", page_icon="Satellite", layout="wide")
    st.title("GeoSR Dashboard")
    st.write("Inspect real Sentinel-2 super-resolution outputs produced by the GeoSR pipeline.")
    with st.sidebar:
        st.header("Visualization")
        name = st.selectbox("Layer", list(LAYER_CATALOG))
        st.caption("Layer availability")
        for layer_name, (filename, _description, _kind) in LAYER_CATALOG.items():
            status = "available" if output_path(filename).is_file() else "Output not available yet"
            st.caption(f"{layer_name}: {status}")
    filename, description, kind = LAYER_CATALOG[name]
    left, right = st.columns([3, 1])
    with left:
        render_layer(name, filename, description, kind)
    with right:
        st.subheader("Pipeline status")
        available = sum(output_path(item[0]).is_file() for item in LAYER_CATALOG.values())
        st.metric("Layers available", f"{available}/{len(LAYER_CATALOG)}")
        if output_path("geosr_quality_evidence.tif").is_file():
            st.caption("Quality evidence exists; band meanings are undocumented here.")


if __name__ == "__main__":
    main()
