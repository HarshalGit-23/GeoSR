# GeoSR
Deep Learning Based Super Resolution Mapping from Sentinel-2 Imagery

## Dashboard

The Streamlit dashboard is a read-only viewer for GeoTIFF outputs from the GeoSR pipeline. It does not run inference or generate raster values. Missing files are shown as unavailable until the pipeline creates them in `outputs/`.

Install the dashboard dependencies and launch it from the project root:

```powershell
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The sidebar selects Original Sentinel-2, Bicubic, GeoSR, NDVI, NDVI Difference, or Relative Quality layers. Expected output filenames are defined near the top of `app/dashboard.py`, leaving room for M4 validation metadata.
