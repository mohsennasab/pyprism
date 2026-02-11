# PRISM Data Processing Toolkit

A Python toolkit for downloading, processing, and visualizing PRISM (Parameter-elevation Regressions on Independent Slopes Model) climate data. Developed for FFRD (Flood Frequency and Risk Determination) hydrological and water resources engineering applications.

**Author:** HydroMohsen  
**Date:** January 2025  
**Data Source:** Oregon State University PRISM Climate Group (https://prism.oregonstate.edu/)

---

## Table of Contents

1. [Overview](#overview)
2. [Target Projection (FFRD Standard)](#target-projection-ffrd-standard)
3. [Features](#features)
4. [Installation](#installation)
5. [PRISM Data Background](#prism-data-background)
6. [Script 1: Download PRISM Data](#script-1-download-prism-data)
7. [Script 2: Clip to Watershed](#script-2-clip-to-watershed)
8. [Script 3: Gap-Fill Missing Dates](#script-3-gap-fill-missing-dates)
9. [Script 4: Visualization and Statistics](#script-4-visualization-and-statistics)
10. [Workflow Example](#workflow-example)

---

## Overview

This toolkit provides a complete workflow for working with PRISM daily climate data:

```
Download → Reproject to FFRD CRS → Clip → Gap-Fill → Visualize
```

All outputs are automatically reprojected to the **FFRD Standard Projection** (USA Contiguous Albers Equal Area Conic with US Survey Feet units).

---

## Target Projection (FFRD Standard)

All data processed by this toolkit is reprojected to the FFRD standard coordinate system:

**USA_Contiguous_Albers_Equal_Area_Conic_FFRD**

| Parameter | Value |
|-----------|-------|
| Datum | NAD83 (North American Datum 1983) |
| Spheroid | GRS 1980 |
| Projection | Albers Conic Equal Area |
| Central Meridian | -96.0° |
| Standard Parallel 1 | 29.5° |
| Standard Parallel 2 | 45.5° |
| Latitude of Origin | 23.0° |
| False Easting | 0.0 |
| False Northing | 0.0 |
| **Linear Units** | **US Survey Feet (0.3048 m)** |

### WKT Definition

```
PROJCS["USA_Contiguous_Albers_Equal_Area_Conic",
    GEOGCS["GCS_North_American_1983",
        DATUM["D_North_American_1983",
            SPHEROID["GRS_1980",6378137.0,298.257222101]],
        PRIMEM["Greenwich",0.0],
        UNIT["Degree",0.0174532925199433]],
    PROJECTION["Albers_Conic_Equal_Area"],
    PARAMETER["False_Easting",0.0],
    PARAMETER["False_Northing",0.0],
    PARAMETER["Central_Meridian",-96.0],
    PARAMETER["Standard_Parallel_1",29.5],
    PARAMETER["Standard_Parallel_2",45.5],
    PARAMETER["Latitude_Of_Origin",23.0],
    UNIT["Foot",0.3048]]
```

### Proj4 String

```
+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=ft +no_defs
```

---

## Features

### Data Download (Script 1)
- Download daily PRISM data via web service API
- Support for **800m** and **4km** resolutions
- Support for **AN** (All Networks) and **LT** (Long Term) dataset types
- **Automatic reprojection to FFRD CRS**
- Comprehensive metadata including download timing and missing days

### Watershed Clipping (Script 2)
- Support for GeoJSON and Shapefile (zipped) boundaries
- **Automatic boundary reprojection to FFRD CRS**
- Missing file detection and logging

### Gap-Filling (Script 3)
- Pixel-by-pixel linear interpolation
- Handles single-day and consecutive gaps
- Maximum gap threshold protection (default: 10 days)
- PNG visualizations of interpolated rasters
- Excel log of all interpolations

### Visualization (Script 4)
- Time series charts with mean/max/min
- Cumulative charts
- Custom date range analysis
- Spatial distribution maps
- Excel statistics export
- Support for SI and US Customary units

---

## Installation

### Prerequisites

- Python 3.8 or higher
- GDAL (recommended for optimal performance)

### Install Dependencies

```bash
# Create a new conda environment with Python and GDAL
conda create -n pyprism python=3.11 gdal -c conda-forge -y

# Activate the environment
conda activate pyprism

# Install required packages
pip install -r requirements.txt
```

---

## PRISM Data Background

### What is PRISM?

PRISM (Parameter-elevation Regressions on Independent Slopes Model) is a climate mapping system that produces high-resolution gridded estimates of climate variables for the contiguous United States.

### Available Variables

| Variable | Description | SI Units | US Units |
|----------|-------------|----------|----------|
| `ppt` | Total precipitation (rain + melted snow) | mm | inches |
| `tmax` | Maximum temperature | °C | °F |
| `tmin` | Minimum temperature | °C | °F |
| `tmean` | Mean temperature | °C | °F |
| `tdmean` | Mean dew point temperature | °C | °F |
| `vpdmin` | Minimum vapor pressure deficit | hPa | inHg |
| `vpdmax` | Maximum vapor pressure deficit | hPa | inHg |

### Dataset Types

| Type | Name | Description |
|------|------|-------------|
| **AN** | All Networks | Uses all available station data for best estimate at each time step |
| **LT** | Long Term | Uses only long-term stations (≥20 years) for temporal consistency |

### Spatial Resolution

| Resolution | Grid Spacing | Approximate Cell Size |
|------------|--------------|----------------------|
| **800m** | ~30 arc-seconds | ~800 meters |
| **4km** | ~2.5 arc-minutes | ~4 kilometers |

### Original Data Specifications

- **Original CRS:** NAD83 (EPSG:4269)
- **Target CRS:** FFRD Albers Equal Area Conic (US Feet)
- **Time Zone:** Pacific Time (data updates)
- **PRISM Day Definition:** 1200 UTC to 1200 UTC (7 AM-7 AM EST)
- **Daily Data Availability:** 1981-present

---

## Script 1: Download PRISM Data

### Purpose

Downloads PRISM daily data for specified variables, dates, and resolution. **Automatically reprojects to FFRD CRS.**

### Usage

```bash
python 01_download_prism.py \
    --variable ppt \
    --start 2020-01-01 \
    --end 2020-12-31 \
    --resolution 800m \
    --dataset_type AN \
    --output_dir ./prism_data
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--variable` | Yes | Climate variable (ppt, tmax, tmin, etc.) |
| `--start` | Yes | Start date (YYYY-MM-DD) |
| `--end` | Yes | End date (YYYY-MM-DD) |
| `--resolution` | Yes | Spatial resolution (800m or 4km) |
| `--dataset_type` | No | AN or LT (default: AN) |
| `--output_dir` | Yes | Output directory |
| `--unit_system` | No | SI or US (default: SI) |
| `--delay` | No | Delay between downloads in seconds (default: 1.0) |

### Output Structure

```
prism_data/
├── raw/                        # Downloaded zip files
├── processed/                  # Reprojected GeoTIFFs (FFRD CRS)
├── logs/                       # Download logs
├── ffrd_projection.prj         # Projection file
├── ffrd_projection_info.txt    # Detailed projection info
└── metadata_*.txt              # Comprehensive metadata
```

---

## Script 2: Clip to Watershed

### Purpose

Clips reprojected PRISM GeoTIFFs to a watershed boundary. **Both rasters and boundary are ensured to be in FFRD CRS.**

### Usage

```bash
python 02_clip_to_watershed.py \
    --input_dir ./prism_data/processed \
    --boundary ./my_watershed.geojson \
    --output_dir ./clipped_data
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input_dir` | Yes | Directory with reprojected GeoTIFFs |
| `--boundary` | Yes | Boundary file (GeoJSON, .shp, or .zip) |
| `--output_dir` | Yes | Output directory |

### Output Structure

```
clipped_data/
├── clipped/                    # Clipped GeoTIFFs (FFRD CRS)
├── boundary/                   # Reprojected boundary files (FFRD CRS)
│   ├── *_ffrd.geojson
│   └── *_ffrd_shp/
├── logs/                       # Processing logs
├── ffrd_projection.prj         # Projection file
└── clip_metadata_*.txt         # Clipping metadata
```

---

## Script 3: Gap-Fill Missing Dates

### Purpose

Identifies missing dates in the clipped data and creates interpolated rasters using pixel-by-pixel linear interpolation.

### Interpolation Methodology

For a gap of length K between valid days A and B:

```
For each missing day i (position 1 to K):
pixel(i) = pixel(A) + (pixel(B) - pixel(A)) × (i / (K+1))
```

**Example:** If days 1, 2, 3 are missing between day 0 and day 4:
- Day 1 = Day0 + (Day4 - Day0) × 1/4 = 25% toward Day4
- Day 2 = Day0 + (Day4 - Day0) × 2/4 = 50% toward Day4
- Day 3 = Day0 + (Day4 - Day0) × 3/4 = 75% toward Day4

### Usage

```bash
python 03_gap_fill.py \
    --input_dir ./clipped_data/clipped \
    --output_dir ./gap_filled \
    --max_gap 10 \
    --variable ppt
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input_dir` | Yes | Directory with clipped GeoTIFFs |
| `--output_dir` | Yes | Output directory |
| `--max_gap` | No | Maximum gap to interpolate (default: 10 days) |
| `--variable` | Yes | Variable type for visualization |
| `--unit_system` | No | SI or US (default: SI) |

---

## Script 4: Visualization and Statistics

### Purpose

Creates publication-ready visualizations and exports comprehensive statistics to Excel.

### Usage

```bash
python 04_visualize.py \
    --input_dir ./clipped_data/clipped \
    --output_dir ./visualizations \
    --variable ppt \
    --unit_system SI \
    --custom_start 2020-06-01 \
    --custom_end 2020-08-31 \
    --boundary ./my_watershed.geojson
```

---

## Workflow Example

### Complete Workflow

```bash
# Step 1: Download precipitation data for 2020 (800m, AN)
python 01_download_prism.py \
    --variable ppt \
    --start 2020-01-01 \
    --end 2020-12-31 \
    --resolution 800m \
    --dataset_type AN \
    --output_dir ./prism_ppt

# Step 2: Clip to watershed (automatically uses FFRD CRS)
python 02_clip_to_watershed.py \
    --input_dir ./prism_ppt/processed \
    --boundary ./my_watershed.geojson \
    --output_dir ./clipped

# Step 3: Gap-fill any missing dates
python 03_gap_fill.py \
    --input_dir ./clipped/clipped \
    --output_dir ./gap_filled \
    --max_gap 10 \
    --variable ppt

# Step 4: Create visualizations
python 04_visualize.py \
    --input_dir ./clipped/clipped \
    --output_dir ./viz \
    --variable ppt \
    --unit_system US \
    --custom_start 2020-06-01 \
    --custom_end 2020-08-31 \
    --boundary ./my_watershed.geojson
```

---

## References

- Daly, C., et al. (2008). Physiographically sensitive mapping of climatological temperature and precipitation across the conterminous United States. *International Journal of Climatology*, 28(15), 2031-2064.

- PRISM Climate Group. (2025). *Descriptions of PRISM Spatial Datasets*. Oregon State University.

---

## License

This toolkit is provided for educational and research purposes. Please ensure compliance with PRISM's terms of use when downloading and using PRISM data.

**PRISM Data Citation:**
> PRISM Climate Group, Oregon State University, https://prism.oregonstate.edu.
