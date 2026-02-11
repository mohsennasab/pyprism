"""
PRISM Data Processing Toolkit - Script 2: Clip to Watershed
============================================================
Reads reprojected PRISM GeoTIFFs and clips them to a user-provided 
watershed boundary (GeoJSON or Shapefile).

Features:
- Supports GeoJSON and Shapefile (as .zip) boundary inputs
- Automatic reprojection of boundary to FFRD CRS
- Comprehensive logging and error tracking
- Missing file detection

Target Projection (FFRD Standard):
    USA_Contiguous_Albers_Equal_Area_Conic_FFRD
    - Datum: NAD83
    - Projection: Albers Equal Area Conic
    - Central Meridian: -96.0
    - Standard Parallel 1: 29.5
    - Standard Parallel 2: 45.5
    - Latitude of Origin: 23.0
    - Units: US Survey Feet

Author: HydroMohsen
Date: January 2025

Usage:
    python 02_clip_to_watershed.py --input_dir ./prism_data/processed \
        --boundary ./watershed.geojson --output_dir ./clipped_data
"""

import os
import sys
import json
import time
import logging
import argparse
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import re

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely.geometry import mapping
from tqdm import tqdm
from pyproj import CRS as PyprojCRS

# =============================================================================
# FFRD PROJECTION DEFINITION
# =============================================================================
# USA Contiguous Albers Equal Area Conic - FFRD Standard
# This is the official projection used for FFRD (Flood Frequency and Risk Determination)
# projects. Units are in US Survey Feet.

FFRD_PROJECTION_WKT = '''PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_FFRD",
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
    UNIT["Foot",0.3048]]'''

# Proj4 string equivalent (alternative format)
FFRD_PROJECTION_PROJ4 = (
    "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 "
    "+x_0=0 +y_0=0 +datum=NAD83 +units=ft +no_defs"
)

FFRD_PROJECTION_NAME = "USA_Contiguous_Albers_Equal_Area_Conic_FFRD"


def get_ffrd_crs_rasterio() -> CRS:
    """Get the FFRD CRS object for rasterio."""
    return CRS.from_wkt(FFRD_PROJECTION_WKT)


def get_ffrd_crs_pyproj() -> PyprojCRS:
    """Get the FFRD CRS object for pyproj/geopandas."""
    return PyprojCRS.from_wkt(FFRD_PROJECTION_WKT)


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger('prism_clipper')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_boundary(boundary_path: Path, logger: logging.Logger) -> Optional[gpd.GeoDataFrame]:
    """
    Load boundary from GeoJSON or zipped Shapefile.
    
    Parameters
    ----------
    boundary_path : Path
        Path to boundary file (GeoJSON or .zip containing Shapefile)
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    Optional[gpd.GeoDataFrame]
        GeoDataFrame with boundary geometry, or None if loading failed
    """
    try:
        if boundary_path.suffix.lower() == '.zip':
            # Extract and read shapefile from zip
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(boundary_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the .shp file
                shp_files = list(Path(temp_dir).rglob("*.shp"))
                if not shp_files:
                    logger.error("No .shp file found in the zip archive")
                    return None
                
                gdf = gpd.read_file(shp_files[0])
                logger.info(f"Loaded shapefile from zip: {shp_files[0].name}")
                
        elif boundary_path.suffix.lower() in ['.geojson', '.json']:
            gdf = gpd.read_file(boundary_path)
            logger.info(f"Loaded GeoJSON: {boundary_path.name}")
            
        elif boundary_path.suffix.lower() == '.shp':
            gdf = gpd.read_file(boundary_path)
            logger.info(f"Loaded Shapefile: {boundary_path.name}")
            
        else:
            logger.error(f"Unsupported boundary format: {boundary_path.suffix}")
            return None
        
        logger.info(f"Boundary CRS: {gdf.crs}")
        logger.info(f"Boundary bounds: {gdf.total_bounds}")
        logger.info(f"Number of features: {len(gdf)}")
        
        return gdf
        
    except Exception as e:
        logger.error(f"Failed to load boundary: {str(e)}")
        return None


def reproject_boundary_to_ffrd(gdf: gpd.GeoDataFrame, 
                                logger: logging.Logger) -> gpd.GeoDataFrame:
    """
    Reproject boundary to FFRD CRS.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    gpd.GeoDataFrame
        Reprojected GeoDataFrame
    """
    if gdf.crs is None:
        logger.warning("Boundary has no CRS defined. Assuming EPSG:4326 (WGS84)")
        gdf = gdf.set_crs("EPSG:4326")
    
    # Get FFRD CRS
    ffrd_crs = get_ffrd_crs_pyproj()
    
    logger.info(f"Reprojecting boundary from {gdf.crs} to {FFRD_PROJECTION_NAME}")
    gdf = gdf.to_crs(ffrd_crs)
    
    return gdf


def save_reprojected_boundary(gdf: gpd.GeoDataFrame, output_dir: Path, 
                               original_name: str, logger: logging.Logger) -> Path:
    """
    Save the reprojected boundary to output directory.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Reprojected GeoDataFrame
    output_dir : Path
        Output directory
    original_name : str
        Original boundary filename
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    Path
        Path to saved boundary file
    """
    # Save as GeoJSON and Shapefile
    base_name = Path(original_name).stem
    geojson_path = output_dir / f"{base_name}_ffrd.geojson"
    shp_dir = output_dir / f"{base_name}_ffrd_shp"
    shp_dir.mkdir(exist_ok=True)
    shp_path = shp_dir / f"{base_name}_ffrd.shp"
    
    # Save the projection file
    prj_path = shp_dir / f"{base_name}_ffrd.prj"
    with open(prj_path, 'w') as f:
        f.write(FFRD_PROJECTION_WKT)
    
    gdf.to_file(geojson_path, driver='GeoJSON')
    gdf.to_file(shp_path)
    
    logger.info(f"Saved reprojected boundary (FFRD CRS) to: {geojson_path}")
    logger.info(f"Saved reprojected boundary (FFRD CRS) to: {shp_path}")
    
    return geojson_path


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract date from PRISM filename.
    
    Expected format: prism_<var>_<res>_<YYYYMMDD>.tif
    
    Parameters
    ----------
    filename : str
        Filename to parse
    
    Returns
    -------
    Optional[str]
        Date string (YYYYMMDD) or None if not found
    """
    # Try pattern: prism_ppt_800m_20200101.tif
    match = re.search(r'(\d{8})\.tif$', filename)
    if match:
        return match.group(1)
    
    # Try PRISM original pattern: PRISM_ppt_stable_4kmD2_20200101_bil
    match = re.search(r'(\d{8})_bil', filename)
    if match:
        return match.group(1)
    
    return None


def clip_raster(raster_path: Path, geometry, output_path: Path,
                logger: logging.Logger) -> Tuple[bool, str]:
    """
    Clip a raster to a geometry.
    
    Parameters
    ----------
    raster_path : Path
        Path to input raster
    geometry : geometry
        Clipping geometry (must be in same CRS as raster)
    output_path : Path
        Path for output raster
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    Tuple[bool, str]
        (success, error_message)
    """
    try:
        with rasterio.open(raster_path) as src:
            # Clip raster to geometry
            out_image, out_transform = mask(src, [geometry], crop=True, all_touched=True)
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"
            })
            
            # Write output
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        return True, ""
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Clipping failed for {raster_path.name}: {error_msg}")
        return False, error_msg


def get_raster_files(input_dir: Path) -> List[Path]:
    """Get sorted list of raster files in directory."""
    rasters = list(input_dir.glob("*.tif"))
    rasters.sort()
    return rasters


def write_clip_metadata(metadata_path: Path, metadata: dict):
    """Write clipping metadata to JSON and text files."""
    # JSON version
    json_path = metadata_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Human-readable text version
    txt_path = metadata_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PRISM WATERSHED CLIPPING METADATA\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PROCESSING INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Processing Start Time: {metadata['processing_start_time']}\n")
        f.write(f"Processing End Time: {metadata['processing_end_time']}\n")
        f.write(f"Total Processing Duration: {metadata['total_processing_duration_seconds']:.2f} seconds\n\n")
        
        f.write("INPUT DATA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Input Directory: {metadata['input_directory']}\n")
        f.write(f"Total Input Files: {metadata['total_input_files']}\n")
        f.write(f"Boundary File: {metadata['boundary_file']}\n")
        f.write(f"Original Boundary CRS: {metadata['original_boundary_crs']}\n\n")
        
        f.write("TARGET PROJECTION (FFRD STANDARD)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Projection Name: {metadata['target_crs_name']}\n")
        f.write(f"Datum: NAD83\n")
        f.write(f"Projection: Albers Equal Area Conic\n")
        f.write(f"Central Meridian: -96.0°\n")
        f.write(f"Standard Parallel 1: 29.5°\n")
        f.write(f"Standard Parallel 2: 45.5°\n")
        f.write(f"Latitude of Origin: 23.0°\n")
        f.write(f"Linear Units: US Survey Feet\n\n")
        
        f.write("OUTPUT DATA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Output Directory: {metadata['output_directory']}\n")
        f.write(f"Successful Clips: {metadata['successful_clips']}\n")
        f.write(f"Failed Clips: {metadata['failed_clips']}\n\n")
        
        f.write("DATE RANGE\n")
        f.write("-" * 40 + "\n")
        f.write(f"First Date: {metadata['first_date']}\n")
        f.write(f"Last Date: {metadata['last_date']}\n")
        f.write(f"Expected Days: {metadata['expected_days']}\n")
        f.write(f"Actual Days: {metadata['actual_days']}\n\n")
        
        if metadata['missing_files']:
            f.write("MISSING FILES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of Missing Files: {len(metadata['missing_files'])}\n")
            for file in metadata['missing_files'][:20]:
                f.write(f"  - {file}\n")
            if len(metadata['missing_files']) > 20:
                f.write(f"  ... and {len(metadata['missing_files']) - 20} more\n")
            f.write("\n")
        
        if metadata['clip_errors']:
            f.write("CLIPPING ERRORS\n")
            f.write("-" * 40 + "\n")
            for error in metadata['clip_errors'][:20]:
                f.write(f"  - {error['file']}: {error['error']}\n")
            if len(metadata['clip_errors']) > 20:
                f.write(f"  ... and {len(metadata['clip_errors']) - 20} more\n")
            f.write("\n")
        
        f.write("WKT DEFINITION\n")
        f.write("-" * 40 + "\n")
        f.write(FFRD_PROJECTION_WKT + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Metadata\n")


def save_projection_file(output_dir: Path):
    """Save the FFRD projection as a .prj file for reference."""
    prj_path = output_dir / "ffrd_projection.prj"
    with open(prj_path, 'w') as f:
        f.write(FFRD_PROJECTION_WKT)


def main():
    parser = argparse.ArgumentParser(
        description='Clip PRISM rasters to watershed boundary using FFRD CRS.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Clip with GeoJSON boundary
    python 02_clip_to_watershed.py --input_dir ./prism_data/processed \\
        --boundary ./watershed.geojson --output_dir ./clipped_data

    # Clip with zipped Shapefile
    python 02_clip_to_watershed.py --input_dir ./prism_data/processed \\
        --boundary ./watershed_shp.zip --output_dir ./clipped_data

Supported Boundary Formats:
    - GeoJSON (.geojson, .json)
    - Shapefile (.shp or .zip containing .shp and associated files)

Target Projection (automatically applied):
    USA_Contiguous_Albers_Equal_Area_Conic_FFRD
    - Datum: NAD83
    - Units: US Survey Feet
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing reprojected PRISM GeoTIFFs')
    parser.add_argument('--boundary', type=str, required=True,
                        help='Path to watershed boundary (GeoJSON or zipped Shapefile)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for clipped rasters')
    
    args = parser.parse_args()
    
    # Parse arguments
    input_dir = Path(args.input_dir)
    boundary_path = Path(args.boundary)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Validate boundary file
    if not boundary_path.exists():
        print(f"Error: Boundary file does not exist: {boundary_path}")
        sys.exit(1)
    
    # Create output directories
    clipped_dir = output_dir / "clipped"
    boundary_dir = output_dir / "boundary"
    logs_dir = output_dir / "logs"
    
    for d in [clipped_dir, boundary_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save projection file for reference
    save_projection_file(output_dir)
    
    # Set up logging
    log_file = logs_dir / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("PRISM Watershed Clipper Started")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Boundary file: {boundary_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target CRS: {FFRD_PROJECTION_NAME}")
    
    # Load boundary
    gdf = load_boundary(boundary_path, logger)
    if gdf is None:
        print("Error: Failed to load boundary file")
        sys.exit(1)
    
    original_crs = str(gdf.crs) if gdf.crs else "Unknown"
    
    # Reproject boundary to FFRD CRS
    gdf = reproject_boundary_to_ffrd(gdf, logger)
    
    # Save reprojected boundary
    save_reprojected_boundary(gdf, boundary_dir, boundary_path.name, logger)
    
    # Get union of all geometries for clipping
    clip_geometry = gdf.unary_union
    logger.info(f"Clipping geometry type: {clip_geometry.geom_type}")
    
    # Get list of raster files
    raster_files = get_raster_files(input_dir)
    total_files = len(raster_files)
    
    if total_files == 0:
        logger.error("No .tif files found in input directory")
        print("Error: No .tif files found in input directory")
        sys.exit(1)
    
    logger.info(f"Found {total_files} raster files to process")
    
    # Initialize tracking
    processing_start = datetime.now()
    successful_clips = 0
    failed_clips = 0
    clip_errors = []
    processed_dates = []
    
    # Process each raster
    for raster_path in tqdm(raster_files, desc="Clipping rasters"):
        # Generate output filename (same as input)
        output_path = clipped_dir / raster_path.name
        
        # Extract date for tracking
        date_str = extract_date_from_filename(raster_path.name)
        if date_str:
            processed_dates.append(date_str)
        
        # Clip raster
        success, error = clip_raster(raster_path, clip_geometry, output_path, logger)
        
        if success:
            successful_clips += 1
        else:
            failed_clips += 1
            clip_errors.append({"file": raster_path.name, "error": error})
    
    processing_end = datetime.now()
    total_duration = (processing_end - processing_start).total_seconds()
    
    # Analyze date coverage
    processed_dates.sort()
    first_date = processed_dates[0] if processed_dates else None
    last_date = processed_dates[-1] if processed_dates else None
    
    # Calculate expected days (if we have dates)
    expected_days = 0
    missing_files = []
    if first_date and last_date:
        from datetime import datetime as dt, timedelta
        start_dt = dt.strptime(first_date, "%Y%m%d")
        end_dt = dt.strptime(last_date, "%Y%m%d")
        expected_days = (end_dt - start_dt).days + 1
        
        # Find missing dates
        date_set = set(processed_dates)
        current = start_dt
        while current <= end_dt:
            date_str = current.strftime("%Y%m%d")
            if date_str not in date_set:
                missing_files.append(date_str)
            current += timedelta(days=1)
    
    # Compile metadata
    metadata = {
        "processing_start_time": processing_start.isoformat(),
        "processing_end_time": processing_end.isoformat(),
        "total_processing_duration_seconds": total_duration,
        "input_directory": str(input_dir),
        "boundary_file": str(boundary_path),
        "original_boundary_crs": original_crs,
        "target_crs_name": FFRD_PROJECTION_NAME,
        "target_crs_wkt": FFRD_PROJECTION_WKT,
        "target_crs_proj4": FFRD_PROJECTION_PROJ4,
        "output_directory": str(clipped_dir),
        "total_input_files": total_files,
        "successful_clips": successful_clips,
        "failed_clips": failed_clips,
        "first_date": first_date,
        "last_date": last_date,
        "expected_days": expected_days,
        "actual_days": len(processed_dates),
        "missing_files": missing_files,
        "clip_errors": clip_errors
    }
    
    # Write metadata
    metadata_path = output_dir / f"clip_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    write_clip_metadata(metadata_path, metadata)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Clipping Complete")
    logger.info("=" * 60)
    logger.info(f"Successful clips: {successful_clips}/{total_files}")
    logger.info(f"Failed clips: {failed_clips}")
    logger.info(f"Total processing time: {total_duration:.2f} seconds")
    
    if missing_files:
        logger.warning(f"Missing dates: {len(missing_files)}")
    
    logger.info(f"Clipped rasters saved to: {clipped_dir}")
    logger.info(f"Metadata saved to: {metadata_path}.txt")
    
    print(f"\n✓ Clipping complete! {successful_clips}/{total_files} files processed.")
    print(f"  Clipped data (FFRD CRS): {clipped_dir}")
    print(f"  Reprojected boundary: {boundary_dir}")
    print(f"  Metadata: {metadata_path}.txt")
    
    if missing_files:
        print(f"\n⚠ Warning: {len(missing_files)} missing dates detected")


if __name__ == "__main__":
    main()
