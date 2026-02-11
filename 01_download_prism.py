"""
PRISM Data Processing Toolkit - Script 1: Data Downloader
==========================================================
Downloads PRISM daily data for user-specified variables and saves with 
comprehensive metadata including download timing, missing days, and 
original data information.

Features:
- Download daily PRISM data via web service
- Support for 800m and 4km resolutions
- Support for AN (All Networks) and LT (Long Term) dataset types
- Automatic reprojection to FFRD Albers Equal Area Conic (NAD83, US Feet)
- Comprehensive metadata and logging

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
    python 01_download_prism.py --variable ppt --start 2020-01-01 --end 2020-12-31 \
        --resolution 800m --dataset_type AN --output_dir ./prism_data --unit_system SI
"""

import os
import sys
import json
import time
import logging
import argparse
import zipfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import urllib.error

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from tqdm import tqdm

from config import (
    PRISMVariable, PRISMResolution, PRISMDatasetType, UnitSystem,
    PRISM_WEB_SERVICE_BASE, PRISM_CRS, PRISMMetadata, PRISM_UNITS,
    PRISM_DAILY_START_YEAR
)

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


def get_ffrd_crs() -> CRS:
    """
    Get the FFRD CRS object.
    
    Returns
    -------
    rasterio.crs.CRS
        The FFRD Albers Equal Area Conic CRS
    """
    return CRS.from_wkt(FFRD_PROJECTION_WKT)


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger('prism_downloader')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def generate_date_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Generate list of dates between start and end (inclusive)."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def build_download_url(variable: PRISMVariable, resolution: PRISMResolution,
                       date: datetime, dataset_type: PRISMDatasetType,
                       region: str = "us") -> str:
    """
    Build the PRISM web service download URL.
    
    URL format: https://services.nacse.org/prism/data/get/<region>/<res>/<element>/<date>
    For LT data, add /lt at the end.
    """
    date_str = date.strftime("%Y%m%d")
    url = f"{PRISM_WEB_SERVICE_BASE}/{region}/{resolution.value}/{variable.value}/{date_str}"
    
    if dataset_type == PRISMDatasetType.LT:
        url += "/lt"
    
    return url


def download_single_day(url: str, output_path: Path, logger: logging.Logger,
                        max_retries: int = 3, retry_delay: int = 5) -> Tuple[bool, float, str]:
    """Download a single day's PRISM data."""
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            urllib.request.urlretrieve(url, output_path)
            download_time = time.time() - start_time
            
            # Check if downloaded file is valid (not an error text file)
            if output_path.stat().st_size < 1000:
                with open(output_path, 'r') as f:
                    content = f.read(500)
                    if 'error' in content.lower() or 'limit' in content.lower():
                        logger.warning(f"Rate limit or error detected. Content: {content[:200]}")
                        os.remove(output_path)
                        return False, 0, "Rate limit exceeded or server error"
            
            return True, download_time, ""
            
        except urllib.error.HTTPError as e:
            error_msg = f"HTTP Error {e.code}: {e.reason}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
        except urllib.error.URLError as e:
            error_msg = f"URL Error: {e.reason}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return False, 0, error_msg


def extract_zip(zip_path: Path, extract_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Extract downloaded zip file and return path to the .tif or .bil file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        for ext in ['.tif', '.bil']:
            for f in extract_dir.glob(f"*{ext}"):
                return f
        
        logger.error(f"No raster file found in {zip_path}")
        return None
        
    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {zip_path}")
        return None


def reproject_raster_to_ffrd(input_path: Path, output_path: Path,
                              logger: logging.Logger) -> bool:
    """
    Reproject a raster to the FFRD Albers Equal Area Conic coordinate system.
    
    Target CRS: USA_Contiguous_Albers_Equal_Area_Conic_FFRD
    - Datum: NAD83
    - Units: US Survey Feet
    """
    try:
        target_crs = get_ffrd_crs()
        
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'driver': 'GTiff',
                'compress': 'lzw'
            })
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
        
        logger.debug(f"Reprojected to FFRD CRS: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Reprojection failed for {input_path}: {str(e)}")
        return False


def write_metadata(metadata_path: Path, metadata: dict):
    """Write metadata to a JSON file and a human-readable text file."""
    # JSON version
    json_path = metadata_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Human-readable text version
    txt_path = metadata_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PRISM DATA DOWNLOAD METADATA\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DOWNLOAD INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Download Start Time: {metadata['download_start_time']}\n")
        f.write(f"Download End Time: {metadata['download_end_time']}\n")
        f.write(f"Total Download Duration: {metadata['total_download_duration_seconds']:.2f} seconds\n")
        f.write(f"Total Files Downloaded: {metadata['successful_downloads']}\n")
        f.write(f"Failed Downloads: {metadata['failed_downloads']}\n\n")
        
        f.write("DATA PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Variable: {metadata['variable']}\n")
        f.write(f"Resolution: {metadata['resolution']}\n")
        f.write(f"Dataset Type: {metadata['dataset_type']}\n")
        f.write(f"Date Range: {metadata['start_date']} to {metadata['end_date']}\n")
        f.write(f"Total Days Requested: {metadata['total_days_requested']}\n\n")
        
        f.write("ORIGINAL DATA SPECIFICATIONS (PRISM)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original CRS: {metadata['original_crs']}\n")
        f.write(f"Original Units: {metadata['original_units']}\n")
        f.write(f"Time Zone: {metadata['time_zone']}\n")
        f.write(f"Day Definition: {metadata['day_definition']}\n")
        f.write(f"Data Source: {metadata['data_source']}\n")
        f.write(f"Website: {metadata['website']}\n\n")
        
        f.write("TARGET PROJECTION (FFRD STANDARD)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Projection Name: {metadata['target_crs_name']}\n")
        f.write(f"Datum: NAD83\n")
        f.write(f"Projection: Albers Equal Area Conic\n")
        f.write(f"Central Meridian: -96.0°\n")
        f.write(f"Standard Parallel 1: 29.5°\n")
        f.write(f"Standard Parallel 2: 45.5°\n")
        f.write(f"Latitude of Origin: 23.0°\n")
        f.write(f"Linear Units: US Survey Feet\n")
        f.write(f"Reprojected Files Location: {metadata['processed_folder']}\n\n")
        
        f.write("WKT DEFINITION\n")
        f.write("-" * 40 + "\n")
        f.write(metadata['target_crs_wkt'] + "\n\n")
        
        if metadata['missing_dates']:
            f.write("MISSING DATES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of Missing Days: {len(metadata['missing_dates'])}\n")
            for date in metadata['missing_dates']:
                f.write(f"  - {date}\n")
            f.write("\n")
        else:
            f.write("MISSING DATES: None\n\n")
        
        if metadata['download_errors']:
            f.write("DOWNLOAD ERRORS\n")
            f.write("-" * 40 + "\n")
            for error in metadata['download_errors']:
                f.write(f"  - {error['date']}: {error['error']}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Metadata\n")


def save_projection_file(output_dir: Path):
    """Save the FFRD projection as a .prj file for reference."""
    prj_path = output_dir / "ffrd_projection.prj"
    with open(prj_path, 'w') as f:
        f.write(FFRD_PROJECTION_WKT)
    
    txt_path = output_dir / "ffrd_projection_info.txt"
    with open(txt_path, 'w') as f:
        f.write("FFRD Standard Projection\n")
        f.write("=" * 60 + "\n\n")
        f.write("Name: USA_Contiguous_Albers_Equal_Area_Conic_FFRD\n\n")
        f.write("Parameters:\n")
        f.write("  - Datum: NAD83 (North American Datum 1983)\n")
        f.write("  - Spheroid: GRS 1980\n")
        f.write("  - Projection: Albers Equal Area Conic\n")
        f.write("  - Central Meridian: -96.0°\n")
        f.write("  - Standard Parallel 1: 29.5°\n")
        f.write("  - Standard Parallel 2: 45.5°\n")
        f.write("  - Latitude of Origin: 23.0°\n")
        f.write("  - False Easting: 0.0\n")
        f.write("  - False Northing: 0.0\n")
        f.write("  - Linear Units: US Survey Feet (0.3048 m)\n\n")
        f.write("WKT:\n")
        f.write("-" * 60 + "\n")
        f.write(FFRD_PROJECTION_WKT + "\n\n")
        f.write("Proj4:\n")
        f.write("-" * 60 + "\n")
        f.write(FFRD_PROJECTION_PROJ4 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download PRISM daily data and reproject to FFRD Albers Equal Area Conic.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download precipitation for 2020, 800m resolution
    python 01_download_prism.py --variable ppt --start 2020-01-01 --end 2020-12-31 \\
        --resolution 800m --dataset_type AN --output_dir ./prism_data

    # Download temperature with US Customary units
    python 01_download_prism.py --variable tmax --start 2020-06-01 --end 2020-08-31 \\
        --resolution 4km --dataset_type AN --output_dir ./prism_data --unit_system US

Available Variables:
    ppt     - Total precipitation (rain + melted snow) [mm or inches]
    tmax    - Maximum temperature [°C or °F]
    tmin    - Minimum temperature [°C or °F]
    tmean   - Mean temperature [°C or °F]
    tdmean  - Mean dew point temperature [°C or °F]
    vpdmin  - Minimum vapor pressure deficit [hPa or inHg]
    vpdmax  - Maximum vapor pressure deficit [hPa or inHg]

Dataset Types:
    AN - All Networks: Best estimate using all available station data
    LT - Long Term: Focused on temporal consistency (fewer stations)

Target Projection (automatically applied):
    USA_Contiguous_Albers_Equal_Area_Conic_FFRD
    - Datum: NAD83
    - Units: US Survey Feet
        """
    )
    
    parser.add_argument('--variable', type=str, required=True,
                        choices=['ppt', 'tmax', 'tmin', 'tmean', 'tdmean', 'vpdmin', 'vpdmax'],
                        help='PRISM climate variable to download')
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--resolution', type=str, required=True,
                        choices=['800m', '4km'],
                        help='Spatial resolution')
    parser.add_argument('--dataset_type', type=str, default='AN',
                        choices=['AN', 'LT'],
                        help='Dataset type: AN (All Networks) or LT (Long Term)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for downloaded data')
    parser.add_argument('--unit_system', type=str, default='SI',
                        choices=['SI', 'US'],
                        help='Unit system for outputs (SI or US Customary)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between downloads in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Parse arguments
    variable = PRISMVariable(args.variable)
    resolution = PRISMResolution(args.resolution)
    dataset_type = PRISMDatasetType(args.dataset_type)
    unit_system = UnitSystem(args.unit_system)
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Validate date range
    if start_date.year < PRISM_DAILY_START_YEAR:
        print(f"Error: Daily PRISM data only available from {PRISM_DAILY_START_YEAR} onwards.")
        sys.exit(1)
    
    if start_date > end_date:
        print("Error: Start date must be before or equal to end date.")
        sys.exit(1)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    logs_dir = output_dir / "logs"
    
    for d in [raw_dir, processed_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save projection file for reference
    save_projection_file(output_dir)
    
    # Set up logging
    log_file = logs_dir / f"download_{variable.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("PRISM Data Downloader Started")
    logger.info("=" * 60)
    logger.info(f"Variable: {variable.value}")
    logger.info(f"Resolution: {resolution.value}")
    logger.info(f"Dataset Type: {dataset_type.value}")
    logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Target CRS: {FFRD_PROJECTION_NAME}")
    logger.info(f"Unit System: {unit_system.value}")
    
    # Generate date range
    dates = generate_date_range(start_date, end_date)
    total_days = len(dates)
    logger.info(f"Total days to download: {total_days}")
    
    # Initialize metadata
    prism_meta = PRISMMetadata(variable, resolution, dataset_type)
    
    download_start_time = datetime.now()
    successful_downloads = 0
    failed_downloads = 0
    missing_dates = []
    download_errors = []
    download_times = []
    
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download loop with progress bar
        for date in tqdm(dates, desc="Downloading PRISM data"):
            date_str = date.strftime("%Y%m%d")
            zip_path = raw_dir / f"prism_{variable.value}_{resolution.value}_{date_str}.zip"
            output_tif = processed_dir / f"prism_{variable.value}_{resolution.value}_{date_str}.tif"
            
            # Build URL
            url = build_download_url(variable, resolution, date, dataset_type)
            logger.debug(f"Downloading: {url}")
            
            # Download
            success, dl_time, error = download_single_day(url, zip_path, logger)
            
            if success:
                download_times.append(dl_time)
                
                # Extract
                extract_dir = temp_path / date_str
                extract_dir.mkdir(exist_ok=True)
                raster_path = extract_zip(zip_path, extract_dir, logger)
                
                if raster_path:
                    # Reproject to FFRD CRS
                    if reproject_raster_to_ffrd(raster_path, output_tif, logger):
                        successful_downloads += 1
                        logger.debug(f"Successfully processed: {date_str}")
                    else:
                        failed_downloads += 1
                        missing_dates.append(date_str)
                        download_errors.append({"date": date_str, "error": "Reprojection failed"})
                else:
                    failed_downloads += 1
                    missing_dates.append(date_str)
                    download_errors.append({"date": date_str, "error": "Extraction failed"})
            else:
                failed_downloads += 1
                missing_dates.append(date_str)
                download_errors.append({"date": date_str, "error": error})
            
            # Delay between downloads
            time.sleep(args.delay)
    
    download_end_time = datetime.now()
    total_duration = (download_end_time - download_start_time).total_seconds()
    
    # Compile metadata
    metadata = {
        "download_start_time": download_start_time.isoformat(),
        "download_end_time": download_end_time.isoformat(),
        "total_download_duration_seconds": total_duration,
        "average_download_time_seconds": np.mean(download_times) if download_times else 0,
        "variable": variable.value,
        "resolution": resolution.value,
        "dataset_type": dataset_type.value,
        "start_date": args.start,
        "end_date": args.end,
        "total_days_requested": total_days,
        "successful_downloads": successful_downloads,
        "failed_downloads": failed_downloads,
        "missing_dates": missing_dates,
        "download_errors": download_errors,
        "original_crs": PRISM_CRS,
        "original_units": PRISM_UNITS[variable]["SI"],
        "target_crs_name": FFRD_PROJECTION_NAME,
        "target_crs_wkt": FFRD_PROJECTION_WKT,
        "target_crs_proj4": FFRD_PROJECTION_PROJ4,
        "target_units": "US Survey Feet",
        "unit_system": unit_system.value,
        "time_zone": prism_meta.time_zone,
        "day_definition": prism_meta.day_definition,
        "data_source": prism_meta.data_source,
        "website": prism_meta.website,
        "raw_folder": str(raw_dir),
        "processed_folder": str(processed_dir)
    }
    
    # Write metadata
    metadata_path = output_dir / f"metadata_{variable.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    write_metadata(metadata_path, metadata)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Download Complete")
    logger.info("=" * 60)
    logger.info(f"Successful downloads: {successful_downloads}/{total_days}")
    logger.info(f"Failed downloads: {failed_downloads}")
    logger.info(f"Total time: {total_duration:.2f} seconds")
    if download_times:
        logger.info(f"Average download time: {np.mean(download_times):.2f} seconds")
    if missing_dates:
        logger.warning(f"Missing dates: {len(missing_dates)}")
        for date in missing_dates[:10]:
            logger.warning(f"  - {date}")
        if len(missing_dates) > 10:
            logger.warning(f"  ... and {len(missing_dates) - 10} more")
    
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Processed GeoTIFFs saved to: {processed_dir}")
    
    print(f"\n✓ Download complete! {successful_downloads}/{total_days} files processed.")
    print(f"  Raw data: {raw_dir}")
    print(f"  Processed data (FFRD CRS): {processed_dir}")
    print(f"  Metadata: {metadata_path}.txt")
    print(f"  Projection info: {output_dir / 'ffrd_projection_info.txt'}")


if __name__ == "__main__":
    main()
