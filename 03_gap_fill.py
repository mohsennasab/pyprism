"""
PRISM Data Processing Toolkit - Script 3: Gap-Fill Missing Dates
================================================================
Identifies missing dates in clipped PRISM data and creates interpolated
rasters using pixel-by-pixel linear interpolation.

Features:
- Automatic detection of missing dates
- Pixel-by-pixel linear interpolation
- Handling of single gaps and consecutive gaps
- Maximum gap threshold (default: 10 days)
- PNG visualizations of interpolated rasters
- Excel log of interpolated dates

Interpolation Methodology:
--------------------------
1. SINGLE DAY GAP:
   If day N is missing but days N-1 and N+1 exist:
   pixel(N) = (pixel(N-1) + pixel(N+1)) / 2
   
2. CONSECUTIVE GAPS (up to max_gap days):
   If days N to N+k are missing (k <= max_gap):
   For each missing day i in [N, N+k]:
   pixel(i) = pixel(N-1) + (pixel(N+k+1) - pixel(N-1)) * (i - (N-1)) / ((N+k+1) - (N-1))
   
   This is equivalent to linear interpolation between the bracketing valid days.

3. EDGE CASES:
   - Gaps at start: No interpolation (require preceding day)
   - Gaps at end: No interpolation (require following day)
   - Gaps > max_gap: Skip and log warning

Note: This script expects input data to already be in the FFRD CRS
(USA_Contiguous_Albers_Equal_Area_Conic_FFRD, US Survey Feet).

Author: HydroMohsen
Date: January 2025

Usage:
    python 03_gap_fill.py --input_dir ./clipped_data/clipped \
        --output_dir ./gap_filled --max_gap 10 --unit_system SI --variable ppt
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from config import (
    PRISMVariable, UnitSystem, PRISM_UNITS, convert_units,
    FFRD_PROJECTION_NAME, FFRD_PROJECTION_WKT
)


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger('prism_gapfill')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
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


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract date (YYYYMMDD) from filename."""
    match = re.search(r'(\d{8})', filename)
    return match.group(1) if match else None


def get_filename_template(raster_files: List[Path]) -> Tuple[str, str]:
    """Extract filename template from existing files."""
    if not raster_files:
        return "prism_", ".tif"
    
    sample = raster_files[0].name
    match = re.search(r'(\d{8})', sample)
    if match:
        date_str = match.group(1)
        idx = sample.find(date_str)
        prefix = sample[:idx]
        suffix = sample[idx + 8:]
        return prefix, suffix
    
    return "prism_", ".tif"


def scan_raster_dates(input_dir: Path, logger: logging.Logger) -> Tuple[Dict[str, Path], str, str]:
    """Scan directory for raster files and extract dates."""
    raster_files = sorted(input_dir.glob("*.tif"))
    date_to_file = {}
    
    for f in raster_files:
        date_str = extract_date_from_filename(f.name)
        if date_str:
            date_to_file[date_str] = f
    
    if not date_to_file:
        return {}, "", ""
    
    sorted_dates = sorted(date_to_file.keys())
    return date_to_file, sorted_dates[0], sorted_dates[-1]


def find_missing_dates(date_to_file: Dict[str, Path], first_date: str, 
                       last_date: str, logger: logging.Logger) -> List[str]:
    """Find all missing dates in the date range."""
    existing_dates = set(date_to_file.keys())
    start_dt = datetime.strptime(first_date, "%Y%m%d")
    end_dt = datetime.strptime(last_date, "%Y%m%d")
    
    missing = []
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        if date_str not in existing_dates:
            missing.append(date_str)
        current += timedelta(days=1)
    
    return missing


def identify_gap_sequences(missing_dates: List[str]) -> List[List[str]]:
    """Group consecutive missing dates into sequences."""
    if not missing_dates:
        return []
    
    sequences = []
    current_seq = [missing_dates[0]]
    
    for i in range(1, len(missing_dates)):
        prev_dt = datetime.strptime(missing_dates[i-1], "%Y%m%d")
        curr_dt = datetime.strptime(missing_dates[i], "%Y%m%d")
        
        if (curr_dt - prev_dt).days == 1:
            current_seq.append(missing_dates[i])
        else:
            sequences.append(current_seq)
            current_seq = [missing_dates[i]]
    
    sequences.append(current_seq)
    return sequences


def find_bracketing_dates(gap_sequence: List[str], date_to_file: Dict[str, Path],
                          first_date: str, last_date: str) -> Tuple[Optional[str], Optional[str]]:
    """Find the dates that bracket a gap sequence."""
    gap_start = datetime.strptime(gap_sequence[0], "%Y%m%d")
    gap_end = datetime.strptime(gap_sequence[-1], "%Y%m%d")
    
    # Find preceding date
    prev_date = None
    check_date = gap_start - timedelta(days=1)
    first_dt = datetime.strptime(first_date, "%Y%m%d")
    while check_date >= first_dt:
        date_str = check_date.strftime("%Y%m%d")
        if date_str in date_to_file:
            prev_date = date_str
            break
        check_date -= timedelta(days=1)
    
    # Find following date
    next_date = None
    check_date = gap_end + timedelta(days=1)
    last_dt = datetime.strptime(last_date, "%Y%m%d")
    while check_date <= last_dt:
        date_str = check_date.strftime("%Y%m%d")
        if date_str in date_to_file:
            next_date = date_str
            break
        check_date += timedelta(days=1)
    
    return prev_date, next_date


def interpolate_pixel_by_pixel(raster_before: np.ndarray, raster_after: np.ndarray,
                               gap_length: int, position: int) -> np.ndarray:
    """
    Perform linear interpolation for a single position within a gap.
    
    Linear interpolation formula:
    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    """
    total_distance = gap_length + 1
    weight = position / total_distance
    
    # Handle nodata values
    nodata_mask = np.isnan(raster_before) | np.isnan(raster_after)
    
    # Linear interpolation
    interpolated = raster_before + (raster_after - raster_before) * weight
    
    # Restore nodata
    interpolated[nodata_mask] = np.nan
    
    return interpolated


def create_interpolated_raster(template_path: Path, data: np.ndarray, 
                               output_path: Path, logger: logging.Logger) -> bool:
    """Create a new raster file with interpolated data."""
    try:
        with rasterio.open(template_path) as src:
            meta = src.meta.copy()
            meta.update({
                'driver': 'GTiff',
                'compress': 'lzw'
            })
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(data.astype(meta['dtype']), 1)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create raster {output_path}: {str(e)}")
        return False


def create_visualization(raster_path: Path, output_path: Path, 
                        variable: PRISMVariable, unit_system: UnitSystem,
                        date_str: str, logger: logging.Logger):
    """Create PNG visualization of interpolated raster."""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            
            # Convert units if needed
            if unit_system == UnitSystem.US_CUSTOMARY:
                data = convert_units(data, variable, to_us=True)
            
            # Get unit label
            unit_label = PRISM_UNITS[variable][unit_system.value]
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Mask nodata
            masked_data = np.ma.masked_invalid(data)
            
            # Choose colormap based on variable
            if variable == PRISMVariable.PPT:
                cmap = 'Blues'
                vmin = 0
                vmax = np.nanpercentile(data, 99) if not np.all(np.isnan(data)) else 1
            elif variable in [PRISMVariable.TMAX, PRISMVariable.TMIN, PRISMVariable.TMEAN]:
                cmap = 'RdYlBu_r'
                vmin = np.nanpercentile(data, 1) if not np.all(np.isnan(data)) else 0
                vmax = np.nanpercentile(data, 99) if not np.all(np.isnan(data)) else 1
            else:
                cmap = 'viridis'
                vmin = np.nanpercentile(data, 1) if not np.all(np.isnan(data)) else 0
                vmax = np.nanpercentile(data, 99) if not np.all(np.isnan(data)) else 1
            
            # Plot
            im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Format date for title
            date_formatted = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
            
            # Colorbar and labels
            cbar = plt.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label(f'{variable.value.upper()} ({unit_label})', fontsize=12)
            
            ax.set_title(f'PRISM {variable.value.upper()} - {date_formatted}\n(Interpolated)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Column (FFRD CRS - US Feet)', fontsize=10)
            ax.set_ylabel('Row (FFRD CRS - US Feet)', fontsize=10)
            
            # Add interpolation note
            ax.text(0.02, 0.02, 'Linear Interpolation', transform=ax.transAxes,
                   fontsize=9, style='italic', color='gray',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        logger.error(f"Failed to create visualization for {date_str}: {str(e)}")


def write_gapfill_metadata(metadata_path: Path, metadata: dict):
    """Write gap-fill metadata to JSON and text files."""
    # JSON version
    json_path = metadata_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Human-readable text version
    txt_path = metadata_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PRISM GAP-FILL INTERPOLATION METADATA\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PROCESSING INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Processing Start Time: {metadata['processing_start_time']}\n")
        f.write(f"Processing End Time: {metadata['processing_end_time']}\n")
        f.write(f"Total Processing Duration: {metadata['total_processing_duration_seconds']:.2f} seconds\n\n")
        
        f.write("COORDINATE SYSTEM\n")
        f.write("-" * 40 + "\n")
        f.write(f"CRS: {FFRD_PROJECTION_NAME}\n")
        f.write(f"Units: US Survey Feet\n\n")
        
        f.write("INTERPOLATION PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Maximum Gap Allowed: {metadata['max_gap_days']} days\n")
        f.write(f"Interpolation Method: {metadata['interpolation_method']}\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 40 + "\n")
        f.write(metadata['methodology_description'])
        f.write("\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Missing Dates Found: {metadata['total_missing_dates']}\n")
        f.write(f"Successfully Interpolated: {metadata['successfully_interpolated']}\n")
        f.write(f"Skipped (Gap Too Large): {metadata['skipped_large_gap']}\n")
        f.write(f"Skipped (Edge Cases): {metadata['skipped_edge_cases']}\n\n")
        
        if metadata['interpolated_dates']:
            f.write("INTERPOLATED DATES\n")
            f.write("-" * 40 + "\n")
            for item in metadata['interpolated_dates']:
                f.write(f"  {item['date']}: gap_length={item['gap_length']}, ")
                f.write(f"bracketed by {item['before_date']} and {item['after_date']}\n")
            f.write("\n")
        
        if metadata['skipped_gaps']:
            f.write("SKIPPED GAPS (EXCEEDING THRESHOLD)\n")
            f.write("-" * 40 + "\n")
            f.write("WARNING: The following gaps exceed the maximum threshold.\n")
            f.write("Manual inspection is recommended.\n\n")
            for gap in metadata['skipped_gaps']:
                f.write(f"  Gap of {gap['length']} days: {gap['dates'][0]} to {gap['dates'][-1]}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Metadata\n")


def main():
    parser = argparse.ArgumentParser(
        description='Gap-fill missing PRISM dates using linear interpolation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interpolation Methodology:
-------------------------
This script uses pixel-by-pixel linear interpolation to fill missing dates.

For a gap of length N between valid days A and B:
  - Each missing day i (where i = 1 to N) is interpolated as:
    pixel[i] = pixel[A] + (pixel[B] - pixel[A]) * (i / (N+1))

This ensures smooth transitions across gaps while respecting the spatial
variability in the data.

Example:
  If days 1, 2, 3 are missing between day 0 and day 4:
  - Day 1 = Day0 + (Day4 - Day0) * 1/4 = 25% toward Day4
  - Day 2 = Day0 + (Day4 - Day0) * 2/4 = 50% toward Day4
  - Day 3 = Day0 + (Day4 - Day0) * 3/4 = 75% toward Day4

Note: Input data should be in FFRD CRS (USA_Contiguous_Albers_Equal_Area_Conic_FFRD)
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing clipped PRISM GeoTIFFs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for interpolated rasters')
    parser.add_argument('--max_gap', type=int, default=10,
                        help='Maximum gap length to interpolate (default: 10 days)')
    parser.add_argument('--variable', type=str, required=True,
                        choices=['ppt', 'tmax', 'tmin', 'tmean', 'tdmean', 'vpdmin', 'vpdmax'],
                        help='PRISM variable (for visualization)')
    parser.add_argument('--unit_system', type=str, default='SI',
                        choices=['SI', 'US'],
                        help='Unit system for visualizations')
    
    args = parser.parse_args()
    
    # Parse arguments
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    max_gap = args.max_gap
    variable = PRISMVariable(args.variable)
    unit_system = UnitSystem(args.unit_system)
    
    # Validate
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directories
    interpolated_dir = output_dir / "interpolated_rasters"
    viz_dir = output_dir / "visualizations"
    logs_dir = output_dir / "logs"
    
    for d in [interpolated_dir, viz_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = logs_dir / f"gapfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("PRISM Gap-Fill Interpolation Started")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Maximum gap: {max_gap} days")
    logger.info(f"Variable: {variable.value}")
    logger.info(f"Unit system: {unit_system.value}")
    logger.info(f"CRS: {FFRD_PROJECTION_NAME}")
    
    # Scan for existing dates
    date_to_file, first_date, last_date = scan_raster_dates(input_dir, logger)
    
    if not date_to_file:
        logger.error("No valid raster files found in input directory")
        print("Error: No valid raster files found")
        sys.exit(1)
    
    logger.info(f"Found {len(date_to_file)} raster files")
    logger.info(f"Date range: {first_date} to {last_date}")
    
    # Get filename template
    prefix, suffix = get_filename_template(list(date_to_file.values()))
    logger.info(f"Filename template: {prefix}YYYYMMDD{suffix}")
    
    # Find missing dates
    missing_dates = find_missing_dates(date_to_file, first_date, last_date, logger)
    total_missing = len(missing_dates)
    
    if total_missing == 0:
        logger.info("No missing dates found. Dataset is complete!")
        print("✓ No missing dates found. Dataset is complete!")
        return
    
    logger.info(f"Found {total_missing} missing dates")
    
    # Group into sequences
    gap_sequences = identify_gap_sequences(missing_dates)
    logger.info(f"Identified {len(gap_sequences)} gap sequence(s)")
    
    # Initialize tracking
    processing_start = datetime.now()
    successfully_interpolated = 0
    skipped_large_gap = 0
    skipped_edge_cases = 0
    interpolated_dates = []
    skipped_gaps = []
    
    # Process each gap sequence
    for gap_seq in tqdm(gap_sequences, desc="Processing gaps"):
        gap_length = len(gap_seq)
        
        # Check if gap exceeds maximum
        if gap_length > max_gap:
            logger.warning(f"Gap of {gap_length} days exceeds maximum ({max_gap})")
            logger.warning(f"  Dates: {gap_seq[0]} to {gap_seq[-1]}")
            skipped_large_gap += len(gap_seq)
            skipped_gaps.append({
                "length": gap_length,
                "dates": gap_seq
            })
            continue
        
        # Find bracketing dates
        before_date, after_date = find_bracketing_dates(
            gap_seq, date_to_file, first_date, last_date
        )
        
        # Check for edge cases
        if before_date is None or after_date is None:
            logger.warning(f"Cannot interpolate gap at edge: {gap_seq}")
            skipped_edge_cases += len(gap_seq)
            continue
        
        # Load bracketing rasters
        with rasterio.open(date_to_file[before_date]) as src:
            raster_before = src.read(1).astype(np.float32)
            raster_before[raster_before == src.nodata] = np.nan
        
        with rasterio.open(date_to_file[after_date]) as src:
            raster_after = src.read(1).astype(np.float32)
            raster_after[raster_after == src.nodata] = np.nan
        
        # Interpolate each day in the gap
        for i, date_str in enumerate(gap_seq, start=1):
            # Perform interpolation
            interpolated = interpolate_pixel_by_pixel(
                raster_before, raster_after, gap_length, i
            )
            
            # Create output filename
            output_filename = f"{prefix}{date_str}{suffix}"
            output_path = interpolated_dir / output_filename
            
            # Save raster
            template_path = date_to_file[before_date]
            if create_interpolated_raster(template_path, interpolated, output_path, logger):
                successfully_interpolated += 1
                
                # Track interpolation
                interpolated_dates.append({
                    "date": date_str,
                    "gap_length": gap_length,
                    "position_in_gap": i,
                    "before_date": before_date,
                    "after_date": after_date
                })
                
                # Create visualization
                viz_path = viz_dir / f"{prefix}{date_str}_interpolated.png"
                create_visualization(output_path, viz_path, variable, unit_system, date_str, logger)
                
                logger.debug(f"Interpolated {date_str} (gap position {i}/{gap_length})")
    
    processing_end = datetime.now()
    total_duration = (processing_end - processing_start).total_seconds()
    
    # Save interpolated dates to Excel
    if interpolated_dates:
        excel_path = output_dir / "interpolated_dates.xlsx"
        df = pd.DataFrame(interpolated_dates)
        df['date_formatted'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date')
        df.to_excel(excel_path, index=False)
        logger.info(f"Saved interpolated dates to: {excel_path}")
    
    # Methodology description
    methodology = """
Pixel-by-Pixel Linear Interpolation:

For each missing date within a gap sequence, the value of each pixel is
calculated using linear interpolation between the corresponding pixels
in the preceding and following valid rasters.

Formula:
  pixel(t) = pixel(t0) + (pixel(t1) - pixel(t0)) * (t - t0) / (t1 - t0)

Where:
  t0 = date of last valid raster before the gap
  t1 = date of first valid raster after the gap
  t  = date being interpolated

This method:
  - Preserves spatial variability at each time step
  - Creates smooth temporal transitions
  - Handles multi-day gaps consistently
  - Maintains NoData pixels from source rasters
"""
    
    # Compile metadata
    metadata = {
        "processing_start_time": processing_start.isoformat(),
        "processing_end_time": processing_end.isoformat(),
        "total_processing_duration_seconds": total_duration,
        "input_directory": str(input_dir),
        "output_directory": str(interpolated_dir),
        "max_gap_days": max_gap,
        "interpolation_method": "Pixel-by-pixel linear interpolation",
        "methodology_description": methodology,
        "total_missing_dates": total_missing,
        "successfully_interpolated": successfully_interpolated,
        "skipped_large_gap": skipped_large_gap,
        "skipped_edge_cases": skipped_edge_cases,
        "interpolated_dates": interpolated_dates,
        "skipped_gaps": skipped_gaps
    }
    
    # Write metadata
    metadata_path = output_dir / f"gapfill_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    write_gapfill_metadata(metadata_path, metadata)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Gap-Fill Complete")
    logger.info("=" * 60)
    logger.info(f"Total missing dates: {total_missing}")
    logger.info(f"Successfully interpolated: {successfully_interpolated}")
    logger.info(f"Skipped (gap > {max_gap} days): {skipped_large_gap}")
    logger.info(f"Skipped (edge cases): {skipped_edge_cases}")
    logger.info(f"Total processing time: {total_duration:.2f} seconds")
    
    print(f"\n✓ Gap-fill complete!")
    print(f"  Interpolated rasters: {interpolated_dir}")
    print(f"  Visualizations: {viz_dir}")
    print(f"  Excel log: {output_dir / 'interpolated_dates.xlsx'}")
    print(f"  Metadata: {metadata_path}.txt")
    
    if skipped_gaps:
        print(f"\n⚠ WARNING: {len(skipped_gaps)} gap(s) exceeded the {max_gap}-day threshold")
        print("  Review the metadata file for details on these gaps.")


if __name__ == "__main__":
    main()
