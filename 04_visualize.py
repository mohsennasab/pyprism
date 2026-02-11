"""
PRISM Data Processing Toolkit - Script 4: Visualization and Statistics
======================================================================
Creates comprehensive visualizations and statistics from PRISM data.

Visualizations:
1. Time Series Line Chart: Daily values with mean, max, min over watershed
2. Cumulative Chart: Cumulative values over entire period of record
3. Custom Period Cumulative: Cumulative values for user-specified date range
4. Spatial Distribution Map: Average spatial distribution over selected period

Author: HydroMohsen
Date: January 2025

Usage:
    python 04_visualize.py --input_dir ./clipped_data/clipped \
        --output_dir ./visualizations --variable ppt --unit_system SI \
        --custom_start 2020-06-01 --custom_end 2020-08-31 \
        --boundary ./watershed.geojson
"""

import os
import sys
import json
import logging
import argparse
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from config import PRISMVariable, UnitSystem, PRISM_UNITS, convert_units


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging."""
    logger = logging.getLogger('prism_viz')
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


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract datetime from filename."""
    match = re.search(r'(\d{8})', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    return None


def get_unit_label(variable: PRISMVariable, unit_system: UnitSystem) -> str:
    """Get appropriate unit label."""
    return PRISM_UNITS[variable][unit_system.value]


def get_variable_label(variable: PRISMVariable, unit_system: UnitSystem) -> str:
    """Get full variable label with units."""
    labels = {
        PRISMVariable.PPT: "Precipitation",
        PRISMVariable.TMAX: "Maximum Temperature",
        PRISMVariable.TMIN: "Minimum Temperature",
        PRISMVariable.TMEAN: "Mean Temperature",
        PRISMVariable.TDMEAN: "Mean Dew Point",
        PRISMVariable.VPDMIN: "Minimum VPD",
        PRISMVariable.VPDMAX: "Maximum VPD"
    }
    return f"{labels[variable]} ({get_unit_label(variable, unit_system)})"


def load_raster_stats(raster_path: Path, variable: PRISMVariable, 
                      unit_system: UnitSystem) -> Dict[str, float]:
    """Load a raster and compute statistics."""
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        
        if unit_system == UnitSystem.US_CUSTOMARY:
            data = convert_units(data, variable, to_us=True)
        
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return {'mean': np.nan, 'min': np.nan, 'max': np.nan, 'std': np.nan, 'valid_pixels': 0}
        
        return {
            'mean': np.nanmean(valid_data),
            'min': np.nanmin(valid_data),
            'max': np.nanmax(valid_data),
            'std': np.nanstd(valid_data),
            'valid_pixels': len(valid_data)
        }


def load_all_raster_data(input_dir: Path, variable: PRISMVariable,
                         unit_system: UnitSystem, logger: logging.Logger) -> pd.DataFrame:
    """Load all rasters and compute time series statistics."""
    raster_files = sorted(input_dir.glob("*.tif"))
    
    records = []
    for raster_path in tqdm(raster_files, desc="Loading rasters"):
        date = extract_date_from_filename(raster_path.name)
        if date is None:
            continue
        
        stats = load_raster_stats(raster_path, variable, unit_system)
        stats['date'] = date
        stats['filename'] = raster_path.name
        records.append(stats)
    
    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} rasters")
    if len(df) > 0:
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_boundary(boundary_path: Path, logger: logging.Logger) -> Optional[gpd.GeoDataFrame]:
    """Load boundary from GeoJSON or zipped Shapefile."""
    try:
        if boundary_path.suffix.lower() == '.zip':
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(boundary_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                shp_files = list(Path(temp_dir).rglob("*.shp"))
                if not shp_files:
                    return None
                gdf = gpd.read_file(shp_files[0])
        elif boundary_path.suffix.lower() in ['.geojson', '.json']:
            gdf = gpd.read_file(boundary_path)
        elif boundary_path.suffix.lower() == '.shp':
            gdf = gpd.read_file(boundary_path)
        else:
            return None
        return gdf
    except Exception as e:
        logger.error(f"Failed to load boundary: {e}")
        return None


def compute_average_spatial(input_dir: Path, start_date: Optional[datetime],
                            end_date: Optional[datetime], variable: PRISMVariable,
                            unit_system: UnitSystem, logger: logging.Logger) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """Compute average spatial distribution over a date range."""
    raster_files = sorted(input_dir.glob("*.tif"))
    
    # Filter by date range
    filtered_files = []
    for f in raster_files:
        date = extract_date_from_filename(f.name)
        if date is None:
            continue
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        filtered_files.append(f)
    
    if not filtered_files:
        return None, None
    
    logger.info(f"Computing average from {len(filtered_files)} rasters")
    
    # Load first raster for metadata
    with rasterio.open(filtered_files[0]) as src:
        meta = src.meta.copy()
        shape = (src.height, src.width)
        nodata = src.nodata
    
    # Accumulate
    sum_array = np.zeros(shape, dtype=np.float64)
    count_array = np.zeros(shape, dtype=np.int32)
    
    for raster_path in tqdm(filtered_files, desc="Computing spatial average"):
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32)
            if nodata is not None:
                valid_mask = data != nodata
            else:
                valid_mask = ~np.isnan(data)
            
            sum_array[valid_mask] += data[valid_mask]
            count_array[valid_mask] += 1
    
    # Compute average
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_array = np.where(count_array > 0, sum_array / count_array, np.nan)
    
    # Convert units
    if unit_system == UnitSystem.US_CUSTOMARY:
        avg_array = convert_units(avg_array, variable, to_us=True)
    
    return avg_array, meta


# ============================================================================
# VISUALIZATION 1: Time Series with Mean, Max, Min
# ============================================================================
def create_timeseries_plot(df: pd.DataFrame, variable: PRISMVariable,
                           unit_system: UnitSystem, output_path: Path,
                           logger: logging.Logger):
    """
    Create time series line chart showing daily mean, max, and min.
    
    Y-axis: Variable value
    X-axis: Date
    Lines: Mean (solid), Max (dashed), Min (dotted)
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot lines
    ax.fill_between(df['date'], df['min'], df['max'], alpha=0.3, color='steelblue',
                    label='Range (Min-Max)')
    ax.plot(df['date'], df['mean'], color='steelblue', linewidth=1.5,
            label='Watershed Mean')
    ax.plot(df['date'], df['max'], color='darkblue', linewidth=0.8, linestyle='--',
            alpha=0.7, label='Daily Maximum')
    ax.plot(df['date'], df['min'], color='lightblue', linewidth=0.8, linestyle=':',
            alpha=0.7, label='Daily Minimum')
    
    # Formatting
    unit = get_unit_label(variable, unit_system)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(get_variable_label(variable, unit_system), fontsize=12)
    ax.set_title(f'PRISM {variable.value.upper()} Time Series\nWatershed Statistics',
                fontsize=14, fontweight='bold')
    
    # Date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved time series plot: {output_path}")


# ============================================================================
# VISUALIZATION 2: Cumulative Over Period of Record
# ============================================================================
def create_cumulative_plot(df: pd.DataFrame, variable: PRISMVariable,
                           unit_system: UnitSystem, output_path: Path,
                           logger: logging.Logger):
    """
    Create cumulative chart over entire period of record.
    
    For precipitation: cumulative sum
    For temperature/VPD: cumulative mean (running average)
    """
    df_sorted = df.sort_values('date').copy()
    
    # Calculate cumulative
    if variable == PRISMVariable.PPT:
        df_sorted['cumulative'] = df_sorted['mean'].cumsum()
        ylabel = f'Cumulative {get_variable_label(variable, unit_system)}'
        title_suffix = 'Cumulative'
    else:
        # For temperature, use running mean
        df_sorted['cumulative'] = df_sorted['mean'].expanding().mean()
        ylabel = f'Running Mean {get_variable_label(variable, unit_system)}'
        title_suffix = 'Running Average'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df_sorted['date'], df_sorted['cumulative'], color='darkgreen', linewidth=2)
    ax.fill_between(df_sorted['date'], 0, df_sorted['cumulative'], alpha=0.3, color='green')
    
    # Add final value annotation
    final_val = df_sorted['cumulative'].iloc[-1]
    ax.annotate(f'Total: {final_val:.2f}', 
                xy=(df_sorted['date'].iloc[-1], final_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.8))
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'PRISM {variable.value.upper()} - {title_suffix}\nPeriod of Record',
                fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved cumulative plot: {output_path}")


# ============================================================================
# VISUALIZATION 3: Custom Period Cumulative
# ============================================================================
def create_custom_period_cumulative(df: pd.DataFrame, start_date: datetime,
                                    end_date: datetime, variable: PRISMVariable,
                                    unit_system: UnitSystem, output_path: Path,
                                    logger: logging.Logger):
    """Create cumulative chart for user-specified date range."""
    # Filter to date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_period = df[mask].sort_values('date').copy()
    
    if len(df_period) == 0:
        logger.warning(f"No data found for period {start_date} to {end_date}")
        return
    
    # Calculate cumulative
    if variable == PRISMVariable.PPT:
        df_period['cumulative'] = df_period['mean'].cumsum()
        ylabel = f'Cumulative {get_variable_label(variable, unit_system)}'
    else:
        df_period['cumulative'] = df_period['mean'].expanding().mean()
        ylabel = f'Running Mean {get_variable_label(variable, unit_system)}'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_period['date'], df_period['cumulative'], color='purple', linewidth=2)
    ax.fill_between(df_period['date'], 0, df_period['cumulative'], alpha=0.3, color='purple')
    
    final_val = df_period['cumulative'].iloc[-1]
    ax.annotate(f'Total: {final_val:.2f}', 
                xy=(df_period['date'].iloc[-1], final_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple', alpha=0.8))
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'PRISM {variable.value.upper()} - Custom Period\n'
                f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved custom period cumulative plot: {output_path}")


# ============================================================================
# VISUALIZATION 4: Spatial Distribution Map
# ============================================================================
def create_spatial_map(avg_array: np.ndarray, meta: dict, 
                       boundary: Optional[gpd.GeoDataFrame],
                       start_date: Optional[datetime], end_date: Optional[datetime],
                       variable: PRISMVariable, unit_system: UnitSystem,
                       output_path: Path, logger: logging.Logger):
    """Create spatial distribution map showing average values."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Choose colormap
    if variable == PRISMVariable.PPT:
        cmap = 'Blues'
    elif variable in [PRISMVariable.TMAX, PRISMVariable.TMIN, PRISMVariable.TMEAN]:
        cmap = 'RdYlBu_r'
    else:
        cmap = 'viridis'
    
    # Mask invalid values
    masked_data = np.ma.masked_invalid(avg_array)
    
    # Get extent from transform
    transform = meta['transform']
    extent = [
        transform.c,
        transform.c + transform.a * avg_array.shape[1],
        transform.f + transform.e * avg_array.shape[0],
        transform.f
    ]
    
    # Plot raster
    im = ax.imshow(masked_data, cmap=cmap, extent=extent, origin='upper')
    
    # Add boundary outline if provided
    if boundary is not None:
        try:
            boundary_reproj = boundary.to_crs(meta['crs'])
            boundary_reproj.boundary.plot(ax=ax, color='black', linewidth=2)
        except Exception as e:
            logger.warning(f"Could not overlay boundary: {e}")
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(get_variable_label(variable, unit_system), fontsize=11)
    
    # Title
    if start_date and end_date:
        period_str = f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
    else:
        period_str = 'Full Period of Record'
    
    ax.set_title(f'PRISM {variable.value.upper()} - Spatial Distribution\n'
                f'Average over {period_str}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting', fontsize=11)
    ax.set_ylabel('Northing', fontsize=11)
    
    # Add statistics annotation
    valid_data = avg_array[~np.isnan(avg_array)]
    if len(valid_data) > 0:
        stats_text = (f'Mean: {np.mean(valid_data):.2f}\n'
                     f'Min: {np.min(valid_data):.2f}\n'
                     f'Max: {np.max(valid_data):.2f}\n'
                     f'Std: {np.std(valid_data):.2f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved spatial map: {output_path}")


# ============================================================================
# STATISTICS EXPORT
# ============================================================================
def export_statistics(df: pd.DataFrame, variable: PRISMVariable,
                      unit_system: UnitSystem, output_path: Path,
                      logger: logging.Logger):
    """Export comprehensive statistics to Excel."""
    unit = get_unit_label(variable, unit_system)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Daily statistics
        daily_df = df[['date', 'mean', 'min', 'max', 'std', 'valid_pixels']].copy()
        daily_df.columns = ['Date', f'Mean ({unit})', f'Min ({unit})', 
                           f'Max ({unit})', f'Std Dev ({unit})', 'Valid Pixels']
        daily_df.to_excel(writer, sheet_name='Daily Statistics', index=False)
        
        # Summary statistics
        summary_data = {
            'Statistic': [
                'Period Start', 'Period End', 'Total Days',
                'Overall Mean', 'Overall Min', 'Overall Max',
                'Overall Std Dev', 'Total (for ppt)'
            ],
            'Value': [
                df['date'].min().strftime('%Y-%m-%d'),
                df['date'].max().strftime('%Y-%m-%d'),
                len(df),
                df['mean'].mean(),
                df['min'].min(),
                df['max'].max(),
                df['mean'].std(),
                df['mean'].sum() if variable == PRISMVariable.PPT else 'N/A'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Monthly statistics
        df_monthly = df.copy()
        df_monthly['year_month'] = df_monthly['date'].dt.to_period('M')
        
        if variable == PRISMVariable.PPT:
            monthly_agg = df_monthly.groupby('year_month').agg({
                'mean': ['sum', 'mean', 'count'],
                'max': 'max',
                'min': 'min'
            })
        else:
            monthly_agg = df_monthly.groupby('year_month').agg({
                'mean': ['mean', 'std', 'count'],
                'max': 'max',
                'min': 'min'
            })
        
        monthly_agg.columns = ['_'.join(col).strip() for col in monthly_agg.columns.values]
        monthly_agg = monthly_agg.reset_index()
        monthly_agg['year_month'] = monthly_agg['year_month'].astype(str)
        monthly_agg.to_excel(writer, sheet_name='Monthly Statistics', index=False)
        
        # Annual statistics
        df_annual = df.copy()
        df_annual['year'] = df_annual['date'].dt.year
        
        if variable == PRISMVariable.PPT:
            annual_agg = df_annual.groupby('year').agg({
                'mean': ['sum', 'mean', 'count'],
                'max': 'max',
                'min': 'min'
            })
        else:
            annual_agg = df_annual.groupby('year').agg({
                'mean': ['mean', 'std', 'count'],
                'max': 'max',
                'min': 'min'
            })
        
        annual_agg.columns = ['_'.join(col).strip() for col in annual_agg.columns.values]
        annual_agg = annual_agg.reset_index()
        annual_agg.to_excel(writer, sheet_name='Annual Statistics', index=False)
    
    logger.info(f"Saved statistics: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create visualizations and statistics from PRISM data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Visualizations Created:
1. Time series line chart (mean, max, min over watershed)
2. Cumulative chart over period of record
3. Custom period cumulative (if dates specified)
4. Spatial distribution map

Examples:
    # Basic usage
    python 04_visualize.py --input_dir ./clipped_data/clipped \\
        --output_dir ./visualizations --variable ppt --unit_system SI

    # With custom period and boundary
    python 04_visualize.py --input_dir ./clipped_data/clipped \\
        --output_dir ./visualizations --variable ppt --unit_system US \\
        --custom_start 2020-06-01 --custom_end 2020-08-31 \\
        --boundary ./watershed.geojson
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing clipped PRISM GeoTIFFs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--variable', type=str, required=True,
                        choices=['ppt', 'tmax', 'tmin', 'tmean', 'tdmean', 'vpdmin', 'vpdmax'],
                        help='PRISM variable')
    parser.add_argument('--unit_system', type=str, default='SI',
                        choices=['SI', 'US'],
                        help='Unit system (SI or US Customary)')
    parser.add_argument('--custom_start', type=str, default=None,
                        help='Custom period start date (YYYY-MM-DD)')
    parser.add_argument('--custom_end', type=str, default=None,
                        help='Custom period end date (YYYY-MM-DD)')
    parser.add_argument('--boundary', type=str, default=None,
                        help='Watershed boundary for map overlay (optional)')
    parser.add_argument('--spatial_start', type=str, default=None,
                        help='Start date for spatial map (YYYY-MM-DD, default: full period)')
    parser.add_argument('--spatial_end', type=str, default=None,
                        help='End date for spatial map (YYYY-MM-DD, default: full period)')
    
    args = parser.parse_args()
    
    # Parse arguments
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    variable = PRISMVariable(args.variable)
    unit_system = UnitSystem(args.unit_system)
    
    # Parse dates
    custom_start = datetime.strptime(args.custom_start, "%Y-%m-%d") if args.custom_start else None
    custom_end = datetime.strptime(args.custom_end, "%Y-%m-%d") if args.custom_end else None
    spatial_start = datetime.strptime(args.spatial_start, "%Y-%m-%d") if args.spatial_start else None
    spatial_end = datetime.strptime(args.spatial_end, "%Y-%m-%d") if args.spatial_end else None
    
    # Validate
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directories
    plots_dir = output_dir / "plots"
    stats_dir = output_dir / "statistics"
    logs_dir = output_dir / "logs"
    
    for d in [plots_dir, stats_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = logs_dir / f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("PRISM Visualization Started")
    logger.info("=" * 60)
    logger.info(f"Variable: {variable.value}")
    logger.info(f"Unit system: {unit_system.value}")
    
    # Load boundary if provided
    boundary = None
    if args.boundary:
        boundary = load_boundary(Path(args.boundary), logger)
    
    # Load all raster statistics
    df = load_all_raster_data(input_dir, variable, unit_system, logger)
    
    if len(df) == 0:
        logger.error("No valid raster data found")
        print("Error: No valid raster data found")
        sys.exit(1)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Time Series Plot
    ts_path = plots_dir / f"01_timeseries_{variable.value}.png"
    create_timeseries_plot(df, variable, unit_system, ts_path, logger)
    
    # 2. Cumulative Plot
    cum_path = plots_dir / f"02_cumulative_{variable.value}.png"
    create_cumulative_plot(df, variable, unit_system, cum_path, logger)
    
    # 3. Custom Period Cumulative (if dates provided)
    if custom_start and custom_end:
        custom_path = plots_dir / f"03_custom_period_{variable.value}.png"
        create_custom_period_cumulative(df, custom_start, custom_end, variable, 
                                        unit_system, custom_path, logger)
    
    # 4. Spatial Distribution Map
    avg_array, meta = compute_average_spatial(input_dir, spatial_start, spatial_end,
                                              variable, unit_system, logger)
    if avg_array is not None:
        spatial_path = plots_dir / f"04_spatial_distribution_{variable.value}.png"
        create_spatial_map(avg_array, meta, boundary, spatial_start, spatial_end,
                          variable, unit_system, spatial_path, logger)
    
    # Export statistics
    stats_path = stats_dir / f"statistics_{variable.value}.xlsx"
    export_statistics(df, variable, unit_system, stats_path, logger)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Visualization Complete")
    logger.info("=" * 60)
    
    print(f"\n✓ Visualization complete!")
    print(f"  Plots: {plots_dir}")
    print(f"  Statistics: {stats_dir}")


if __name__ == "__main__":
    main()
