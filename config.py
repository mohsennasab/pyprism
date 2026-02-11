"""
PRISM Data Processing Toolkit - Configuration Module
=====================================================
Shared configuration settings for all PRISM processing scripts.

Includes the FFRD Standard Projection definition used throughout the toolkit.

Author: HydroMohsen
Date: January 2025
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

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

FFRD_PROJECTION_PROJ4 = (
    "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 "
    "+x_0=0 +y_0=0 +datum=NAD83 +units=ft +no_defs"
)

FFRD_PROJECTION_NAME = "USA_Contiguous_Albers_Equal_Area_Conic_FFRD"


class UnitSystem(Enum):
    """Unit system selection for outputs."""
    SI = "SI"  # Metric (mm, °C, hPa)
    US_CUSTOMARY = "US"  # US Customary (inches, °F, inHg)


class PRISMVariable(Enum):
    """Available PRISM climate variables."""
    PPT = "ppt"       # Total precipitation (rain + melted snow)
    TMAX = "tmax"     # Maximum temperature
    TMIN = "tmin"     # Minimum temperature
    TMEAN = "tmean"   # Mean temperature (derived: (tmax+tmin)/2)
    TDMEAN = "tdmean" # Mean dew point temperature
    VPDMIN = "vpdmin" # Minimum vapor pressure deficit
    VPDMAX = "vpdmax" # Maximum vapor pressure deficit


class PRISMResolution(Enum):
    """Available PRISM spatial resolutions."""
    RES_800M = "800m"  # 800 meter (~30 arc-second)
    RES_4KM = "4km"    # 4 kilometer (~2.5 arc-minute)


class PRISMDatasetType(Enum):
    """PRISM dataset types (temporal consistency vs best estimate)."""
    AN = "AN"  # All Networks - best estimate possible
    LT = "LT"  # Long Term - focused on temporal consistency


# PRISM original coordinate system
PRISM_CRS = "EPSG:4269"  # NAD83

# PRISM original units (as documented)
PRISM_UNITS = {
    PRISMVariable.PPT: {"SI": "mm", "US": "inches", "conversion_to_US": 1/25.4},
    PRISMVariable.TMAX: {"SI": "°C", "US": "°F", "conversion_to_US": lambda c: c * 9/5 + 32},
    PRISMVariable.TMIN: {"SI": "°C", "US": "°F", "conversion_to_US": lambda c: c * 9/5 + 32},
    PRISMVariable.TMEAN: {"SI": "°C", "US": "°F", "conversion_to_US": lambda c: c * 9/5 + 32},
    PRISMVariable.TDMEAN: {"SI": "°C", "US": "°F", "conversion_to_US": lambda c: c * 9/5 + 32},
    PRISMVariable.VPDMIN: {"SI": "hPa", "US": "inHg", "conversion_to_US": 1/33.8639},
    PRISMVariable.VPDMAX: {"SI": "hPa", "US": "inHg", "conversion_to_US": 1/33.8639},
}

# PRISM data availability
PRISM_DAILY_START_YEAR = 1981  # Daily data starts from 1981
PRISM_MONTHLY_START_YEAR = 1895  # Monthly data starts from 1895

# Web service base URLs (as per PRISM documentation - updated March 2025)
PRISM_WEB_SERVICE_BASE = "https://services.nacse.org/prism/data/get"
PRISM_RELEASE_DATE_SERVICE = "https://services.nacse.org/prism/data/get/releaseDate"

# PRISM time zone: Data uses Pacific Time for updates
PRISM_TIMEZONE = "America/Los_Angeles"

# PRISM Day definition: 1200 UTC-1200 UTC (7 AM-7AM EST)
PRISM_DAY_DEFINITION = "1200 UTC to 1200 UTC (PRISM Day)"


@dataclass
class PRISMMetadata:
    """Metadata for PRISM datasets."""
    variable: PRISMVariable
    resolution: PRISMResolution
    dataset_type: PRISMDatasetType
    original_crs: str = PRISM_CRS
    original_units: str = ""
    time_zone: str = PRISM_TIMEZONE
    day_definition: str = PRISM_DAY_DEFINITION
    data_source: str = "Oregon State University PRISM Climate Group"
    website: str = "https://prism.oregonstate.edu/"
    
    def __post_init__(self):
        self.original_units = PRISM_UNITS[self.variable]["SI"]


def get_unit_label(variable: PRISMVariable, unit_system: UnitSystem) -> str:
    """Get the appropriate unit label for a variable and unit system."""
    return PRISM_UNITS[variable][unit_system.value]


def convert_units(value, variable: PRISMVariable, to_us: bool = False):
    """
    Convert PRISM data values between SI and US Customary units.
    
    PRISM data is originally in SI units:
    - Precipitation: mm
    - Temperature: °C
    - Vapor pressure deficit: hPa
    
    Parameters
    ----------
    value : float or array-like
        Value(s) to convert
    variable : PRISMVariable
        The PRISM variable type
    to_us : bool
        If True, convert from SI to US Customary
        If False, return value unchanged (already in SI)
    
    Returns
    -------
    Converted value(s)
    """
    if not to_us:
        return value
    
    conversion = PRISM_UNITS[variable]["conversion_to_US"]
    if callable(conversion):
        return conversion(value)
    else:
        return value * conversion
