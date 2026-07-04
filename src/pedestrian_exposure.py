"""Pedestrian-exposure signal sampled from WorldPop population-density rasters.

WorldPop (hub.worldpop.org) publishes free, no-auth-required 100m gridded
population-count/density GeoTIFFs per country (CC BY 4.0). This module
samples the mean density within a buffer of each road segment as a direct
proxy for "how many people are near this road" -- a more direct signal than
urban_flag (binary) or helmet-wearing rate (a motorcyclist-exposure proxy,
not a pedestrian one). See docs/methodology.md for the full rationale and
its resolution limitations.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from .utils import METRIC_CRS

# Buffer radius (metres) around each segment's line geometry used to sample
# population density -- approximates the pedestrian catchment of the road,
# not just its footprint. Matches WorldPop's native 100m cell size, since a
# much smaller buffer would just be re-sampling a single pixel differently.
POP_BUFFER_METRES = 100

# Per-country path to a WorldPop population-density GeoTIFF, downloaded
# manually (see README Setup) -- not committed, not fetched automatically.
WORLDPOP_RASTER_PATHS = {
    "India": "data/external/worldpop/ind_pop_density_2020.tif",
    "Thailand": "data/external/worldpop/tha_pop_density_2020.tif",
}


def sample_population_density(gdf, raster_paths=WORLDPOP_RASTER_PATHS, buffer_m=POP_BUFFER_METRES):
    """Return a Series of mean WorldPop density within buffer_m of each segment.

    Processes each country in gdf against its own raster and metric CRS
    (reprojecting via METRIC_CRS, the same dict used for the helmet-layer
    spatial join in utils.py). A country with no raster configured, or
    whose raster file isn't present locally, gets NaN for all its segments
    -- never 0, since "unknown exposure" and "zero exposure" are different
    claims, and the rest of the pipeline must keep working without the
    (large, manually-downloaded) real rasters present.
    """
    result = pd.Series(np.nan, index=gdf.index, dtype=float)
    for country, raster_path in raster_paths.items():
        mask = gdf["country"] == country
        if not mask.any():
            continue
        try:
            metric_crs = METRIC_CRS[country]
            buffered = gdf.loc[mask].to_crs(metric_crs).geometry.buffer(buffer_m)
            buffered_4326 = buffered.to_crs("EPSG:4326")
            stats = zonal_stats(buffered_4326, raster_path, stats=["mean"], nodata=-99999)
            # zonal_stats reports None (not NaN) when a buffer has no
            # overlapping raster cells at all; coerce so the assignment
            # doesn't upcast this float64 column to object dtype.
            result.loc[mask] = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
        except (FileNotFoundError, rasterio.errors.RasterioIOError):
            # Raster not downloaded locally yet -- leave these segments NaN
            # rather than crashing the whole pipeline.
            continue
    return result


def compute_pedestrian_exposure(gdf, raster_paths=WORLDPOP_RASTER_PATHS, buffer_m=POP_BUFFER_METRES):
    """Add pop_density_raw (mean WorldPop density) and pop_density_norm (0-1 min-max) to gdf.

    NaNs in pop_density_raw are filled with the column mean before scaling
    so a handful of missing rasters/segments don't distort the min-max
    range for everyone else. If the entire column is NaN (no rasters
    available at all), pop_density_norm stays NaN rather than fabricating
    a value.
    """
    gdf = gdf.copy()
    gdf["pop_density_raw"] = sample_population_density(gdf, raster_paths, buffer_m)

    filled = gdf["pop_density_raw"].fillna(gdf["pop_density_raw"].mean())
    if filled.isna().all():
        gdf["pop_density_norm"] = np.nan
        return gdf

    lo, hi = filled.min(), filled.max()
    span = hi - lo
    gdf["pop_density_norm"] = (filled - lo) / span if span > 0 else 0.0
    return gdf
