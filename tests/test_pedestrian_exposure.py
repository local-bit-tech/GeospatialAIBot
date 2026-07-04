"""Tests for src/pedestrian_exposure.py against a tiny synthetic GeoTIFF -- no
real WorldPop download needed.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from src.pedestrian_exposure import compute_pedestrian_exposure, sample_population_density


@pytest.fixture
def test_raster(tmp_path):
    """A 20x20 constant-value (50.0) GeoTIFF covering roughly lon 76.99-77.01, lat 19.99-20.01."""
    path = tmp_path / "india_pop.tif"
    width = height = 20
    transform = from_origin(76.99, 20.01, 0.001, 0.001)
    data = np.full((height, width), 50.0, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-99999,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _segment(coords, country):
    return gpd.GeoDataFrame(
        {"country": [country]}, geometry=[LineString(coords)], crs="EPSG:4326"
    )


def test_sample_population_density_inside_raster(test_raster):
    gdf = _segment([(77.0, 20.0), (77.002, 20.002)], "India")
    result = sample_population_density(gdf, raster_paths={"India": test_raster})
    assert result.iloc[0] == pytest.approx(50.0)


def test_sample_population_density_outside_raster_extent(test_raster):
    gdf = _segment([(90.0, 25.0), (90.002, 25.002)], "India")
    result = sample_population_density(gdf, raster_paths={"India": test_raster})
    assert pd.isna(result.iloc[0])


def test_sample_population_density_missing_file_does_not_raise(tmp_path):
    gdf = _segment([(77.0, 20.0), (77.002, 20.002)], "India")
    missing_path = str(tmp_path / "does_not_exist.tif")
    result = sample_population_density(gdf, raster_paths={"India": missing_path})
    assert pd.isna(result.iloc[0])


def test_sample_population_density_country_with_no_raster_configured(test_raster):
    gdf = _segment([(77.0, 20.0), (77.002, 20.002)], "Thailand")
    result = sample_population_density(gdf, raster_paths={"India": test_raster})
    assert pd.isna(result.iloc[0])


def test_compute_pedestrian_exposure_adds_raw_and_norm_columns(test_raster):
    gdf = gpd.GeoDataFrame(
        {"country": ["India", "Thailand"]},
        geometry=[
            LineString([(77.0, 20.0), (77.002, 20.002)]),
            LineString([(100.0, 14.0), (100.002, 14.002)]),
        ],
        crs="EPSG:4326",
    )
    result = compute_pedestrian_exposure(gdf, raster_paths={"India": test_raster})

    assert result["pop_density_raw"].iloc[0] == pytest.approx(50.0)
    assert pd.isna(result["pop_density_raw"].iloc[1])
    # NaN raw values are filled with the column mean before scaling, so the
    # Thailand segment (no raster) still gets a defined norm, not a crash.
    assert not result["pop_density_norm"].isna().any()


def test_compute_pedestrian_exposure_all_missing_stays_nan():
    gdf = _segment([(77.0, 20.0), (77.002, 20.002)], "India")
    result = compute_pedestrian_exposure(gdf, raster_paths={})
    assert pd.isna(result["pop_density_raw"].iloc[0])
    assert pd.isna(result["pop_density_norm"].iloc[0])
