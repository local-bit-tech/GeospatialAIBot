"""Tests for the vru_exposure blend in src/features.py, including the
regression guard that omitting pop_density_norm must reproduce the
original 0.40*urban_flag + 0.60*helmet_risk formula exactly.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from src.features import VRU_EXPOSURE_WEIGHTS, _blend, compute_vru_exposure


def _segments():
    return gpd.GeoDataFrame(
        {"country": ["India", "India", "India"], "urban_flag": [1.0, 0.0, 1.0]},
        geometry=[
            LineString([(75.0, 20.0), (75.01, 20.01)]),
            LineString([(76.0, 21.0), (76.01, 21.01)]),
            LineString([(75.5, 20.5), (75.51, 20.51)]),
        ],
        crs="EPSG:4326",
    )


def _zones():
    """Two zones with different helmet_rate; zone 1 covers segments 0 and 2, zone 2 covers segment 1."""
    return gpd.GeoDataFrame(
        {"helmet_rate": [0.9, 0.3]},
        geometry=[box(74.5, 19.5, 75.6, 20.6), box(75.9, 20.9, 76.6, 21.6)],
        crs="EPSG:4326",
    )


def test_no_helmet_no_pop_density_falls_back_to_urban_flag_alone():
    gdf = _segments()
    result = compute_vru_exposure(gdf)
    assert result["vru_exposure"].tolist() == pytest.approx(gdf["urban_flag"].tolist())


def test_omitting_pop_density_reproduces_original_two_component_formula():
    """Regression guard: segments without WorldPop coverage must score
    identically to the pre-pedestrian-exposure 0.40/0.60 formula."""
    gdf = _segments()
    result = compute_vru_exposure(gdf, helmet_layers={"India": _zones()})

    # low_helmet_risk = 1 - helmet_rate -> [0.1, 0.7, 0.1], min-max normalised
    # across the three segments -> [0.0, 1.0, 0.0].
    expected = [
        0.40 * 1.0 + 0.60 * 0.0,
        0.40 * 0.0 + 0.60 * 1.0,
        0.40 * 1.0 + 0.60 * 0.0,
    ]
    assert result["vru_exposure"].tolist() == pytest.approx(expected)


def test_all_three_components_blend_at_full_strength_weights():
    gdf = _segments()
    pop_density_norm = pd.Series([0.2, 0.8, 0.5], index=gdf.index)
    result = compute_vru_exposure(gdf, helmet_layers={"India": _zones()}, pop_density_norm=pop_density_norm)

    w = VRU_EXPOSURE_WEIGHTS
    helmet_norm = [0.0, 1.0, 0.0]
    urban = gdf["urban_flag"].tolist()
    expected = [
        w["urban_flag"] * urban[i] + w["helmet_risk"] * helmet_norm[i] + w["pop_density"] * pop_density_norm.iloc[i]
        for i in range(3)
    ]
    assert result["vru_exposure"].tolist() == pytest.approx(expected)


def test_blend_renormalises_when_a_component_is_missing():
    series_a = pd.Series([1.0, 0.0])
    series_b = pd.Series([0.0, 1.0])
    result = _blend({"a": (series_a, 0.3), "b": (None, 0.7)})
    assert result.tolist() == pytest.approx(series_a.tolist())


def test_blend_raises_if_all_components_missing():
    with pytest.raises(ValueError):
        _blend({"a": (None, 0.5), "b": (None, 0.5)})


def test_blend_treats_all_nan_series_as_absent_not_zero():
    """Regression guard: a component passed as a real (non-None) Series that
    is entirely NaN -- e.g. pop_density_norm when no WorldPop raster is
    downloaded for any country -- must not NaN-propagate through the whole
    blend. It should renormalise away exactly like an omitted component."""
    urban = pd.Series([1.0, 0.0, 1.0])
    all_nan_pop_density = pd.Series([float("nan"), float("nan"), float("nan")])
    result = _blend({"urban_flag": (urban, 0.24), "pop_density": (all_nan_pop_density, 0.40)})
    assert result.tolist() == pytest.approx(urban.tolist())
    assert not result.isna().any()


def test_blend_renormalises_per_row_for_partial_nan_series():
    """A component that's only NaN on some rows should renormalise per-row,
    not drag the whole column to NaN or drop the component everywhere."""
    a = pd.Series([1.0, 1.0])
    b = pd.Series([0.0, float("nan")])
    result = _blend({"a": (a, 0.5), "b": (b, 0.5)})
    # row 0: both present -> 0.5*1 + 0.5*0 = 0.5
    # row 1: b is NaN -> falls back to a alone -> 1.0
    assert result.tolist() == pytest.approx([0.5, 1.0])


def test_compute_vru_exposure_all_nan_pop_density_falls_back_gracefully():
    """End-to-end regression for the bug: an all-NaN pop_density_norm column
    (real coverage 0%, not None) must not collapse vru_exposure to NaN."""
    gdf = _segments()
    all_nan = pd.Series([float("nan")] * len(gdf), index=gdf.index)
    result = compute_vru_exposure(gdf, helmet_layers={"India": _zones()}, pop_density_norm=all_nan)
    assert not result["vru_exposure"].isna().any()

    expected = [
        0.40 * 1.0 + 0.60 * 0.0,
        0.40 * 0.0 + 0.60 * 1.0,
        0.40 * 1.0 + 0.60 * 0.0,
    ]
    assert result["vru_exposure"].tolist() == pytest.approx(expected)
