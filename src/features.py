"""Feature engineering for the Speed Safety Score: gap, mismatch, exposure, risk."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import METRIC_CRS, assign_zone_attribute

# Safe System posted-speed ranges (km/h) by Overture road class.
SAFE_SYSTEM_SPEED_RANGES = {
    "motorway": (80, 110),
    "trunk": (60, 90),
    "primary": (50, 70),
    "secondary": (40, 60),
    "tertiary": (30, 50),
    "residential": (10, 30),
    "living_street": (5, 20),
    "unclassified": (20, 40),
}

# Fatality-probability lookup for a pedestrian/cyclist struck at a given posted speed.
BIO_RISK_BANDS = [
    (20, 0.05),
    (30, 0.10),
    (40, 0.30),
    (50, 0.80),
    (60, 0.90),
]
BIO_RISK_ABOVE_60 = 1.00


def compute_speed_gap(gdf):
    """Add speed_gap (F85 minus limit, clipped at 0) and its 0-1 normalised version."""
    gap = gdf["F85thPercentileSpeed"] - gdf["SpeedLimit"]
    gdf["speed_gap"] = gap.clip(lower=0)
    scaler = MinMaxScaler()
    gdf["speed_gap_norm"] = scaler.fit_transform(gdf[["speed_gap"]].fillna(0))
    return gdf


def compute_road_mismatch(gdf):
    """Add road_mismatch: how far the posted limit exceeds the class's Safe System max, clipped at 1."""
    class_max = gdf["road_class"].map(lambda c: SAFE_SYSTEM_SPEED_RANGES.get(c, (np.nan, np.nan))[1])
    # A road is only "mismatched" when the posted limit is strictly above the
    # class ceiling. A limit below the ceiling (e.g. 55 km/h on a secondary
    # road capped at 60) is compliant and scores 0, not a positive value.
    over = (gdf["SpeedLimit"] - class_max).clip(lower=0)
    gdf["road_mismatch"] = (over / class_max).clip(upper=1.0)
    return gdf


def compute_urban_flag(gdf):
    """Add urban_flag: 1.0 where UrbanPC > 0.5, else 0.0."""
    gdf["urban_flag"] = (gdf["UrbanPC"] > 0.5).astype(float)
    return gdf


# Full-strength weights for vru_exposure's three components, used when
# helmet-wearing data and pop_density_norm are both supplied. Chosen so
# that dropping any one component and renormalising the rest by their
# weight sum recovers a clean sub-formula: urban_flag:helmet_risk are kept
# in a fixed 0.40:0.60 ratio (urban_flag alone gets renormalised weight 1.0
# if helmet data is unavailable too), the same proportions the score used
# before pop_density existed -- so a segment with no WorldPop coverage
# scores identically to the pre-pedestrian-exposure formula, not silently
# under-weighted.
VRU_EXPOSURE_WEIGHTS = {
    "urban_flag": 0.24,
    "helmet_risk": 0.36,
    "pop_density": 0.40,
}


def _blend(components):
    """Weighted average of {name: (series, weight)} pairs, skipping series that are None.

    Renormalisation happens per row against whichever components actually
    have a non-NaN value on that row, not just whichever components were
    passed in at all. This matters because a component can be "supplied" as
    a real Series that is nonetheless entirely (or partially) NaN -- e.g.
    pop_density_norm when no WorldPop raster is available for a country --
    and treating "supplied" as "usable" would let a handful of NaNs silently
    propagate through the weighted sum and NaN-out every row's blend,
    instead of gracefully falling back to the remaining components.
    """
    active = {name: (series, weight) for name, (series, weight) in components.items() if series is not None}
    if not active:
        raise ValueError("At least one component must be supplied to _blend")
    values = pd.concat({name: series for name, (series, _) in active.items()}, axis=1)
    weights = pd.Series({name: weight for name, (_, weight) in active.items()})
    valid = values.notna()
    effective_weights = valid.mul(weights, axis=1)
    weight_sums = effective_weights.sum(axis=1)
    weighted = (values.fillna(0) * effective_weights).sum(axis=1)
    return (weighted / weight_sums).where(weight_sums > 0)


def compute_vru_exposure(gdf, helmet_layers=None, pop_density_norm=None):
    """Add vru_exposure: urban_flag + low-helmet-compliance risk + population-density proxy.

    helmet_layers: optional dict of {country: zones_geodataframe} with a
    helmet_rate column (0-1). When supplied, the inverse of the local helmet
    wearing rate (i.e. exposure of unprotected riders) is spatially joined to
    each segment. Resolution differs sharply between countries (4 zones for
    Maharashtra vs 77 provinces for Thailand), so this is a coarse proxy, not
    a precise exposure model.

    pop_density_norm: optional 0-1 normalised population-density Series
    (see src/pedestrian_exposure.py, a WorldPop-derived proxy for how many
    people are near the segment) aligned to gdf's index.

    Either or both may be omitted (e.g. no helmet layer available, or
    WorldPop rasters not downloaded locally) -- see VRU_EXPOSURE_WEIGHTS and
    _blend for how the formula degrades gracefully in that case.
    """
    components = {"urban_flag": (gdf["urban_flag"], VRU_EXPOSURE_WEIGHTS["urban_flag"])}

    if helmet_layers:
        low_helmet_risk = pd.Series(np.nan, index=gdf.index)
        for country, zones in helmet_layers.items():
            country_mask = gdf["country"] == country
            if not country_mask.any():
                continue
            helmet_rate = assign_zone_attribute(gdf.loc[country_mask], zones, METRIC_CRS[country])
            low_helmet_risk.loc[country_mask] = 1 - helmet_rate

        scaler = MinMaxScaler()
        low_helmet_risk_norm = pd.Series(
            scaler.fit_transform(low_helmet_risk.fillna(low_helmet_risk.mean()).to_frame()).flatten(),
            index=gdf.index,
        )
        components["helmet_risk"] = (low_helmet_risk_norm, VRU_EXPOSURE_WEIGHTS["helmet_risk"])

    if pop_density_norm is not None:
        components["pop_density"] = (pop_density_norm, VRU_EXPOSURE_WEIGHTS["pop_density"])

    gdf["vru_exposure"] = _blend(components).clip(0, 1)
    return gdf


def compute_recommended_speed_limit(gdf):
    """Add recommended_speed_limit and speed_limit_gap for each segment.

    recommended_speed_limit interpolates within the segment's road-class
    Safe System range (SAFE_SYSTEM_SPEED_RANGES), pulled toward the class
    minimum as vru_exposure rises toward 1 and toward the class maximum as
    it falls toward 0, rounded to the nearest 10 km/h (how limits are
    actually posted). Segments whose road_class has no defined range get
    NaN. speed_limit_gap is the posted SpeedLimit minus this recommendation
    -- positive means the posted limit is above what Safe System principles
    suggest for that segment's exposure.
    """
    ranges = gdf["road_class"].map(SAFE_SYSTEM_SPEED_RANGES)
    class_min = ranges.map(lambda r: r[0] if isinstance(r, tuple) else np.nan)
    class_max = ranges.map(lambda r: r[1] if isinstance(r, tuple) else np.nan)
    recommended = class_min + (class_max - class_min) * (1 - gdf["vru_exposure"])
    gdf["recommended_speed_limit"] = (recommended / 10).round() * 10
    gdf["speed_limit_gap"] = gdf["SpeedLimit"] - gdf["recommended_speed_limit"]
    return gdf


def _bio_risk_for_speed(speed):
    """Look up the fatality-probability band for a single posted speed limit."""
    if pd.isna(speed):
        return np.nan
    for ceiling, prob in BIO_RISK_BANDS:
        if speed <= ceiling:
            return prob
    return BIO_RISK_ABOVE_60


def compute_bio_risk(gdf):
    """Add bio_risk: speed-based fatality probability multiplied by vru_exposure."""
    fatality_prob = gdf["SpeedLimit"].map(_bio_risk_for_speed)
    gdf["bio_risk"] = fatality_prob * gdf["vru_exposure"]
    return gdf


def compute_confidence_weight(gdf):
    """Add confidence_weight: 0.5 for segments longer than 10km, else 1.0."""
    gdf["confidence_weight"] = np.where(gdf["RoadLength_km"] > 10, 0.5, 1.0)
    return gdf


def compute_mapillary_url(gdf):
    """Add mapillary_url built from the centre point of the StreetImageLink bounding coordinates."""

    def _to_url(value):
        if pd.isna(value):
            return None
        try:
            lon1, lat1, lon2, lat2 = (float(v) for v in str(value).split(","))
        except ValueError:
            return None
        center_lon = (lon1 + lon2) / 2
        center_lat = (lat1 + lat2) / 2
        return f"https://www.mapillary.com/app/?lat={center_lat}&lng={center_lon}&z=16"

    gdf["mapillary_url"] = gdf["StreetImageLink"].map(_to_url)
    return gdf


def engineer_features(gdf, helmet_layers=None, pop_density_norm=None):
    """Run all feature engineering steps on a reliable-segments GeoDataFrame, in order."""
    gdf = compute_speed_gap(gdf)
    gdf = compute_road_mismatch(gdf)
    gdf = compute_urban_flag(gdf)
    gdf = compute_vru_exposure(gdf, helmet_layers, pop_density_norm)
    gdf = compute_recommended_speed_limit(gdf)
    gdf = compute_bio_risk(gdf)
    gdf = compute_confidence_weight(gdf)
    gdf = compute_mapillary_url(gdf)
    return gdf
