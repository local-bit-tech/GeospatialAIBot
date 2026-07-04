# Methodology

## Overview

This project scores road segments in India (Maharashtra) and Thailand on how
misaligned their posted speed limits are with Safe System principles and
vulnerable road user (VRU) exposure. It does **not** train a predictive
model — the Speed Safety Score is a transparent, hand-weighted formula built
from four risk components, chosen so the reasoning behind every segment's
score is auditable. See `src/train_model.py` for an intentionally unfinished
scaffold if a future iteration wants to fit an actual predictive model.

## Data sources and schema reconciliation

The two source files were exported at different times by different teams and
**do not share a schema**. Thailand's columns match the Agilysis data-guide
PDF (`RoadClass`, `SampleSizeTotal`, no `DISSOLVE_ID`/`UrbanPC`); Maharashtra's
match the field list in the challenge brief (`DISSOLVE_ID`, `class`,
`UrbanPC`, `ExcludeFromSpeedSPI`, `Sample_Size_Total`). `src/utils.py:harmonize_schema`
reconciles both onto a common column set:

| Unified column | India (Maharashtra) source | Thailand source |
|---|---|---|
| `segment_id` | `DISSOLVE_ID` | `OBJECTID` |
| `road_class` | `class` (== `RoadClass`, verified identical) | `RoadClass` |
| `Sample_Size_Total` | `Sample_Size_Total` | `SampleSizeTotal` |
| `RoadLength_km` | `Shape_Length / 1000` | `Shape_Length / 1000` |
| `UrbanPC` | native field | derived: 1.0 if `LandUse == 'URBAN'` else 0.0 |

The two files are **disjoint geographic datasets** (different countries), so
the "join match rate" check the brief asks for returns ~0% overlap by
design — they are concatenated, not joined on a shared key.

`RoadLength` is explicitly marked "ignore, use Shape Length" in the Agilysis
data guide. Both fields were verified to encode the same length (Shape_Length
in metres, RoadLength in km, related by an exact ×1000 factor), so
`RoadLength_km` is derived from `Shape_Length` throughout.

`RankedPercentile` is a 0–1 fraction in the Maharashtra file but a 0–100 scale
in the Thailand file; it is rescaled to 0–100 for both before any
cross-country comparison.

## Known data anomaly

Some segments report `F85thPercentileSpeed` more than 20 km/h above
`SpeedLimit` while `PercentOverLimit` reads 0. These are flagged in
notebook 01 but **not excluded** — the underlying cause (sparse sampling,
sample taken at a different point than the posted limit applies, etc.) isn't
resolvable from the data alone.

## Reliability filter

A segment is "reliable" if (where the column exists) `ExcludeFromSpeedSPI ==
0`, `AnalysisStatus == 'Valid'`, and `Sample_Size_Total >= 1000`. Everything
else is "low confidence" and excluded from scoring, but kept in
`data/processed/segments_low_confidence.geojson` and shown on the map as a
separate, off-by-default "Insufficient data" layer.

## Feature definitions

- **`speed_gap`** = `max(F85thPercentileSpeed - SpeedLimit, 0)`, min-max
  normalised to `speed_gap_norm`. Only positive gaps are a risk signal — a
  segment where traffic travels *below* the limit gets 0, not a negative
  score.
- **`road_mismatch`** compares the posted `SpeedLimit` against the Safe
  System speed range for its road class. It is only positive when the
  posted limit **exceeds** the class ceiling (motorway 110, trunk 90,
  primary 70, secondary 60, tertiary 50, residential 30, living_street 20,
  unclassified 40), clipped at 1.0. **Note:** the brief's worked example
  claims a secondary road posted at 55 km/h is "5 km/h above" the 60 km/h
  secondary ceiling — that's arithmetically backwards (55 < 60). Per the
  stated definition, that case correctly scores `road_mismatch = 0`; this
  was verified against the 1,148 real Maharashtra segments posted at 55
  km/h on secondary roads.
- **`urban_flag`** = 1.0 if `UrbanPC > 0.5` else 0.0.
- **`vru_exposure`** blends three signals, each contributing at a
  "full-strength" weight (`src/features.py:VRU_EXPOSURE_WEIGHTS`):
  `urban_flag` (0.24), a region-level low-helmet-compliance risk signal
  (0.36): `1 - helmet_wearing_rate`, spatially joined from the
  helmet-wearing survey layers in `Archive/*.gpkg` and min-max normalised,
  and `pop_density_norm` (0.40): a min-max normalised population-density
  proxy sampled from [WorldPop](https://hub.worldpop.org/project/list)
  100m gridded population rasters within a 100m buffer of each segment
  (see `src/pedestrian_exposure.py`). Population density gets the single
  highest weight because it is the most direct pedestrian-exposure signal
  of the three — a measurement of people near the road rather than a
  correlate of it.

  **Resolution differs sharply between countries** for the helmet-wearing
  term — 4 zones for Maharashtra (Mumbai, Pune, Maharashtra Rural,
  Maharashtra Urban) vs. 77 provinces for Thailand — so it remains a coarse
  proxy, especially for India. The WorldPop term is itself a proxy too:
  "people living/working within 100m of this segment" is not the same as
  "pedestrians walking or crossing this road," and its 100m native cell
  size sets a hard floor on spatial precision — same honesty-about-limits
  spirit as the helmet-resolution caveat above.

  Any component may be unavailable (no helmet layer supplied, or WorldPop
  rasters not downloaded locally) — `_blend` in `src/features.py` drops
  that component and renormalises the remaining weights to sum to 1. This
  is deliberately constructed so that dropping `pop_density_norm` recovers
  the *exact* original `0.40*urban_flag + 0.60*helmet_risk` formula (0.24
  and 0.36 renormalised over their 0.60 sum give back 0.40 and 0.60) — a
  segment without WorldPop coverage is not silently under-weighted, it
  scores identically to the pre-pedestrian-exposure formula. With neither
  helmet data nor population density supplied, the function falls back to
  `urban_flag` alone, as before.
- **`bio_risk`** = a fatality-probability lookup on `SpeedLimit` (≤20: 0.05,
  ≤30: 0.10, ≤40: 0.30, ≤50: 0.80, ≤60: 0.90, >60: 1.00), multiplied by
  `vru_exposure`.
- **`confidence_weight`** = 0.5 if `RoadLength_km > 10` else 1.0, down-weighting
  long segments where a sparse point sample may not represent the whole
  segment.
- **`mapillary_url`** is built from the centroid of the two endpoint
  coordinates in `StreetImageLink` (the data guide describes this field as
  endpoint lon/lat pairs, not a sorted bounding box — but centroid averaging
  produces the correct midpoint either way).
- **`recommended_speed_limit`** interpolates within the segment's road-class
  Safe System range (the same `motorway`/`trunk`/.../`unclassified` ranges
  used by `road_mismatch`), pulled toward the class **minimum** as
  `vru_exposure` rises toward 1 and toward the class **maximum** as it falls
  toward 0:
  `class_min + (class_max - class_min) * (1 - vru_exposure)`, rounded to the
  nearest 10 km/h (how limits are actually posted). Segments whose
  `road_class` has no defined Safe System range get `NaN`.
- **`speed_limit_gap`** = posted `SpeedLimit` minus `recommended_speed_limit`.
  Positive means the posted limit sits above what Safe System principles
  suggest for that segment's vulnerable-road-user exposure — this is the
  field to hand a transport ministry official for "what should this
  segment's limit actually be." Across the 14,546 reliable segments: mean
  gap +12.8 km/h, 73.2% of segments posted above their recommendation.

## Speed Safety Score

```
speed_safety_score = round(
    (0.30 * speed_gap_norm
   + 0.25 * road_mismatch
   + 0.30 * bio_risk
   + 0.15 * vru_exposure)
  * confidence_weight * 100,
  1
)
```

Risk tiers: High risk ≥ 70, Medium risk ≥ 40, Low risk < 40, Insufficient
data for segments excluded by the reliability filter.

## Validation results (current run)

*Note: the numbers below were re-run with the current pipeline (including
`compute_pedestrian_exposure`) but with 0% WorldPop coverage — no raster has
been downloaded to `data/external/worldpop/` yet, so `vru_exposure` is
running its graceful two-component fallback (`_blend` in `src/features.py`
renormalises away the missing `pop_density_norm` term for every segment).
That's why these numbers are close to, but not byte-identical to, the
pre-pedestrian-exposure figures from an earlier run. Once real WorldPop
rasters are in place, re-run notebooks 02–04 and refresh this section again.*

- **Correlation with `RankedPercentile`: -0.195** (both countries
  individually: India -0.12, Thailand -0.23). This is **negative**, contrary
  to the brief's expectation of a positive correlation. Root cause:
  `RankedPercentile` ranks segments by **travel volume share** (per the
  Agilysis data guide: "allows presentation of roads by percentage
  traffic"), not by safety risk. A busy, well-engineered motorway can carry
  a large share of national travel while being comparatively safe (wide
  lanes, good geometry, lower `bio_risk` at higher design speeds), while a
  low-traffic rural secondary road posted above its Safe System range scores
  high on risk but negligible on travel share. The two metrics are simply
  measuring different things, and the negative correlation is being
  reported here rather than masked.
- **Risk tier distribution is lopsided**: 14,160 Low risk vs. 386 Medium
  risk vs. **zero** High risk segments among the 14,546 reliable segments
  (max score observed: 53.3, never reaching the 70 threshold). The fixed
  0/40/70 thresholds, taken literally from the brief, do not suit this
  formula's actual achievable range — `bio_risk` rarely reaches its
  theoretical maximum because `vru_exposure` (which it's multiplied by) is
  itself bounded by the 40/60 blend and rarely approaches 1.0. This is
  reported as-is rather than re-calibrated; a follow-up iteration may want
  to either rescale the score to use its full observed range or set
  data-driven thresholds (e.g. tier cutoffs at observed percentiles) instead
  of fixed absolute values.
- **Sensitivity analysis**: 84 valid weight combinations (±0.10 per weight,
  0.05 steps, summing to 1.0) were tested against the baseline top-20%
  highest-scoring segments. Average overlap: **90.2%** — above the 70%
  threshold, so the score is robust to reasonable weight re-calibration.

## Map performance trade-off

Embedding all ~70,000 segments (reliable + low-confidence) in one Folium
HTML file at full geometry/attribute fidelity produced a 61MB file —
impractical for GitHub Pages and slow to pan/zoom in a browser. The
low-confidence ("Insufficient data") layer is therefore: (a) simplified more
aggressively (0.0015° tolerance vs. 0.0003° for scored segments), (b) given a
minimal tooltip with no popup, and (c) off by default via the layer control.
This brought the file to ~30MB while keeping every segment inspectable on
demand.

## Limitations

- `SpeedLimit` and the original `LandUse` classification are estimates from
  secondary sources (per the Agilysis data guide) and should be treated as
  approximate; `UrbanPC` is the more reliable numeric field, used here.
  These are limitations of the upstream data, not of this pipeline.
- The VRU exposure proxy now includes a WorldPop population-density signal
  (see `vru_exposure` above) alongside the regional helmet-wearing surveys,
  but still relies on a 100m-buffer density estimate rather than actual
  segment-level pedestrian/cyclist counts or footfall — school/market
  proximity (e.g. Overture Places POI density) and graded urbanization
  (GHS-SMOD) remain future work; see README's Future Enhancements.
- This is a rule-based score, not a validated predictive model. See
  `src/train_model.py` for a deliberately unfinished scaffold to take this
  further with real outcome data.
