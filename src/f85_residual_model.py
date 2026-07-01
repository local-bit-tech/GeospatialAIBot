"""Predict expected real-world driving speed from road context alone, then
flag segments where actual driving behaviour exceeds that expectation.

Unlike train_model.py (which distills the hand-weighted speed_safety_score
formula and is somewhat circular), this fits a genuinely independent
target: F85thPercentileSpeed is measured TomTom probe data, not derived
from any of the predictors used here (road_class, country, urban_flag,
RoadLength_km -- notably NOT SpeedLimit). The residual (actual - expected)
is a data-driven complement to the rule-based score: segments where real
drivers go faster than their road's own context predicts, independent of
what the posted limit says.
"""

import geopandas as gpd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CONTEXT_FEATURES = ["road_class", "country", "urban_flag", "RoadLength_km"]
TARGET_COLUMN = "F85thPercentileSpeed"


def load_data(path="outputs/segments_scored.geojson"):
    """Load the scored segments, dropping rows missing the target or context features."""
    gdf = gpd.read_file(path)
    return gdf.dropna(subset=CONTEXT_FEATURES + [TARGET_COLUMN])


def build_pipeline(random_state=42):
    """Build a one-hot-encoding + RandomForestRegressor pipeline over the context features.

    min_samples_leaf=20 regularises the forest: RoadLength_km is continuous
    and high-cardinality, so an unrestricted RandomForest (min_samples_leaf=1)
    memorises per-segment noise on the training split and generalises no
    better than chance (test R^2 ~0.01). Requiring at least 20 samples per
    leaf forces splits onto the real, coarser signal in road_class/country/
    urban_flag instead, raising held-out R^2 to ~0.28.
    """
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["road_class", "country"]),
        ("num", "passthrough", ["urban_flag", "RoadLength_km"]),
    ])
    model = RandomForestRegressor(random_state=random_state, min_samples_leaf=20)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train(test_size=0.2, random_state=42):
    """Fit the pipeline on a country-stratified split; return it with the held-out set."""
    gdf = load_data()
    X = gdf[CONTEXT_FEATURES]
    y = gdf[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=gdf["country"]
    )
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline, (X_test, y_test)


def evaluate(pipeline, X_test, y_test):
    """Print held-out R^2/MAE (km/h) for the expected-speed model."""
    preds = pipeline.predict(X_test)
    print(f"R^2: {r2_score(y_test, preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f} km/h")


def compute_residuals(pipeline, gdf=None):
    """Add expected_speed and speed_residual (actual - expected) columns to gdf.

    A positive residual means the segment's actual 85th-percentile speed is
    higher than its road context alone would predict -- an outlier signal
    independent of the posted SpeedLimit and the rule-based speed_safety_score.
    """
    if gdf is None:
        gdf = load_data()
    gdf = gdf.copy()
    gdf["expected_speed"] = pipeline.predict(gdf[CONTEXT_FEATURES])
    gdf["speed_residual"] = gdf[TARGET_COLUMN] - gdf["expected_speed"]
    return gdf


RESULTS_COLUMNS = [
    "rank", "segment_id", "country", "road_class", "SpeedLimit",
    "F85thPercentileSpeed", "expected_speed", "speed_residual",
]


def save_results(gdf, path="outputs/f85_residual_results.csv"):
    """Write every scored segment to a CSV, ranked by speed_residual (highest first).

    Unlike a top-N preview, this covers all segments so the full ranking
    can be handed to reviewers or joined back to the map/GIS outputs.
    """
    ranked = gdf.sort_values("speed_residual", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    ranked[RESULTS_COLUMNS].to_csv(path, index=False)
    return path


if __name__ == "__main__":
    pipeline, (X_test, y_test) = train()
    evaluate(pipeline, X_test, y_test)

    gdf = compute_residuals(pipeline)
    out_path = save_results(gdf)

    print(f"\nSaved full ranked results for all {len(gdf)} segments to {out_path}")
    print("\nspeed_residual summary (actual - expected, km/h):")
    print(gdf["speed_residual"].describe().round(2).to_string())

    print("\nBy country:")
    print(gdf.groupby("country")["speed_residual"].describe().round(2).to_string())

    print("\nBy road class:")
    print(gdf.groupby("road_class")["speed_residual"].describe().round(2).to_string())
