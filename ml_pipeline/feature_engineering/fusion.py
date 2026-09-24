"""
Multi-source data fusion: combines precipitation, terrain, Sentinel-derived,
and observed river-level features into a single flat feature vector.

River Feature Hierarchy:
- PRIMARY: Brahmaputra at Tezpur / Ganeshghat (regional mainstream flood context)
- SECONDARY: Jia Bharali at N.T. Road Crossing (local flash-flood signal)

Model Compatibility:
- Preserves existing XGBoost model features (`river_level`, `river_level_change_*`,
  `distance_to_danger_level`) using the Jia Bharali data upon which the current model
  was trained, preventing semantic skew without forced retraining.
- Exposes independent `primary_*` (Brahmaputra) and `secondary_*` (Jia Bharali) features
  for the API, UI, and future model retraining.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.core.config import settings
from ml_pipeline.data_ingestion.india_wris import (
    fetch_primary_level_timeseries,
    fetch_secondary_level_timeseries,
)
from ml_pipeline.data_ingestion.precipitation import (
    compute_accumulation_features,
    fetch_imerg_timeseries,
)
from ml_pipeline.data_ingestion.river_levels import (
    compute_threshold_features,
    get_station_thresholds,
)
from ml_pipeline.data_ingestion.sentinel import (
    fetch_landcover_summary,
    fetch_surface_water_fraction,
)
from ml_pipeline.data_ingestion.terrain import fetch_terrain_features
from ml_pipeline.feature_engineering.river_level_features import compute_river_level_features


def build_feature_vector(as_of: datetime | None = None, use_mock: bool | None = None) -> dict:
    """
    Returns one flat dict of features for the pilot region, ready to
    feed into either the physics-informed risk score or the XGBoost
    model. `as_of` defaults to now; pass a historical timestamp when
    building the training dataset from compiled flood events.
    """
    as_of = as_of or datetime.now(timezone.utc)

    precip_df = fetch_imerg_timeseries(end_time=as_of, use_mock=use_mock)
    rainfall_features = compute_accumulation_features(precip_df)

    terrain_features = fetch_terrain_features(use_mock=use_mock)
    water_fraction = fetch_surface_water_fraction(end_time=as_of, use_mock=use_mock)
    landcover = fetch_landcover_summary(use_mock=use_mock)

    # -----------------------------------------------------------------------
    # 1. Primary River Gauge: Brahmaputra at Tezpur
    # -----------------------------------------------------------------------
    primary_series, primary_times = fetch_primary_level_timeseries(use_mock=use_mock)
    primary_river_features = compute_river_level_features(primary_series, primary_times)

    primary_observed_level = None
    if primary_series is not None and len(primary_series) > 0:
        primary_observed_level = float(primary_series.iloc[-1])

    primary_thresh = get_station_thresholds(settings.PRIMARY_GAUGE_LOCATION)
    primary_dl = float(primary_thresh["danger_level_m"]) if primary_thresh else settings.PRIMARY_DANGER_LEVEL_M
    primary_hfl = float(primary_thresh["hfl_m"]) if primary_thresh else settings.PRIMARY_HFL_M
    primary_tf = compute_threshold_features(primary_observed_level, primary_dl, primary_hfl)

    primary_dist_dl = (primary_dl - primary_observed_level) if primary_observed_level is not None else None

    # -----------------------------------------------------------------------
    # 2. Secondary River Gauge: Jia Bharali at N.T. Road Crossing
    # -----------------------------------------------------------------------
    sec_series, sec_times = fetch_secondary_level_timeseries(use_mock=use_mock)
    sec_river_features = compute_river_level_features(sec_series, sec_times)

    sec_observed_level = None
    if sec_series is not None and len(sec_series) > 0:
        sec_observed_level = float(sec_series.iloc[-1])

    sec_thresh = get_station_thresholds(settings.SECONDARY_GAUGE_LOCATION)
    sec_dl = float(sec_thresh["danger_level_m"]) if sec_thresh else settings.SECONDARY_DANGER_LEVEL_M
    sec_hfl = float(sec_thresh["hfl_m"]) if sec_thresh else settings.SECONDARY_HFL_M
    sec_tf = compute_threshold_features(sec_observed_level, sec_dl, sec_hfl)

    sec_dist_dl = (sec_dl - sec_observed_level) if sec_observed_level is not None else None

    features = {
        "as_of": as_of.isoformat(),
        **rainfall_features,
        **terrain_features,
        "surface_water_fraction": water_fraction,
        **{f"landcover_{k}": v for k, v in landcover.items()},

        # --- PRIMARY Gauge Features (Brahmaputra at Tezpur) ---
        "primary_river_level": primary_observed_level,
        "primary_river_level_change_1h": primary_river_features.get("level_change_1h"),
        "primary_river_level_change_3h": primary_river_features.get("level_change_3h"),
        "primary_river_level_change_6h": primary_river_features.get("level_change_6h"),
        "primary_rate_of_rise": primary_river_features.get("rate_of_rise"),
        "primary_distance_to_danger_level": primary_dist_dl,
        "primary_danger_ratio": primary_tf.get("danger_ratio"),
        "primary_hfl_ratio": primary_tf.get("hfl_ratio"),
        **{f"primary_{k}": v for k, v in primary_river_features.items() if k not in ["rate_of_rise"]},

        # --- SECONDARY Gauge Features (Jia Bharali at N.T. Road Xing) ---
        "secondary_river_level": sec_observed_level,
        "secondary_river_level_change_1h": sec_river_features.get("level_change_1h"),
        "secondary_river_level_change_3h": sec_river_features.get("level_change_3h"),
        "secondary_river_level_change_6h": sec_river_features.get("level_change_6h"),
        "secondary_rate_of_rise": sec_river_features.get("rate_of_rise"),
        "secondary_distance_to_danger_level": sec_dist_dl,
        "secondary_danger_ratio": sec_tf.get("danger_ratio"),
        "secondary_hfl_ratio": sec_tf.get("hfl_ratio"),
        **{f"secondary_{k}": v for k, v in sec_river_features.items() if k not in ["rate_of_rise"]},

        # --- MODEL COMPATIBILITY (Existing XGBoost Contract) ---
        # The existing XGBoost model was trained on historical Jia Bharali gauge records.
        # Preserve existing feature semantics by routing the secondary (Jia Bharali) features
        # into the legacy slots, preventing feature skew without retraining.
        "river_level": sec_observed_level,
        "river_level_change_1h": sec_river_features.get("level_change_1h"),
        "river_level_change_3h": sec_river_features.get("level_change_3h"),
        "river_level_change_6h": sec_river_features.get("level_change_6h"),
        "distance_to_danger_level": sec_dist_dl,
        "danger_ratio": sec_tf.get("danger_ratio"),
        "hfl_ratio": sec_tf.get("hfl_ratio"),
        **sec_river_features,
    }
    return features


def feature_vector_to_dataframe(features: dict) -> pd.DataFrame:
    """Single-row DataFrame, convenient for model input / logging."""
    return pd.DataFrame([features])
