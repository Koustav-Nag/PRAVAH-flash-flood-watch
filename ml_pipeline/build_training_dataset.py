"""
Builds the ML-ready training dataset (sonitpur_flood_training.csv)
from raw rainfall data + terrain features + flood targets.

This script fuses:
  1. raw_rainfall.csv       — real GPM-IMERG rainfall (12,864 rows)
  2. Terrain features       — from GEE SRTM/HydroSHEDS (fetched once, reused)
  3. Flood targets          — computed from flood_events.csv (leakage-safe)

Run with:
    python -m ml_pipeline.build_training_dataset

Output:
    data/historical/ml_ready/sonitpur_flood_training.csv

After this runs, train the model with:
    python -m ml_pipeline.train --target flood_next_24h
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ml_pipeline.data_ingestion.historical_events import (
    FLOOD_EVENTS_PATH,
    ML_READY_PATH,
    RAW_RAINFALL_PATH,
    load_flood_events,
    load_raw_rainfall,
)
from ml_pipeline.data_ingestion.target_builder import build_targets_for_series
from ml_pipeline.data_ingestion.terrain import fetch_terrain_features

# --- Constants ---
# Accumulation windows in hours → number of 30-min steps
WINDOWS = {
    "rainfall_1h":  2,
    "rainfall_3h":  6,
    "rainfall_6h":  12,
    "rainfall_12h": 24,
    "rainfall_24h": 48,
    "rainfall_48h": 96,
    "rainfall_72h": 144,
}


def compute_rainfall_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling accumulation features for one pixel/station group.
    Uses trailing windows (no lookahead) — leakage-safe by design.
    """
    group = group.sort_values("timestamp").copy()
    for col, steps in WINDOWS.items():
        group[col] = group["rainfall_mm"].rolling(window=steps, min_periods=1).sum()

    # Antecedent saturation: 7-day accumulation prior to each timestamp
    group["antecedent_rainfall_7d_mm"] = group["rainfall_mm"].rolling(
        window=336, min_periods=1
    ).sum()

    # Rainfall persistence: fraction of last 6h with non-trivial rain
    group["rainfall_persistence_6h_frac"] = (
        (group["rainfall_mm"] > 0.25)
        .rolling(window=12, min_periods=1)
        .mean()
    )

    return group


def main():
    # --- Load raw rainfall ---
    if not RAW_RAINFALL_PATH.exists():
        logger.error(
            f"{RAW_RAINFALL_PATH} not found. "
            "Run: python -m ml_pipeline.fetch_historical_rainfall"
        )
        sys.exit(1)

    rainfall = load_raw_rainfall()
    if rainfall.empty:
        logger.error(
            "raw_rainfall.csv is empty. "
            "Run: python -m ml_pipeline.fetch_historical_rainfall"
        )
        sys.exit(1)

    logger.info(f"Loaded {len(rainfall)} rainfall rows.")

    # --- Compute rolling accumulation features per station ---
    logger.info("Computing rolling accumulation features...")
    rainfall = (
        rainfall
        .groupby("station_id", group_keys=False)
        .apply(compute_rainfall_features)
        .reset_index(drop=True)
    )

    # --- Fetch terrain features (once for the region) ---
    logger.info("Fetching terrain features from GEE (SRTM/HydroSHEDS)...")
    terrain = fetch_terrain_features()
    logger.info(f"Terrain features: {terrain}")

    # Attach terrain features as static columns (same value for all rows —
    # this is a region-level prototype; per-grid-cell values come later)
    for key, val in terrain.items():
        rainfall[key] = val

    # --- Assign event_id to each timestamp ---
    logger.info("Assigning event_id labels...")
    events = load_flood_events()
    if events.empty:
        logger.error(
            "flood_events.csv is empty. "
            "Run the flood events setup step first."
        )
        sys.exit(1)

    def assign_event_id(row):
        """Assign event_id if timestamp falls within a flood event window."""
        ts = row["timestamp"]
        for _, event in events.iterrows():
            start = pd.to_datetime(event["flood_start"])
            end = pd.to_datetime(event["flood_end"]) if pd.notna(event.get("flood_end")) else start
            # Window: 14 days before to 7 days after
            if (start - pd.Timedelta(days=14)) <= ts <= (end + pd.Timedelta(days=7)):
                return event["event_id"]
        return None

    rainfall["event_id"] = rainfall.apply(assign_event_id, axis=1)
    rainfall["district"] = "Sonitpur"

    event_rows = rainfall["event_id"].notna().sum()
    logger.info(f"Assigned event_id to {event_rows}/{len(rainfall)} rows.")

    # --- Build flood targets (leakage-safe: looks strictly forward) ---
    logger.info("Building flood targets...")
    targets = build_targets_for_series(rainfall["timestamp"], events)
    rainfall = pd.concat([rainfall.reset_index(drop=True), targets], axis=1)

    # --- Placeholder columns (not yet sourced) ---
    # These are in the ML-ready schema but require additional data sources.
    # Left as None/NaN — XGBoost handles NaN natively, so training still
    # works; these columns will be filled once the data sources are integrated.
    for col in ["river_level", "river_level_change_1h", "river_level_change_3h",
                "river_level_change_6h", "distance_to_danger_level",
                "soil_moisture", "temperature", "vegetation_index", "land_cover"]:
        if col not in rainfall.columns:
            rainfall[col] = None

    # --- Select and order final columns to match ML_READY_COLUMNS schema ---
    output_cols = [
        "timestamp", "latitude", "longitude", "district", "event_id",
        "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_12h",
        "rainfall_24h", "rainfall_48h", "rainfall_72h",
        "river_level", "river_level_change_1h", "river_level_change_3h",
        "river_level_change_6h", "distance_to_danger_level",
        "soil_moisture", "temperature", "vegetation_index",
        "elevation", "slope", "aspect", "flow_accumulation",
        "distance_to_river", "drainage_density", "land_cover",
        "flood_next_6h", "flood_next_12h", "flood_next_24h",
        "data_source", "quality_flag",
    ]

    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in rainfall.columns]
    final_df = rainfall[output_cols].copy()

    # --- Write output ---
    ML_READY_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(ML_READY_PATH, index=False)

    flood_rows = (final_df["flood_next_24h"] == 1).sum()
    logger.info(
        f"Wrote {len(final_df)} rows to {ML_READY_PATH}. "
        f"Flood-positive (24h): {flood_rows} rows ({flood_rows/len(final_df)*100:.1f}%). "
        f"Non-flood: {len(final_df) - flood_rows} rows."
    )
    logger.info("Done. Now run: python -m ml_pipeline.train --target flood_next_24h")


if __name__ == "__main__":
    main()
