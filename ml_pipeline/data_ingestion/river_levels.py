"""
River danger-level / historical-HFL reference data.

Sourced from a user-provided official danger-level table (river-wise
gauge stations across Assam). This is STATIC reference data (danger
levels, historical highest-flood-level), not a live feed — it becomes
directly useful once real water-level readings are obtained (see
project README: Assam WRD data-request pathway), at which point
"current_level - danger_level" becomes a strong, simple risk feature.

For now, it's used to (a) surface the two Sonitpur gauge stations'
thresholds for reference/explainability, and (b) anchor the two
historical Sonitpur flood events used to bootstrap model training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DANGER_LEVELS_PATH = Path("data/reference/river_danger_levels_assam.csv")


def load_danger_levels() -> pd.DataFrame:
    if not DANGER_LEVELS_PATH.exists():
        return pd.DataFrame(
            columns=["river", "gauge_location", "district", "danger_level_m", "hfl_m", "hfl_date"]
        )
    return pd.read_csv(DANGER_LEVELS_PATH, parse_dates=["hfl_date"])


def get_sonitpur_gauge_stations() -> pd.DataFrame:
    """Returns just the Sonitpur-district gauge stations (Tezpur, Jia Bharali)."""
    df = load_danger_levels()
    return df[df["district"] == "Sonitpur"].reset_index(drop=True)


def compute_danger_level_ratio(current_level_m: float, gauge_location: str) -> float | None:
    """
    Returns current_level / danger_level for the named gauge (>1 means
    already above danger level). Returns None if the gauge or current
    reading isn't available — callers should treat that as "unknown",
    not "safe".
    """
    stations = get_sonitpur_gauge_stations()
    match = stations[stations["gauge_location"] == gauge_location]
    if match.empty or current_level_m is None:
        return None
    danger_level = float(match.iloc[0]["danger_level_m"])
    return current_level_m / danger_level


def get_station_thresholds(gauge_location: str) -> dict | None:
    """
    Looks up a gauge station by `gauge_location` in the danger levels CSV.
    Returns a dict with keys: `river`, `gauge_location`, `district`, `danger_level_m`, `hfl_m`, `hfl_date`
    Returns `None` if the station is not found.
    """
    df = load_danger_levels()
    match = df[df["gauge_location"] == gauge_location]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def compute_threshold_features(level_m: float | None, danger_level_m: float, hfl_m: float | None = None) -> dict:
    """
    Computes threshold-based features (danger_ratio, level_margin_m, hfl_ratio, hfl_margin_m).
    Returns None for any metric when input is unavailable. NEVER interprets None as 0.
    """
    features = {
        "danger_ratio": None,
        "level_margin_m": None,
        "hfl_ratio": None,
        "hfl_margin_m": None,
    }

    if level_m is not None:
        features["danger_ratio"] = float(level_m / danger_level_m)
        features["level_margin_m"] = float(level_m - danger_level_m)

        if hfl_m is not None:
            features["hfl_ratio"] = float(level_m / hfl_m)
            features["hfl_margin_m"] = float(level_m - hfl_m)

    return features
