"""
Precipitation ingestion via GPM/IMERG (Global Precipitation Measurement,
Integrated Multi-satellitE Retrievals), pulled through Google Earth
Engine, with an offline mock path for development without live GEE
credentials.

GPM/IMERG Late Run gives ~30-minute precipitation estimates globally,
which is what we use to derive the rolling accumulation windows
described in the architecture (15min ... 24h).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import settings
from ml_pipeline.data_ingestion.gee_client import get_region_geometry, is_gee_available

IMERG_COLLECTION = "NASA/GPM_L3/IMERG_V07"

ACCUMULATION_WINDOWS_HOURS = {
    "rainfall_1h": 1,
    "rainfall_3h": 3,
    "rainfall_6h": 6,
    "rainfall_12h": 12,
    "rainfall_24h": 24,
    "rainfall_48h": 48,
    "rainfall_72h": 72,
}


def fetch_imerg_timeseries(
    end_time: datetime | None = None,
    lookback_hours: int = 216,  # 9 days: covers rainfall_72h + 7d antecedent
    use_mock: bool | None = None,
) -> pd.DataFrame:
    """
    Returns a half-hourly precipitation time series (mm/30min) for the
    pilot region, averaged over the region for now (grid-cell-level
    disaggregation happens once this is validated end-to-end).

    Columns: [timestamp, precip_mm]
    """
    end_time = end_time or datetime.utcnow()
    start_time = end_time - timedelta(hours=lookback_hours)

    if use_mock is None:
        use_mock = not is_gee_available()

    if use_mock:
        return _mock_imerg_timeseries(start_time, end_time)

    return _fetch_imerg_from_gee(start_time, end_time)


def _fetch_imerg_from_gee(start_time: datetime, end_time: datetime) -> pd.DataFrame:
    import ee

    region = get_region_geometry()
    collection = (
        ee.ImageCollection(IMERG_COLLECTION)
        .filterDate(start_time.isoformat(), end_time.isoformat())
        .filterBounds(region)
        .select("precipitation")
    )

    def _reduce(image):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=10000, bestEffort=True
        )
        return image.set(
            {
                "timestamp": image.date().format("YYYY-MM-dd'T'HH:mm:ss"),
                "precip_mm": stats.get("precipitation"),
            }
        )

    features = collection.map(_reduce)
    data = features.reduceColumns(
        ee.Reducer.toList(2), ["timestamp", "precip_mm"]
    ).get("list").getInfo()

    df = pd.DataFrame(data, columns=["timestamp", "precip_mm"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _mock_imerg_timeseries(start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Synthetic half-hourly rainfall series with a plausible storm burst,
    for pipeline development / demo without live GEE access. NOT for
    reporting real numbers to judges — clearly label as simulated.
    """
    logger.info("Using MOCK IMERG precipitation data (no GEE credentials).")
    timestamps = pd.date_range(start_time, end_time, freq="30min")
    rng = np.random.default_rng(seed=42)

    baseline = rng.gamma(shape=0.3, scale=1.2, size=len(timestamps))
    # Inject a synthetic convective burst in the back third of the window,
    # since flash-flood demos need a visible spike.
    burst_start = int(len(timestamps) * 0.7)
    burst_len = max(2, int(len(timestamps) * 0.1))
    baseline[burst_start : burst_start + burst_len] += rng.gamma(
        shape=4.0, scale=6.0, size=burst_len
    )

    return pd.DataFrame({"timestamp": timestamps, "precip_mm": baseline})


def compute_accumulation_features(precip_df: pd.DataFrame) -> dict:
    """
    Given a half-hourly precip_df (timestamp, precip_mm) ending at
    "now", compute the rolling accumulation features used across the
    architecture (rain_15min_mm ... rain_24h_mm), plus antecedent
    rainfall and rainfall persistence indicators.
    """
    if precip_df.empty:
        raise ValueError("precip_df is empty — cannot compute accumulation features")

    df = precip_df.sort_values("timestamp").set_index("timestamp")
    now = df.index.max()

    features = {}
    for name, hours in ACCUMULATION_WINDOWS_HOURS.items():
        window_start = now - timedelta(hours=hours)
        features[name] = float(df.loc[df.index > window_start, "precip_mm"].sum())

    # Antecedent rainfall: 7-day accumulation prior to the last 24h,
    # a classic soil-saturation proxy for flash-flood susceptibility.
    antecedent_start = now - timedelta(days=8)
    antecedent_end = now - timedelta(hours=24)
    antecedent_mask = (df.index > antecedent_start) & (df.index <= antecedent_end)
    features["antecedent_rainfall_7d_mm"] = float(df.loc[antecedent_mask, "precip_mm"].sum())

    # Rainfall persistence: fraction of the last 6h with non-trivial rain,
    # since sustained moderate rain over saturated hilly terrain is a
    # known flash-flood driver distinct from a single sharp spike.
    last_6h = df.loc[df.index > (now - timedelta(hours=6))]
    features["rainfall_persistence_6h_frac"] = (
        float((last_6h["precip_mm"] > 0.25).mean()) if not last_6h.empty else 0.0
    )

    # Rainfall anomaly placeholder: current 24h accumulation vs a
    # long-term monthly climatology mean. Requires a climatology
    # baseline dataset — wired in once historical data is compiled.
    features["rainfall_anomaly_ratio"] = None

    return features
