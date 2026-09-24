"""
Temporal features and hydrological projection from observed river-level time series.
These features must only use information available at prediction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_level_lags(level_series: pd.Series | None, timestamps: pd.DatetimeIndex | None) -> dict:
    """
    Computes lagged river levels for 30m, 1h, 3h, 6h, 12h, 24h.
    These features must only use information available at prediction time.
    """
    lags = {
        "level_lag_30m": None,
        "level_lag_1h": None,
        "level_lag_3h": None,
        "level_lag_6h": None,
        "level_lag_12h": None,
        "level_lag_24h": None,
    }

    if level_series is None or timestamps is None or len(level_series) == 0:
        return lags

    idx = pd.to_datetime(timestamps)
    df = pd.DataFrame({"level": level_series.values}, index=idx).sort_index()
    now = pd.to_datetime(df.index[-1])

    lag_offsets = {
        "level_lag_30m": pd.Timedelta(minutes=30),
        "level_lag_1h": pd.Timedelta(hours=1),
        "level_lag_3h": pd.Timedelta(hours=3),
        "level_lag_6h": pd.Timedelta(hours=6),
        "level_lag_12h": pd.Timedelta(hours=12),
        "level_lag_24h": pd.Timedelta(hours=24),
    }

    for key, offset in lag_offsets.items():
        target_time = now - offset
        past_data = df.loc[:target_time]
        if not past_data.empty:
            lags[key] = float(past_data.iloc[-1]["level"])

    return lags


def compute_level_changes(level_series: pd.Series | None, timestamps: pd.DatetimeIndex | None) -> dict:
    """
    Computes level changes over 1h, 3h, 6h windows.
    These features must only use information available at prediction time.
    """
    changes = {
        "level_change_1h": None,
        "level_change_3h": None,
        "level_change_6h": None,
    }

    if level_series is None or timestamps is None or len(level_series) == 0:
        return changes

    idx = pd.to_datetime(timestamps)
    df = pd.DataFrame({"level": level_series.values}, index=idx).sort_index()
    now = pd.to_datetime(df.index[-1])
    current_level = df.iloc[-1]["level"]

    lag_offsets = {
        "level_change_1h": pd.Timedelta(hours=1),
        "level_change_3h": pd.Timedelta(hours=3),
        "level_change_6h": pd.Timedelta(hours=6),
    }

    for key, offset in lag_offsets.items():
        target_time = now - offset
        past_data = df.loc[:target_time]
        if not past_data.empty:
            past_level = past_data.iloc[-1]["level"]
            changes[key] = float(current_level - past_level)

    return changes


def compute_rate_of_rise(level_series: pd.Series | None, timestamps: pd.DatetimeIndex | None, window_hours: int = 1) -> float | None:
    """
    Computes rate of rise in m/hour over the specified window.
    These features must only use information available at prediction time.
    """
    if level_series is None or timestamps is None or len(level_series) == 0:
        return None

    idx = pd.to_datetime(timestamps)
    df = pd.DataFrame({"level": level_series.values}, index=idx).sort_index()
    now = pd.to_datetime(df.index[-1])
    current_level = df.iloc[-1]["level"]

    target_time = now - pd.Timedelta(hours=window_hours)
    past_data = df.loc[:target_time]

    if past_data.empty:
        return None

    past_level = past_data.iloc[-1]["level"]
    past_time = pd.to_datetime(past_data.index[-1])

    time_diff_hours = (now - past_time).total_seconds() / 3600.0
    if time_diff_hours == 0:
        return None

    return float((current_level - past_level) / time_diff_hours)


def compute_river_level_features(level_series: pd.Series | None, timestamps: pd.DatetimeIndex | None) -> dict:
    """
    Convenience function that calls lags, changes, and rate of rise functions
    and merges results into a single dict.
    If level_series is empty or None, returns dict with all keys set to None.
    """
    lags = compute_level_lags(level_series, timestamps)
    changes = compute_level_changes(level_series, timestamps)
    ror = compute_rate_of_rise(level_series, timestamps, window_hours=1)

    return {
        **lags,
        **changes,
        "rate_of_rise": ror,
    }


def project_future_river_level(
    current_level_m: float | None,
    rate_of_rise_m_per_hr: float | None = None,
    horizon_hours: int = 1,
    forecast_precip_mm: float = 0.0,
    base_level_m: float = 75.80,
    decay_timescale_hours: float = 8.0,
    min_clamp_m: float | None = None,
    max_clamp_m: float | None = None,
) -> float | None:
    """
    Hydrological stage projection model for a river gauge station.
    Station-specific:
    - Brahmaputra (Tezpur): base_level ~63.80m, clamping [60.0, 68.0]
    - Jia Bharali (NT Road Xing): base_level ~75.80m, clamping [74.0, 80.0]

    Grounding:
    1. Momentum / rate-of-rise trend: dissipated exponentially (timescale ~8h)
    2. Catchment precipitation response: ~0.015 m stage rise per mm of forecast rainfall
       with basin routing efficiency scaled by horizon
    3. Stage recession: relaxation towards base level if dry and falling

    Returns projected stage in metres, or None if current_level_m is unavailable.
    """
    if current_level_m is None:
        return None

    h = float(horizon_hours)
    ror = rate_of_rise_m_per_hr if rate_of_rise_m_per_hr is not None else 0.0

    # Guard against extreme noise in rate of rise
    ror = max(-0.5, min(0.5, ror))

    # 1. Momentum trend component
    tau = decay_timescale_hours
    trend_rise = ror * tau * (1.0 - np.exp(-h / tau))

    # 2. Forecast rainfall runoff response
    routing_efficiency = min(1.0, h / 4.0)
    rain_rise = forecast_precip_mm * 0.015 * routing_efficiency

    # 3. Recession towards seasonal baseflow if above base level with minimal rain
    recession = 0.0
    if current_level_m > base_level_m and forecast_precip_mm < 1.0 and ror <= 0.0:
        excess = current_level_m - base_level_m
        recession = excess * (1.0 - np.exp(-h / (tau * 3.0)))

    projected = current_level_m + trend_rise + rain_rise - recession

    # Physical clamping to plausible gauge bounds
    clamp_min = min_clamp_m if min_clamp_m is not None else (base_level_m - 2.0)
    clamp_max = max_clamp_m if max_clamp_m is not None else (base_level_m + 4.5)
    projected = max(clamp_min, min(clamp_max, projected))

    return round(float(projected), 2)
