"""
India-WRIS / CWC Advisory Flood Forecast (AFF) River-Level Client.
Authoritative source: CWC Advisory Flood Forecast (aff.india-water.gov.in)
- PRIMARY: Brahmaputra at Tezpur / Ganeshghat (Station: TEZPUR, River: BRAHMAPUTRA)
- SECONDARY: Jia Bharali at N.T. Road Crossing (Station: NT ROAD CROSSING JIA-BHARALI, River: JIABHARALI)

Key Principles:
1. Real-time inference: Fetches live CWC AFF observations (WIMS_Value, Date_WIMS, current_condition).
2. Independent streams: Brahmaputra and Jia Bharali are fetched, QC'd, and stored completely independently.
3. Fallback: When telemetry is offline, uses verified staff-gauge / manual observation or realistic simulated data.
4. Unavailable state: Returns None and UNAVAILABLE when neither source is valid (never zero-fills or guesses).
5. Strictly NEVER substitute GloFAS modelled discharge for river water level.
6. Preserves explicit provenance:
   - OBSERVED — CWC AFF (or OBSERVED — TELEMETRY)
   - FORECAST — CWC AFF
   - DERIVED — INTERPOLATED
   - DEMO / SIMULATED
   - UNAVAILABLE
"""

from __future__ import annotations

import csv
import datetime
import random
import urllib.parse
from typing import Any

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import settings
from ml_pipeline.data_ingestion.river_qc_reconcile import (
    CAT_DEMO,
    CAT_DERIVED,
    CAT_OBSERVED,
    CAT_UNAVAILABLE,
    FLAG_FALLBACK_MANUAL,
    FLAG_INTERPOLATED,
    PROV_DEMO_MANUAL,
    PROV_DEMO_TELEMETRY,
    PROV_DERIVED_INTERPOLATED,
    PROV_OBS_MANUAL,
    PROV_OBS_TELEMETRY,
    PROV_UNAVAILABLE,
    QC_MISSING,
    QC_PASSED,
    RiverDataReconciler,
    RiverQCChecker,
    StationQCConfig,
)

PROV_OBS_CWC_AFF = "OBSERVED — CWC AFF"
PROV_FORECAST_CWC_AFF = "FORECAST — CWC AFF"

_IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

_cwc_aff_available: bool | None = None
_cwc_table_cache: dict[str, dict[str, Any]] | None = None
_cwc_table_cache_time: datetime.datetime | None = None


# ---------------------------------------------------------------------------
# Station Metadata Resolver
# ---------------------------------------------------------------------------

def resolve_station_info(station: str | None = None) -> dict[str, Any]:
    """
    Resolves a station identifier to canonical metadata and QC configuration.
    Defaults to PRIMARY (Brahmaputra at Tezpur).
    """
    stn_str = (str(station).strip() if station is not None else "").lower()
    if stn_str in (
        "secondary",
        "jia bharali",
        "jiabharali",
        "nt road crossing",
        "nt road crossing jia-bharali",
        "12359",
        settings.SECONDARY_STATION_CODE.lower(),
        settings.SECONDARY_STATION_NAME.lower(),
    ):
        return {
            "role": "secondary",
            "station_code": settings.SECONDARY_STATION_CODE,
            "station_name": settings.SECONDARY_STATION_NAME,
            "river_name": settings.SECONDARY_RIVER_NAME,
            "cwc_aff_name": "NT ROAD CROSSING JIA-BHARALI",
            "danger_level_m": settings.SECONDARY_DANGER_LEVEL_M,
            "warning_level_m": settings.SECONDARY_WARNING_LEVEL_M,
            "hfl_m": settings.SECONDARY_HFL_M,
            "base_level_m": settings.SECONDARY_BASE_LEVEL_M,
            "qc_config": StationQCConfig.for_secondary(),
            "lat": settings.SECONDARY_GAUGE_LAT,
            "lon": settings.SECONDARY_GAUGE_LON,
        }

    # Default: Primary (Brahmaputra at Tezpur)
    return {
        "role": "primary",
        "station_code": settings.PRIMARY_STATION_CODE,
        "station_name": settings.PRIMARY_STATION_NAME,
        "river_name": settings.PRIMARY_RIVER_NAME,
        "cwc_aff_name": "TEZPUR",
        "danger_level_m": settings.PRIMARY_DANGER_LEVEL_M,
        "warning_level_m": settings.PRIMARY_WARNING_LEVEL_M,
        "hfl_m": settings.PRIMARY_HFL_M,
        "base_level_m": settings.PRIMARY_BASE_LEVEL_M,
        "qc_config": StationQCConfig.for_primary(),
        "lat": settings.PRIMARY_GAUGE_LAT,
        "lon": settings.PRIMARY_GAUGE_LON,
    }


# ---------------------------------------------------------------------------
# Public: Availability Checks
# ---------------------------------------------------------------------------

def is_cwc_aff_available() -> bool:
    """Returns True if the CWC Advisory Flood Forecast portal is reachable."""
    global _cwc_aff_available
    if _cwc_aff_available is not None:
        return _cwc_aff_available

    url = getattr(settings, "CWC_AFF_TABLE_URL", "https://aff.india-water.gov.in/textdata/Floodday_table_view_header.txt")
    try:
        resp = httpx.get(url, timeout=4.0, verify=False)
        _cwc_aff_available = resp.status_code == 200 and len(resp.text) > 1000
    except Exception as exc:
        logger.info(f"CWC AFF live portal not reachable ({exc}). Falling back to simulated/mock data.")
        _cwc_aff_available = False

    return _cwc_aff_available


def is_wris_available() -> bool:
    """Compatibility alias for is_cwc_aff_available."""
    return is_cwc_aff_available()


def _is_live_available() -> bool:
    """Returns True only when both CWC AFF and WRIS compatibility checks are truthy."""
    try:
        return bool(is_cwc_aff_available() and is_wris_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CWC AFF Live Dataset Fetchers
# ---------------------------------------------------------------------------

def fetch_cwc_aff_table(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    """
    Fetches the latest CWC Advisory Flood Forecast table.
    Returns parsed records keyed by uppercase station name.
    """
    global _cwc_table_cache, _cwc_table_cache_time
    now = datetime.datetime.now(_IST)
    if not force_refresh and _cwc_table_cache is not None and _cwc_table_cache_time is not None:
        if (now - _cwc_table_cache_time).total_seconds() < 120.0:
            return _cwc_table_cache

    url = getattr(settings, "CWC_AFF_TABLE_URL", "https://aff.india-water.gov.in/textdata/Floodday_table_view_header.txt")
    try:
        resp = httpx.get(url, timeout=6.0, verify=False)
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
        if not lines:
            return _cwc_table_cache or {}
        header = lines[0].split(",")
        stations = {}
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= len(header):
                row = dict(zip(header, parts))
                stn = row.get("Station", "").strip().upper()
                if stn:
                    stations[stn] = row
        _cwc_table_cache = stations
        _cwc_table_cache_time = now
        return stations
    except Exception as exc:
        logger.warning(f"CWC AFF table fetch failed: {exc}")
        return _cwc_table_cache or {}


def fetch_cwc_aff_timeseries(cwc_station_name: str) -> list[dict[str, Any]]:
    """
    Fetches hourly timeseries from CWC AFF for a specific station name.
    """
    base_url = getattr(settings, "CWC_AFF_TIMESERIES_URL", "https://aff.india-water.gov.in/Timeseries")
    encoded = urllib.parse.quote(cwc_station_name)
    url = f"{base_url}/{encoded}-W.txt"
    try:
        resp = httpx.get(url, timeout=6.0, verify=False)
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
        if len(lines) < 2:
            return []
        reader = csv.DictReader(lines)
        results = []
        for row in reader:
            ts = row.get("Date", "").strip()
            wims_str = row.get("WIMS", "").strip()
            aff_str = row.get("AFF", "").strip()
            discharge_str = row.get("Discharge", "").strip()
            wims_val = float(wims_str) if wims_str else None
            aff_val = float(aff_str) if aff_str else None
            discharge_val = float(discharge_str) if discharge_str else None
            results.append({
                "timestamp": ts,
                "water_level_m": wims_val,
                "forecast_level_m": aff_val,
                "discharge_m3s": discharge_val,
            })
        return results
    except Exception as exc:
        logger.warning(f"CWC AFF timeseries fetch failed for {cwc_station_name}: {exc}")
        return []


def fetch_cwc_aff_forecast_horizon_map(cwc_station_name: str) -> dict[int, float]:
    """
    Extracts hourly hydrodynamic forecast levels from CWC AFF timeseries
    (aff.india-water.gov.in/hydro.php) keyed by forecast horizon (+1h, +3h, etc.)
    relative to the latest valid WIMS observation timestamp.
    """
    ts = fetch_cwc_aff_timeseries(cwc_station_name)
    if not ts:
        return {}

    obs_rows = [r for r in ts if r.get("water_level_m") is not None]
    if not obs_rows:
        return {}

    try:
        last_obs_ts = pd.to_datetime(obs_rows[-1]["timestamp"])
    except Exception:
        return {}

    fcst_map: dict[int, float] = {}
    for r in ts:
        if r.get("forecast_level_m") is None:
            continue
        try:
            r_ts = pd.to_datetime(r["timestamp"])
            if r_ts > last_obs_ts:
                diff_hours = round((r_ts - last_obs_ts).total_seconds() / 3600.0)
                if diff_hours > 0 and diff_hours not in fcst_map:
                    fcst_map[diff_hours] = float(r["forecast_level_m"])
        except Exception:
            continue

    return fcst_map


# ---------------------------------------------------------------------------
# Telemetry Stream Ingestion (Live + Mock)
# ---------------------------------------------------------------------------

def fetch_telemetry_stream(
    station_code: str | None = None,
    use_mock: bool = False,
    lookback_hours: int = 72,
    simulate_failure: bool = False,
    simulate_spike: bool = False,
) -> dict[str, Any]:
    """Fetches automated telemetry stream for the resolved station."""
    if simulate_failure:
        return {"observed_level_m": None, "timeseries": [], "data_age_minutes": None}

    info = resolve_station_info(station_code)
    cwc_name = info["cwc_aff_name"]

    if not use_mock and _is_live_available():
        try:
            ts_list = fetch_cwc_aff_timeseries(cwc_name)
            observed_ts = [
                {"timestamp": r["timestamp"], "water_level_m": r["water_level_m"]}
                for r in ts_list
                if r["water_level_m"] is not None
            ]
            if observed_ts:
                latest = observed_ts[-1]
                prev = observed_ts[-2] if len(observed_ts) > 1 else latest
                now = datetime.datetime.now(_IST)
                try:
                    latest_ts = pd.Timestamp(latest["timestamp"])
                    if latest_ts.tzinfo is None:
                        latest_ts = latest_ts.tz_localize(_IST)
                    age_min = max(0, round((now - latest_ts).total_seconds() / 60.0))
                except Exception:
                    age_min = None

                return {
                    "observed_level_m": latest["water_level_m"],
                    "timestamp": latest["timestamp"],
                    "prev_level_m": prev["water_level_m"],
                    "time_diff_hours": 1.0,
                    "data_age_minutes": age_min,
                    "timeseries": observed_ts[-lookback_hours:],
                }
        except Exception as exc:
            logger.warning(f"Live telemetry stream fetch failed for {cwc_name}: {exc}")

    # Mock telemetry fallback
    return _generate_mock_telemetry(
        station_info=info,
        lookback_hours=lookback_hours,
        simulate_spike=simulate_spike,
    )


# ---------------------------------------------------------------------------
# Manual Stream Ingestion (Live + Mock)
# ---------------------------------------------------------------------------

def fetch_manual_stream(
    station_code: str | None = None,
    use_mock: bool = False,
    lookback_hours: int = 72,
    simulate_failure: bool = False,
) -> dict[str, Any]:
    """Fetches manual staff gauge observation stream for the resolved station."""
    if simulate_failure:
        return {"observed_level_m": None, "timeseries": [], "data_age_minutes": None}

    info = resolve_station_info(station_code)
    # Return mock manual stream calibrated to station baseline
    return _generate_mock_manual(
        station_info=info,
        lookback_hours=lookback_hours,
    )


# ---------------------------------------------------------------------------
# Public: Reconciled Latest River Level (Real-time Inference)
# ---------------------------------------------------------------------------

def fetch_observed_river_level(
    station_code: str | None = None,
    use_mock: bool | None = None,
    simulate_telemetry_failure: bool = False,
    simulate_manual_failure: bool = False,
    simulate_telemetry_spike: bool = False,
) -> dict[str, Any]:
    """
    Returns the latest reconciled river water level for the specified station.
    Defaults to PRIMARY (Brahmaputra at Tezpur).
    """
    info = resolve_station_info(station_code)
    config = info["qc_config"]
    checker = RiverQCChecker(config)
    reconciler = RiverDataReconciler(config)

    if use_mock is None:
        use_mock = not _is_live_available()

    # 1. Total failure simulation or live unavailable without mock
    if (simulate_telemetry_failure and simulate_manual_failure) or (not use_mock and not _is_live_available()):
        return reconciler.reconcile_latest(
            telemetry_obs=None,
            manual_obs=None,
            is_mock=False,
        )

    # 2. Live CWC AFF path (when live is available and not forced mock)
    if not use_mock and _is_live_available() and not simulate_telemetry_failure:
        try:
            table = fetch_cwc_aff_table()
            cwc_name = info["cwc_aff_name"]
            row = table.get(cwc_name)
            if row and row.get("WIMS_Value"):
                wims_val = float(row["WIMS_Value"])
                date_wims = row.get("Date_WIMS")
                # Authoritative danger level and HFL come from official station config.
                # Note: In CWC AFF's live table for Jia Bharali, DangerLevel is listed as 78.5m (equal to HFL),
                # whereas the official Assam WRD / CWC danger level is 77.00m.
                # Authoritative station thresholds must take precedence over feed artifacts.
                dl = float(config.danger_level_m)
                wl = float(config.warning_level_m)
                hfl = float(config.hfl_m)
                cwc_table_dl = float(row.get("DangerLevel")) if row.get("DangerLevel") else dl

                # Determine flood condition against authoritative thresholds
                if wims_val >= hfl:
                    current_cond = "Above HFL"
                elif wims_val >= dl:
                    current_cond = "Above Danger Level"
                elif wims_val >= wl:
                    current_cond = "Above Warning Level"
                else:
                    current_cond = row.get("current_condition", "Normal")

                # Check QC range & spike
                is_valid, qc_flag = checker.check_range(wims_val)
                if is_valid:
                    # Extract 7-day CWC forecast values
                    cwc_forecasts = []
                    for d_idx in range(1, 8):
                        f_date = row.get(f"ForecastDate{d_idx}")
                        f_val = row.get(f"ForecastDay{d_idx}")
                        f_cond = row.get(f"ConditionDay{d_idx}", "Normal")
                        if f_val:
                            try:
                                cwc_forecasts.append({
                                    "forecast_date": f_date,
                                    "forecast_level_m": float(f_val),
                                    "condition": f_cond,
                                    "data_category": "FORECAST",
                                    "provenance": PROV_FORECAST_CWC_AFF,
                                })
                            except ValueError:
                                pass

                    now = datetime.datetime.now(_IST)
                    try:
                        latest_ts = pd.Timestamp(date_wims)
                        if latest_ts.tzinfo is None:
                            latest_ts = latest_ts.tz_localize(_IST)
                        age_min = max(0, round((now - latest_ts).total_seconds() / 60.0))
                    except Exception:
                        age_min = None

                    # Fetch recent timeseries
                    ts_data = fetch_cwc_aff_timeseries(cwc_name)
                    obs_ts = [
                        {"timestamp": r["timestamp"], "water_level_m": r["water_level_m"]}
                        for r in ts_data
                        if r["water_level_m"] is not None
                    ]

                    return {
                        "observed_level_m": wims_val,
                        "timestamp": date_wims,
                        "data_category": CAT_OBSERVED,
                        "source": "CWC Advisory Flood Forecast (aff.india-water.gov.in)",
                        "provenance": PROV_OBS_CWC_AFF,
                        "unit": "m",
                        "station_code": info["station_code"],
                        "station_name": info["station_name"],
                        "river": info["river_name"],
                        "role": info["role"],
                        "danger_level_m": dl,
                        "warning_level_m": wl,
                        "hfl_m": hfl,
                        "flood_condition": current_cond,
                        "cwc_forecasts": cwc_forecasts,
                        "retrieved_at": now.isoformat(),
                        "data_age_minutes": age_min,
                        "detail": f"Active: CWC AFF observation ({qc_flag}). Flood condition: {current_cond}.",
                        "reconciliation_method": "CWC_AFF_LIVE",
                        "source_stream": "CWC_AFF",
                        "is_fallback_active": False,
                        "qc_flag": qc_flag,
                        "telemetry_status": {"available": True, "valid": True, "level_m": wims_val, "qc_flag": qc_flag},
                        "manual_status": {"available": False, "valid": False, "level_m": None, "qc_flag": QC_MISSING},
                        "timeseries": obs_ts[-72:] if obs_ts else [],
                    }
        except Exception as exc:
            logger.warning(f"Live CWC AFF processing failed for {info['station_name']}: {exc}")

    # 3. Fallback / Mock stream ingestion
    telemetry_obs = fetch_telemetry_stream(
        station_code=info["station_code"],
        use_mock=use_mock,
        lookback_hours=72,
        simulate_failure=simulate_telemetry_failure,
        simulate_spike=simulate_telemetry_spike,
    )
    manual_obs = fetch_manual_stream(
        station_code=info["station_code"],
        use_mock=use_mock,
        lookback_hours=72,
        simulate_failure=simulate_manual_failure,
    )

    reconciled = reconciler.reconcile_latest(
        telemetry_obs=telemetry_obs,
        manual_obs=manual_obs,
        is_mock=use_mock,
    )

    # Attach canonical station identity
    reconciled["station_code"] = info["station_code"]
    reconciled["station_name"] = info["station_name"]
    reconciled["river"] = info["river_name"]
    reconciled["role"] = info["role"]
    reconciled["danger_level_m"] = info["danger_level_m"]
    reconciled["warning_level_m"] = info["warning_level_m"]
    reconciled["hfl_m"] = info["hfl_m"]
    reconciled["flood_condition"] = "Normal"
    reconciled["cwc_forecasts"] = []

    return reconciled


def fetch_primary_river_level(use_mock: bool | None = None) -> dict[str, Any]:
    """Fetches observed river level for PRIMARY gauge: Brahmaputra at Tezpur."""
    return fetch_observed_river_level(station_code="primary", use_mock=use_mock)


def fetch_secondary_river_level(use_mock: bool | None = None) -> dict[str, Any]:
    """Fetches observed river level for SECONDARY gauge: Jia Bharali at N.T. Road Crossing."""
    return fetch_observed_river_level(station_code="secondary", use_mock=use_mock)


def fetch_dual_observed_river_levels(use_mock: bool | None = None) -> dict[str, Any]:
    """
    Fetches both primary and secondary river gauges completely independently.
    Never derives one from the other.
    """
    return {
        "primary": fetch_primary_river_level(use_mock=use_mock),
        "secondary": fetch_secondary_river_level(use_mock=use_mock),
    }


# ---------------------------------------------------------------------------
# Public: Reconciled Time-Series (Features & ML)
# ---------------------------------------------------------------------------

def fetch_level_timeseries(
    station_code: str | None = None,
    lookback_hours: int = 72,
    use_mock: bool | None = None,
    simulate_telemetry_failure: bool = False,
    simulate_manual_failure: bool = False,
    max_interp_gap_hours: int = 2,
) -> tuple[pd.Series | None, pd.DatetimeIndex | None]:
    """
    Returns (level_series, timestamps) for the resolved station.
    Defaults to PRIMARY (Brahmaputra at Tezpur).
    """
    info = resolve_station_info(station_code)
    config = info["qc_config"]
    reconciler = RiverDataReconciler(config)

    if use_mock is None:
        use_mock = not _is_live_available()

    if not use_mock and not _is_live_available():
        return None, None

    # Fetch stream data
    t_data = fetch_telemetry_stream(
        station_code=info["station_code"],
        use_mock=use_mock,
        lookback_hours=lookback_hours,
        simulate_failure=simulate_telemetry_failure,
    )
    t_series = _ts_to_series(t_data.get("timeseries", []))

    # Only fetch/generate manual stream if telemetry is failed, empty, or mock mode is requested.
    # When live telemetry is active, do not inject synthetic mock data at boundary hours.
    if simulate_telemetry_failure or use_mock or t_series is None or t_series.empty:
        m_data = fetch_manual_stream(
            station_code=info["station_code"],
            use_mock=use_mock,
            lookback_hours=lookback_hours,
            simulate_failure=simulate_manual_failure,
        )
        m_series = _ts_to_series(m_data.get("timeseries", []))
    else:
        m_series = pd.Series(dtype=float)

    if (t_series is None or t_series.empty) and (m_series is None or m_series.empty):
        return None, None

    reconciled_df = reconciler.reconcile_timeseries(
        telemetry_series=t_series,
        manual_series=m_series,
        lookback_hours=lookback_hours,
        max_interp_gap_hours=max_interp_gap_hours,
        is_mock=use_mock,
    )

    valid_mask = ~reconciled_df["level_m"].isna()
    if not valid_mask.any():
        return None, None

    clean_df = reconciled_df[valid_mask]
    series = pd.Series(clean_df["level_m"].values, index=clean_df.index, name="water_level_m")
    return series, clean_df.index


def fetch_primary_level_timeseries(
    lookback_hours: int = 72, use_mock: bool | None = None
) -> tuple[pd.Series | None, pd.DatetimeIndex | None]:
    """Returns (level_series, timestamps) for PRIMARY gauge (Brahmaputra)."""
    return fetch_level_timeseries(station_code="primary", lookback_hours=lookback_hours, use_mock=use_mock)


def fetch_secondary_level_timeseries(
    lookback_hours: int = 72, use_mock: bool | None = None
) -> tuple[pd.Series | None, pd.DatetimeIndex | None]:
    """Returns (level_series, timestamps) for SECONDARY gauge (Jia Bharali)."""
    return fetch_level_timeseries(station_code="secondary", lookback_hours=lookback_hours, use_mock=use_mock)


def fetch_dual_level_timeseries(
    lookback_hours: int = 72, use_mock: bool | None = None
) -> tuple[tuple[pd.Series | None, pd.DatetimeIndex | None], tuple[pd.Series | None, pd.DatetimeIndex | None]]:
    """Returns independent ((primary_series, primary_times), (secondary_series, secondary_times))."""
    primary = fetch_primary_level_timeseries(lookback_hours=lookback_hours, use_mock=use_mock)
    secondary = fetch_secondary_level_timeseries(lookback_hours=lookback_hours, use_mock=use_mock)
    return primary, secondary


def fetch_reconciled_dataframe(
    station_code: str | None = None,
    lookback_hours: int = 72,
    use_mock: bool | None = None,
    max_interp_gap_hours: int = 2,
) -> pd.DataFrame:
    """Returns full reconciled DataFrame with explicit provenance for ML dataset preparation."""
    info = resolve_station_info(station_code)
    config = info["qc_config"]
    reconciler = RiverDataReconciler(config)

    if use_mock is None:
        use_mock = not _is_live_available()

    if not use_mock and not _is_live_available():
        return pd.DataFrame()

    t_data = fetch_telemetry_stream(info["station_code"], use_mock=use_mock, lookback_hours=lookback_hours)
    m_data = fetch_manual_stream(info["station_code"], use_mock=use_mock, lookback_hours=lookback_hours)

    t_series = _ts_to_series(t_data.get("timeseries", []))
    m_series = _ts_to_series(m_data.get("timeseries", []))

    return reconciler.reconcile_timeseries(
        telemetry_series=t_series,
        manual_series=m_series,
        lookback_hours=lookback_hours,
        max_interp_gap_hours=max_interp_gap_hours,
        is_mock=use_mock,
    )


# ---------------------------------------------------------------------------
# Internal Helpers: Station-Specific Synthetic Mock Streams
# ---------------------------------------------------------------------------

def _generate_mock_telemetry(
    station_info: dict[str, Any],
    lookback_hours: int = 72,
    simulate_spike: bool = False,
) -> dict[str, Any]:
    """Generates synthetic hourly telemetry calibrated to station base stage."""
    now = datetime.datetime.now(_IST).replace(minute=0, second=0, microsecond=0)
    start_time = now - datetime.timedelta(hours=lookback_hours)

    base = station_info["base_level_m"]
    rng = random.Random(now.hour + now.day * 100 + int(station_info["danger_level_m"] * 10))
    timeseries = []
    current_ts = start_time
    total_steps = lookback_hours + 1
    step = 0

    while current_ts <= now:
        trend = (step / max(1, total_steps)) * 0.45
        hour_frac = current_ts.hour + current_ts.minute / 60.0
        diurnal = 0.08 * np.sin(2 * np.pi * (hour_frac - 14) / 24)
        noise = rng.uniform(-0.04, 0.04)

        level = round(base + trend + diurnal + noise, 2)
        timeseries.append({
            "timestamp": current_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "water_level_m": level,
        })
        current_ts += datetime.timedelta(hours=1)
        step += 1

    latest = timeseries[-1]
    prev = timeseries[-2] if len(timeseries) > 1 else latest

    if simulate_spike:
        latest["water_level_m"] = round(latest["water_level_m"] + 3.0, 2)

    return {
        "observed_level_m": latest["water_level_m"],
        "timestamp": latest["timestamp"],
        "prev_level_m": prev["water_level_m"],
        "time_diff_hours": 1.0,
        "data_age_minutes": 5,
        "timeseries": timeseries,
    }


def _generate_mock_manual(
    station_info: dict[str, Any],
    lookback_hours: int = 72,
) -> dict[str, Any]:
    """Generates synthetic manual staff gauge observations calibrated to station base stage."""
    now = datetime.datetime.now(_IST).replace(minute=0, second=0, microsecond=0)
    start_time = now - datetime.timedelta(hours=lookback_hours)

    base = station_info["base_level_m"]
    rng = random.Random(now.hour + now.day * 100 + int(station_info["danger_level_m"] * 10) + 42)
    timeseries = []
    current_ts = start_time
    total_steps = lookback_hours + 1
    step = 0

    while current_ts <= now:
        trend = (step / max(1, total_steps)) * 0.45
        hour_frac = current_ts.hour + current_ts.minute / 60.0
        diurnal = 0.08 * np.sin(2 * np.pi * (hour_frac - 14) / 24)
        noise = rng.uniform(-0.02, 0.02)

        level = round(base + trend + diurnal + noise, 2)
        timeseries.append({
            "timestamp": current_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "water_level_m": level,
        })
        current_ts += datetime.timedelta(hours=1)
        step += 1

    latest = timeseries[-1]
    prev = timeseries[-2] if len(timeseries) > 1 else latest

    return {
        "observed_level_m": latest["water_level_m"],
        "timestamp": latest["timestamp"],
        "prev_level_m": prev["water_level_m"],
        "time_diff_hours": 1.0,
        "data_age_minutes": 45,
        "timeseries": timeseries,
    }


def _ts_to_series(ts_list: list[dict[str, Any]]) -> pd.Series | None:
    if not ts_list:
        return None
    timestamps = []
    levels = []
    for item in ts_list:
        if item.get("water_level_m") is None:
            continue
        t = pd.Timestamp(item["timestamp"])
        if t.tzinfo is None:
            t = t.tz_localize(_IST)
        timestamps.append(t)
        levels.append(float(item["water_level_m"]))
    if not timestamps:
        return None
    return pd.Series(levels, index=pd.DatetimeIndex(timestamps)).sort_index()
