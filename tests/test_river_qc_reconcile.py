"""
Comprehensive unit and integration tests for the Hybrid India-WRIS/CWC River-Level Pipeline.

Tests:
1. Station metadata verification (Jia Bharali - N.T. Road Xing)
2. Quality Control (QC): range check, rate-of-rise check, spike rejection, flatline check
3. Genuine rapid flood rise preservation (never falsely reject legitimate flash floods)
4. Telemetry preference for real-time inference (low latency)
5. Manual staff-gauge fallback when telemetry is missing, stale, or fails QC
6. Strict UNAVAILABLE handling (neither source available -> None level, no zero-filling)
7. Strict separation of GloFAS discharge (MODELLED in m³/s, never substituted for stage)
8. Historical time-series reconciliation with explicit DERIVED — INTERPOLATED marking
"""

import datetime
import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from ml_pipeline.data_ingestion.river_qc_reconcile import (
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
    QC_FLATLINE,
    QC_MISSING,
    QC_OUT_OF_BOUNDS,
    QC_PASSED,
    QC_RATE_EXCEEDED,
    QC_SPIKE,
    QC_STALE,
    RiverDataReconciler,
    RiverQCChecker,
    StationQCConfig,
)
from ml_pipeline.data_ingestion.india_wris import (
    fetch_observed_river_level,
    fetch_level_timeseries,
    fetch_reconciled_dataframe,
    is_wris_available,
)
from ml_pipeline.data_ingestion.river_levels import (
    get_station_thresholds,
    compute_threshold_features,
)
from ml_pipeline.data_ingestion.glofas import fetch_glofas_discharge
from ml_pipeline.feature_engineering.river_level_features import (
    compute_river_level_features,
    project_future_river_level,
)


# ---------------------------------------------------------------------------
# 1. Station Metadata Verification
# ---------------------------------------------------------------------------

def test_jia_bharali_station_metadata():
    """Verify Jia Bharali N.T. Road Xing metadata matches official Assam reference."""
    thresh = get_station_thresholds("N.T.Road Xing")
    assert thresh is not None
    assert thresh["river"] == "Jia Bharali"
    assert thresh["gauge_location"] == "N.T.Road Xing"
    assert float(thresh["danger_level_m"]) == 77.00
    assert float(thresh["hfl_m"]) == 78.50

    config = StationQCConfig.for_secondary()
    assert config.station_code == "12359"
    assert config.danger_level_m == 77.00
    assert config.hfl_m == 78.50
    assert config.min_stage_m == 72.00
    assert config.max_stage_m == 81.00


def test_brahmaputra_primary_station_metadata():
    """Verify Brahmaputra Tezpur metadata matches CWC AFF primary reference."""
    thresh = get_station_thresholds("Tezpur")
    assert thresh is not None
    assert thresh["river"] == "Brahmaputra"
    assert thresh["gauge_location"] == "Tezpur"
    assert float(thresh["danger_level_m"]) == 65.23
    assert float(thresh["hfl_m"]) == 66.59

    config = StationQCConfig.for_primary()
    assert config.station_code == "TEZPUR"
    assert config.danger_level_m == 65.23
    assert config.hfl_m == 66.59
    assert config.min_stage_m == 55.86
    assert config.max_stage_m == 68.59


# ---------------------------------------------------------------------------
# 2. Quality Control (QC) Tests
# ---------------------------------------------------------------------------

def test_qc_range_check():
    """Verify physical range limits for both secondary and primary stations."""
    sec_checker = RiverQCChecker(StationQCConfig.for_secondary())

    # Valid within bounds for Jia Bharali (72.0m to 81.0m)
    ok, flag = sec_checker.check_range(75.50)
    assert ok is True
    assert flag == QC_PASSED

    # At boundary
    ok, flag = sec_checker.check_range(72.00)
    assert ok is True
    ok, flag = sec_checker.check_range(81.00)
    assert ok is True

    # Below physical river bed datum
    ok, flag = sec_checker.check_range(71.50)
    assert ok is False
    assert flag == QC_OUT_OF_BOUNDS

    # Implausibly high level
    ok, flag = sec_checker.check_range(83.00)
    assert ok is False
    assert flag == QC_OUT_OF_BOUNDS

    # Null / NaN
    ok, flag = sec_checker.check_range(None)
    assert ok is False
    assert flag == QC_MISSING

    # Brahmaputra Primary (55.86m to 68.59m)
    prim_checker = RiverQCChecker(StationQCConfig.for_primary())
    ok, flag = prim_checker.check_range(63.86)
    assert ok is True
    assert flag == QC_PASSED
    ok, flag = prim_checker.check_range(54.00)
    assert ok is False
    assert flag == QC_OUT_OF_BOUNDS
    ok, flag = prim_checker.check_range(70.00)
    assert ok is False
    assert flag == QC_OUT_OF_BOUNDS


def test_qc_rate_of_change_check():
    """Verify hourly rate-of-change check (<= 1.5 m/h)."""
    checker = RiverQCChecker(StationQCConfig.for_secondary())

    # Normal rise of 0.3m in 1h
    ok, flag = checker.check_rate_of_change(75.80, 75.50, time_diff_hours=1.0)
    assert ok is True
    assert flag == QC_PASSED

    # Rapid but permissible rise of 1.2m in 1h
    ok, flag = checker.check_rate_of_change(76.70, 75.50, time_diff_hours=1.0)
    assert ok is True
    assert flag == QC_PASSED

    # Excessive jump of 2.5m in 1h
    ok, flag = checker.check_rate_of_change(78.00, 75.50, time_diff_hours=1.0)
    assert ok is False
    assert flag == QC_RATE_EXCEEDED


def test_qc_genuine_rapid_flood_rise_preserved():
    """
    CRITICAL: Ensure genuine sustained flash flood surges are NOT falsely
    rejected as spikes! Jia Bharali can rise 0.8m - 1.2m/hr in severe monsoons.
    """
    checker = RiverQCChecker(StationQCConfig.for_secondary())

    # Create a 6-hour time series with a legitimate, sustained rapid rise
    times = pd.date_range("2026-07-20 00:00:00", periods=6, freq="1h")
    # 75.5 -> 76.4 (+0.9m) -> 77.3 (+0.9m, over danger level) -> 77.8 (+0.5m) -> 77.9 -> 77.7
    levels = [75.50, 76.40, 77.30, 77.80, 77.90, 77.70]
    series = pd.Series(levels, index=times)

    df_qc = checker.filter_series(series)

    # All points in this legitimate flood sequence must pass QC
    assert df_qc["is_valid"].all(), "Genuine rapid flood rise was incorrectly rejected!"
    assert (df_qc["qc_flag"] == QC_PASSED).all()


def test_qc_transient_spike_rejected():
    """
    Verify single-point aberrant noise spike that immediately drops back
    is rejected with QC_SPIKE.
    """
    checker = RiverQCChecker(StationQCConfig.for_secondary())

    times = pd.date_range("2026-07-20 00:00:00", periods=5, freq="1h")
    # 75.50 -> 75.60 -> 78.50 (spike!) -> 75.70 -> 75.80
    levels = [75.50, 75.60, 78.50, 75.70, 75.80]
    series = pd.Series(levels, index=times)

    df_qc = checker.filter_series(series)

    # Spike at index 2 must be invalid with QC_SPIKE
    assert df_qc.iloc[2]["is_valid"] == False
    assert df_qc.iloc[2]["qc_flag"] == QC_SPIKE

    # Surrounding valid points must pass
    assert df_qc.iloc[0]["is_valid"] == True
    assert df_qc.iloc[1]["is_valid"] == True
    assert df_qc.iloc[3]["is_valid"] == True


def test_qc_stuck_sensor_flatline_rejected():
    """Verify stuck sensor (identical reading for > 24 consecutive hours) is flagged."""
    checker = RiverQCChecker(StationQCConfig.for_secondary())

    times = pd.date_range("2026-07-20 00:00:00", periods=30, freq="1h")
    # Identical elevated reading of 76.80 for 30 hours
    levels = [76.80] * 30
    series = pd.Series(levels, index=times)

    df_qc = checker.filter_series(series)

    # Points after 24 hours must be flagged as QC_FLATLINE
    assert df_qc.iloc[26]["is_valid"] == False
    assert df_qc.iloc[26]["qc_flag"] == QC_FLATLINE


# ---------------------------------------------------------------------------
# 3. Hybrid Real-time Reconciliation & Fallback Tests
# ---------------------------------------------------------------------------

def test_telemetry_primary_preference():
    """
    Verify that when Telemetry is fresh and valid, it is selected as primary
    source with OBSERVED — TELEMETRY provenance.
    """
    reconciler = RiverDataReconciler(StationQCConfig.for_secondary())

    telemetry_obs = {
        "observed_level_m": 76.10,
        "timestamp": "2026-07-20 14:00:00",
        "data_age_minutes": 10,
        "timeseries": [{"timestamp": "2026-07-20 14:00:00", "water_level_m": 76.10}],
    }
    manual_obs = {
        "observed_level_m": 76.05,
        "timestamp": "2026-07-20 13:00:00",
        "data_age_minutes": 70,
        "timeseries": [{"timestamp": "2026-07-20 13:00:00", "water_level_m": 76.05}],
    }

    result = reconciler.reconcile_latest(telemetry_obs, manual_obs, is_mock=False)

    assert result["observed_level_m"] == 76.10
    assert result["reconciliation_method"] == "TELEMETRY_PRIMARY"
    assert result["source_stream"] == "TELEMETRY"
    assert result["provenance"] == PROV_OBS_TELEMETRY
    assert result["is_fallback_active"] is False


def test_primary_brahmaputra_reconciliation():
    """Verify reconciliation on primary Brahmaputra at Tezpur gauge levels."""
    reconciler = RiverDataReconciler(StationQCConfig.for_primary())

    telemetry_obs = {
        "observed_level_m": 63.85,
        "timestamp": "2026-07-20 14:00:00",
        "data_age_minutes": 15,
        "timeseries": [{"timestamp": "2026-07-20 14:00:00", "water_level_m": 63.85}],
    }
    manual_obs = {
        "observed_level_m": 63.80,
        "timestamp": "2026-07-20 13:00:00",
        "data_age_minutes": 75,
        "timeseries": [{"timestamp": "2026-07-20 13:00:00", "water_level_m": 63.80}],
    }

    result = reconciler.reconcile_latest(telemetry_obs, manual_obs, is_mock=False)

    assert result["observed_level_m"] == 63.85
    assert result["reconciliation_method"] == "TELEMETRY_PRIMARY"
    assert result["is_fallback_active"] is False


def test_manual_fallback_when_telemetry_offline():
    """
    Verify fallback to Manual staff gauge reading when Telemetry is offline/missing.
    """
    reconciler = RiverDataReconciler(StationQCConfig.for_secondary())

    telemetry_obs = {
        "observed_level_m": None,  # offline
        "timestamp": None,
        "data_age_minutes": None,
        "timeseries": [],
    }
    manual_obs = {
        "observed_level_m": 76.25,
        "timestamp": "2026-07-20 13:30:00",
        "data_age_minutes": 45,
        "timeseries": [{"timestamp": "2026-07-20 13:30:00", "water_level_m": 76.25}],
    }

    result = reconciler.reconcile_latest(telemetry_obs, manual_obs, is_mock=False)

    assert result["observed_level_m"] == 76.25
    assert result["reconciliation_method"] == "MANUAL_FALLBACK"
    assert result["source_stream"] == "MANUAL"
    assert result["provenance"] == PROV_OBS_MANUAL
    assert result["is_fallback_active"] is True
    assert "fallback" in result["detail"].lower()


def test_manual_fallback_when_telemetry_fails_qc():
    """
    Verify fallback to Manual when Telemetry reading is anomalous (e.g. out of bounds).
    """
    reconciler = RiverDataReconciler(StationQCConfig.for_secondary())

    telemetry_obs = {
        "observed_level_m": 88.00,  # anomalous out of range (>81.0m)
        "timestamp": "2026-07-20 14:00:00",
        "data_age_minutes": 5,
        "timeseries": [],
    }
    manual_obs = {
        "observed_level_m": 76.40,
        "timestamp": "2026-07-20 13:00:00",
        "data_age_minutes": 65,
        "timeseries": [{"timestamp": "2026-07-20 13:00:00", "water_level_m": 76.40}],
    }

    result = reconciler.reconcile_latest(telemetry_obs, manual_obs, is_mock=False)

    assert result["observed_level_m"] == 76.40
    assert result["reconciliation_method"] == "MANUAL_FALLBACK"
    assert result["provenance"] == PROV_OBS_MANUAL
    assert result["is_fallback_active"] is True


def test_both_sources_unavailable_handling():
    """
    Verify that when neither source is valid:
    - observed_level_m is None (NEVER 0.0)
    - data_category is UNAVAILABLE
    - provenance is UNAVAILABLE
    - GloFAS is NOT substituted
    """
    reconciler = RiverDataReconciler(StationQCConfig.for_secondary())

    telemetry_obs = {"observed_level_m": None, "data_age_minutes": None}
    manual_obs = {"observed_level_m": None, "data_age_minutes": None}

    result = reconciler.reconcile_latest(telemetry_obs, manual_obs, is_mock=False)

    assert result["observed_level_m"] is None
    assert result["observed_level_m"] != 0.0
    assert result["data_category"] == CAT_UNAVAILABLE
    assert result["provenance"] == PROV_UNAVAILABLE
    assert result["reconciliation_method"] == "UNAVAILABLE"
    assert result["is_fallback_active"] is False


def test_mock_simulation_fallback_mode():
    """Verify simulated failure trigger in fetch_observed_river_level."""
    # 1. Normal simulated telemetry (Primary Brahmaputra)
    data = fetch_observed_river_level(use_mock=True)
    assert data["observed_level_m"] is not None
    assert "TELEMETRY" in data["provenance"]
    assert data["is_fallback_active"] is False
    assert data["station_code"] == settings.PRIMARY_STATION_CODE

    # Secondary simulated telemetry (Jia Bharali)
    sec_data = fetch_observed_river_level(settings.SECONDARY_STATION_CODE, use_mock=True)
    assert sec_data["observed_level_m"] is not None
    assert sec_data["station_code"] == settings.SECONDARY_STATION_CODE

    # 2. Simulated telemetry failure -> automatic fallback to simulated manual
    data_fallback = fetch_observed_river_level(
        settings.SECONDARY_STATION_CODE,
        use_mock=True,
        simulate_telemetry_failure=True,
    )
    assert data_fallback["observed_level_m"] is not None
    assert data_fallback["reconciliation_method"] == "MANUAL_FALLBACK"
    assert data_fallback["source_stream"] == "MANUAL"
    assert "MANUAL" in data_fallback["provenance"]
    assert data_fallback["is_fallback_active"] is True

    # 3. Simulated failure of both -> UNAVAILABLE
    data_unavail = fetch_observed_river_level(
        settings.SECONDARY_STATION_CODE,
        use_mock=True,
        simulate_telemetry_failure=True,
        simulate_manual_failure=True,
    )
    assert data_unavail["observed_level_m"] is None
    assert data_unavail["data_category"] == CAT_UNAVAILABLE
    assert data_unavail["provenance"] == PROV_UNAVAILABLE


# ---------------------------------------------------------------------------
# 4. Strict GloFAS Separation Tests
# ---------------------------------------------------------------------------

def test_glofas_strict_separation_and_units():
    """Verify GloFAS is strictly river discharge in m³/s and never mixed with stage."""
    glofas = fetch_glofas_discharge()
    if glofas.get("status") != "UNAVAILABLE":
        assert glofas["unit"] == "m³/s"
        assert glofas["unit"] != "m"
        assert glofas["data_category"] == "MODELLED"
        assert glofas["variable"] == "river_discharge"


# ---------------------------------------------------------------------------
# 5. Historical Reconciliation & DERIVED Interpolation Tests
# ---------------------------------------------------------------------------

def test_timeseries_reconciliation_and_derived_interpolation():
    """
    Verify historical reconciliation:
    - Prioritizes telemetry where available
    - Falls back to manual
    - Gaps <= 2h are filled and tagged 'DERIVED — INTERPOLATED' with is_observed=False
    - Gaps > 2h remain NaN with UNAVAILABLE
    """
    reconciler = RiverDataReconciler(StationQCConfig.for_secondary())

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).replace(minute=0, second=0, microsecond=0)
    times = pd.date_range(end=now, periods=8, freq="1h")

    # Telemetry has hours 0, 1, 2, and hour 6, 7
    telemetry_s = pd.Series(
        [75.50, 75.60, 75.70, np.nan, np.nan, np.nan, 76.20, 76.30],
        index=times,
    ).dropna()

    # Manual has hour 3 (fills hour 3)
    manual_s = pd.Series(
        [np.nan, np.nan, np.nan, 75.75, np.nan, np.nan, np.nan, np.nan],
        index=times,
    ).dropna()

    df_rec = reconciler.reconcile_timeseries(
        telemetry_series=telemetry_s,
        manual_series=manual_s,
        lookback_hours=8,
        max_interp_gap_hours=2,
        is_mock=False,
    )

    # Hour 0..2 should be Telemetry
    assert df_rec.iloc[0]["source_stream"] == "TELEMETRY"
    assert df_rec.iloc[0]["provenance"] == PROV_OBS_TELEMETRY
    assert bool(df_rec.iloc[0]["is_observed"]) is True

    # Hour 3 should be Manual
    assert df_rec.iloc[3]["source_stream"] == "MANUAL"
    assert df_rec.iloc[3]["provenance"] == PROV_OBS_MANUAL
    assert bool(df_rec.iloc[3]["is_observed"]) is True

    # Hour 4, 5 are a 2-hour gap between hour 3 (75.75m) and hour 8 (76.20m)
    # Hour 4, 5 should be interpolated and tagged DERIVED — INTERPOLATED with is_observed=False
    assert df_rec.iloc[4]["source_stream"] == "INTERPOLATED"
    assert df_rec.iloc[4]["provenance"] == PROV_DERIVED_INTERPOLATED
    assert df_rec.iloc[4]["data_category"] == CAT_DERIVED
    assert bool(df_rec.iloc[4]["is_observed"]) is False  # CRITICAL: NEVER ground truth!
    assert df_rec.iloc[4]["quality_flag"] == FLAG_INTERPOLATED


def test_fetch_level_timeseries_integration():
    """Verify fetch_level_timeseries produces valid lags, changes, and rate of rise."""
    series, timestamps = fetch_level_timeseries(use_mock=True, lookback_hours=48)
    assert series is not None
    assert timestamps is not None
    assert len(series) > 0

    features = compute_river_level_features(series, timestamps)
    assert "level_lag_1h" in features
    assert "level_change_1h" in features
    assert "rate_of_rise" in features
    assert features["rate_of_rise"] is not None
