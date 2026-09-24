import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from ml_pipeline.data_ingestion.india_wris import (
    fetch_observed_river_level,
    fetch_level_timeseries,
    is_wris_available,
)
from ml_pipeline.data_ingestion.river_levels import (
    compute_threshold_features,
    get_station_thresholds,
)
from ml_pipeline.data_ingestion.glofas import fetch_glofas_discharge
from ml_pipeline.feature_engineering.river_level_features import compute_river_level_features


@pytest.fixture
def client():
    return TestClient(app)


def test_wris_station_metadata():
    """Verify both Brahmaputra (Primary) and Jia Bharali (Secondary) station thresholds and codes."""
    # Secondary: Jia Bharali
    thresh_sec = get_station_thresholds(settings.SECONDARY_GAUGE_LOCATION)
    assert thresh_sec is not None
    assert thresh_sec["river"] == "Jia Bharali"
    assert thresh_sec["gauge_location"] == "N.T.Road Xing"
    assert float(thresh_sec["danger_level_m"]) == 77.00
    assert float(thresh_sec["hfl_m"]) == 78.50
    assert settings.SECONDARY_STATION_CODE == "12359"

    # Primary: Brahmaputra
    thresh_prim = get_station_thresholds(settings.PRIMARY_GAUGE_LOCATION)
    assert thresh_prim is not None
    assert thresh_prim["river"] == "Brahmaputra"
    assert thresh_prim["gauge_location"] == "Tezpur"
    assert float(thresh_prim["danger_level_m"]) == 65.23
    assert float(thresh_prim["hfl_m"]) == 66.59
    assert settings.PRIMARY_STATION_CODE == "TEZPUR"


def test_wris_mock_fallback_structure():
    """Test that mock/simulated fallback returns realistic gauge observations for both rivers."""
    # Primary: Brahmaputra
    prim_data = fetch_observed_river_level("primary", use_mock=True)
    assert prim_data["station_code"] == settings.PRIMARY_STATION_CODE
    assert prim_data["river"] == settings.PRIMARY_RIVER_NAME
    assert prim_data["unit"] == "m"
    assert isinstance(prim_data["observed_level_m"], float)
    assert 55.0 <= prim_data["observed_level_m"] <= 68.0
    assert prim_data["data_category"] in ("DEMO / SIMULATED", "OBSERVED")
    assert prim_data["timestamp"] is not None
    assert prim_data["data_age_minutes"] is not None
    assert len(prim_data["timeseries"]) > 50

    # Secondary: Jia Bharali
    sec_data = fetch_observed_river_level("secondary", use_mock=True)
    assert sec_data["station_code"] == settings.SECONDARY_STATION_CODE
    assert sec_data["river"] in ("Jia Bharali", "JIABHARALI")
    assert sec_data["unit"] == "m"
    assert isinstance(sec_data["observed_level_m"], float)
    assert 74.0 <= sec_data["observed_level_m"] <= 78.0
    assert sec_data["data_category"] in ("DEMO / SIMULATED", "OBSERVED")


def test_wris_unavailable_handling(monkeypatch):
    """
    Test that when WRIS / CWC AFF API is unreachable and mock is disabled,
    it cleanly returns UNAVAILABLE without crashing or fabricating numbers.
    """
    monkeypatch.setattr("ml_pipeline.data_ingestion.india_wris.is_wris_available", lambda: False)
    monkeypatch.setattr("ml_pipeline.data_ingestion.india_wris.is_cwc_aff_available", lambda: False)

    data = fetch_observed_river_level(use_mock=False)

    assert data["observed_level_m"] is None
    assert data["data_category"] == "UNAVAILABLE"
    assert data["provenance"] == "UNAVAILABLE"
    assert data["timeseries"] == []
    assert data["data_age_minutes"] is None


def test_never_substitute_zero_or_glofas_for_observed_level(monkeypatch):
    """
    Ensure observed level is NEVER 0.0 or substituted with GloFAS discharge.
    """
    monkeypatch.setattr("ml_pipeline.data_ingestion.india_wris.is_wris_available", lambda: False)
    monkeypatch.setattr("ml_pipeline.data_ingestion.india_wris.is_cwc_aff_available", lambda: False)

    # 1. Unavailable state must return None, NEVER 0.0
    unavail = fetch_observed_river_level(use_mock=False)
    assert unavail["observed_level_m"] is not 0.0
    assert unavail["observed_level_m"] is None

    # 2. GloFAS discharge must be strictly separate
    glofas = fetch_glofas_discharge()
    if glofas.get("status") != "UNAVAILABLE":
        assert glofas["unit"] == "m³/s"
        assert glofas["data_category"] == "MODELLED"
        assert glofas["data_category"] != "OBSERVED"
        assert glofas["variable"] == "river_discharge"


def test_wris_threshold_calculation():
    """Verify threshold analysis correctly evaluates observed level against danger level."""
    thresh = get_station_thresholds(settings.JIA_BHARALI_GAUGE_LOCATION)
    dl = float(thresh["danger_level_m"])
    hfl = float(thresh["hfl_m"])

    # Test below danger level
    metrics_safe = compute_threshold_features(76.20, dl, hfl)
    assert metrics_safe["danger_ratio"] < 1.0
    assert metrics_safe["level_margin_m"] < 0.0  # 76.20 - 77.00 = -0.80

    # Test above danger level
    metrics_flood = compute_threshold_features(77.50, dl, hfl)
    assert metrics_flood["danger_ratio"] > 1.0
    assert metrics_flood["level_margin_m"] > 0.0  # 77.50 - 77.00 = +0.50


def test_wris_timeseries_and_river_features():
    """Test time-series extraction and feature computation (lags, changes, rate of rise)."""
    series, timestamps = fetch_level_timeseries(use_mock=True, lookback_hours=72)

    assert series is not None
    assert timestamps is not None
    assert len(series) > 0
    assert len(timestamps) == len(series)

    features = compute_river_level_features(series, timestamps)
    assert "level_lag_1h" in features
    assert "level_lag_24h" in features
    assert "level_change_1h" in features
    assert "rate_of_rise" in features
    assert features["level_lag_1h"] is not None


def test_api_river_status_endpoint(client):
    """Test GET /data/river-status includes primary (Brahmaputra) and secondary (Jia Bharali)."""
    resp = client.get("/data/river-status")
    assert resp.status_code == 200
    data = resp.json()

    assert "primary_river" in data
    assert "secondary_river" in data
    assert "observed_river_level" in data
    assert "glofas_discharge" in data
    assert "danger_level_thresholds" in data

    prim = data["primary_river"]
    assert prim["river"] in ("Brahmaputra", "BRAHMAPUTRA")
    assert prim["unit"] == "m"
    assert "provenance" in prim

    sec = data["secondary_river"]
    assert sec["river"] in ("Jia Bharali", "JIABHARALI")
    assert sec["unit"] == "m"
    assert "provenance" in sec

    # Threshold analysis must be present when level is available
    if prim["observed_level_m"] is not None:
        assert data["threshold_analysis"] is not None
        assert "danger_level_m" in data["threshold_analysis"]
        assert "distance_to_danger_m" in data["threshold_analysis"]


def test_api_sources_endpoint(client):
    """Test GET /data/sources contains India-WRIS / CWC gauge status."""
    resp = client.get("/data/sources")
    assert resp.status_code == 200
    data = resp.json()

    source_names = [s["name"] for s in data["sources"]]
    assert any("India-WRIS" in name or "CWC" in name for name in source_names)


def test_api_risk_forecast_endpoint(client):
    """Test GET /risk/forecast includes observed and projected river levels for both rivers."""
    resp = client.get("/risk/forecast")
    assert resp.status_code == 200
    data = resp.json()

    assert "current" in data
    assert "horizons" in data
    assert "+1h" in data["horizons"]
    assert "observed_river_level" in data["current"]

    h1 = data["horizons"]["+1h"]
    assert "predicted_level_m" in h1
    assert h1["predicted_level_m"] is not None
    assert isinstance(h1["predicted_level_m"], float)
    # Brahmaputra projected stage
    assert 55.0 <= h1["predicted_level_m"] <= 69.0
    assert "Projected:" in h1["predicted_level_label"]

    # Secondary predicted level (Jia Bharali)
    assert "secondary_predicted_level_m" in h1
    if h1["secondary_predicted_level_m"] is not None:
        assert 74.0 <= h1["secondary_predicted_level_m"] <= 81.0


def test_project_future_river_level_logic():
    """Verify hydrological stage extrapolation logic."""
    from ml_pipeline.feature_engineering.river_level_features import project_future_river_level

    # 1. When current level is None, returns None
    assert project_future_river_level(None, horizon_hours=1) is None

    # 2. Rising trend + precipitation causes projected stage to increase
    proj_rise = project_future_river_level(
        current_level_m=76.0,
        rate_of_rise_m_per_hr=0.05,
        horizon_hours=3,
        forecast_precip_mm=10.0,
    )
    assert proj_rise is not None
    assert proj_rise > 76.0

    # 3. Dry condition with negative rate of rise allows natural recession
    proj_fall = project_future_river_level(
        current_level_m=76.5,
        rate_of_rise_m_per_hr=-0.02,
        horizon_hours=6,
        forecast_precip_mm=0.0,
    )
    assert proj_fall is not None
    assert proj_fall < 76.5
