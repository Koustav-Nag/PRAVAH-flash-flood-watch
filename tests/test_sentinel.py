import pytest
from datetime import datetime
from ml_pipeline.data_ingestion.sentinel import (
    fetch_surface_water_fraction,
    fetch_landcover_summary,
)
from ml_pipeline.feature_engineering.fusion import build_feature_vector


def test_fetch_surface_water_fraction_mock():
    water_frac = fetch_surface_water_fraction(use_mock=True)
    assert isinstance(water_frac, float)
    assert 0.0 <= water_frac <= 1.0


def test_fetch_landcover_summary_mock():
    lc = fetch_landcover_summary(use_mock=True)
    assert isinstance(lc, dict)
    expected_keys = {"forest_frac", "cropland_frac", "built_up_frac", "bare_soil_frac", "water_frac"}
    assert expected_keys.issubset(set(lc.keys()))
    
    total = sum(lc.values())
    assert 0.95 <= total <= 1.05
    for k, v in lc.items():
        assert 0.0 <= v <= 1.0


def test_sentinel_gee_fallback_on_failure(monkeypatch):
    """Verify that when GEE is marked available but calls fail, functions return fallback without crashing."""
    monkeypatch.setattr("ml_pipeline.data_ingestion.sentinel.is_gee_available", lambda: True)
    
    def raise_gee_error(*args, **kwargs):
        raise RuntimeError("GEE connection error or quota exceeded")
    
    monkeypatch.setattr("ml_pipeline.data_ingestion.sentinel._fetch_surface_water_from_gee", raise_gee_error)
    monkeypatch.setattr("ml_pipeline.data_ingestion.sentinel._fetch_landcover_from_gee", raise_gee_error)

    # Should not raise RuntimeError; should log warning and return mock baseline
    water_frac = fetch_surface_water_fraction(use_mock=None)
    assert isinstance(water_frac, float)
    assert 0.0 <= water_frac <= 1.0

    lc = fetch_landcover_summary(use_mock=None)
    assert isinstance(lc, dict)
    assert "water_frac" in lc
    assert "forest_frac" in lc


def test_fusion_includes_sentinel_features():
    features = build_feature_vector(use_mock=True)
    assert "surface_water_fraction" in features
    assert isinstance(features["surface_water_fraction"], float)
    
    assert "landcover_forest_frac" in features
    assert "landcover_cropland_frac" in features
    assert "landcover_built_up_frac" in features
    assert "landcover_bare_soil_frac" in features
    assert "landcover_water_frac" in features
