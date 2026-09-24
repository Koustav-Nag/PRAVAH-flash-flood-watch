import pytest
from ml_pipeline.data_ingestion.river_levels import (
    get_station_thresholds,
    compute_threshold_features,
    compute_danger_level_ratio
)

def test_get_station_thresholds():
    # Jia Bharali
    data = get_station_thresholds('N.T.Road Xing')
    assert data is not None
    assert data["danger_level_m"] == 77.0
    
    # Brahmaputra
    data = get_station_thresholds('Tezpur')
    assert data is not None
    assert data["danger_level_m"] == 65.23
    
    # Nonexistent
    assert get_station_thresholds('NonexistentStation') is None

def test_compute_threshold_features():
    # Normal case
    res = compute_threshold_features(level_m=75.0, danger_level_m=77.0, hfl_m=78.5)
    assert res is not None
    assert res["danger_ratio"] == pytest.approx(75.0 / 77.0)
    assert res["level_margin_m"] == pytest.approx(75.0 - 77.0)
    assert res["hfl_ratio"] == pytest.approx(75.0 / 78.5)
    assert res["hfl_margin_m"] == pytest.approx(75.0 - 78.5)
    
    # At danger level (margin=0)
    res = compute_threshold_features(level_m=77.0, danger_level_m=77.0)
    assert res["level_margin_m"] == 0.0
    assert res["danger_ratio"] == 1.0
    
    # Above danger level (margin>0)
    res = compute_threshold_features(level_m=79.0, danger_level_m=77.0)
    assert res["level_margin_m"] > 0
    assert res["danger_ratio"] > 1.0
    
    # None level
    res = compute_threshold_features(level_m=None, danger_level_m=77.0)
    assert res["danger_ratio"] is None
    assert res["level_margin_m"] is None
    assert res["hfl_ratio"] is None
    assert res["hfl_margin_m"] is None
    
    # None hfl
    res = compute_threshold_features(level_m=75.0, danger_level_m=77.0, hfl_m=None)
    assert res["hfl_ratio"] is None
    assert res["hfl_margin_m"] is None

def test_danger_ratio_is_not_probability():
    """Verify danger_ratio is just a ratio, can be > 1.0"""
    res = compute_threshold_features(level_m=100.0, danger_level_m=50.0)
    assert res["danger_ratio"] == 2.0

def test_existing_compute_danger_level_ratio():
    # Existing functionality must still work
    ratio = compute_danger_level_ratio(77.0, 'N.T.Road Xing')
    assert ratio == 1.0
