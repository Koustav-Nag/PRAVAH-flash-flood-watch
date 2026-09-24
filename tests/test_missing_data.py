import pytest
import pandas as pd
from ml_pipeline.data_ingestion.river_levels import (
    compute_danger_level_ratio,
    compute_threshold_features
)
from ml_pipeline.data_ingestion.precipitation import compute_accumulation_features

def test_compute_danger_level_ratio_missing():
    assert compute_danger_level_ratio(None, 'N.T.Road Xing') is None
    assert compute_danger_level_ratio(75.0, 'NonexistentStation') is None
    
def test_compute_accumulation_features_empty():
    with pytest.raises(ValueError, match="precip_df is empty"):
        compute_accumulation_features(pd.DataFrame())

def test_compute_threshold_features_none():
    res = compute_threshold_features(level_m=None, danger_level_m=77.0)
    assert res["danger_ratio"] is None
    assert res["level_margin_m"] is None
    assert res["hfl_ratio"] is None
    assert res["hfl_margin_m"] is None

def test_none_not_interpreted_as_zero():
    # Verify that passing None does not compute as 0.0
    res = compute_threshold_features(level_m=None, danger_level_m=77.0)
    assert res["level_margin_m"] is not 0.0
    assert res["danger_ratio"] is not 0.0
