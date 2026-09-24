import pandas as pd
import pytest
from ml_pipeline.data_ingestion.target_builder import build_targets_for_timestamp, chronological_split
from ml_pipeline.data_ingestion.precipitation import compute_accumulation_features

def test_build_targets_for_timestamp():
    t = pd.Timestamp("2023-01-01 10:00:00")
    
    # Event inside (t, t+horizon]
    events = pd.DataFrame({"flood_start": [pd.Timestamp("2023-01-01 12:00:00")]})
    targets = build_targets_for_timestamp(t, events)
    assert targets["flood_next_6h"] == 1
    assert targets["flood_next_12h"] == 1
    
    # Event at exactly t does NOT set target=1 (must be strictly after t)
    events_at_t = pd.DataFrame({"flood_start": [t]})
    targets_at_t = build_targets_for_timestamp(t, events_at_t)
    assert targets_at_t["flood_next_6h"] == 0

def test_chronological_split():
    events = pd.DataFrame({
        "event_id": [1, 2, 3, 4, 5],
        "flood_start": [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-02"),
            pd.Timestamp("2023-01-03"),
            pd.Timestamp("2023-01-04"),
            pd.Timestamp("2023-01-05"),
        ]
    })
    
    splits = chronological_split(events, train_frac=0.6, val_frac=0.2)
    assert splits["train"] == [1, 2, 3]
    assert splits["val"] == [4]
    assert splits["test"] == [5]

def test_feature_leakage():
    """Test that features at time t cannot include information from time t+1"""
    t = pd.Timestamp("2023-01-01 10:00:00")
    df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2023-01-01 09:00:00"),
            pd.Timestamp("2023-01-01 10:00:00"),
            pd.Timestamp("2023-01-01 11:00:00")  # Future info
        ],
        "precip_mm": [10.0, 5.0, 100.0]
    })
    
    # To compute features at time t, we must only pass data up to t
    df_at_t = df[df["timestamp"] <= t]
    features = compute_accumulation_features(df_at_t)
    
    # Should not include the 100.0 from t+1
    # 1h window: (09:00, 10:00] -> contains 5.0
    assert features["rainfall_1h"] == 5.0
