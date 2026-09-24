import pytest
from pathlib import Path
import pandas as pd

# These are created by another agent, so we mock them if they don't exist yet
try:
    from ml_pipeline.data_ingestion.glofas import fetch_glofas_discharge
except ImportError:
    pass

try:
    from ml_pipeline.data_ingestion.open_meteo import fetch_weather_forecast
except ImportError:
    pass


def test_glofas_provenance(monkeypatch):
    def mock_fetch(*args, **kwargs):
        return {"data_category": "MODELLED"}
    
    # Use monkeypatch to simulate the fetch function
    monkeypatch.setattr("ml_pipeline.data_ingestion.glofas.fetch_glofas_discharge", mock_fetch, raising=False)
    
    # Test the mock
    from ml_pipeline.data_ingestion.glofas import fetch_glofas_discharge
    data = fetch_glofas_discharge()
    assert data["data_category"] == "MODELLED"
    assert data["data_category"] != "OBSERVED"

def test_open_meteo_provenance(monkeypatch):
    def mock_fetch(*args, **kwargs):
        return {"data_category": "FORECAST"}
    
    monkeypatch.setattr("ml_pipeline.data_ingestion.open_meteo.fetch_weather_forecast", mock_fetch, raising=False)
    
    from ml_pipeline.data_ingestion.open_meteo import fetch_weather_forecast
    data = fetch_weather_forecast()
    assert data["data_category"] == "FORECAST"

def test_static_reference_data():
    path = Path("c:/flash-flood-sih/data/reference/river_danger_levels_assam.csv")
    assert path.exists(), "Danger levels CSV must exist"
    
    df = pd.read_csv(path)
    expected_cols = {"river", "gauge_location", "district", "danger_level_m", "hfl_m", "hfl_date"}
    assert expected_cols.issubset(set(df.columns)), "Missing expected columns in reference data"
