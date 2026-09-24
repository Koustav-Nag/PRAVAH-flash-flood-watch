"""
Central configuration for the Flash Flood Prediction System.

Loads settings from environment variables / a .env file. Nothing here
should contain secrets directly — the actual GEE service-account key
lives in a JSON file whose *path* is referenced below, and that file
must never be committed to version control.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- App ---
    APP_NAME: str = "Flash Flood Prediction System"
    ENV: str = "development"

    # --- Google Earth Engine ---
    # Path to the service-account JSON key file. See README.md ->
    # "Google Earth Engine setup" for how to create one.
    GEE_SERVICE_ACCOUNT_EMAIL: str | None = None
    GEE_SERVICE_ACCOUNT_KEY_PATH: str | None = None
    GEE_PROJECT_ID: str | None = None

    # --- CWC Advisory Flood Forecast (AFF) / India-WRIS ---
    CWC_AFF_BASE_URL: str = "https://aff.india-water.gov.in"
    CWC_AFF_TABLE_URL: str = "https://aff.india-water.gov.in/textdata/Floodday_table_view_header.txt"
    CWC_AFF_TIMESERIES_URL: str = "https://aff.india-water.gov.in/Timeseries"

    # --- PRIMARY River Gauge: Brahmaputra at Tezpur / Ganeshghat ---
    PRIMARY_RIVER_NAME: str = "Brahmaputra"
    PRIMARY_STATION_NAME: str = "Tezpur"
    PRIMARY_STATION_CODE: str = "TEZPUR"
    PRIMARY_GAUGE_LOCATION: str = "Tezpur"
    PRIMARY_GAUGE_LAT: float = 26.61667
    PRIMARY_GAUGE_LON: float = 92.79731
    PRIMARY_DANGER_LEVEL_M: float = 65.23
    PRIMARY_WARNING_LEVEL_M: float = 64.23
    PRIMARY_HFL_M: float = 66.59
    PRIMARY_MIN_STAGE_M: float = 55.86
    PRIMARY_MAX_STAGE_M: float = 68.59
    PRIMARY_BASE_LEVEL_M: float = 63.80
    PRIMARY_MAX_RATE_OF_CHANGE_M_PER_HR: float = 1.00
    PRIMARY_MAX_SPIKE_DEVIATION_M: float = 1.00
    PRIMARY_MAX_FLATLINE_HOURS: int = 24

    # --- SECONDARY River Gauge: Jia Bharali at N.T. Road Crossing (Local Flash-Flood Signal) ---
    SECONDARY_RIVER_NAME: str = "Jia Bharali"
    SECONDARY_STATION_NAME: str = "NT ROAD CROSSING JIA-BHARALI"
    SECONDARY_STATION_CODE: str = "12359"
    SECONDARY_GAUGE_LOCATION: str = "N.T.Road Xing"
    SECONDARY_GAUGE_LAT: float = 26.8105
    SECONDARY_GAUGE_LON: float = 92.87972
    SECONDARY_DANGER_LEVEL_M: float = 77.00
    SECONDARY_WARNING_LEVEL_M: float = 76.00
    SECONDARY_HFL_M: float = 78.50
    SECONDARY_MIN_STAGE_M: float = 72.00
    SECONDARY_MAX_STAGE_M: float = 81.00
    SECONDARY_BASE_LEVEL_M: float = 75.80
    SECONDARY_MAX_RATE_OF_CHANGE_M_PER_HR: float = 1.50
    SECONDARY_MAX_SPIKE_DEVIATION_M: float = 1.20
    SECONDARY_MAX_FLATLINE_HOURS: int = 24

    # Legacy WRIS aliases (preserved for backward compatibility)
    WRIS_API_BASE_URL: str = "https://indiawris.gov.in/wims/api"
    WRIS_API_KEY: str | None = None
    WRIS_STATION_CODE: str = "12359"
    WRIS_STATION_NAME: str = "N.T. Road Crossing"
    WRIS_RIVER_NAME: str = "Jia Bharali"
    WRIS_TELEMETRY_ENDPOINT: str = "/getStationData"
    WRIS_MANUAL_ENDPOINT: str = "/getManualData"
    WRIS_DANGER_LEVEL_M: float = 77.00
    WRIS_WARNING_LEVEL_M: float = 76.00
    WRIS_HFL_M: float = 78.50
    WRIS_MIN_STAGE_M: float = 72.00
    WRIS_MAX_STAGE_M: float = 81.00
    WRIS_MAX_RATE_OF_CHANGE_M_PER_HR: float = 1.50
    WRIS_MAX_SPIKE_DEVIATION_M: float = 1.20
    WRIS_MAX_FLATLINE_HOURS: int = 24
    WRIS_TELEMETRY_MAX_AGE_HOURS: float = 3.0
    WRIS_MANUAL_MAX_AGE_HOURS: float = 24.0

    # --- Pilot region: Sonitpur District, Assam ---
    REGION_NAME: str = "Sonitpur, Assam"
    REGION_MIN_LON: float = 92.60
    REGION_MIN_LAT: float = 26.55
    REGION_MAX_LON: float = 93.45
    REGION_MAX_LAT: float = 27.20

    # Legacy gauge location aliases
    JIA_BHARALI_GAUGE_LAT: float = 26.8105
    JIA_BHARALI_GAUGE_LON: float = 92.87972
    JIA_BHARALI_GAUGE_LOCATION: str = "N.T.Road Xing"

    # --- Forecast configuration ---
    NOWCAST_HORIZON_HOURS: int = 2
    SHORT_TERM_HORIZON_HOURS: int = 6
    FORECAST_HORIZONS_HOURS: list[int] = [1, 3, 6, 12, 24]

    # --- Risk thresholds (physics-informed baseline, tune with data) ---
    RISK_LOW_MAX: float = 0.25
    RISK_MEDIUM_MAX: float = 0.5
    RISK_HIGH_MAX: float = 0.75
    # anything above RISK_HIGH_MAX => "severe"


settings = Settings()
