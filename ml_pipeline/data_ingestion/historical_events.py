"""
Historical flood event + raw historical dataset access, aligned to the
data-collection spec: raw data, event metadata, and the ML-ready
training table are kept as SEPARATE files (see data/historical/), never
merged in place. Nothing in this module fabricates values — every
loader returns exactly what's on disk, which starts EMPTY (schema
only) until real historical data is compiled and added.

File layout:
    data/historical/raw/raw_rainfall.csv         - Dataset A (rainfall)
    data/historical/raw/raw_river_data.csv       - Dataset A (river)
    data/historical/metadata/flood_events.csv    - event metadata
    data/historical/metadata/terrain_features.csv
    data/historical/metadata/data_dictionary.csv
    data/historical/metadata/source_metadata.csv
    data/historical/ml_ready/sonitpur_flood_training.csv  - Dataset B

Populating these is a data-collection task (ASDMA, Assam WRD, CWC,
NRSC/ISRO, IMD, satellite archives) — do not generate synthetic
historical observations. See data_dictionary.csv for every field's
definition, units, and source expectations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_RAINFALL_PATH = Path("data/historical/raw/raw_rainfall.csv")
RAW_RIVER_PATH = Path("data/historical/raw/raw_river_data.csv")
FLOOD_EVENTS_PATH = Path("data/historical/metadata/flood_events.csv")
TERRAIN_FEATURES_PATH = Path("data/historical/metadata/terrain_features.csv")
ML_READY_PATH = Path("data/historical/ml_ready/sonitpur_flood_training.csv")

EVENT_SCHEMA_COLUMNS = [
    "event_id",
    "flood_start",
    "flood_end",
    "district",
    "subdistrict",
    "affected_area",
    "affected_population",
    "severity_information",
    "primary_source",
    "secondary_source",
    "source_reference",
]

ML_READY_COLUMNS = [
    "timestamp", "latitude", "longitude", "district", "event_id",
    "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_12h",
    "rainfall_24h", "rainfall_48h", "rainfall_72h",
    "river_level", "river_level_change_1h", "river_level_change_3h",
    "river_level_change_6h", "distance_to_danger_level",
    "soil_moisture", "temperature", "vegetation_index",
    "elevation", "slope", "aspect", "flow_accumulation",
    "distance_to_river", "drainage_density", "land_cover",
    "flood_next_6h", "flood_next_12h", "flood_next_24h",
    "data_source", "quality_flag",
]


def _load_or_empty(path: Path, columns: list[str], parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, parse_dates=parse_dates or [])


def load_flood_events() -> pd.DataFrame:
    """Event metadata table (flood_events.csv). Empty until compiled."""
    return _load_or_empty(FLOOD_EVENTS_PATH, EVENT_SCHEMA_COLUMNS, parse_dates=["flood_start", "flood_end"])


def load_raw_rainfall() -> pd.DataFrame:
    return _load_or_empty(
        RAW_RAINFALL_PATH,
        ["timestamp", "station_id", "latitude", "longitude", "rainfall_mm", "data_source", "quality_flag"],
        parse_dates=["timestamp"],
    )


def load_raw_river_data() -> pd.DataFrame:
    return _load_or_empty(
        RAW_RIVER_PATH,
        ["timestamp", "station_id", "gauge_location", "latitude", "longitude", "river_level_m",
         "river_discharge", "danger_level_m", "data_source", "quality_flag"],
        parse_dates=["timestamp"],
    )


def load_ml_ready_dataset() -> pd.DataFrame:
    """Dataset B — sonitpur_flood_training.csv. Empty until built from raw + events."""
    return _load_or_empty(ML_READY_PATH, ML_READY_COLUMNS, parse_dates=["timestamp"])


def append_flood_event(
    event_id: str,
    flood_start: str,
    flood_end: str,
    district: str,
    primary_source: str,
    source_reference: str,
    subdistrict: str = "",
    affected_area: str = "",
    affected_population: str = "",
    severity_information: str = "",
    secondary_source: str = "",
) -> None:
    """
    Appends one verified flood event to flood_events.csv. Every event
    MUST carry a primary_source and source_reference — this is a hard
    requirement per the data-quality spec, not optional metadata.
    """
    if not primary_source or not source_reference:
        raise ValueError(
            "primary_source and source_reference are required for every "
            "flood event — do not record events without a traceable source."
        )

    df = load_flood_events()
    new_row = pd.DataFrame([{
        "event_id": event_id,
        "flood_start": flood_start,
        "flood_end": flood_end,
        "district": district,
        "subdistrict": subdistrict,
        "affected_area": affected_area,
        "affected_population": affected_population,
        "severity_information": severity_information,
        "primary_source": primary_source,
        "secondary_source": secondary_source,
        "source_reference": source_reference,
    }])
    df = pd.concat([df, new_row], ignore_index=True)

    # Normalize both date columns to a single consistent string format
    # before writing — otherwise a mix of already-parsed Timestamps
    # (from prior appends' load_flood_events call) and fresh plain
    # strings round-trips inconsistently and breaks re-parsing later.
    for col in ("flood_start", "flood_end"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
        df[col] = df[col].fillna("")

    FLOOD_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FLOOD_EVENTS_PATH, index=False)
