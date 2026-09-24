"""
Fetches real GPM/IMERG rainfall (via Earth Engine) for the time window
around each compiled historical flood event, and appends it to
data/historical/raw/raw_rainfall.csv.

This script REFUSES to run against mock data — it calls the GEE
ingestion path explicitly (use_mock=False) and fails loudly if
credentials aren't configured, rather than silently writing simulated
numbers into what's supposed to be the raw historical record. Set up
GEE credentials first (see ml_pipeline/data_ingestion/gee_client.py
docstring), then:

    python -m ml_pipeline.fetch_historical_rainfall

Per the data-collection spec, each event gets:
    ~14 days before flood_start  ->  ~7 days after flood_end
(or flood_start + 7 days, for events with no recorded flood_end)

Rainfall is fetched as a region-average over the configured Sonitpur
bounding box (see app/core/config.py) — not per individual gauge,
since exact station coordinates for the Jia Bharali and other gauges
weren't available in the compiled source data. This is recorded
honestly in the data_source field rather than implying station-level
precision the data doesn't have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from app.core.config import settings
from ml_pipeline.data_ingestion.gee_client import is_gee_available
from ml_pipeline.data_ingestion.historical_events import RAW_RAINFALL_PATH, load_flood_events
from ml_pipeline.data_ingestion.precipitation import fetch_imerg_timeseries

PRE_EVENT_DAYS = 14
POST_EVENT_DAYS = 7

REGION_CENTROID_LAT = (settings.REGION_MIN_LAT + settings.REGION_MAX_LAT) / 2
REGION_CENTROID_LON = (settings.REGION_MIN_LON + settings.REGION_MAX_LON) / 2
DATA_SOURCE_LABEL = "GPM_L3/IMERG_V07 via Earth Engine (region-averaged over Sonitpur bounding box)"


def fetch_window_for_event(event: pd.Series) -> pd.DataFrame:
    flood_start = pd.to_datetime(event["flood_start"])
    flood_end = pd.to_datetime(event["flood_end"]) if pd.notna(event.get("flood_end")) else flood_start

    window_start = flood_start - pd.Timedelta(days=PRE_EVENT_DAYS)
    window_end = flood_end + pd.Timedelta(days=POST_EVENT_DAYS)
    lookback_hours = (window_end - window_start).total_seconds() / 3600

    logger.info(
        f"Fetching IMERG for event '{event['event_id']}': "
        f"{window_start} -> {window_end} ({lookback_hours:.0f}h)"
    )

    # use_mock=False is deliberate: this must fail rather than
    # silently substitute simulated rainfall into the raw record.
    precip_df = fetch_imerg_timeseries(
        end_time=window_end.to_pydatetime(), lookback_hours=lookback_hours, use_mock=False
    )

    precip_df["station_id"] = "sonitpur_region_avg"
    precip_df["latitude"] = REGION_CENTROID_LAT
    precip_df["longitude"] = REGION_CENTROID_LON
    precip_df["data_source"] = DATA_SOURCE_LABEL
    precip_df["quality_flag"] = "satellite-derived"
    precip_df = precip_df.rename(columns={"precip_mm": "rainfall_mm"})
    return precip_df[["timestamp", "station_id", "latitude", "longitude", "rainfall_mm", "data_source", "quality_flag"]]


def main():
    if not is_gee_available():
        logger.error(
            "Earth Engine is not available (no credentials configured). "
            "This script will not run against mock data — set up GEE "
            "credentials first (see gee_client.py) and try again."
        )
        sys.exit(1)

    events = load_flood_events()
    if events.empty:
        logger.error("No flood events found in flood_events.csv — nothing to fetch around.")
        sys.exit(1)

    all_rows = []
    failed_events = []
    for _, event in events.iterrows():
        try:
            all_rows.append(fetch_window_for_event(event))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to fetch rainfall for event '{event['event_id']}': {exc}")
            failed_events.append(event["event_id"])

    if not all_rows:
        logger.error("No rainfall data was successfully fetched for any event.")
        sys.exit(1)

    new_data = pd.concat(all_rows, ignore_index=True)

    existing = pd.read_csv(RAW_RAINFALL_PATH, parse_dates=["timestamp"]) if RAW_RAINFALL_PATH.exists() else pd.DataFrame()
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "station_id"]).sort_values("timestamp")

    RAW_RAINFALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RAW_RAINFALL_PATH, index=False)

    logger.info(
        f"Wrote {len(new_data)} new rows ({len(combined)} total) to {RAW_RAINFALL_PATH}. "
        f"Succeeded: {len(all_rows)}/{len(events)} events. Failed: {failed_events or 'none'}"
    )


if __name__ == "__main__":
    main()
