"""
Leakage-safe target generation: flood_next_6h / flood_next_12h /
flood_next_24h.

Hard rule (per the data-collection spec): at observation timestamp t,
the target may use information about events starting AFTER t, but
every PREDICTOR column at row t must only use information available
at or before t. This module only builds targets — it never touches
predictor columns, and callers must build predictors from a strictly
trailing window before joining these targets on.
"""

from __future__ import annotations

import pandas as pd

HORIZONS_HOURS = {"flood_next_6h": 6, "flood_next_12h": 12, "flood_next_24h": 24}


def build_targets_for_timestamp(timestamp: pd.Timestamp, flood_events: pd.DataFrame) -> dict:
    """
    Given one observation timestamp and the flood_events table (with
    flood_start columns), returns the three binary targets.

    flood_next_Xh = 1 iff some event's flood_start falls in
    (timestamp, timestamp + Xh] — strictly after t, which is what
    makes this leakage-safe: the label looks forward, the predictors
    (built elsewhere) must never look forward.
    """
    targets = {}
    if flood_events.empty:
        return {k: 0 for k in HORIZONS_HOURS}

    starts = pd.to_datetime(flood_events["flood_start"])
    ts = pd.to_datetime(timestamp)
    for target_col, hours in HORIZONS_HOURS.items():
        window_end = ts + pd.Timedelta(hours=hours)
        in_window = (starts > ts) & (starts <= window_end)
        targets[target_col] = int(in_window.any())
    return targets


def build_targets_for_series(timestamps: pd.Series, flood_events: pd.DataFrame) -> pd.DataFrame:
    """Vectorized convenience wrapper over build_targets_for_timestamp."""
    rows = [build_targets_for_timestamp(ts, flood_events) for ts in timestamps]
    return pd.DataFrame(rows, index=timestamps.index)


def chronological_split(
    flood_events: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> dict:
    """
    Splits flood_events chronologically by flood_start into
    train/validation/test event_id sets, keeping each event whole in
    exactly one split (per spec section 13 — never split a single
    event's observations across sets).

    Returns {"train": [...event_ids], "val": [...], "test": [...]}.
    Any observation-level dataset should be split by joining on
    event_id to these sets (non-event / normal-period rows need a
    separate chronological cutoff, e.g. by timestamp, decided when
    that data exists).
    """
    if flood_events.empty:
        return {"train": [], "val": [], "test": []}

    ordered = flood_events.sort_values("flood_start")
    n = len(ordered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": ordered.iloc[:n_train]["event_id"].tolist(),
        "val": ordered.iloc[n_train:n_train + n_val]["event_id"].tolist(),
        "test": ordered.iloc[n_train + n_val:]["event_id"].tolist(),
    }
