"""
Training entrypoint for the XGBoost flash-flood model — rebuilt
around the data-collection spec's Dataset B (ML-ready table) and a
chronological, event-grouped split (never a random row-level split,
which would leak information between train and test for a time-series
problem — see target_builder.chronological_split).

Run with:
    python -m ml_pipeline.train --target flood_next_24h

Prerequisites:
    - data/historical/ml_ready/sonitpur_flood_training.csv populated
      with real observations (predictors) and computed targets
      (flood_next_6h / flood_next_12h / flood_next_24h). This file is
      currently EMPTY — populating it is a data-collection task, not
      something this script can do (no synthetic data is generated
      anywhere in this pipeline).
    - data/historical/metadata/flood_events.csv populated, used here
      only to build the chronological event split.
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from ml_pipeline.data_ingestion.historical_events import load_flood_events, load_ml_ready_dataset
from ml_pipeline.data_ingestion.target_builder import chronological_split
from ml_pipeline.models.xgboost_model import FlashFloodXGBModel


def build_split_dataframes(target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ml_ready = load_ml_ready_dataset()
    events = load_flood_events()

    if ml_ready.empty:
        raise RuntimeError(
            "data/historical/ml_ready/sonitpur_flood_training.csv is empty. "
            "This dataset must be built from real, sourced historical "
            "observations (see the data-collection spec and "
            "data_dictionary.csv) — no synthetic data will be generated "
            "here."
        )
    if target_col not in ml_ready.columns:
        raise ValueError(f"Target column '{target_col}' not found in the ML-ready dataset.")

    split = chronological_split(events)

    # Rows with a non-null event_id are assigned by their event's split;
    # rows with no event_id (ordinary/non-flood periods) are split by
    # timestamp using the same train/val fraction boundary, so normal
    # periods are chronologically consistent with the event split too.
    train_ids = set(split["train"]) | set(split["val"])
    test_ids = set(split["test"])

    has_event = ml_ready["event_id"].notna()
    train_mask = has_event & ml_ready["event_id"].isin(train_ids)
    test_mask = has_event & ml_ready["event_id"].isin(test_ids)

    non_event = ml_ready[~has_event].sort_values("timestamp")
    cutoff = non_event["timestamp"].quantile(0.8) if not non_event.empty else None
    if cutoff is not None:
        train_mask |= (~has_event) & (ml_ready["timestamp"] <= cutoff)
        test_mask |= (~has_event) & (ml_ready["timestamp"] > cutoff)

    return ml_ready[train_mask].copy(), ml_ready[test_mask].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", default="flood_next_24h",
        choices=["flood_next_6h", "flood_next_12h", "flood_next_24h"],
    )
    args = parser.parse_args()

    train_df, test_df = build_split_dataframes(args.target)
    logger.info(
        f"Training on {len(train_df)} rows, testing on {len(test_df)} rows "
        f"(chronological, event-grouped split; target={args.target})"
    )

    model = FlashFloodXGBModel()
    metrics = model.train_from_prepared_split(train_df, test_df, label_col=args.target)
    logger.info(f"Training metrics: {metrics}")

    model.save()


if __name__ == "__main__":
    main()
