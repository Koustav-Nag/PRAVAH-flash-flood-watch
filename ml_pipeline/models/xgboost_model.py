"""
XGBoost flash-flood risk model.

Trains a gradient-boosted classifier on the fused feature vectors
(see feature_engineering/fusion.py) against a binary/ordinal flood
label compiled from historical events (see historical_events.py).

This module intentionally keeps model I/O simple (a single joblib
file) since the prototype's priority is a working, explainable
end-to-end pipeline, not model-serving infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

MODEL_PATH = Path("data/processed/xgboost_flood_model.joblib")

# Features expected by the model — kept as an explicit list (rather
# than "whatever fusion.py returns") so schema drift is caught early.
# Names match data/historical/ml_ready/sonitpur_flood_training.csv
# (the spec's official ML-ready schema).
FEATURE_COLUMNS = [
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_12h",
    "rainfall_24h",
    "rainfall_48h",
    "rainfall_72h",
    "river_level",
    "river_level_change_1h",
    "river_level_change_3h",
    "river_level_change_6h",
    "distance_to_danger_level",
    "soil_moisture",
    "temperature",
    "vegetation_index",
    "elevation",
    "slope",
    "aspect",
    "flow_accumulation",
    "distance_to_river",
    "drainage_density",
]


class FlashFloodXGBModel:
    def __init__(self):
        self.model: xgb.XGBClassifier | None = None

    def is_trained(self) -> bool:
        return self.model is not None

    def train(self, training_df: pd.DataFrame, label_col: str = "flood_occurred") -> dict:
        """
        training_df must contain FEATURE_COLUMNS + label_col (0/1).
        Returns evaluation metrics on a held-out split.

        NOTE: with a small compiled historical-event dataset (typical
        for a first SIH pass), treat these metrics as indicative, not
        production-grade — flag this honestly in the demo.
        """
        missing = set(FEATURE_COLUMNS) - set(training_df.columns)
        if missing:
            raise ValueError(f"training_df is missing required columns: {missing}")

        X = training_df[FEATURE_COLUMNS]
        y = training_df[label_col]

        min_class_count = y.value_counts().min() if y.nunique() > 1 else 0

        # With only a handful of compiled historical events (typical for
        # a first SIH pass), a stratified held-out split isn't reliable
        # or sometimes even possible. Fit on the full dataset and report
        # that no held-out evaluation was performed, rather than crash
        # or silently produce a meaningless split.
        if len(training_df) < 10 or min_class_count < 3:
            logger.warning(
                f"Only {len(training_df)} training examples "
                f"(smallest class has {min_class_count}) — too few for a "
                "reliable held-out split. Fitting on all available data; "
                "treat this model as indicative only until more historical "
                "events are compiled."
            )
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
            )
            self.model.fit(X, y)
            return {
                "warning": "Trained on full dataset — no held-out evaluation "
                           "possible with this few examples.",
                "n_examples": len(training_df),
                "n_positive": int(y.sum()),
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
        )

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {"report": classification_report(y_test, y_pred, output_dict=True)}
        if y_test.nunique() > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

        logger.info(f"XGBoost training complete. ROC-AUC: {metrics.get('roc_auc', 'n/a')}")
        return metrics

    def train_from_prepared_split(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str = "flood_next_24h"
    ) -> dict:
        """
        Trains on a caller-provided chronological split (see
        target_builder.chronological_split) rather than a random one —
        required once real event data exists, since randomly splitting
        individual rows of a time-series flood dataset leaks
        information between train and test (spec section 13).
        """
        missing = set(FEATURE_COLUMNS) - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df is missing required columns: {missing}")

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[label_col]
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        if test_df.empty:
            return {"warning": "No test split available — trained without held-out evaluation."}

        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[label_col]
        y_pred = self.model.predict(X_test)
        metrics = {"report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)}
        if y_test.nunique() > 1:
            y_proba = self.model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        return metrics

    def predict_proba(self, features: dict) -> float | None:
        """
        Returns predicted flood probability in [0, 1], or None if the
        model isn't trained yet — callers should fall back to the
        physics-informed score in that case.
        """
        if not self.is_trained():
            logger.warning("XGBoost model not trained — no prediction available.")
            return None

        # Not every FEATURE_COLUMNS entry is produced by the live nowcast
        # pipeline yet (e.g. river_level, soil_moisture — no live source
        # wired in). Missing -> NaN, which XGBoost handles natively,
        # rather than None (which would break the DataFrame dtype).
        row = {col: features.get(col, float("nan")) for col in FEATURE_COLUMNS}
        row = {k: (float("nan") if v is None else v) for k, v in row.items()}
        X = pd.DataFrame([row])
        return float(self.model.predict_proba(X)[0, 1])

    def feature_importance(self) -> dict:
        """Returns feature importances for explainability output."""
        if not self.is_trained():
            return {}
        return dict(zip(FEATURE_COLUMNS, self.model.feature_importances_.tolist()))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path = MODEL_PATH) -> bool:
        if not path.exists():
            logger.warning(f"No saved model found at {path}")
            return False
        self.model = joblib.load(path)
        return True
