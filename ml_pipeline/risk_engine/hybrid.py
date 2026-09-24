"""
Hybrid risk engine: combines the XGBoost model's predictive
probability with the physics-informed score.

Strategy:
- If the XGBoost model is trained: final score = weighted blend of
  (XGBoost probability) and (physics-informed score). The blend
  favors XGBoost but keeps the physics score as a sanity-check/
  smoothing term, which also guards against a model that has only
  seen a handful of historical events and could otherwise swing wildly.
- If the XGBoost model is NOT trained/available: fall back entirely
  to the physics-informed score, so the system is always demoable.
"""

from __future__ import annotations

from ml_pipeline.models.xgboost_model import FlashFloodXGBModel
from ml_pipeline.risk_engine.physics_informed import (
    categorize_risk,
    compute_physics_informed_score,
)

XGB_BLEND_WEIGHT = 0.65  # weight given to the ML model when available
PHYSICS_BLEND_WEIGHT = 1 - XGB_BLEND_WEIGHT


def compute_hybrid_risk(features: dict, xgb_model: FlashFloodXGBModel) -> dict:
    physics_result = compute_physics_informed_score(features)

    xgb_proba = xgb_model.predict_proba(features) if xgb_model.is_trained() else None

    if xgb_proba is not None:
        final_score = (
            XGB_BLEND_WEIGHT * xgb_proba + PHYSICS_BLEND_WEIGHT * physics_result["score"]
        )
        source = "hybrid (xgboost + physics-informed)"
    else:
        final_score = physics_result["score"]
        source = "physics-informed (fallback — xgboost model not trained yet)"

    return {
        "final_score": final_score,
        "final_category": categorize_risk(final_score),
        "source": source,
        "xgboost_probability": xgb_proba,
        "physics_informed": physics_result,
    }
