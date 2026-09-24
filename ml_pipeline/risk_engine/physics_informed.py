"""
Physics-informed flash-flood risk score.

This is a transparent, weighted-indicator scoring model grounded in
established hydrological drivers of flash floods in hilly terrain:
rainfall intensity, accumulated rainfall, antecedent soil saturation,
terrain steepness/drainage density, and current observed surface
water. It exists for two reasons:

1. INTERPRETABILITY — every score component can be explained to a
   judge ("risk is high because 3h rainfall is 40mm on steep terrain
   with saturated antecedent conditions"), unlike a black-box model.
2. FALLBACK — it works even before the XGBoost model has been trained
   on enough historical events, and it keeps the system functional
   if the ML path fails or a feature is missing.

Each sub-score is normalized to [0, 1]; the final score is a weighted
sum, also in [0, 1]. Weights are a documented starting point — they
should be tuned once real historical events are compiled and can be
used to check the score's ranking against known outcomes.
"""

from __future__ import annotations

from app.core.config import settings

WEIGHTS = {
    "rainfall_intensity": 0.30,
    "rainfall_accumulation": 0.25,
    "antecedent_saturation": 0.15,
    "terrain_susceptibility": 0.20,
    "current_water_signal": 0.10,
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _rainfall_intensity_score(features: dict) -> float:
    # Normalize 1h rainfall against a commonly-cited "heavy rain" threshold
    # (~30mm/hr) used in Indian meteorological rainfall-intensity categories.
    rain_1h = features.get("rainfall_1h") or 0.0
    return _clip01(rain_1h / 30.0)


def _rainfall_accumulation_score(features: dict) -> float:
    # Blend 6h and 24h accumulation; 6h captures the flash-flood-relevant
    # burst, 24h captures sustained event totals.
    rain_6h = features.get("rainfall_6h") or 0.0
    rain_24h = features.get("rainfall_24h") or 0.0
    score_6h = _clip01(rain_6h / 60.0)
    score_24h = _clip01(rain_24h / 150.0)
    return 0.6 * score_6h + 0.4 * score_24h


def _antecedent_saturation_score(features: dict) -> float:
    antecedent = features.get("antecedent_rainfall_7d_mm") or 0.0
    persistence = features.get("rainfall_persistence_6h_frac") or 0.0
    score_antecedent = _clip01(antecedent / 100.0)
    return 0.7 * score_antecedent + 0.3 * _clip01(persistence)


def _terrain_susceptibility_score(features: dict) -> float:
    slope = features.get("slope") or 0.0
    drainage = features.get("drainage_density") or 0.0
    # Steeper terrain + higher drainage density => faster runoff
    # concentration => higher flash-flood susceptibility.
    score_slope = _clip01(slope / 30.0)
    score_drainage = _clip01(drainage)
    return 0.6 * score_slope + 0.4 * score_drainage


def _current_water_signal_score(features: dict) -> float:
    water_fraction = features.get("surface_water_fraction") or 0.0
    # Baseline water fraction in hilly terrain is low (~1-6%); treat
    # anything meaningfully above that as an active-flooding signal.
    return _clip01((water_fraction - 0.05) / 0.20)


def compute_physics_informed_score(features: dict) -> dict:
    """
    Returns a dict with the overall score, its risk category, and a
    breakdown of each component — the breakdown is what makes this
    explainable to a judge or an end user.
    """
    components = {
        "rainfall_intensity": _rainfall_intensity_score(features),
        "rainfall_accumulation": _rainfall_accumulation_score(features),
        "antecedent_saturation": _antecedent_saturation_score(features),
        "terrain_susceptibility": _terrain_susceptibility_score(features),
        "current_water_signal": _current_water_signal_score(features),
    }

    overall_score = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    overall_score = _clip01(overall_score)

    return {
        "score": overall_score,
        "category": categorize_risk(overall_score),
        "components": components,
        "weights": WEIGHTS,
    }


def categorize_risk(score: float) -> str:
    if score <= settings.RISK_LOW_MAX:
        return "low"
    if score <= settings.RISK_MEDIUM_MAX:
        return "medium"
    if score <= settings.RISK_HIGH_MAX:
        return "high"
    return "severe"
