"""
Combined "nearest safe place" recommender for alert dispatch.

Strategy: prefer a known shelter within a reasonable distance; if none
exists (empty/sparse registry, or nearest shelter is implausibly far
for the given warning lead time), fall back to the nearest
meaningfully-higher-ground point.
"""

from __future__ import annotations

from ml_pipeline.safe_place_engine.elevation_fallback import find_nearest_high_ground
from ml_pipeline.safe_place_engine.registry import find_nearest_shelter

MAX_REASONABLE_SHELTER_DISTANCE_KM = 15.0


def recommend_safe_place(lat: float, lon: float, use_mock_elevation: bool | None = None) -> dict:
    """
    Returns a dict describing the recommended safe place:
        {
          "type": "shelter" | "high_ground" | "none_found",
          "name": str | None,
          "latitude", "longitude",
          "distance_km",
          "details": {...source-specific fields...}
        }
    """
    shelter = find_nearest_shelter(lat, lon)

    if shelter and shelter["distance_km"] <= MAX_REASONABLE_SHELTER_DISTANCE_KM:
        return {
            "type": "shelter",
            "name": shelter["name"],
            "latitude": shelter["latitude"],
            "longitude": shelter["longitude"],
            "distance_km": round(shelter["distance_km"], 2),
            "details": {
                "capacity": shelter.get("capacity"),
                "source": shelter.get("source"),
            },
        }

    high_ground = find_nearest_high_ground(lat, lon, use_mock=use_mock_elevation)
    if high_ground:
        return {
            "type": "high_ground",
            "name": "Nearest higher-ground point (no shelter within range)",
            "latitude": high_ground["latitude"],
            "longitude": high_ground["longitude"],
            "distance_km": round(high_ground["distance_km"], 2),
            "details": {
                "elevation_m": round(high_ground["elevation_m"], 1),
                "elevation_gain_m": round(high_ground["elevation_gain_m"], 1),
            },
        }

    # Nothing found — still return the nearest known shelter if one
    # exists at all, even if far, rather than nothing.
    if shelter:
        return {
            "type": "shelter",
            "name": shelter["name"],
            "latitude": shelter["latitude"],
            "longitude": shelter["longitude"],
            "distance_km": round(shelter["distance_km"], 2),
            "details": {
                "capacity": shelter.get("capacity"),
                "source": shelter.get("source"),
                "note": "Nearest option found, but beyond the normal recommended range.",
            },
        }

    return {"type": "none_found", "name": None, "latitude": None, "longitude": None,
            "distance_km": None, "details": {}}
