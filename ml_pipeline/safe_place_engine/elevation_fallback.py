"""
Elevation-based fallback for "nearest safe place".

Used when no known shelter is close enough (or the registry is
empty). Samples a ring of candidate points around the person's
location and picks the nearest one that represents a meaningful
elevation gain — a standard flash-flood-response heuristic ("get to
higher ground") when no formal shelter is reachable in time.
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger

from ml_pipeline.data_ingestion.gee_client import is_gee_available
from ml_pipeline.safe_place_engine.registry import haversine_km

MIN_ELEVATION_GAIN_M = 15.0  # minimum gain to be worth recommending
SEARCH_RADIUS_KM = 5.0
N_CANDIDATE_RINGS = 3
POINTS_PER_RING = 8


def _candidate_points(lat: float, lon: float) -> list[tuple[float, float]]:
    """Generates a ring of candidate lat/lon points around (lat, lon)."""
    points = []
    for ring in range(1, N_CANDIDATE_RINGS + 1):
        radius_km = (SEARCH_RADIUS_KM / N_CANDIDATE_RINGS) * ring
        for i in range(POINTS_PER_RING):
            bearing = (2 * math.pi / POINTS_PER_RING) * i
            dlat = (radius_km / 111.0) * math.cos(bearing)
            dlon = (radius_km / (111.0 * math.cos(math.radians(lat)))) * math.sin(bearing)
            points.append((lat + dlat, lon + dlon))
    return points


def _sample_elevation(lat: float, lon: float, use_mock: bool | None = None) -> float:
    use_mock = not is_gee_available() if use_mock is None else use_mock

    if use_mock:
        # Deterministic pseudo-terrain so results are stable across calls
        # for the same point (useful for demoing / testing).
        rng = np.random.default_rng(seed=int(abs(lat * 1000) + abs(lon * 1000)))
        return float(200 + 150 * math.sin(lat * 10) + 100 * math.cos(lon * 10) + rng.uniform(-10, 10))

    import ee

    point = ee.Geometry.Point([lon, lat])
    elevation = ee.Image("USGS/SRTMGL1_003").reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=30
    ).get("elevation")
    return float(elevation.getInfo())


def find_nearest_high_ground(lat: float, lon: float, use_mock: bool | None = None) -> dict | None:
    """
    Returns the nearest candidate point with a meaningful elevation
    gain over the origin, as:
        {latitude, longitude, elevation_m, elevation_gain_m, distance_km}
    or None if no candidate meets MIN_ELEVATION_GAIN_M.
    """
    origin_elevation = _sample_elevation(lat, lon, use_mock=use_mock)
    candidates = _candidate_points(lat, lon)

    best = None
    for c_lat, c_lon in candidates:
        elevation = _sample_elevation(c_lat, c_lon, use_mock=use_mock)
        gain = elevation - origin_elevation
        if gain < MIN_ELEVATION_GAIN_M:
            continue

        distance_km = haversine_km(lat, lon, c_lat, c_lon)
        candidate = {
            "latitude": c_lat,
            "longitude": c_lon,
            "elevation_m": elevation,
            "elevation_gain_m": gain,
            "distance_km": distance_km,
        }
        # Prefer the closest candidate that still clears the minimum gain,
        # rather than the single highest point (which may be far away) —
        # closer + "high enough" beats far + "highest".
        if best is None or distance_km < best["distance_km"]:
            best = candidate

    if best is None:
        logger.warning(
            f"No high-ground candidate found within {SEARCH_RADIUS_KM}km "
            f"clearing {MIN_ELEVATION_GAIN_M}m elevation gain."
        )
    return best
