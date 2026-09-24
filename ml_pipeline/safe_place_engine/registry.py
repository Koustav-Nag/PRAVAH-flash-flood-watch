"""
Known-shelter registry and nearest-shelter lookup.

IMPORTANT: data/safe_places/sonitpur_shelters.csv currently ships with
PLACEHOLDER entries (plausible-sounding names/locations, NOT verified
real relief camps) so the pipeline and dashboard are demoable. Before
any real deployment or judge-facing claim of "real shelters", replace
this file with a verified list from ASDMA / the Sonitpur district
administration.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

SHELTERS_PATH = Path("data/safe_places/sonitpur_shelters.csv")


def load_shelters() -> pd.DataFrame:
    if not SHELTERS_PATH.exists():
        return pd.DataFrame(columns=["name", "type", "latitude", "longitude", "capacity", "source"])
    return pd.read_csv(SHELTERS_PATH)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_nearest_shelter(lat: float, lon: float) -> dict | None:
    """
    Returns the nearest known shelter to (lat, lon) as a dict with an
    added distance_km field, or None if the registry is empty.
    """
    shelters = load_shelters()
    if shelters.empty:
        return None

    shelters = shelters.copy()
    shelters["distance_km"] = shelters.apply(
        lambda row: haversine_km(lat, lon, row["latitude"], row["longitude"]), axis=1
    )
    nearest = shelters.sort_values("distance_km").iloc[0]
    return nearest.to_dict()
