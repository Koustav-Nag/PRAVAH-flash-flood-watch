"""
Terrain / DEM feature extraction, using SRTM (30m) via Earth Engine.

These features are largely static (terrain doesn't change hour to
hour) so in production they'd be computed once per region and cached,
not re-fetched on every prediction request. They also serve as the
DEM-derived PROXY for river/water-level data, since real-time CWC /
Assam WRD hydrological feeds require a formal data request and are
not yet wired in (see project notes).
"""

from __future__ import annotations

import numpy as np
import math
from loguru import logger

from ml_pipeline.data_ingestion.gee_client import get_region_geometry, is_gee_available

SRTM_COLLECTION = "USGS/SRTMGL1_003"


def fetch_terrain_features(use_mock: bool | None = None) -> dict:
    """
    Returns region-level summary terrain statistics:
      - mean_elevation_m, elevation_range_m
      - mean_slope_deg
      - drainage_density_proxy (0-1, higher = more channelized terrain)
      - distance_to_river_proxy_km (mean distance to nearest high-flow-
        accumulation cell — stands in for "distance to river" until a
        real hydrography layer / CWC gauge network is integrated)

    NOTE: this is a REGION-LEVEL summary for the prototype. The
    architecture is designed to move to per-grid-cell / per-sub-basin
    values once the demo needs spatial risk maps rather than a single
    region-wide score.

    Keys match the ML-ready dataset schema: elevation, slope, aspect,
    flow_accumulation, distance_to_river, drainage_density.
    """
    use_mock = not is_gee_available() if use_mock is None else use_mock

    if use_mock:
        logger.info("Using MOCK SRTM terrain features (no GEE credentials).")
        rng = np.random.default_rng(seed=11)
        return {
            "elevation": float(rng.uniform(150, 600)),
            "elevation_range_m": float(rng.uniform(300, 1200)),
            "slope": float(rng.uniform(8, 25)),
            "aspect": float(rng.uniform(0, 360)),
            "flow_accumulation": float(rng.uniform(0, 5000)),
            "drainage_density": float(rng.uniform(0.3, 0.7)),
            "distance_to_river": float(rng.uniform(0.5, 5.0)),
        }

    import ee

    region = get_region_geometry()
    dem = ee.Image(SRTM_COLLECTION).clip(region)
    slope = ee.Terrain.slope(dem)

    elevation_stats = dem.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=region,
        scale=30,
        bestEffort=True,
    ).getInfo()

    slope_stats = slope.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=30, bestEffort=True
    ).getInfo()

    # Flow accumulation requires a hydrologically-conditioned DEM
    # (fill sinks -> flow direction -> flow accumulation). GEE doesn't
    # ship this natively; the standard approach is the WWF HydroSHEDS
    # flow-accumulation asset, which we use as the drainage-density
    # and distance-to-river proxy instead of computing it from scratch.
    flow_acc = ee.Image("WWF/HydroSHEDS/15ACC")
    flow_stats = flow_acc.clip(region).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=450, bestEffort=True
    ).getInfo()

    return {
        "elevation": elevation_stats.get("elevation_mean"),
        "elevation_range_m": (
            elevation_stats.get("elevation_max", 0) - elevation_stats.get("elevation_min", 0)
        ),
        "slope": slope_stats.get("slope"),
        "aspect": None,  # TODO: ee.Terrain.aspect(dem), not yet wired in
        "flow_accumulation": flow_stats.get("b1"),
        "drainage_density": min(1.0, math.log1p(flow_stats.get("b1") or 0) / math.log1p(10000)),  # log-normalized to [0,1]; 10k cells ≈ high drainage
        "distance_to_river": None,  # derived downstream from flow_acc raster, TODO
    }
