"""
Sentinel-1 (SAR, all-weather) and Sentinel-2 (optical) ingestion.

Sentinel-1 is used for surface-water/flood-extent detection (works
through cloud cover, which matters a lot during active storm events).
Sentinel-2 (via Dynamic World) is used for land-use/land-cover context
(built-up area, vegetation, bare soil) that feeds into the terrain-vulnerability
features. Both are pulled through Google Earth Engine (GEE).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from loguru import logger

from app.core.config import settings
from ml_pipeline.data_ingestion.gee_client import get_region_geometry, is_gee_available

SENTINEL1_COLLECTION = "COPERNICUS/S1_GRD"
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
SRTM_COLLECTION = "USGS/SRTMGL1_003"


def _now_utc() -> datetime:
    """Helper to return current naive UTC datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _mock_surface_water_fraction() -> float:
    logger.info("Using MOCK Sentinel-1 surface water fraction.")
    rng = np.random.default_rng(seed=7)
    return float(rng.uniform(0.01, 0.06))  # plausible baseline for hilly terrain


def _mock_landcover_summary() -> dict:
    logger.info("Using MOCK Sentinel-2 land-cover summary.")
    return {
        "forest_frac": 0.55,
        "cropland_frac": 0.25,
        "built_up_frac": 0.08,
        "bare_soil_frac": 0.07,
        "water_frac": 0.05,
    }


def fetch_surface_water_fraction(
    end_time: datetime | None = None,
    lookback_days: int = 12,
    use_mock: bool | None = None,
) -> float:
    """
    Estimates current surface-water fraction over the pilot region
    from the most recent Sentinel-1 VH-band pass, using SAR thresholding:
    open water produces specular reflection and appears dark in VH backscatter.

    Applies:
      1. Instrument mode filter ('IW')
      2. Polarisation selection ('VH')
      3. Speckle noise reduction (30m focal median filter)
      4. Terrain slope masking (slopes >= 5 deg excluded via SRTM to avoid radar shadow false positives)
      5. VH thresholding (< -17 dB)

    Returns a float in [0, 1] — fraction of the region flagged as open water.
    Falls back to mock baseline if GEE is unavailable or upon failure.
    """
    end_time = end_time or _now_utc()

    if use_mock is None:
        use_mock = not is_gee_available()

    if use_mock:
        return _mock_surface_water_fraction()

    try:
        return _fetch_surface_water_from_gee(end_time, lookback_days)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Failed to fetch live Sentinel-1 surface water fraction from GEE ({exc}). "
            "Falling back to mock baseline."
        )
        return _mock_surface_water_fraction()


def _fetch_surface_water_from_gee(end_time: datetime, lookback_days: int) -> float:
    import ee

    region = get_region_geometry()
    start_time = end_time - timedelta(days=lookback_days)

    collection = (
        ee.ImageCollection(SENTINEL1_COLLECTION)
        .filterDate(start_time.isoformat(), end_time.isoformat())
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select("VH")
    )

    count = collection.size().getInfo()
    if count == 0:
        logger.warning(
            f"No Sentinel-1 IW VH scenes found in last {lookback_days} days. "
            "Widening search to 30 days..."
        )
        start_time = end_time - timedelta(days=30)
        collection = (
            ee.ImageCollection(SENTINEL1_COLLECTION)
            .filterDate(start_time.isoformat(), end_time.isoformat())
            .filterBounds(region)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select("VH")
        )
        count = collection.size().getInfo()
        if count == 0:
            logger.warning("No Sentinel-1 scenes available in past 30 days. Using mock fallback.")
            return _mock_surface_water_fraction()

    # Most recent pass
    image = collection.sort("system:time_start", False).first()

    # 1. 30m focal median filter to suppress radar speckle noise
    smoothed = image.focalMedian(30, "circle", "meters")

    # 2. Water threshold: VH backscatter < -17 dB is characteristic of specular open water
    water = smoothed.lt(-17)

    # 3. Slope mask: water does not pool on steep terrain (slope >= 5°), eliminates mountain shadows
    try:
        dem = ee.Image(SRTM_COLLECTION).clip(region)
        slope = ee.Terrain.slope(dem)
        water_mask = water.And(slope.lt(5)).rename("water")
    except Exception as slope_exc:  # noqa: BLE001
        logger.warning(f"Slope masking skipped due to error: {slope_exc}")
        water_mask = water.rename("water")

    stats = water_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=30,
        bestEffort=True,
        maxPixels=1e8,
    )
    val = stats.get("water").getInfo()
    if val is None:
        return 0.0
    return float(np.clip(val, 0.0, 1.0))


def fetch_landcover_summary(
    as_of: datetime | None = None,
    lookback_days: int = 45,
    use_mock: bool | None = None,
) -> dict:
    """
    Returns a land-cover breakdown (fractions) for the pilot region
    using the Dynamic World dataset (Sentinel-2 10m L2A near-real-time LULC).
    Used as a vulnerability modifier in runoff and flood susceptibility.

    Keys:
      - forest_frac: trees, shrubs, and grassland
      - cropland_frac: agricultural crops
      - built_up_frac: urban and infrastructure
      - bare_soil_frac: bare ground and sparse vegetation
      - water_frac: surface water and flooded vegetation
    """
    as_of = as_of or _now_utc()

    if use_mock is None:
        use_mock = not is_gee_available()

    if use_mock:
        return _mock_landcover_summary()

    try:
        return _fetch_landcover_from_gee(as_of, lookback_days)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Failed to fetch live Sentinel-2 / Dynamic World land cover from GEE ({exc}). "
            "Falling back to mock baseline."
        )
        return _mock_landcover_summary()


def _fetch_landcover_from_gee(as_of: datetime, lookback_days: int) -> dict:
    import ee

    region = get_region_geometry()
    end_date = as_of
    start_date = end_date - timedelta(days=lookback_days)

    dw_col = (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterDate(start_date.isoformat(), end_date.isoformat())
        .filterBounds(region)
    )

    count = dw_col.size().getInfo()
    if count == 0:
        logger.warning(
            f"No Dynamic World scenes found in last {lookback_days} days. "
            "Widening search to 90 days..."
        )
        start_date = end_date - timedelta(days=90)
        dw_col = (
            ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .filterBounds(region)
        )
        count = dw_col.size().getInfo()
        if count == 0:
            logger.warning("No Dynamic World scenes available in past 90 days. Using mock fallback.")
            return _mock_landcover_summary()

    # Mode composite of the 'label' band
    # Classes: 0:water, 1:trees, 2:grass, 3:flooded_veg, 4:crops, 5:shrub, 6:built, 7:bare, 8:snow
    classification = dw_col.select("label").reduce(ee.Reducer.mode())

    pixel_counts = classification.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=region,
        scale=100,  # 100m scale avoids memory limits and runs rapidly
        bestEffort=True,
        maxPixels=1e8,
    )

    hist = pixel_counts.get("label_mode").getInfo() or {}
    total_pixels = sum(hist.values())
    if not total_pixels:
        logger.warning("Empty histogram from Dynamic World reduction. Using mock fallback.")
        return _mock_landcover_summary()

    # Map Dynamic World classes into project feature schema
    # 0: water, 3: flooded_vegetation
    water_count = hist.get("0", 0) + hist.get("3", 0)
    # 1: trees, 2: grass, 5: shrub_and_scrub
    forest_count = hist.get("1", 0) + hist.get("2", 0) + hist.get("5", 0)
    # 4: crops
    cropland_count = hist.get("4", 0)
    # 6: built
    built_count = hist.get("6", 0)
    # 7: bare, 8: snow_and_ice
    bare_count = hist.get("7", 0) + hist.get("8", 0)

    return {
        "forest_frac": round(float(forest_count / total_pixels), 3),
        "cropland_frac": round(float(cropland_count / total_pixels), 3),
        "built_up_frac": round(float(built_count / total_pixels), 3),
        "bare_soil_frac": round(float(bare_count / total_pixels), 3),
        "water_frac": round(float(water_count / total_pixels), 3),
    }
