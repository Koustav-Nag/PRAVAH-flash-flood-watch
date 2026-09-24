"""
Google Earth Engine client bootstrap.

This module centralizes GEE authentication so every ingestion module
(precipitation, Sentinel imagery, DEM/terrain) shares one initialized
session. It is written so the REST of the pipeline works even before
real GEE credentials exist: call `is_gee_available()` to check, and
every ingestion function has a `use_mock=True` path that returns
realistically-shaped synthetic data instead of failing.

--- Authentication (three methods, tried in order) ---

Method 1 — Service account (recommended for production/cloud):
    Set in your .env file:
        GEE_SERVICE_ACCOUNT_EMAIL=your-sa@your-project.iam.gserviceaccount.com
        GEE_SERVICE_ACCOUNT_KEY_PATH=/absolute/path/to/key.json
        GEE_PROJECT_ID=your-project-id

Method 2 — Default credentials (recommended for local dev):
    Just set GEE_PROJECT_ID in your .env. No key file needed.
    Run once to authenticate:
        earthengine authenticate
    or:
        gcloud auth application-default login
    Then the pipeline uses your personal Google account automatically.

Method 3 — Mock mode (no GEE at all):
    Leave all GEE_* variables empty. Pipeline uses synthetic data
    for demo purposes.
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger

from app.core.config import settings

_gee_initialized = False


@lru_cache(maxsize=1)
def is_gee_available() -> bool:
    """
    Attempts to initialize Earth Engine using one of three methods:
      1. Service account credentials (if configured in .env)
      2. Default credentials / personal account (if GEE_PROJECT_ID is set)
      3. Returns False — callers fall back to mock data

    Never raises — callers use this to decide whether to hit real GEE
    or fall back to mock data.
    """
    try:
        import ee  # local import so the package is optional until needed
    except (ImportError, ModuleNotFoundError):
        return False

    # --- Method 1: Service account ---
    if settings.GEE_SERVICE_ACCOUNT_EMAIL and settings.GEE_SERVICE_ACCOUNT_KEY_PATH:
        try:
            credentials = ee.ServiceAccountCredentials(
                settings.GEE_SERVICE_ACCOUNT_EMAIL,
                settings.GEE_SERVICE_ACCOUNT_KEY_PATH,
            )
            ee.Initialize(credentials, project=settings.GEE_PROJECT_ID)
            _gee_initialized = True
            logger.info("Earth Engine initialized with service account credentials.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Service account auth failed ({exc}). "
                "Trying default credentials..."
            )

    # --- Method 2: Default credentials (personal Google account) ---
    if settings.GEE_PROJECT_ID:
        try:
            ee.Initialize(project=settings.GEE_PROJECT_ID)
            _gee_initialized = True
            logger.info(
                "Earth Engine initialized with default credentials "
                f"(project: {settings.GEE_PROJECT_ID})."
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Earth Engine initialization failed: {exc}")
            return False

    # --- Method 3: No credentials configured ---
    logger.warning(
        "GEE credentials not configured. Ingestion modules will use mock data. "
        "Set GEE_PROJECT_ID in .env to use default credentials, or set "
        "GEE_SERVICE_ACCOUNT_EMAIL + GEE_SERVICE_ACCOUNT_KEY_PATH for a "
        "service account. See gee_client.py docstring for setup steps."
    )
    return False


def get_region_geometry():
    """
    Returns an ee.Geometry.Rectangle for the configured pilot region.
    Only call this after confirming is_gee_available() is True.
    """
    import ee

    return ee.Geometry.Rectangle(
        [
            settings.REGION_MIN_LON,
            settings.REGION_MIN_LAT,
            settings.REGION_MAX_LON,
            settings.REGION_MAX_LAT,
        ]
    )
