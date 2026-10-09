"""
Flash Flood Prediction System — FastAPI entrypoint.

Run with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import settings
from ml_pipeline.data_ingestion.gee_client import is_gee_available
from ml_pipeline.data_ingestion.glofas import fetch_glofas_discharge
from ml_pipeline.data_ingestion.india_wris import (
    fetch_cwc_aff_forecast_horizon_map,
    fetch_observed_river_level,
    is_wris_available,
)
from ml_pipeline.data_ingestion.open_meteo import fetch_weather_forecast
from ml_pipeline.data_ingestion.precipitation import fetch_imerg_timeseries
from ml_pipeline.data_ingestion.river_levels import (
    compute_threshold_features,
    get_sonitpur_gauge_stations,
    get_station_thresholds,
)
from ml_pipeline.feature_engineering.fusion import build_feature_vector
from ml_pipeline.feature_engineering.river_level_features import project_future_river_level
from ml_pipeline.models.xgboost_model import FlashFloodXGBModel
from ml_pipeline.risk_engine.hybrid import compute_hybrid_risk
from ml_pipeline.safe_place_engine.registry import load_shelters
from ml_pipeline.safe_place_engine.safe_place import recommend_safe_place

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pravah.api")

# Loaded once at startup; stays untrained (physics-only fallback) until
# a training dataset exists and train.py has been run.
xgb_model = FlashFloodXGBModel()

# ---------------------------------------------------------------------------
# Self-ping keep-alive (prevents Render free-tier cold start)
# ---------------------------------------------------------------------------
# Render spins down free services after ~15 min of inactivity.
# This background task pings /ping every 13 min to keep it warm.
# Set RENDER_EXTERNAL_URL in your Render environment, or SELF_PING_URL
# in .env.  Disabled when neither is set (e.g. local dev).

SELF_PING_INTERVAL_SECONDS = int(os.getenv("SELF_PING_INTERVAL", "780"))  # 13 min
SELF_PING_URL: str | None = os.getenv(
    "SELF_PING_URL",
    os.getenv("RENDER_EXTERNAL_URL"),  # Render sets this automatically
)


async def _self_ping_loop() -> None:
    """Periodically hit our own /ping endpoint to prevent cold starts."""
    if not SELF_PING_URL:
        logger.info("Self-ping disabled (RENDER_EXTERNAL_URL / SELF_PING_URL not set)")
        return

    url = f"{SELF_PING_URL.rstrip('/')}/ping"
    logger.info("Self-ping enabled → %s every %ds", url, SELF_PING_INTERVAL_SECONDS)

    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(SELF_PING_INTERVAL_SECONDS)
            try:
                resp = await client.get(url)
                logger.info("Self-ping → %s  %d", url, resp.status_code)
            except Exception:
                logger.warning("Self-ping failed for %s", url, exc_info=True)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model + launch keep-alive.  Shutdown: cancel task."""
    # --- startup ---
    xgb_model.load()  # no-op / warns if no saved model exists yet
    logger.info("Model initialization completed; trained=%s", xgb_model.is_trained())

    ping_task = asyncio.create_task(_self_ping_loop())

    yield

    # --- shutdown ---
    ping_task.cancel()
    try:
        await ping_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-source flash-flood early-warning prototype — Sonitpur, Assam pilot",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.head("/")
def root():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "region": settings.REGION_NAME,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "region": settings.REGION_NAME,
        "gee_available": is_gee_available(),
        "xgboost_model_trained": xgb_model.is_trained(),
    }


@app.get("/region/bounds")
def get_region_bounds():
    return {
        "region": settings.REGION_NAME,
        "min_lon": settings.REGION_MIN_LON,
        "min_lat": settings.REGION_MIN_LAT,
        "max_lon": settings.REGION_MAX_LON,
        "max_lat": settings.REGION_MAX_LAT,
    }


@app.get("/safe-places/shelters")
def list_shelters():
    return load_shelters().to_dict(orient="records")


def compute_hydrological_alert_summary(primary_obs: dict, secondary_obs: dict) -> dict:
    """
    Evaluates physical river stage against statutory thresholds (DL/HFL)
    independently from the meteorological/ML flash-flood surge risk score.
    """
    alerts = []
    is_above_danger = False
    is_above_warning = False

    # Check Primary (Brahmaputra)
    p_lvl = primary_obs.get("observed_level_m")
    p_dl = float(primary_obs.get("danger_level_m", settings.PRIMARY_DANGER_LEVEL_M))
    p_wl = float(primary_obs.get("warning_level_m", settings.PRIMARY_WARNING_LEVEL_M))
    p_name = primary_obs.get("river", settings.PRIMARY_RIVER_NAME)
    p_stn = primary_obs.get("station_name", settings.PRIMARY_STATION_NAME)

    if p_lvl is not None:
        if p_lvl >= p_dl:
            is_above_danger = True
            alerts.append({
                "river": p_name,
                "station": p_stn,
                "role": "primary",
                "level_m": p_lvl,
                "threshold_m": p_dl,
                "margin_m": round(p_lvl - p_dl, 2),
                "condition": "Above Danger Level",
            })
        elif p_lvl >= p_wl:
            is_above_warning = True
            alerts.append({
                "river": p_name,
                "station": p_stn,
                "role": "primary",
                "level_m": p_lvl,
                "threshold_m": p_wl,
                "margin_m": round(p_lvl - p_wl, 2),
                "condition": "Above Warning Level",
            })

    # Check Secondary (Jia Bharali)
    s_lvl = secondary_obs.get("observed_level_m")
    s_dl = float(secondary_obs.get("danger_level_m", settings.SECONDARY_DANGER_LEVEL_M))
    s_wl = float(secondary_obs.get("warning_level_m", settings.SECONDARY_WARNING_LEVEL_M))
    s_name = secondary_obs.get("river", settings.SECONDARY_RIVER_NAME)
    s_stn = secondary_obs.get("station_name", settings.SECONDARY_STATION_NAME)

    if s_lvl is not None:
        if s_lvl >= s_dl:
            is_above_danger = True
            alerts.append({
                "river": s_name,
                "station": s_stn,
                "role": "secondary",
                "level_m": s_lvl,
                "threshold_m": s_dl,
                "margin_m": round(s_lvl - s_dl, 2),
                "condition": "Above Danger Level",
            })
        elif s_lvl >= s_wl:
            is_above_warning = True
            alerts.append({
                "river": s_name,
                "station": s_stn,
                "role": "secondary",
                "level_m": s_lvl,
                "threshold_m": s_wl,
                "margin_m": round(s_lvl - s_wl, 2),
                "condition": "Above Warning Level",
            })

    status = "ABOVE_DANGER" if is_above_danger else ("WARNING" if is_above_warning else "NORMAL")
    summary = (
        f"{', '.join(a['river'] for a in alerts)} currently exceeding flood thresholds"
        if alerts else
        "All monitored gauge stations currently below danger and warning levels"
    )

    return {
        "status": status,
        "is_above_danger": is_above_danger,
        "is_above_warning": is_above_warning,
        "active_exceedances": alerts,
        "summary": summary,
        "note": (
            "Hydrological alert is an active physical sensor warning. "
            "Model risk score measures rainfall-induced flash flood surge probability."
        ),
    }


@app.get("/risk/current")
def get_current_risk():
    """
    Computes current flash-flood risk for the pilot region using live
    (or mock, if GEE isn't configured) multi-source data, fused and
    scored through the hybrid risk engine.
    """
    try:
        features = build_feature_vector()
        result = compute_hybrid_risk(features, xgb_model)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    primary_obs = fetch_observed_river_level(settings.PRIMARY_STATION_CODE)
    secondary_obs = fetch_observed_river_level(settings.SECONDARY_STATION_CODE)
    hydro_alert = compute_hydrological_alert_summary(primary_obs, secondary_obs)

    return {
        "region": settings.REGION_NAME,
        "as_of": features.get("as_of"),
        "risk": result,
        "features": features,
        "hydrological_alert": hydro_alert,
    }


@app.get("/risk/explain")
def explain_current_risk():
    """
    Same as /risk/current but shaped for a UI 'why is this the risk
    level' panel — returns only the explainability-relevant fields.
    """
    features = build_feature_vector()
    result = compute_hybrid_risk(features, xgb_model)

    return {
        "final_category": result["final_category"],
        "final_score": result["final_score"],
        "source": result["source"],
        "physics_components": result["physics_informed"]["components"],
        "physics_weights": result["physics_informed"]["weights"],
        "xgboost_feature_importance": xgb_model.feature_importance(),
    }


@app.get("/data/rainfall-series")
def rainfall_series(lookback_hours: int = 72):
    """
    Half-hourly rainfall series for the pilot region (live GEE or mock
    fallback — same source used by the risk engine), shaped for a
    frontend trend chart. Capped at 7 days to keep the payload small.
    """
    lookback_hours = max(1, min(lookback_hours, 168))
    precip_df = fetch_imerg_timeseries(lookback_hours=lookback_hours)
    return {
        "lookback_hours": lookback_hours,
        "points": [
            {"timestamp": row["timestamp"].isoformat(), "rainfall_mm": round(float(row["precip_mm"]), 3)}
            for _, row in precip_df.iterrows()
        ],
    }


@app.get("/data/sources")
def data_sources_status():
    """
    Per-source status using the specification's six data categories:
    OBSERVED, MODELLED, FORECAST, STATIC REFERENCE, DEMO / SIMULATED,
    UNAVAILABLE.
    """
    gee_auth_ok = is_gee_available()
    wris_ok = is_wris_available()
    return {
        "gee_authenticated": gee_auth_ok,
        "wris_authenticated": wris_ok,
        "sources": [
            {
                "name": "Rainfall (GPM/IMERG)",
                "data_category": "DEMO / SIMULATED" if not gee_auth_ok else "OBSERVED",
                "detail": "Half-hourly satellite precipitation. Live nowcast requires a billed GEE project.",
            },
            {
                "name": "Weather forecast (Open-Meteo)",
                "data_category": "FORECAST",
                "detail": "Hourly precipitation and probability forecasts up to +72h.",
            },
            {
                "name": "GloFAS discharge (Open-Meteo Flood API)",
                "data_category": "MODELLED",
                "detail": "Modelled river discharge in m³/s. Not observed river level.",
            },
            {
                "name": "Terrain (SRTM/HydroSHEDS)",
                "data_category": "STATIC / DERIVED",
                "detail": "Static DEM-derived elevation, slope, drainage density, and flow accumulation (SRTM/HydroSHEDS).",
            },
            {
                "name": "Surface water (Sentinel-1)",
                "data_category": "OBSERVED" if gee_auth_ok else "DEMO / SIMULATED",
                "detail": (
                    "SAR-based water fraction from live Sentinel-1 C-band pass via Google Earth Engine."
                    if gee_auth_ok else
                    "SAR-based water fraction. Simulated baseline (live fetch requires configured GEE credentials)."
                ),
            },
            {
                "name": "Land cover (Sentinel-2)",
                "data_category": "OBSERVED" if gee_auth_ok else "DEMO / SIMULATED",
                "detail": (
                    "Live Dynamic World (Sentinel-2 10m) land-use classification via Google Earth Engine."
                    if gee_auth_ok else
                    "Dynamic World land-cover summary. Simulated baseline (live fetch requires configured GEE credentials)."
                ),
            },
            {
                "name": "River gauge — Primary: Brahmaputra (Tezpur)",
                "data_category": "OBSERVED" if wris_ok else "DEMO / SIMULATED",
                "detail": (
                    "Authoritative CWC river gauge for Brahmaputra at Tezpur / Ganeshghat (Station: TEZPUR). "
                    "Provides primary regional river stage context via CWC Advisory Flood Forecast (AFF) / WIMS."
                    if wris_ok else
                    "Simulated gauge data for Brahmaputra at Tezpur. Live telemetry connects to CWC AFF (aff.india-water.gov.in)."
                ),
            },
            {
                "name": "River gauge — Secondary: Jia Bharali (N.T. Road Xing)",
                "data_category": "OBSERVED" if wris_ok else "DEMO / SIMULATED",
                "detail": (
                    "Authoritative CWC river gauge for Jia Bharali at N.T. Road Crossing (Station: NT ROAD CROSSING JIA-BHARALI, Code: 12359). "
                    "Provides secondary/local flash-flood signal via CWC Advisory Flood Forecast (AFF) / WIMS."
                    if wris_ok else
                    "Simulated gauge data for Jia Bharali N.T. Road Xing. Live telemetry connects to CWC AFF (aff.india-water.gov.in)."
                ),
            },
            {
                "name": "Danger levels (Assam WRD / CWC reference)",
                "data_category": "STATIC REFERENCE",
                "detail": "Official danger level and HFL thresholds per gauge station (Tezpur: 65.23m / 66.59m; N.T. Road Xing: 77.00m / 78.50m).",
            },
        ],
    }


@app.get("/data/river-status")
def river_status():
    """
    Current river status:
    - PRIMARY: Brahmaputra at Tezpur (observed stage, DL 65.23m, HFL 66.59m, condition)
    - SECONDARY: Jia Bharali at N.T. Road Xing (observed stage, DL 77.00m, HFL 78.50m, condition)
    - GloFAS modelled discharge (kept strictly separate)
    - Danger-level thresholds per station
    """
    stations = get_sonitpur_gauge_stations()

    # Ingest Primary and Secondary gauges independently
    primary_obs = fetch_observed_river_level(settings.PRIMARY_STATION_CODE)
    secondary_obs = fetch_observed_river_level(settings.SECONDARY_STATION_CODE)

    # Threshold analysis for Primary (Brahmaputra at Tezpur)
    primary_analysis = None
    if primary_obs.get("observed_level_m") is not None:
        p_dl = float(primary_obs.get("danger_level_m", settings.PRIMARY_DANGER_LEVEL_M))
        p_hfl = float(primary_obs.get("hfl_m", settings.PRIMARY_HFL_M))
        p_curr = float(primary_obs["observed_level_m"])
        primary_analysis = compute_threshold_features(p_curr, p_dl, p_hfl)
        primary_analysis["danger_level_m"] = p_dl
        primary_analysis["hfl_m"] = p_hfl
        primary_analysis["distance_to_danger_m"] = float(round(p_dl - p_curr, 2))
        primary_analysis["is_above_danger"] = bool(p_curr >= p_dl)
        primary_analysis["is_above_hfl"] = bool(p_curr >= p_hfl)

    # Threshold analysis for Secondary (Jia Bharali at N.T. Road Xing)
    secondary_analysis = None
    if secondary_obs.get("observed_level_m") is not None:
        s_dl = float(secondary_obs.get("danger_level_m", settings.SECONDARY_DANGER_LEVEL_M))
        s_hfl = float(secondary_obs.get("hfl_m", settings.SECONDARY_HFL_M))
        s_curr = float(secondary_obs["observed_level_m"])
        secondary_analysis = compute_threshold_features(s_curr, s_dl, s_hfl)
        secondary_analysis["danger_level_m"] = s_dl
        secondary_analysis["hfl_m"] = s_hfl
        secondary_analysis["distance_to_danger_m"] = float(round(s_dl - s_curr, 2))
        secondary_analysis["is_above_danger"] = bool(s_curr >= s_dl)
        secondary_analysis["is_above_hfl"] = bool(s_curr >= s_hfl)

    # GloFAS modelled discharge (free API) — kept strictly separate
    glofas = fetch_glofas_discharge(
        latitude=settings.PRIMARY_GAUGE_LAT,
        longitude=settings.PRIMARY_GAUGE_LON,
    )

    # Weather forecast (free API)
    weather = fetch_weather_forecast()

    # Static danger-level thresholds
    thresholds = []
    for _, row in stations.iterrows():
        thresholds.append({
            "river": row["river"],
            "gauge_location": row["gauge_location"],
            "danger_level_m": float(row["danger_level_m"]),
            "hfl_m": float(row["hfl_m"]),
            "hfl_date": str(row["hfl_date"]).split("T")[0] if row["hfl_date"] else None,
            "data_category": "STATIC REFERENCE",
        })

    # Hydrological alert summary
    hydro_alert = compute_hydrological_alert_summary(primary_obs, secondary_obs)

    return {
        "region": settings.REGION_NAME,
        "hydrological_alert": hydro_alert,
        "primary_river": {
            **primary_obs,
            "threshold_analysis": primary_analysis,
        },
        "secondary_river": {
            **secondary_obs,
            "threshold_analysis": secondary_analysis,
        },
        # Backwards compatibility: observed_river_level represents Primary
        "observed_river_level": primary_obs,
        "threshold_analysis": primary_analysis,
        "secondary_observed_river_level": secondary_obs,
        "secondary_threshold_analysis": secondary_analysis,
        "glofas_discharge": glofas,
        "weather_forecast": weather,
        "danger_level_thresholds": thresholds,
    }


@app.get("/risk/forecast")
def risk_forecast():
    """
    Multi-horizon risk forecast (+1h, +3h, +6h, +12h, +24h).
    Blends the hybrid risk assessment with weather forecasts and
    current observed river level context from both CWC gauge stations.
    """
    try:
        features = build_feature_vector()
        current_risk = compute_hybrid_risk(features, xgb_model)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    weather = fetch_weather_forecast()
    primary_obs = fetch_observed_river_level(settings.PRIMARY_STATION_CODE)
    secondary_obs = fetch_observed_river_level(settings.SECONDARY_STATION_CODE)

    primary_level = primary_obs.get("observed_level_m")
    primary_ror = features.get("primary_rate_of_rise")

    sec_level = secondary_obs.get("observed_level_m")
    sec_ror = features.get("secondary_rate_of_rise")

    hourly_precip = weather.get("hourly", [])

    # Authoritative CWC AFF hydrodynamic forecast mappings (aff.india-water.gov.in/hydro.php)
    primary_cwc_fcst_map = fetch_cwc_aff_forecast_horizon_map("TEZPUR")
    secondary_cwc_fcst_map = fetch_cwc_aff_forecast_horizon_map("NT ROAD CROSSING JIA-BHARALI")

    # Build per-horizon forecast entries
    horizons = {}
    for h in settings.FORECAST_HORIZONS_HOURS:
        key = f"+{h}h"

        weather_at_h = None
        if "forecast_horizons" in weather:
            weather_at_h = weather["forecast_horizons"].get(key)

        cum_precip = sum(
            float(item.get("precipitation_mm") or 0.0)
            for item in hourly_precip[:h]
        )

        # Primary gauge (Brahmaputra at Tezpur) - prioritize authoritative CWC Hydrodynamic Forecast
        if h in primary_cwc_fcst_map:
            primary_projected = round(float(primary_cwc_fcst_map[h]), 2)
            primary_source = "CWC Hydrodynamic Forecast (aff.india-water.gov.in/hydro.php)"
            primary_cat = "FORECAST"
        else:
            primary_projected = project_future_river_level(
                current_level_m=primary_level,
                rate_of_rise_m_per_hr=primary_ror,
                horizon_hours=h,
                forecast_precip_mm=cum_precip,
                base_level_m=settings.PRIMARY_BASE_LEVEL_M,
                min_clamp_m=58.0,
                max_clamp_m=68.0,
            )
            primary_source = "Hydrological Projection Model"
            primary_cat = "MODEL PREDICTION"

        # Secondary gauge (Jia Bharali at N.T. Road Xing) - prioritize authoritative CWC Hydrodynamic Forecast
        if h in secondary_cwc_fcst_map:
            sec_projected = round(float(secondary_cwc_fcst_map[h]), 2)
            sec_source = "CWC Hydrodynamic Forecast (aff.india-water.gov.in/hydro.php)"
            sec_cat = "FORECAST"
        else:
            sec_projected = project_future_river_level(
                current_level_m=sec_level,
                rate_of_rise_m_per_hr=sec_ror,
                horizon_hours=h,
                forecast_precip_mm=cum_precip,
                base_level_m=settings.SECONDARY_BASE_LEVEL_M,
                min_clamp_m=74.0,
                max_clamp_m=80.0,
            )
            sec_source = "Hydrological Projection Model"
            sec_cat = "MODEL PREDICTION"

        horizons[key] = {
            "risk_score": current_risk["final_score"],
            "risk_category": current_risk["final_category"],
            "observed_level_m": primary_level,
            "predicted_level_m": primary_projected,
            "predicted_level_label": (
                f"Projected: {primary_projected:.2f} m (Brahmaputra)"
                if primary_projected is not None else
                "Level: unavailable"
            ),
            "primary_predicted_level_m": primary_projected,
            "secondary_predicted_level_m": sec_projected,
            "primary_forecast_source": primary_source,
            "secondary_forecast_source": sec_source,
            "weather_forecast": weather_at_h,
            "data_category": primary_cat,
        }

    # Static danger-level references for context
    primary_ref = get_station_thresholds(settings.PRIMARY_GAUGE_LOCATION)
    secondary_ref = get_station_thresholds(settings.SECONDARY_GAUGE_LOCATION)
    hydro_alert = compute_hydrological_alert_summary(primary_obs, secondary_obs)

    return {
        "region": settings.REGION_NAME,
        "as_of": features.get("as_of"),
        "hydrological_alert": hydro_alert,
        "current": {
            "risk_score": current_risk["final_score"],
            "risk_category": current_risk["final_category"],
            "source": current_risk["source"],
            "observed_river_level": primary_obs,
            "primary_observed_river_level": primary_obs,
            "secondary_observed_river_level": secondary_obs,
            "hydrological_alert": hydro_alert,
        },
        "horizons": horizons,
        "danger_level_reference": primary_ref,
        "primary_danger_level_reference": primary_ref,
        "secondary_danger_level_reference": secondary_ref,
        "primary_cwc_forecasts": primary_obs.get("cwc_forecasts", []),
        "secondary_cwc_forecasts": secondary_obs.get("cwc_forecasts", []),
        "note": (
            f"Primary gauge (Brahmaputra — Tezpur): {primary_level} m ({primary_obs.get('source', 'CWC AFF')}). "
            f"Secondary gauge (Jia Bharali — N.T. Road Xing): {sec_level} m ({secondary_obs.get('source', 'CWC AFF')}). "
            "Timeline river levels are hydrologically projected using observed stage, rate of rise, "
            "and precipitation forecasts (MODEL PREDICTION). Official CWC multi-day advisory forecasts are preserved separately."
            if primary_level is not None else
            "River-level predictions unavailable until observed river-level "
            "time-series data is integrated."
        ),
    }


class DispatchRequest(BaseModel):
    latitude: float
    longitude: float


@app.post("/alerts/dispatch")
def dispatch_alert(req: DispatchRequest):
    """
    Full alert payload for a person at (latitude, longitude): current
    hybrid risk assessment for the region, plus the nearest recommended
    safe place (a known shelter if one is close enough, otherwise the
    nearest meaningfully-higher-ground point).
    """
    try:
        features = build_feature_vector()
        risk = compute_hybrid_risk(features, xgb_model)
        safe_place = recommend_safe_place(req.latitude, req.longitude)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    primary_obs = fetch_observed_river_level(settings.PRIMARY_STATION_CODE)
    secondary_obs = fetch_observed_river_level(settings.SECONDARY_STATION_CODE)
    hydro_alert = compute_hydrological_alert_summary(primary_obs, secondary_obs)

    return {
        "region": settings.REGION_NAME,
        "as_of": features.get("as_of"),
        "origin": {"latitude": req.latitude, "longitude": req.longitude},
        "risk": {
            "category": risk["final_category"],
            "score": risk["final_score"],
            "source": risk["source"],
        },
        "hydrological_alert": hydro_alert,
        "safe_place": safe_place,
    }


@app.post("/alerts/stub-crowd-report")
def stub_crowd_report(report: dict):
    """
    STUB / extension point for future crowd-sourced or local-authority
    flood reports. Not wired into the risk engine in this prototype —
    kept as a clean integration point per the architecture's optional
    crowd-data layer.
    """
    return {
        "status": "received (stub only — not used in risk scoring yet)",
        "received_report": report,
        "timestamp": datetime.utcnow().isoformat(),
    }
