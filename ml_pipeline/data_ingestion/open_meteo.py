from __future__ import annotations

import datetime
import httpx
from loguru import logger
from app.core.config import settings

def fetch_weather_forecast(latitude: float | None = None, longitude: float | None = None) -> dict:
    """
    Fetches weather forecasts from the free Open-Meteo API.
    Uses region centroid from settings if lat/lon not provided.
    """
    if latitude is None:
        latitude = (settings.REGION_MIN_LAT + settings.REGION_MAX_LAT) / 2
    if longitude is None:
        longitude = (settings.REGION_MIN_LON + settings.REGION_MAX_LON) / 2
        
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "precipitation,rain,precipitation_probability",
        "forecast_days": 3,
        "timezone": "Asia/Kolkata"
    }
    
    try:
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        # Get IST current time to match Asia/Kolkata
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        now_naive = now.replace(tzinfo=None)
        retrieved_at = now.isoformat()
        
        hourly_data = []
        if "hourly" in data and "time" in data["hourly"]:
            times = data["hourly"]["time"]
            precips = data["hourly"].get("precipitation", [])
            rains = data["hourly"].get("rain", [])
            probs = data["hourly"].get("precipitation_probability", [])
            
            for i, t in enumerate(times):
                hourly_data.append({
                    "timestamp": t,
                    "precipitation_mm": precips[i] if i < len(precips) else None,
                    "rain_mm": rains[i] if i < len(rains) else None,
                    "precipitation_probability": probs[i] if i < len(probs) else None,
                })
        
        # Calculate forecast horizons relative to now
        forecast_horizons = {}
        horizons = {"+1h": 1, "+3h": 3, "+6h": 6, "+12h": 12, "+24h": 24}
        
        for name, hours in horizons.items():
            target_time = now_naive + datetime.timedelta(hours=hours)
            target_iso = target_time.strftime("%Y-%m-%dT%H:00")
            
            # Find closest hourly data matching the target hour
            horizon_data = None
            for h in hourly_data:
                if h["timestamp"].startswith(target_iso):
                    horizon_data = h
                    break
            
            # Fallback to closest if exact hour string match fails
            if horizon_data is None and hourly_data:
                try:
                    horizon_data = min(
                        hourly_data, 
                        key=lambda x: abs((datetime.datetime.fromisoformat(x["timestamp"]) - target_time).total_seconds())
                    )
                except Exception:
                    pass
            
            forecast_horizons[name] = horizon_data

        return {
            "source": "Open-Meteo",
            "data_category": "FORECAST",
            "retrieved_at": retrieved_at,
            "location": {"latitude": latitude, "longitude": longitude},
            "hourly": hourly_data,
            "forecast_horizons": forecast_horizons
        }
        
    except Exception as exc:
        logger.warning(f"Failed to fetch weather forecast from Open-Meteo: {exc}")
        return {
            "status": "UNAVAILABLE",
            "error": str(exc)
        }
