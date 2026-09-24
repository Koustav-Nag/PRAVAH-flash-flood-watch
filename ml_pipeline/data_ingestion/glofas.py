from __future__ import annotations

import datetime
import httpx
from loguru import logger

def fetch_glofas_discharge(latitude: float = 26.92, longitude: float = 92.75) -> dict:
    """
    Fetches modelled river discharge from the Open-Meteo Flood API (GloFAS).
    """
    url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "river_discharge,river_discharge_mean,river_discharge_max,river_discharge_min",
        "forecast_days": 7
    }
    
    try:
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        # Get IST current time
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        retrieved_at = now.isoformat()
        
        daily_data = []
        if "daily" in data and "time" in data["daily"]:
            times = data["daily"]["time"]
            discharges = data["daily"].get("river_discharge", [])
            means = data["daily"].get("river_discharge_mean", [])
            maxs = data["daily"].get("river_discharge_max", [])
            mins = data["daily"].get("river_discharge_min", [])
            
            for i, t in enumerate(times):
                daily_data.append({
                    "date": t,
                    "discharge_m3s": discharges[i] if i < len(discharges) else None,
                    "discharge_mean_m3s": means[i] if i < len(means) else None,
                    "discharge_max_m3s": maxs[i] if i < len(maxs) else None,
                    "discharge_min_m3s": mins[i] if i < len(mins) else None,
                })
                
        return {
            "source": "GloFAS / Open-Meteo Flood API",
            "data_category": "MODELLED",
            "variable": "river_discharge",
            "unit": "m³/s",
            "retrieved_at": retrieved_at,
            "location": {"latitude": latitude, "longitude": longitude},
            "daily": daily_data,
            "note": "GloFAS modelled discharge. Not observed river level. Do not substitute for gauge water level."
        }
        
    except Exception as exc:
        logger.warning(f"Failed to fetch GloFAS discharge: {exc}")
        return {
            "status": "UNAVAILABLE",
            "error": str(exc)
        }
