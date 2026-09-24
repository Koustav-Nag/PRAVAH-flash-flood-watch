# 🛠 Technology Stack

> Complete technology reference for the Flash Flood Prediction & Early Warning System.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Backend](#-backend)
- [Frontend](#-frontend)
- [Machine Learning & Data Science](#-machine-learning--data-science)
- [Geospatial & Remote Sensing](#-geospatial--remote-sensing)
- [External Data Sources & APIs](#-external-data-sources--apis)
- [Testing & Quality](#-testing--quality)
- [DevOps & Tooling](#-devops--tooling)

---

## Architecture Overview

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | React 18 + Vite 8 | Interactive dashboard UI |
| **Backend API** | FastAPI + Uvicorn | REST API serving risk data |
| **ML Pipeline** | XGBoost + scikit-learn | Flash flood prediction model |
| **Data Ingestion** | Google Earth Engine, CWC/AFF, Open-Meteo, GloFAS | Multi-source data fusion |
| **Geospatial** | GeoPandas, Shapely, Rasterio, PyProj | Spatial data processing |
| **Config** | Pydantic Settings + python-dotenv | Type-safe environment configuration |

---

## 🐍 Backend

### Core Framework

| Package | Version | Role |
|---|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | 0.115.0 | High-performance async REST API framework with automatic OpenAPI docs |
| [Uvicorn](https://www.uvicorn.org/) | 0.30.6 | ASGI server (HTTP/1.1 and WebSocket support) |
| [Pydantic](https://docs.pydantic.dev/) | 2.9.2 | Data validation, serialisation, and API schema generation |
| [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | 2.5.2 | Environment variable loading with type coercion and `.env` file support |

### Utilities

| Package | Version | Role |
|---|---|---|
| [python-dotenv](https://github.com/theskumar/python-dotenv) | 1.0.1 | `.env` file loading |
| [HTTPX](https://www.python-httpx.org/) | 0.27.2 | Modern async HTTP client for external API calls (CWC, Open-Meteo, GloFAS) |
| [Loguru](https://github.com/Delgan/loguru) | ≥0.7.2 | Structured logging with coloured output and easy configuration |
| [Joblib](https://joblib.readthedocs.io/) | 1.4.2 | Model serialisation (save/load trained XGBoost models) |

### API Design

- **CORS middleware** enabled for cross-origin frontend requests
- **Automatic API documentation** at `/docs` (Swagger UI) and `/redoc`
- **Health endpoint** reports live vs. mocked data sources
- All responses are JSON with Pydantic model validation

---

## ⚛️ Frontend

### Core

| Package | Version | Role |
|---|---|---|
| [React](https://react.dev/) | 18.3.1 | Component-based UI framework |
| [React DOM](https://react.dev/) | 18.3.1 | DOM rendering for React |
| [Vite](https://vite.dev/) | 8.3.0 | Next-generation build tool with instant HMR |
| [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react) | 4.7.0 | React Fast Refresh and JSX support |

### Visualisation & Mapping

| Package | Version | Role |
|---|---|---|
| [Leaflet](https://leafletjs.com/) | 1.9.4 | Interactive tile-based maps |
| [React Leaflet](https://react-leaflet.js.org/) | 4.2.1 | React bindings for Leaflet (markers, polygons, popups) |
| [Recharts](https://recharts.org/) | 2.12.7 | Composable chart components (rainfall trends, hydrographs, risk gauges) |

### Dashboard Components

| Component | File | Purpose |
|---|---|---|
| `RiskGauge` | `RiskGauge.jsx` | Animated circular gauge showing current flood risk score (0–1) |
| `RegionMap` | `RegionMap.jsx` | Leaflet map with risk zones, river overlays, user location, and safe-place markers |
| `RainfallTrend` | `RainfallTrend.jsx` | Recharts time-series chart of recent and forecast rainfall |
| `RiverStatusPanel` | `RiverStatusPanel.jsx` | Live river hydrograph with danger/warning level indicators |
| `ForecastTimeline` | `ForecastTimeline.jsx` | Multi-horizon (1h–24h) forecast risk progression |
| `ComponentBreakdown` | `ComponentBreakdown.jsx` | Bar/radar breakdown of individual risk factors |
| `SafePlaceCard` | `SafePlaceCard.jsx` | Nearest safe-place card with distance, type, and directions |
| `DataSourcesPanel` | `DataSourcesPanel.jsx` | Live status of all data sources (live ✅ / mocked ⏳) |
| `TerrainPanel` | `TerrainPanel.jsx` | Terrain analysis (slope, elevation, drainage density) |

---

## 🤖 Machine Learning & Data Science

### Core ML Stack

| Package | Version | Role |
|---|---|---|
| [XGBoost](https://xgboost.readthedocs.io/) | ≥2.0.0 | Gradient-boosted tree classifier for flash flood prediction |
| [scikit-learn](https://scikit-learn.org/) | ≥1.5.0 | Preprocessing, metrics, cross-validation, and model evaluation |
| [Pandas](https://pandas.pydata.org/) | ≥2.1.0 | Tabular data manipulation, time-series handling, CSV I/O |
| [NumPy](https://numpy.org/) | ≥1.27.0 | Numerical computation and array operations |

### ML Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌────────────────────┐
│  Data Ingestion │───▶│    Feature       │───▶│   Model Training   │
│                 │    │   Engineering    │    │                    │
│ • precipitation │    │ • fusion.py      │    │ • xgboost_model.py │
│ • sentinel      │    │ • river_level    │    │ • train.py         │
│ • terrain       │    │   _features.py   │    │ • chronological    │
│ • india_wris    │    │                  │    │   split            │
│ • open_meteo    │    │                  │    │                    │
│ • glofas        │    │                  │    │                    │
└─────────────────┘    └─────────────────┘    └────────────────────┘
                                                        │
                                                        ▼
                                              ┌────────────────────┐
                                              │   Risk Engine      │
                                              │                    │
                                              │ • physics_informed │
                                              │ • hybrid blending  │
                                              └────────────────────┘
```

### Risk Scoring Approach

| Engine | Description | Availability |
|---|---|---|
| **Physics-Informed** | Weighted heuristic combining rainfall intensity, terrain vulnerability, soil saturation proxy, and river-level ratio | Always available (no training data needed) |
| **XGBoost ML** | Trained classifier on historical flood events with forward-looking targets | Requires populated training dataset |
| **Hybrid** | Blended score — uses both when ML model is available, falls back to physics-only otherwise | Always available |

### Data Integrity Guarantees

- **No synthetic data** — every training row traces to a real observation
- **No temporal leakage** — targets built by `target_builder.py` look only _forward_
- **Chronological splits** — train/test split is time-ordered and event-grouped, never random
- **Provenance tracking** — every flood event requires a primary source URL

---

## 🌍 Geospatial & Remote Sensing

### Libraries

| Package | Version | Role |
|---|---|---|
| [earthengine-api](https://developers.google.com/earth-engine/) | 1.0.0 | Google Earth Engine Python client for satellite data access |
| [geemap](https://geemap.org/) | 0.35.1 | Interactive GEE visualisation and analysis |
| [GeoPandas](https://geopandas.org/) | 1.0.1 | GeoDataFrame operations, spatial joins, and shapefile I/O |
| [Rasterio](https://rasterio.readthedocs.io/) | ≥1.4.0 | Raster data reading/writing (GeoTIFF, elevation grids) |
| [Shapely](https://shapely.readthedocs.io/) | ≥2.0.6 | Geometric operations (buffers, intersections, distance calculations) |
| [PyProj](https://pyproj4.github.io/pyproj/) | ≥3.6.1 | Coordinate reference system transformations |

### Satellite Products Used

| Product | Source | Resolution | Usage |
|---|---|---|---|
| GPM/IMERG | NASA | 0.1° / 30 min | Near-real-time precipitation estimation |
| Sentinel-1 (C-SAR) | ESA/Copernicus | 10 m | Surface water extent detection |
| Sentinel-2 (MSI) | ESA/Copernicus | 10–60 m | Land-cover classification (planned) |
| SRTM | NASA | 30 m | Digital elevation model for terrain analysis |
| HydroSHEDS | WWF/USGS | 15 arc-sec | Flow accumulation, drainage networks |

---

## 📡 External Data Sources & APIs

| Source | API / Method | Auth Required | Usage in System |
|---|---|---|---|
| **Google Earth Engine** | `earthengine-api` Python SDK | GCP Service Account | Satellite imagery (rainfall, SAR, terrain) |
| **CWC Flood Forecast (AFF)** | HTML scraping via `httpx` | None | Live river gauge readings for Brahmaputra & Jia Bharali |
| **Open-Meteo** | REST API | None (free tier) | Weather forecasts (temperature, precipitation, wind) |
| **GloFAS (Copernicus)** | CDS API | CDS account (free) | River discharge forecasts |
| **India-WRIS** | REST API | Optional API key | Historical river data (legacy path) |

### Pilot Region Configuration

| Parameter | Value |
|---|---|
| **District** | Sonitpur, Assam |
| **Primary Gauge** | Brahmaputra at Tezpur (26.617°N, 92.797°E) |
| **Secondary Gauge** | Jia Bharali at N.T. Road Crossing (26.811°N, 92.880°E) |
| **Bounding Box** | 92.60°E – 93.45°E, 26.55°N – 27.20°N |

---

## 🧪 Testing & Quality

| Tool | Version | Purpose |
|---|---|---|
| [pytest](https://docs.pytest.org/) | 8.3.3 | Test framework and runner |

### Test Coverage Areas

| Test File | What It Validates |
|---|---|
| `test_data_provenance.py` | Every flood event has a verifiable primary source |
| `test_temporal_leakage.py` | Feature-target alignment has no future-data contamination |
| `test_threshold_features.py` | Gauge danger/warning level ratio calculations |
| `test_india_wris.py` | CWC/AFF scraper parsing, error handling, and edge cases |
| `test_river_qc_reconcile.py` | Spike detection, flatline checks, multi-source reconciliation |
| `test_sentinel.py` | Sentinel data ingestion and mock fallback |
| `test_missing_data.py` | Graceful degradation when data sources are unavailable |

---

## ⚙️ DevOps & Tooling

### Runtime

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Backend runtime |
| **Node.js** | 18+ | Frontend build toolchain |
| **npm** | 9+ | Frontend package management |

### Configuration Management

| Tool | Purpose |
|---|---|
| **Pydantic Settings** | Type-safe env var loading with validation and defaults |
| **`.env` files** | Local secrets management (never committed to Git) |
| **`.env.example`** | Template for required environment variables |

### Recommended IDE Extensions

| Extension | Purpose |
|---|---|
| Python (Pylance) | Python IntelliSense and type checking |
| ES7+ React Snippets | React component boilerplate |
| Prettier | Frontend code formatting |
| GitLens | Git history and blame annotations |

---

## Version Summary

| Component | Version |
|---|---|
| Python | 3.11+ |
| FastAPI | 0.115.0 |
| React | 18.3.1 |
| Vite | 8.3.0 |
| XGBoost | ≥2.0.0 |
| scikit-learn | ≥1.5.0 |
| Pandas | ≥2.1.0 |
| earthengine-api | 1.0.0 |
| Leaflet | 1.9.4 |
| Recharts | 2.12.7 |
| pytest | 8.3.3 |
