# 🌊 Flash Flood Prediction & Early Warning System

> **SIH 2026 Prototype** — Real-time flash flood risk assessment, forecasting, and evacuation guidance for Sonitpur District, Assam.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8.3-646CFF?logo=vite&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Data Sources](#-data-sources--status)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
  - [Running the Backend](#running-the-backend)
  - [Running the Frontend](#running-the-frontend)
- [API Reference](#-api-reference)
- [ML Pipeline](#-ml-pipeline)
  - [Historical Dataset](#historical-dataset)
  - [Training the Model](#training-the-model)
- [Testing](#-testing)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## Overview

Flash floods in the Brahmaputra basin cause devastating damage with very little warning time. This system fuses **satellite imagery**, **real-time river gauge data**, **weather forecasts**, and **terrain analysis** into a single risk score, then recommends the nearest safe evacuation point.

The prototype targets **Sonitpur District, Assam** — monitoring the Brahmaputra at Tezpur and the Jia Bharali at N.T. Road Crossing — but the architecture is designed to scale to any flood-prone basin.

Every data-ingestion module has a **mock fallback**, so the entire pipeline runs and is demoable _before_ real GEE credentials or a trained model exist. Hit `GET /health` to see which pieces are live vs. mocked.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Hybrid Risk Engine** | Combines a physics-informed heuristic score with XGBoost ML predictions for robust, explainable risk assessment |
| **Multi-Source Data Fusion** | Integrates GPM/IMERG rainfall, Sentinel-1 SAR, SRTM/HydroSHEDS terrain, CWC river gauges, Open-Meteo forecasts, and GloFAS discharge data |
| **Real-Time River Monitoring** | Live scraping of CWC/AFF gauge data with QC, spike detection, flatline checks, and threshold-based alerting |
| **Forecast Horizons** | Risk projections at 1h, 3h, 6h, 12h, and 24h windows using river-level trend extrapolation and weather forecasts |
| **Safe Place Recommendation** | Finds the nearest known shelter or, if none within 15 km, the nearest higher-ground fallback using elevation data |
| **Interactive Dashboard** | React-based UI with live risk gauge, Leaflet map, rainfall trends, river hydrograph, and safe-place navigation |
| **Graceful Degradation** | Every data source has a mock fallback — the system always returns a risk score, even with zero external connectivity |

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                              │
│  GEE (GPM/IMERG, Sentinel-1/2, SRTM/HydroSHEDS)                  │
│  CWC/AFF (Live river gauge scraping)                               │
│  Open-Meteo (Weather forecasts)    GloFAS (Discharge forecasts)    │
└──────────────────────┬─────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                             │
│  fusion.py  →  single fused feature vector per timestamp         │
│  river_level_features.py  →  threshold ratios, rates of change   │
└──────────────────────┬───────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌──────────────────────┐
│ Physics-Informed │     │   XGBoost ML Model   │
│  Risk Heuristic  │     │  (trained on flood   │
│  (always works)  │     │   event history)     │
└────────┬─────────┘     └──────────┬───────────┘
         │                          │
         └────────────┬─────────────┘
                      ▼
           ┌─────────────────────┐
           │  Hybrid Risk Score  │
           │  + Risk Category    │
           └──────────┬──────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐   ┌──────────────────────┐
│  Safe Place      │   │  FastAPI Backend      │
│  Recommendation  │   │  REST API Endpoints   │
│  Engine          │   │  /risk  /alerts       │
└──────────────────┘   └──────────┬────────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  React + Vite       │
                       │  Dashboard UI       │
                       │  (Leaflet, Recharts)│
                       └─────────────────────┘
```

---

## 📂 Project Structure

```
flash-flood-sih/
├── app/                              # FastAPI backend application
│   ├── main.py                       # API entrypoint with all route handlers
│   ├── core/
│   │   └── config.py                 # Pydantic settings (env vars, gauge thresholds)
│   └── api/                          # (Future: modular route files)
│
├── ml_pipeline/                      # Machine learning & data pipeline
│   ├── data_ingestion/               # Data source connectors
│   │   ├── gee_client.py             # Google Earth Engine initialisation
│   │   ├── precipitation.py          # GPM/IMERG rainfall via GEE
│   │   ├── sentinel.py               # Sentinel-1/2 SAR & land cover via GEE
│   │   ├── terrain.py                # SRTM/HydroSHEDS terrain features via GEE
│   │   ├── india_wris.py             # CWC/AFF live river gauge scraper
│   │   ├── open_meteo.py             # Open-Meteo weather forecasts
│   │   ├── glofas.py                 # GloFAS discharge forecasts
│   │   ├── river_levels.py           # Gauge threshold features
│   │   ├── river_qc_reconcile.py     # River data QC & reconciliation
│   │   ├── historical_events.py      # Sonitpur flood event catalogue
│   │   └── target_builder.py         # Forward-looking flood targets (no leakage)
│   ├── feature_engineering/
│   │   ├── fusion.py                 # Multi-source feature vector fusion
│   │   └── river_level_features.py   # River-level trend & projection features
│   ├── models/
│   │   └── xgboost_model.py          # XGBoost flash flood classifier
│   ├── risk_engine/
│   │   ├── physics_informed.py       # Interpretable physics-based risk score
│   │   └── hybrid.py                 # ML + physics blended risk score
│   ├── safe_place_engine/
│   │   ├── registry.py               # Shelter CSV loader
│   │   ├── safe_place.py             # Nearest-shelter recommender
│   │   └── elevation_fallback.py     # Higher-ground fallback when no shelter nearby
│   ├── train.py                      # Model training entrypoint
│   ├── build_training_dataset.py     # Raw data → ML-ready CSV builder
│   └── fetch_historical_rainfall.py  # GEE-based historical rainfall fetcher
│
├── frontend/                         # React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx                   # Main dashboard layout
│   │   ├── main.jsx                  # React entry point
│   │   ├── styles.css                # Global styles
│   │   ├── components/
│   │   │   ├── RiskGauge.jsx         # Animated risk score gauge
│   │   │   ├── RegionMap.jsx         # Leaflet map (risk zones, safe places)
│   │   │   ├── RainfallTrend.jsx     # Recharts rainfall time series
│   │   │   ├── RiverStatusPanel.jsx  # Live river hydrograph & status
│   │   │   ├── ForecastTimeline.jsx  # Multi-horizon forecast display
│   │   │   ├── ComponentBreakdown.jsx# Risk factor breakdown
│   │   │   ├── SafePlaceCard.jsx     # Evacuation route card
│   │   │   ├── DataSourcesPanel.jsx  # Live vs. mock data source status
│   │   │   └── TerrainPanel.jsx      # Terrain analysis display
│   │   └── data/
│   │       └── rivers_sonitpur.json  # Sonitpur river GeoJSON
│   └── package.json
│
├── data/
│   ├── historical/                   # Sourced flood event data
│   │   ├── raw/                      # Raw rainfall & river observations
│   │   ├── metadata/                 # Event catalogue, data dictionary, sources
│   │   └── ml_ready/                 # Training-ready feature table
│   ├── reference/                    # River danger levels, static lookups
│   ├── safe_places/                  # Shelter registry CSV
│   ├── raw/                          # Unprocessed ingested data
│   └── processed/                    # Cleaned intermediate data
│
├── tests/                            # Pytest test suite
│   ├── test_data_provenance.py       # Source attribution checks
│   ├── test_temporal_leakage.py      # Temporal leakage prevention tests
│   ├── test_threshold_features.py    # Gauge threshold feature tests
│   ├── test_india_wris.py            # CWC/AFF scraper tests
│   ├── test_river_qc_reconcile.py    # River QC pipeline tests
│   ├── test_sentinel.py             # Sentinel integration tests
│   └── test_missing_data.py          # Missing data handling tests
│
├── notebooks/                        # Jupyter exploration notebooks
├── .env.example                      # Environment variable template
├── requirements.txt                  # Python dependencies
├── TECH_STACK.md                     # Detailed technology stack documentation
└── README.md                         # ← You are here
```

---

## 📡 Data Sources & Status

| Source | Module | Status |
|---|---|---|
| GPM/IMERG rainfall | `precipitation.py` | ✅ GEE integration written; mock fallback active |
| Sentinel-1 (surface water) | `sentinel.py` | ✅ GEE integration written; mock fallback active |
| Sentinel-2 (land cover) | `sentinel.py` | ⏳ Mock only — live classification not yet built |
| SRTM / HydroSHEDS | `terrain.py` | ✅ GEE integration written; mock fallback active |
| CWC/AFF river gauges | `india_wris.py` | ✅ Live scraping from `aff.india-water.gov.in` |
| Open-Meteo weather | `open_meteo.py` | ✅ Live forecast fetching, no API key needed |
| GloFAS discharge | `glofas.py` | ✅ CDS API integration written |
| River danger levels | `river_levels.py` | ✅ Real thresholds for Tezpur & Jia Bharali |
| Historical flood events | `historical_events.py` | ✅ 6 sourced events (1988–2024) with primary sources |
| Crowd/social reports | `main.py` | ⏳ Stub endpoint only, not used in scoring |
| Shelter registry | `registry.py` | ⚠️ Placeholder data — needs real relief camp list |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm** (for the frontend)
- (Optional) Google Earth Engine service account for live satellite data

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/flash-flood-sih.git
cd flash-flood-sih

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials. The system works in **mock mode** without any credentials — see the [GEE client docstring](ml_pipeline/data_ingestion/gee_client.py) for full setup instructions.

### Running the Backend

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Running the Frontend

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173`. The dashboard expects the backend running at `http://localhost:8000` (override with `VITE_API_BASE` env var).

---

## 📖 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health — shows which data sources are live vs. mocked |
| `GET` | `/risk/current` | Current flood risk score, category, and component breakdown |
| `GET` | `/risk/explain` | Human-readable explanation of the risk assessment |
| `GET` | `/safe-places/shelters` | List of known shelters in the region |
| `POST` | `/alerts/dispatch` | Returns risk + nearest safe place for a given lat/lon |
| `GET` | `/river/status` | Live river gauge readings with threshold analysis |
| `GET` | `/forecast/timeline` | Multi-horizon (1h–24h) risk forecast timeline |

**Example: Dispatch an alert**
```bash
curl -X POST http://localhost:8000/alerts/dispatch \
  -H "Content-Type: application/json" \
  -d '{"latitude": 26.65, "longitude": 92.80}'
```

---

## 🤖 ML Pipeline

### Historical Dataset

The `data/historical/` directory follows a strict data provenance protocol:

```
data/historical/
├── raw/
│   ├── raw_rainfall.csv             # Raw rainfall observations
│   └── raw_river_data.csv           # Raw river gauge readings
├── metadata/
│   ├── flood_events.csv             # 6 sourced Sonitpur flood events (1988–2024)
│   ├── source_metadata.csv          # Primary source & reference URL for each event
│   ├── terrain_features.csv         # Static terrain features
│   └── data_dictionary.csv          # Column definitions, units, and sources
└── ml_ready/
    └── sonitpur_flood_training.csv  # Training feature table (built from raw + events)
```

- **No synthetic data** is generated anywhere in this pipeline
- Every event requires a traceable primary source (official government documents or CWC bulletins)
- Targets (`flood_next_6h/12h/24h`) look only _forward_ from each timestamp — no temporal leakage
- Train/test splitting is chronological and event-grouped, never random row-level

### Training the Model

```bash
python -m ml_pipeline.train --target flood_next_24h
```

> ⚠️ This command intentionally refuses to run while `sonitpur_flood_training.csv` is empty. Until trained, the system gracefully falls back to the physics-informed score only.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

The test suite covers:
- **Data provenance** — verifies every event has a traceable source
- **Temporal leakage** — ensures targets don't leak future data into features
- **Threshold features** — validates gauge-based danger/warning level calculations
- **River QC** — tests spike detection, flatline checks, reconciliation logic
- **Missing data** — confirms graceful fallbacks when data sources are unavailable

---

## 🗺 Roadmap

- [ ] Fetch real GPM/IMERG rainfall for the 6 compiled flood events
- [ ] Acquire CWC/Assam WRD river-level observations for training windows
- [ ] Per-grid-cell / per-sub-basin spatial risk disaggregation
- [ ] Live Sentinel-2 land-cover classification
- [ ] Verified real shelter/relief-camp registry for Sonitpur
- [ ] SMS/push notification dispatch integration
- [ ] Browser geolocation in the dashboard
- [ ] Distance-to-river from HydroSHEDS flow-accumulation raster
- [ ] Wire live water-level readings into danger-level-ratio feature

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

> **Important:** Never commit `.env` files or any credential/key files. See `.gitignore` for excluded patterns.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Central Water Commission (CWC)** — River gauge data and flood forecasts via [AFF Portal](https://aff.india-water.gov.in)
- **Google Earth Engine** — Satellite imagery access (GPM/IMERG, Sentinel, SRTM)
- **Open-Meteo** — Free weather forecast API
- **Copernicus GloFAS** — Global flood awareness discharge forecasts
- **ASDMA (Assam State Disaster Management Authority)** — Flood event documentation
- **HydroSHEDS / WWF** — Hydrological terrain datasets
