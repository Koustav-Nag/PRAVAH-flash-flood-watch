import React, { useEffect, useState, useCallback } from "react";
import RiskGauge from "./components/RiskGauge.jsx";
import ComponentBreakdown from "./components/ComponentBreakdown.jsx";
import RegionMap from "./components/RegionMap.jsx";
import SafePlaceCard from "./components/SafePlaceCard.jsx";
import RainfallTrend from "./components/RainfallTrend.jsx";
import DataSourcesPanel from "./components/DataSourcesPanel.jsx";
import TerrainPanel from "./components/TerrainPanel.jsx";
import RiverStatusPanel from "./components/RiverStatusPanel.jsx";
import ForecastTimeline from "./components/ForecastTimeline.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// Default location: roughly central Sonitpur — replace with the
// browser's geolocation in a real deployment.
const DEFAULT_LAT = 26.72;
const DEFAULT_LON = 92.9;
const RAINFALL_LOOKBACK_HOURS = 72;

export default function App() {
  const [health, setHealth] = useState(null);
  const [bounds, setBounds] = useState(null);
  const [shelters, setShelters] = useState([]);
  const [risk, setRisk] = useState(null);
  const [explain, setExplain] = useState(null);
  const [dispatch, setDispatch] = useState(null);
  const [rainfallSeries, setRainfallSeries] = useState(null);
  const [dataSources, setDataSources] = useState(null);
  const [riverStatus, setRiverStatus] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [lat, setLat] = useState(DEFAULT_LAT);
  const [lon, setLon] = useState(DEFAULT_LON);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadStatic = useCallback(async () => {
    try {
      const [h, b, s, rs, ds, rv, fc] = await Promise.all([
        fetch(`${API_BASE}/health`).then((r) => r.json()),
        fetch(`${API_BASE}/region/bounds`).then((r) => r.json()),
        fetch(`${API_BASE}/safe-places/shelters`).then((r) => r.json()),
        fetch(`${API_BASE}/data/rainfall-series?lookback_hours=${RAINFALL_LOOKBACK_HOURS}`).then((r) => r.json()),
        fetch(`${API_BASE}/data/sources`).then((r) => r.json()),
        fetch(`${API_BASE}/data/river-status`).then((r) => r.json()).catch(() => null),
        fetch(`${API_BASE}/risk/forecast`).then((r) => r.json()).catch(() => null),
      ]);
      setHealth(h);
      setBounds(b);
      setShelters(s);
      setRainfallSeries(rs);
      setDataSources(ds);
      setRiverStatus(rv);
      setForecast(fc);
    } catch (e) {
      setError("Could not reach the backend API. Is it running at " + API_BASE + "?");
    }
  }, []);

  const runDispatch = useCallback(async (latitude, longitude) => {
    setLoading(true);
    setError(null);
    try {
      const [riskRes, explainRes, dispatchRes] = await Promise.all([
        fetch(`${API_BASE}/risk/current`).then((r) => r.json()),
        fetch(`${API_BASE}/risk/explain`).then((r) => r.json()),
        fetch(`${API_BASE}/alerts/dispatch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ latitude, longitude }),
        }).then((r) => r.json()),
      ]);
      setRisk(riskRes);
      setExplain(explainRes);
      setDispatch(dispatchRes);
    } catch (e) {
      setError("Request failed — check that the backend is running.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatic();
    runDispatch(DEFAULT_LAT, DEFAULT_LON);
  }, [loadStatic, runDispatch]);

  const handleLocate = (e) => {
    e.preventDefault();
    runDispatch(parseFloat(lat), parseFloat(lon));
  };

  const category = dispatch?.risk?.category || risk?.risk?.final_category;
  const score = dispatch?.risk?.score ?? risk?.risk?.final_score;
  const terrainFeatures = risk?.features;

  // River threshold status computation (Hydrological Alert State)
  const primaryRiver = riverStatus?.primary_river;
  const secondaryRiver = riverStatus?.secondary_river;
  const primLevel = primaryRiver?.observed_level_m;
  const secLevel = secondaryRiver?.observed_level_m;
  const primDl = primaryRiver?.threshold_analysis?.danger_level_m || 65.23;
  const secDl = secondaryRiver?.threshold_analysis?.danger_level_m || 77.00;
  const primAboveDanger = Boolean(primaryRiver?.threshold_analysis?.is_above_danger);
  const secAboveDanger = Boolean(secondaryRiver?.threshold_analysis?.is_above_danger);
  const isAboveDanger = primAboveDanger || secAboveDanger;
  const secMargin = secondaryRiver?.threshold_analysis?.distance_to_danger_m ?? (secDl - (secLevel ?? 0));

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="masthead">
          <h1>Sonitpur Flash Flood Watch</h1>
          <p>Multi-source early warning — hilly-terrain pilot</p>
        </div>

        <div className="status-row">
          <span>
            <span className={`dot ${health?.gee_available ? "live" : "mock"}`} />
            Earth Engine data: {health?.gee_available ? "live" : "simulated (mock)"}
          </span>
          <span>
            <span className={`dot ${health?.xgboost_model_trained ? "live" : "mock"}`} />
            Prediction model: {health?.xgboost_model_trained ? "trained" : "physics-informed fallback"}
          </span>
        </div>

        {/* 1. ML RISK SCORE (Flash-Flood Runoff Nowcast) */}
        <div className="model-risk-card">
          <div className="card-badge-header">
            <span className="source-badge src-model">MODEL PREDICTION</span>
            <span className="card-sub-header">ML RISK SCORE</span>
          </div>
          <RiskGauge score={score} category={category} />
          <p className="risk-source">{dispatch?.risk?.source || risk?.risk?.source}</p>
          <p className="risk-explainer-note">
            Normalized predictive risk indicator from rainfall nowcasting &amp; terrain runoff.
          </p>
        </div>

        {/* 2. HYDROLOGICAL THRESHOLD ALERT (Physical Gauge Status) */}
        <div className={`river-threshold-card ${isAboveDanger ? "status-danger" : "status-normal"}`}>
          <div className="card-badge-header">
            <span className={`source-badge ${isAboveDanger ? "src-unavailable" : "src-observed"}`}>
              {isAboveDanger ? "ALERT ACTIVE" : "NORMAL"}
            </span>
            <span className="card-sub-header">HYDROLOGICAL ALERT</span>
          </div>

          <div className="threshold-summary-status">
            {isAboveDanger ? (
              <span className="threshold-status-text danger">
                ⚠️ Hydrological threshold alert: Jia Bharali ABOVE DANGER
              </span>
            ) : (
              <span className="threshold-status-text safe">
                ✓ ALL STATIONS BELOW DANGER
              </span>
            )}
          </div>

          <div className="station-quick-list">
            <div className="station-quick-row">
              <span className="stn-name">Jia Bharali (Secondary)</span>
              <span className={`stn-val ${secAboveDanger ? "val-danger" : ""}`}>
                {secLevel != null ? `${secLevel.toFixed(2)} m` : "—"}
                {secAboveDanger ? (
                  <span className="diff-pill">⚠️ +{Math.abs(secMargin).toFixed(2)}m DL</span>
                ) : (
                  <span className="diff-pill-ok">✓ Safe</span>
                )}
              </span>
            </div>
            <div className="station-quick-row">
              <span className="stn-name">Brahmaputra (Primary)</span>
              <span className="stn-val">
                {primLevel != null ? `${primLevel.toFixed(2)} m` : "—"}
                {primAboveDanger ? (
                  <span className="diff-pill">⚠️ Above DL</span>
                ) : (
                  <span className="diff-pill-ok">✓ Safe</span>
                )}
              </span>
            </div>
          </div>

          <p className="threshold-explainer-note">
            {isAboveDanger
              ? "Physical river stage at Jia Bharali exceeds statutory danger level. Model risk remains low because no heavy storm rainfall surge is currently detected."
              : "Direct physical observations against official CWC warning/danger thresholds."}
          </p>
        </div>

        <form className="location-form" onSubmit={handleLocate}>
          <label htmlFor="lat">Your location</label>
          <div className="coords">
            <input
              id="lat"
              type="number"
              step="0.0001"
              value={lat}
              onChange={(e) => setLat(e.target.value)}
              placeholder="Latitude"
            />
            <input
              type="number"
              step="0.0001"
              value={lon}
              onChange={(e) => setLon(e.target.value)}
              placeholder="Longitude"
            />
          </div>
          <button className="primary" type="submit" disabled={loading}>
            {loading ? "Checking…" : "Get nearest safe place"}
          </button>
        </form>

        <div>
          <h2 className="section-title">
            Data sources <span className="hint">data provenance</span>
          </h2>
          <DataSourcesPanel sources={dataSources?.sources} />
        </div>
      </aside>

      <main className="main">
        {error && <div className="error-banner">{error}</div>}

        {/* Operational Hydrological Alert Banner */}
        {isAboveDanger && (
          <div className="operational-alert-banner">
            <div className="alert-banner-header">
              <span className="alert-banner-badge">⚠️ ACTIVE HYDROLOGICAL THRESHOLD ALERT</span>
              <span className="alert-banner-tag">CWC Live Gauge Signal</span>
            </div>
            <div className="alert-banner-body">
              <p className="alert-banner-title">
                <strong>Jia Bharali at N.T. Road Crossing</strong> is currently <strong>{Math.abs(secMargin).toFixed(2)} m ABOVE Danger Level</strong> (Observed: {secLevel?.toFixed(2)} m · DL: {secDl?.toFixed(2)} m · HFL: 78.50 m).
              </p>
              <div className="alert-banner-explainer">
                <strong>Architectural Separation of Operational Signals:</strong>
                <p>
                  <strong>• Hydrological threshold alert:</strong> <em>Jia Bharali ABOVE DANGER</em> — active physical river stage monitored by CWC staff gauge exceeds statutory danger level.
                </p>
                <p>
                  <strong>• ML risk score:</strong> <em>{score != null ? (score * 100).toFixed(0) : "10"}/100 ({category || "Low"})</em> — normalized predictive risk indicator evaluating incoming <em>rainfall-induced flash flood surge wave</em> over the nowcast window (currently low due to minimal storm precipitation).
                </p>
                <p style={{ marginTop: "4px", fontSize: "11px", color: "var(--text-muted)" }}>
                  In operational early-warning architectures, physical river gauge threshold alerts and machine-learning flash flood nowcasts are kept strictly separate as two independent, complementary signals.
                </p>
              </div>
            </div>
          </div>
        )}

        <section>
          <h2 className="section-title">
            Region overview <span className="hint">Sonitpur district, Assam</span>
          </h2>
          <div className="panel map-panel">
            <RegionMap
              bounds={bounds}
              origin={dispatch?.origin || { latitude: lat, longitude: lon }}
              safePlace={dispatch?.safe_place}
              shelters={shelters}
              riskCategory={category}
              riverStatus={riverStatus}
            />
            <div className="legend">
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: "var(--risk-low)" }} />
                Your location
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: "var(--water)" }} />
                Recommended safe place
              </div>
              <div className="legend-item">
                <span className="legend-line" style={{ background: "#0284c7" }} />
                Brahmaputra (Primary)
              </div>
              <div className="legend-item">
                <span className="legend-line" style={{ background: "#7c3aed" }} />
                Jia Bharali (Secondary)
              </div>
              <div className="legend-item">
                <span className="legend-swatch-diamond" style={{ background: "#0284c7" }} />
                Tezpur Gauge (Primary)
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: "#7c3aed" }} />
                N.T. Road Gauge (Secondary)
              </div>
            </div>
          </div>
        </section>

        <section className="two-col">
          <div>
            <h2 className="section-title">
              River status <span className="hint">Brahmaputra (Primary) &amp; Jia Bharali (Secondary)</span>
            </h2>
            <div className="panel">
              <RiverStatusPanel riverStatus={riverStatus} />
            </div>
          </div>

          <div>
            <h2 className="section-title">
              CWC AFF River Forecast <span className="hint">Hydrodynamic model forecast &amp; ML surge (+1h to +24h)</span>
            </h2>
            <div className="panel">
              <ForecastTimeline forecast={forecast} />
            </div>
          </div>
        </section>

        <section className="two-col">
          <div>
            <h2 className="section-title">
              Rainfall <span className="hint">GPM/IMERG, last {RAINFALL_LOOKBACK_HOURS}h</span>
            </h2>
            <div className="panel">
              <RainfallTrend points={rainfallSeries?.points} lookbackHours={RAINFALL_LOOKBACK_HOURS} />
            </div>
          </div>

          <div>
            <h2 className="section-title">
              Terrain &amp; catchment <span className="hint">STATIC / DERIVED</span>
            </h2>
            <div className="panel">
              <TerrainPanel features={terrainFeatures} />
            </div>
          </div>
        </section>

        <section className="two-col">
          <div>
            <h2 className="section-title">
              Why this risk level <span className="hint">physics-informed breakdown</span>
            </h2>
            <div className="panel">
              <ComponentBreakdown components={explain?.physics_components} />
            </div>
          </div>

          <div>
            <h2 className="section-title">
              Alert dispatch <span className="hint">nearest safe place</span>
            </h2>
            <div className="panel">
              <SafePlaceCard safePlace={dispatch?.safe_place} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
