import React from "react";

/**
 * Multi-horizon forecast timeline showing risk predictions at
 * +1h, +3h, +6h, +12h, +24h with danger-level reference.
 *
 * Each value is labelled MODEL PREDICTION or UNAVAILABLE.
 */

const CATEGORY_COLORS = {
  low: "var(--risk-low)",
  medium: "var(--risk-medium)",
  high: "var(--risk-high)",
  severe: "var(--risk-severe)",
};

const BADGE_CLASS = {
  OBSERVED: "src-observed",
  DERIVED: "src-derived",
  MODELLED: "src-modelled",
  FORECAST: "src-forecast",
  "STATIC REFERENCE": "src-static",
  "DEMO / SIMULATED": "src-mock",
  UNAVAILABLE: "src-unavailable",
};

export default function ForecastTimeline({ forecast }) {
  if (!forecast) return <p className="muted">Loading forecast…</p>;

  const {
    current,
    horizons,
    danger_level_reference,
    primary_danger_level_reference,
    secondary_danger_level_reference,
    note,
  } = forecast;

  const primaryRef = primary_danger_level_reference || danger_level_reference;
  const secondaryRef = secondary_danger_level_reference;
  const primaryDl = primaryRef?.danger_level_m;

  return (
    <div className="forecast-timeline">
      {/* Current */}
      <div className="forecast-now">
        <span className="forecast-label">Nowcast Surge</span>
        <span
          className="forecast-score"
          style={{ color: CATEGORY_COLORS[current?.risk_category] || "var(--text)" }}
        >
          {current?.risk_score != null ? (current.risk_score * 100).toFixed(0) : "—"}
        </span>
        <span className="forecast-category">{current?.risk_category || "—"}</span>
        <span className="source-badge src-model">MODEL SURGE</span>
        {current?.hydrological_alert?.is_above_danger && (
          <span
            style={{
              fontSize: "10px",
              color: "#f87171",
              background: "rgba(239, 68, 68, 0.18)",
              border: "1px solid rgba(239, 68, 68, 0.4)",
              padding: "2px 6px",
              borderRadius: "4px",
              fontWeight: 700,
              marginTop: "4px",
              textAlign: "center",
            }}
          >
            ⚠️ Gauge &gt; DL
          </span>
        )}
      </div>

      {/* Horizons */}
      <div className="forecast-horizons">
        {horizons &&
          Object.entries(horizons).map(([key, h]) => (
            <div key={key} className="forecast-horizon-card">
              <span className="horizon-label">{key}</span>
              <span
                className="horizon-score"
                style={{ color: CATEGORY_COLORS[h.risk_category] || "var(--text)" }}
              >
                {h.risk_score != null ? (h.risk_score * 100).toFixed(0) : "—"}
              </span>
              <span className="horizon-category">{h.risk_category || "—"}</span>

              {/* Primary Predicted level (Brahmaputra) */}
              <span
                className={`horizon-level ${
                  h.predicted_level_m != null ? "has-level" : ""
                } ${
                  primaryDl && h.predicted_level_m >= primaryDl
                    ? "level-danger"
                    : ""
                }`}
                title={h.predicted_level_label || ""}
              >
                {h.predicted_level_m != null
                  ? `🌊 ${h.predicted_level_m.toFixed(2)} m`
                  : "Level: unavailable"}
              </span>

              {/* Secondary Predicted level (Jia Bharali) */}
              {h.secondary_predicted_level_m != null && (
                <span
                  style={{
                    fontSize: "11px",
                    color: "var(--text-muted)",
                    display: "block",
                    marginTop: "2px",
                  }}
                  title="Jia Bharali projected stage"
                >
                  Jia Bharali: {h.secondary_predicted_level_m.toFixed(2)} m
                </span>
              )}

              {/* Weather */}
              {h.weather_forecast && (
                <span className="horizon-weather">
                  🌧 {h.weather_forecast.precipitation_mm ?? "—"} mm
                  {h.weather_forecast.precipitation_probability != null &&
                    ` (${h.weather_forecast.precipitation_probability}%)`}
                </span>
              )}

              <span
                className={`source-badge ${BADGE_CLASS[h.data_category] || "src-model"}`}
                title={h.primary_forecast_source || ""}
              >
                {h.data_category === "FORECAST" ? "CWC FORECAST" : (h.data_category || "MODEL")}
              </span>
            </div>
          ))}
      </div>

      {/* Danger level references */}
      <div className="forecast-reference" style={{ flexDirection: "column", alignItems: "flex-start", gap: "6px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span className="source-badge src-static">STATIC REFERENCE</span>
          <span style={{ fontSize: "11px", fontWeight: 600, color: "var(--text-muted)" }}>
            STATION DANGER THRESHOLDS
          </span>
        </div>
        <div style={{ fontSize: "12px", display: "flex", flexDirection: "column", gap: "3px" }}>
          {primaryRef && (
            <div>
              <strong>Primary: {primaryRef.river || "Brahmaputra"}</strong> ({primaryRef.gauge_location || "Tezpur"}) — DL: <strong>{primaryRef.danger_level_m} m</strong>
              {primaryRef.hfl_m && <> · HFL: <strong>{primaryRef.hfl_m} m</strong></>}
            </div>
          )}
          {secondaryRef && (
            <div>
              <strong>Secondary: {secondaryRef.river || "Jia Bharali"}</strong> ({secondaryRef.gauge_location || "N.T. Road Xing"}) — DL: <strong>{secondaryRef.danger_level_m} m</strong>
              {secondaryRef.hfl_m && <> · HFL: <strong>{secondaryRef.hfl_m} m</strong></>}
            </div>
          )}
        </div>
      </div>

      {/* Note */}
      {note && <p className="placeholder-note">{note}</p>}
    </div>
  );
}
