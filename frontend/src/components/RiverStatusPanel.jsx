import React from "react";

/**
 * River status panel showing observed level (India-WRIS/CWC),
 * GloFAS modelled discharge, danger-level thresholds, and data provenance.
 *
 * Each value carries its data_category badge:
 *   OBSERVED | MODELLED | FORECAST | STATIC REFERENCE | DEMO / SIMULATED | UNAVAILABLE
 */

const CATEGORY_CLASS = {
  OBSERVED: "src-observed",
  DERIVED: "src-derived",
  MODELLED: "src-modelled",
  FORECAST: "src-forecast",
  "STATIC REFERENCE": "src-static",
  "DEMO / SIMULATED": "src-mock",
  UNAVAILABLE: "src-unavailable",
};

function Badge({ category }) {
  return (
    <span className={`source-badge ${CATEGORY_CLASS[category] || "src-unavailable"}`}>
      {category}
    </span>
  );
}

function Stat({ label, value, unit, category }) {
  return (
    <div className="river-stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {value !== null && value !== undefined ? (
          <>
            {typeof value === "number" ? value.toFixed(2) : value} {unit}
          </>
        ) : (
          <span className="na">Data unavailable</span>
        )}
      </span>
      {category && <Badge category={category} />}
    </div>
  );
}

function RiverGaugeCard({ title, badgeType, data, isPrimary }) {
  if (!data) return null;
  const isLevelAvailable = data.observed_level_m !== null && data.observed_level_m !== undefined;
  const analysis = data.threshold_analysis;
  const dl = data.danger_level_m || (analysis && analysis.danger_level_m);
  const hfl = data.hfl_m || (analysis && analysis.hfl_m);
  const forecasts = data.cwc_forecasts || [];

  return (
    <div className="river-section">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
          {badgeType && (
            <span style={{
              fontSize: "10px",
              padding: "2px 6px",
              borderRadius: "4px",
              fontWeight: 700,
              background: isPrimary ? "rgba(56, 189, 248, 0.2)" : "rgba(168, 85, 247, 0.2)",
              color: isPrimary ? "#38bdf8" : "#c084fc",
              border: isPrimary ? "1px solid rgba(56, 189, 248, 0.4)" : "1px solid rgba(168, 85, 247, 0.4)",
            }}>
              {badgeType}
            </span>
          )}
          {data.station_code && (
            <span className="small muted">
              Station {data.station_code}
            </span>
          )}
        </div>
      </div>

      <p className="small" style={{ margin: "0 0 6px", fontWeight: 600 }}>
        {data.river} — {data.station_name || data.gauge_location}
      </p>

      {/* Manual Fallback Warning Notice */}
      {data.is_fallback_active && (
        <div
          style={{
            margin: "6px 0 10px",
            padding: "6px 10px",
            borderRadius: "6px",
            fontSize: "11.5px",
            background: "rgba(224, 178, 63, 0.15)",
            border: "1px solid rgba(224, 178, 63, 0.4)",
            color: "#fcd34d",
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          <span>ℹ️</span>
          <span>
            <strong>Manual Fallback Active:</strong> Primary telemetry offline/invalid; utilizing verified staff gauge observation.
          </span>
        </div>
      )}

      <Stat
        label="Current level"
        value={data.observed_level_m}
        unit="m"
        category={data.data_category}
      />

      {data.flood_condition && (
        <div className="river-stat">
          <span className="stat-label">Condition</span>
          <span className="stat-value" style={{ textTransform: "capitalize" }}>
            {data.flood_condition}
          </span>
        </div>
      )}

      {isLevelAvailable && data.timestamp && (
        <p className="muted small" style={{ margin: "4px 0" }}>
          Recorded: <strong>{data.timestamp}</strong>
          {data.data_age_minutes !== null && data.data_age_minutes !== undefined && (
            <span> ({data.data_age_minutes} min ago)</span>
          )}
        </p>
      )}

      {/* Danger Level Comparison */}
      {isLevelAvailable && dl && (
        <div
          style={{
            margin: "8px 0",
            padding: "6px 10px",
            borderRadius: "6px",
            fontSize: "12px",
            background: analysis?.is_above_danger
              ? "rgba(239, 68, 68, 0.15)"
              : "rgba(16, 185, 129, 0.1)",
            border: analysis?.is_above_danger
              ? "1px solid rgba(239, 68, 68, 0.4)"
              : "1px solid rgba(16, 185, 129, 0.3)",
            color: analysis?.is_above_danger ? "#f87171" : "#34d399",
          }}
        >
          {analysis?.is_above_danger ? (
            <span>
              ⚠️ <strong>ABOVE DANGER LEVEL</strong> by{" "}
              {Math.abs(analysis.distance_to_danger_m).toFixed(2)} m (DL: {Number(dl).toFixed(2)} m · HFL: {hfl ? Number(hfl).toFixed(2) : "—"} m)
            </span>
          ) : (
            <span>
              ✓ <strong>{analysis ? analysis.distance_to_danger_m.toFixed(2) : (dl - data.observed_level_m).toFixed(2)} m</strong>{" "}
              below Danger Level ({Number(dl).toFixed(2)} m · HFL: {hfl ? Number(hfl).toFixed(2) : "—"} m)
            </span>
          )}
        </div>
      )}

      <p className="muted small" style={{ margin: "2px 0" }}>
        Provenance:{" "}
        <strong>
          {data.provenance || data.source || "India-WRIS / CWC"}
        </strong>
      </p>

      {data.reconciliation_method && (
        <p className="muted small" style={{ margin: "2px 0" }}>
          Reconciliation Strategy: <strong>{data.reconciliation_method}</strong>
        </p>
      )}

      {data.detail && <p className="muted small" style={{ margin: "2px 0" }}>{data.detail}</p>}

      {/* CWC Multi-Day Official Advisory Forecasts */}
      {forecasts.length > 0 && (
        <div style={{ marginTop: "10px", padding: "8px 10px", background: "rgba(0,0,0,0.15)", borderRadius: "6px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
            <span style={{ fontSize: "11px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase" }}>
              CWC AFF River Advisory Forecast
            </span>
            <Badge category="FORECAST" />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            {forecasts.map((fc, idx) => (
              <div key={idx} style={{ display: "flex", justifyContent: "space-between", fontSize: "11.5px" }}>
                <span className="muted">{fc.forecast_date || fc.date}:</span>
                <span style={{ fontWeight: 600 }}>
                  {(fc.forecast_level_m ?? fc.stage_m) != null
                    ? `${(fc.forecast_level_m ?? fc.stage_m).toFixed(2)} m`
                    : "—"}
                  {fc.condition && fc.condition !== "Normal" && (
                    <span style={{ marginLeft: "6px", fontSize: "10px", color: "#f87171" }}>
                      ({fc.condition})
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function RiverStatusPanel({ riverStatus }) {
  if (!riverStatus) return <p className="muted">Loading river status…</p>;

  const {
    primary_river,
    secondary_river,
    observed_river_level,
    threshold_analysis,
    secondary_observed_river_level,
    secondary_threshold_analysis,
    glofas_discharge,
    danger_level_thresholds,
  } = riverStatus;

  const primaryData = primary_river || (observed_river_level ? {
    ...observed_river_level,
    threshold_analysis,
  } : null);

  const secondaryData = secondary_river || (secondary_observed_river_level ? {
    ...secondary_observed_river_level,
    threshold_analysis: secondary_threshold_analysis,
  } : null);

  const todayDischarge =
    glofas_discharge?.daily?.length > 0 ? glofas_discharge.daily[0] : null;

  return (
    <div className="river-status-panel">
      {/* Primary River Gauge (Brahmaputra at Tezpur) */}
      <RiverGaugeCard
        title="Primary River Gauge"
        badgeType="PRIMARY"
        data={primaryData}
        isPrimary={true}
      />

      {/* Secondary River Gauge (Jia Bharali at N.T. Road Xing) */}
      <RiverGaugeCard
        title="Secondary River Gauge"
        badgeType="SECONDARY"
        data={secondaryData}
        isPrimary={false}
      />

      {/* GloFAS modelled discharge (kept strictly separate) */}
      <div className="river-section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3>GloFAS Discharge (Modelled)</h3>
          <Badge category="MODELLED" />
        </div>
        {glofas_discharge?.status === "UNAVAILABLE" ? (
          <>
            <Stat label="Discharge" value={null} unit="" category="UNAVAILABLE" />
            <p className="muted small">{glofas_discharge?.error}</p>
          </>
        ) : (
          <>
            <Stat
              label="Today"
              value={todayDischarge?.discharge_m3s}
              unit="m³/s"
              category="MODELLED"
            />
            <Stat
              label="Daily max"
              value={todayDischarge?.discharge_max_m3s}
              unit="m³/s"
            />
            <Stat
              label="Daily min"
              value={todayDischarge?.discharge_min_m3s}
              unit="m³/s"
            />
            <p className="muted small">
              Source: {glofas_discharge?.source} · {glofas_discharge?.note}
            </p>
          </>
        )}
      </div>

      {/* Danger-level reference thresholds */}
      {danger_level_thresholds?.length > 0 && (
        <div className="river-section">
          <h3>Danger Level Thresholds</h3>
          {danger_level_thresholds.map((t, i) => (
            <div key={i} className="threshold-card">
              <div className="threshold-header">
                <strong>{t.river}</strong> — {t.gauge_location}
                <Badge category="STATIC REFERENCE" />
              </div>
              <div className="threshold-row">
                <span>
                  Danger level: <strong>{t.danger_level_m} m</strong>
                </span>
                <span>
                  HFL: <strong>{t.hfl_m} m</strong> ({t.hfl_date})
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
