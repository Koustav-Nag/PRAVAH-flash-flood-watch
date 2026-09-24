import React from "react";

/**
 * Data-source status panel — now using the six-category taxonomy:
 *   OBSERVED | MODELLED | FORECAST | STATIC REFERENCE | DEMO / SIMULATED | UNAVAILABLE
 *
 * Backwards-compatible: if a source still has a legacy `status` field
 * instead of `data_category`, it falls back gracefully.
 */

const CATEGORY_LABEL = {
  OBSERVED: "Observed",
  MODELLED: "Modelled",
  FORECAST: "Forecast",
  "STATIC REFERENCE": "Static Reference",
  "STATIC / DERIVED": "Static / Derived",
  "DEMO / SIMULATED": "Simulated",
  UNAVAILABLE: "Not Integrated",
  // legacy compatibility
  live: "Live",
  mock: "Simulated",
  unavailable: "Not Integrated",
};

const CATEGORY_CLASS = {
  OBSERVED: "src-observed",
  MODELLED: "src-modelled",
  FORECAST: "src-forecast",
  "STATIC REFERENCE": "src-static",
  "STATIC / DERIVED": "src-derived",
  "DEMO / SIMULATED": "src-mock",
  UNAVAILABLE: "src-unavailable",
  // legacy
  live: "src-live",
  mock: "src-mock",
  unavailable: "src-unavailable",
};

export default function DataSourcesPanel({ sources }) {
  if (!sources) return <p className="empty-note">Loading data sources…</p>;

  return (
    <ul className="source-list">
      {sources.map((s) => {
        const cat = s.data_category || s.status || "unavailable";
        return (
          <li key={s.name} className="source-row">
            <div className="source-row-top">
              <span className="source-name">{s.name}</span>
              <span className={`source-badge ${CATEGORY_CLASS[cat] || ""}`}>
                {CATEGORY_LABEL[cat] || cat}
              </span>
            </div>
            <p className="source-detail">{s.detail}</p>
          </li>
        );
      })}
    </ul>
  );
}
