import React from "react";

const LABELS = {
  rainfall_intensity: "Rainfall intensity",
  rainfall_accumulation: "Rainfall accumulation",
  antecedent_saturation: "Antecedent rainfall (7d)",
  terrain_susceptibility: "Terrain susceptibility",
  current_water_signal: "Surface water extent (SAR)",
};

export default function ComponentBreakdown({ components }) {
  if (!components) return <p className="empty-note">No breakdown available yet.</p>;

  return (
    <div>
      {Object.entries(components).map(([key, value]) => (
        <div className="component-row" key={key}>
          <span className="component-label">{LABELS[key] || key}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
            />
          </div>
          <span className="component-value">{(value * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}
