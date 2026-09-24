import React from "react";

function Stat({ label, value, unit }) {
  return (
    <div className="terrain-stat">
      <span className="terrain-stat-value">
        {value != null ? value.toFixed(1) : "—"}
        {unit && <span className="terrain-stat-unit">{unit}</span>}
      </span>
      <span className="terrain-stat-label">{label}</span>
    </div>
  );
}

export default function TerrainPanel({ features }) {
  if (!features) return <p className="empty-note">Loading terrain / catchment data…</p>;

  return (
    <div className="terrain-grid">
      <Stat label="Elevation" value={features.elevation} unit=" m" />
      <Stat label="Slope" value={features.slope} unit="°" />
      <Stat label="Flow accumulation" value={features.flow_accumulation} unit="" />
      <Stat label="Distance to river" value={features.distance_to_river} unit=" km" />
      <Stat label="Drainage density" value={features.drainage_density} unit="" />
      <Stat
        label="Surface water fraction"
        value={features.surface_water_fraction != null ? features.surface_water_fraction * 100 : null}
        unit="%"
      />
      <p className="placeholder-note">
        Static terrain parameters derived from SRTM DEM &amp; HydroSHEDS (topographic slope, elevation, drainage density, and flow accumulation).
      </p>
    </div>
  );
}
