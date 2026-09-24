import React, { useEffect, useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Popup,
  Tooltip,
  Rectangle,
  useMap,
  LayersControl,
  LayerGroup,
  GeoJSON,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import riversData from "../data/rivers_sonitpur.json";

const RISK_COLOR = {
  low: "#4caf7d",
  medium: "#e0b23f",
  high: "#e2823a",
  severe: "#d6432f",
};

// Re-centers the map when origin changes without remounting the map.
function RecenterOnChange({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.setView(center, map.getZoom(), { animate: true });
  }, [center, map]);
  return null;
}

function originIcon(color) {
  return L.divIcon({
    className: "",
    html: `<span style="
      display:block;width:16px;height:16px;border-radius:50%;
      background:${color};border:2px solid rgba(255,255,255,0.85);
      box-shadow:0 0 0 6px ${color}33;
    "></span>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });
}

const shelterIcon = L.divIcon({
  className: "",
  html: `<span style="
    display:block;width:12px;height:12px;border-radius:50%;
    background:#1c3126;border:2px solid #3fa9c9;
  "></span>`,
  iconSize: [12, 12],
  iconAnchor: [6, 6],
});

const safePlaceIcon = L.divIcon({
  className: "",
  html: `<span style="
    display:block;width:16px;height:16px;border-radius:50%;
    background:#3fa9c9;border:2px solid rgba(255,255,255,0.85);
  "></span>`,
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

// Primary Brahmaputra gauge marker (diamond shape, rotated square, blue #0284c7, "P")
const primaryGaugeIcon = L.divIcon({
  className: "primary-gauge-marker",
  html: `<div style="
    width:22px;height:22px;background:#0284c7;border:2px solid #ffffff;
    border-radius:4px;transform:rotate(45deg);
    box-shadow:0 0 0 3px rgba(2,132,199,0.4), 0 2px 6px rgba(0,0,0,0.6);
    display:flex;align-items:center;justify-content:center;cursor:pointer;
  "><span style="transform:rotate(-45deg);font-size:11px;font-weight:800;color:#ffffff;line-height:1;">P</span></div>`,
  iconSize: [22, 22],
  iconAnchor: [11, 11],
  popupAnchor: [0, -12],
});

// Secondary Jia Bharali gauge marker (circle, purple #7c3aed, "S")
const secondaryGaugeIcon = L.divIcon({
  className: "secondary-gauge-marker",
  html: `<div style="
    width:20px;height:20px;background:#7c3aed;border:2px solid #ffffff;
    border-radius:50%;box-shadow:0 0 0 3px rgba(124,58,237,0.4), 0 2px 6px rgba(0,0,0,0.6);
    display:flex;align-items:center;justify-content:center;cursor:pointer;
  "><span style="font-size:10px;font-weight:800;color:#ffffff;line-height:1;">S</span></div>`,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
  popupAnchor: [0, -12],
});

// Exact gauge coordinates configured in app/core/config.py
const GAUGE_CONFIG = {
  primary: {
    river: "Brahmaputra",
    station: "Tezpur",
    role: "Primary",
    lat: 26.61667,
    lon: 92.79731,
    defaultDl: 65.23,
    defaultHfl: 66.59,
  },
  secondary: {
    river: "Jia Bharali",
    station: "N.T. Road Crossing",
    role: "Secondary",
    lat: 26.8105,
    lon: 92.87972,
    defaultDl: 77.00,
    defaultHfl: 78.50,
  },
};

function GaugePopupContent({ config, data, isPrimary }) {
  const isLevelAvailable = data?.observed_level_m !== null && data?.observed_level_m !== undefined;
  const level = data?.observed_level_m;
  const analysis = data?.threshold_analysis;
  const dl = data?.danger_level_m || analysis?.danger_level_m || config.defaultDl;
  const hfl = data?.hfl_m || analysis?.hfl_m || config.defaultHfl;
  const isAboveDanger = analysis?.is_above_danger ?? (level != null && dl != null && level >= dl);
  const condition = data?.flood_condition || (isAboveDanger ? "Above Danger Level" : "Normal");
  const provenance = data?.provenance || data?.source || "CWC Advisory Flood Forecast (AFF) / India-WRIS";
  const dataCategory = data?.data_category || (data ? "OBSERVED" : "STATIC REFERENCE");

  return (
    <div style={{ minWidth: "220px", fontSize: "12px", lineHeight: "1.45" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
        <span
          style={{
            fontSize: "10px",
            fontWeight: 700,
            padding: "2px 6px",
            borderRadius: "4px",
            background: isPrimary ? "rgba(56, 189, 248, 0.2)" : "rgba(168, 85, 247, 0.2)",
            color: isPrimary ? "#38bdf8" : "#c084fc",
            border: isPrimary ? "1px solid rgba(56, 189, 248, 0.4)" : "1px solid rgba(168, 85, 247, 0.4)",
            textTransform: "uppercase",
          }}
        >
          {config.role} Gauge
        </span>
        <span className="source-badge src-observed" style={{ fontSize: "9.5px", padding: "1px 6px" }}>
          {dataCategory}
        </span>
      </div>

      <div style={{ fontWeight: 700, fontSize: "13.5px", color: "var(--text)", marginBottom: "2px" }}>
        {config.river} — {data?.station_name || config.station}
      </div>
      <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>
        Coordinates: {config.lat.toFixed(5)}°N, {config.lon.toFixed(5)}°E
      </div>

      <div style={{ background: "rgba(0,0,0,0.25)", padding: "8px 10px", borderRadius: "6px", marginBottom: "8px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
          <span style={{ color: "var(--text-muted)" }}>Current Level:</span>
          <strong style={{ color: isAboveDanger ? "#f87171" : "#34d399", fontSize: "13px" }}>
            {isLevelAvailable ? `${Number(level).toFixed(2)} m` : "Data unavailable"}
          </strong>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
          <span style={{ color: "var(--text-muted)" }}>Condition:</span>
          <span style={{ fontWeight: 600, color: isAboveDanger ? "#f87171" : "var(--text)", textTransform: "capitalize" }}>
            {condition}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)" }}>
          <span>Danger Level:</span>
          <span>{Number(dl).toFixed(2)} m</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)" }}>
          <span>Highest Flood (HFL):</span>
          <span>{Number(hfl).toFixed(2)} m</span>
        </div>
      </div>

      <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
        Provenance: <strong style={{ color: "var(--text)" }}>{provenance}</strong>
      </div>
    </div>
  );
}

export default function RegionMap({ bounds, origin, safePlace, shelters, riskCategory, riverStatus }) {
  const riskColor = RISK_COLOR[riskCategory] || "#3fa9c9";
  const icon = useMemo(() => originIcon(riskColor), [riskColor]);

  const boundsRect = useMemo(
    () =>
      bounds
        ? [
            [bounds.min_lat, bounds.min_lon],
            [bounds.max_lat, bounds.max_lon],
          ]
        : null,
    [bounds]
  );

  const center = useMemo(
    () =>
      origin
        ? [origin.latitude, origin.longitude]
        : bounds
        ? [(bounds.min_lat + bounds.max_lat) / 2, (bounds.min_lon + bounds.max_lon) / 2]
        : [26.72, 92.9],
    [origin, bounds]
  );

  const primaryData = riverStatus?.primary_river || riverStatus?.observed_river_level;
  const secondaryData = riverStatus?.secondary_river || riverStatus?.secondary_observed_river_level;

  if (!bounds) return <p className="empty-note">Loading region…</p>;

  return (
    <MapContainer center={center} zoom={10} scrollWheelZoom={true} className="map-leaflet" preferCanvas={true}>
      <LayersControl position="topright">
        {/* CARTO Voyager raster tile layer (existing raster PNG tile layer) */}
        <LayersControl.BaseLayer checked name="CARTO Voyager">
          <TileLayer
            url="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_3t7j_1_3c8ea119fd25a73cf1f41832"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            subdomains="abcd"
            maxZoom={19}
          />
        </LayersControl.BaseLayer>

        {/* Verified Rivers Overlay (OpenStreetMap via Overpass API - STATIC REFERENCE) */}
        {riversData && riversData.features && riversData.features.length > 0 && (
          <LayersControl.Overlay checked name="Rivers">
            <LayerGroup>
              <GeoJSON
                key="verified-rivers-layer"
                data={riversData}
                style={(feature) => {
                  const isPrimary = feature.properties?.role === "primary";
                  return {
                    color: isPrimary ? "#0284c7" : "#7c3aed",
                    weight: isPrimary ? 4.5 : 3.5,
                    opacity: 0.85,
                    lineCap: "round",
                    lineJoin: "round",
                  };
                }}
                onEachFeature={(feature, layer) => {
                  const props = feature.properties || {};
                  const isPrimary = props.role === "primary";
                  const name = props.name || "River";
                  const roleText = isPrimary ? "Primary River" : "Secondary River";
                  layer.bindTooltip(
                    `<strong>${name}</strong> (${roleText})<br/><span style="font-size:10px;color:#8ba79a;">${props.data_category} · ${props.source}</span>`,
                    { sticky: true }
                  );
                }}
              />
            </LayerGroup>
          </LayersControl.Overlay>
        )}

        {/* Monitored River Gauges Overlay (CWC Stations) */}
        <LayersControl.Overlay checked name="Gauges">
          <LayerGroup>
            {/* Primary Gauge: Brahmaputra at Tezpur */}
            <Marker
              position={[GAUGE_CONFIG.primary.lat, GAUGE_CONFIG.primary.lon]}
              icon={primaryGaugeIcon}
            >
              <Tooltip direction="top" offset={[0, -10]}>
                <strong>Brahmaputra — Tezpur</strong> (Primary Gauge)
              </Tooltip>
              <Popup>
                <GaugePopupContent
                  config={GAUGE_CONFIG.primary}
                  data={primaryData}
                  isPrimary={true}
                />
              </Popup>
            </Marker>

            {/* Secondary Gauge: Jia Bharali at N.T. Road Crossing */}
            <Marker
              position={[GAUGE_CONFIG.secondary.lat, GAUGE_CONFIG.secondary.lon]}
              icon={secondaryGaugeIcon}
            >
              <Tooltip direction="top" offset={[0, -10]}>
                <strong>Jia Bharali — N.T. Road Xing</strong> (Secondary Gauge)
              </Tooltip>
              <Popup>
                <GaugePopupContent
                  config={GAUGE_CONFIG.secondary}
                  data={secondaryData}
                  isPrimary={false}
                />
              </Popup>
            </Marker>
          </LayerGroup>
        </LayersControl.Overlay>

        {/* Shelters Overlay */}
        {shelters && shelters.length > 0 && (
          <LayersControl.Overlay checked name="Shelters">
            <LayerGroup>
              {shelters.map((s) => (
                <Marker key={s.name} position={[s.latitude, s.longitude]} icon={shelterIcon}>
                  <Tooltip direction="top" offset={[0, -6]}>
                    {s.name}
                    {s.capacity ? ` · capacity ${s.capacity}` : ""}
                  </Tooltip>
                </Marker>
              ))}
            </LayerGroup>
          </LayersControl.Overlay>
        )}

        {/* Region Bounds Overlay */}
        {boundsRect && (
          <LayersControl.Overlay checked name="Region Bounds">
            <Rectangle
              bounds={boundsRect}
              pathOptions={{ color: "#2a4536", weight: 1.5, fillOpacity: 0, dashArray: "4 4" }}
            />
          </LayersControl.Overlay>
        )}
      </LayersControl>

      {/* User Context & Safe Place Path: Always visible on top of overlays */}
      {safePlace && safePlace.latitude != null && (
        <>
          {origin && (
            <Polyline
              positions={[
                [origin.latitude, origin.longitude],
                [safePlace.latitude, safePlace.longitude],
              ]}
              pathOptions={{ color: "#3fa9c9", weight: 2, dashArray: "5 6" }}
            />
          )}
          <Marker position={[safePlace.latitude, safePlace.longitude]} icon={safePlaceIcon}>
            <Popup>
              <strong>{safePlace.name || "Recommended safe place"}</strong>
              <br />
              {safePlace.type === "shelter" ? "DEMO / UNVERIFIED SHELTER" : "Higher ground"}
              {safePlace.distance_km ? ` · ${safePlace.distance_km.toFixed(2)} km away` : ""}
            </Popup>
          </Marker>
        </>
      )}

      {origin && (
        <Marker position={[origin.latitude, origin.longitude]} icon={icon}>
          <Tooltip direction="top" offset={[0, -8]} permanent>
            You are here
          </Tooltip>
        </Marker>
      )}

      <RecenterOnChange center={origin ? [origin.latitude, origin.longitude] : null} />
    </MapContainer>
  );
}
