import React from "react";

export default function SafePlaceCard({ safePlace }) {
  if (!safePlace || safePlace.type === "none_found") {
    return <p className="empty-note">No safe place could be determined for this location.</p>;
  }

  const isShelter = safePlace.type === "shelter";

  return (
    <div className="safe-place-card">
      <span className="safe-place-type">{isShelter ? "DEMO / UNVERIFIED SHELTER" : "Higher-ground point"}</span>
      <h3 className="safe-place-name">{safePlace.name}</h3>

      <div className="safe-place-meta">
        <div>
          <strong>{safePlace.distance_km} km</strong>
          distance
        </div>
        {isShelter ? (
          <div>
            <strong>{safePlace.details?.capacity ?? "—"}</strong>
            capacity
          </div>
        ) : (
          <div>
            <strong>+{safePlace.details?.elevation_gain_m ?? "—"} m</strong>
            elevation gain
          </div>
        )}
      </div>

      {isShelter && safePlace.details?.source?.startsWith("PLACEHOLDER") && (
        <p className="placeholder-note">
          Shelter location is placeholder data for this prototype — replace with a verified
          list from ASDMA or the district administration before real use.
        </p>
      )}
      {!isShelter && (
        <p className="placeholder-note">
          No shelter was within range, so the system suggests the nearest point with a
          meaningful elevation gain instead.
        </p>
      )}
    </div>
  );
}
