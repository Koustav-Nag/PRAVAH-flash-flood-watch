import React from "react";

const CATEGORY_COLOR = {
  low: "var(--risk-low)",
  medium: "var(--risk-medium)",
  high: "var(--risk-high)",
  severe: "var(--risk-severe)",
};

/**
 * Semi-circular arc gauge, 0-1 score. Deliberately not a generic
 * donut/speedometer clone — a single sweeping arc that reads at a
 * glance, with the score placed inside where the eye lands first.
 */
export default function RiskGauge({ score, category }) {
  const radius = 80;
  const cx = 100;
  const cy = 100;
  const startAngle = 200;
  const endAngle = -20;
  const clamped = Math.max(0, Math.min(1, score ?? 0));
  const angle = startAngle + (endAngle - startAngle) * clamped;

  const toXY = (deg) => {
    const rad = (deg * Math.PI) / 180;
    return [cx + radius * Math.cos(rad), cy - radius * Math.sin(rad)];
  };

  const [sx, sy] = toXY(startAngle);
  const [ex, ey] = toXY(endAngle);
  const [px, py] = toXY(angle);

  const color = CATEGORY_COLOR[category] || "var(--water)";

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 200 120" width="220" height="132">
        <path
          d={`M ${sx} ${sy} A ${radius} ${radius} 0 0 1 ${ex} ${ey}`}
          fill="none"
          stroke="var(--border)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        <path
          d={`M ${sx} ${sy} A ${radius} ${radius} 0 0 1 ${px} ${py}`}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
        />
        <circle cx={px} cy={py} r="6" fill={color} />
      </svg>
      <div className="gauge-score" style={{ color }}>
        {(clamped * 100).toFixed(0)}
      </div>
      <div className="gauge-category">{category || "unknown"} risk</div>
    </div>
  );
}
