import React from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

function formatTick(iso) {
  const d = new Date(iso);
  return `${d.getHours().toString().padStart(2, "0")}:00`;
}

export default function RainfallTrend({ points, lookbackHours }) {
  if (!points || points.length === 0) {
    return <p className="empty-note">Loading rainfall series…</p>;
  }

  const total = points.reduce((sum, p) => sum + p.rainfall_mm, 0);
  const peak = Math.max(...points.map((p) => p.rainfall_mm));

  return (
    <div>
      <div className="trend-stats">
        <div>
          <span className="trend-stat-value">{total.toFixed(1)} mm</span>
          <span className="trend-stat-label">total, last {lookbackHours}h</span>
        </div>
        <div>
          <span className="trend-stat-value">{peak.toFixed(1)} mm</span>
          <span className="trend-stat-label">peak 30-min rate</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={points} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id="rainfallFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3fa9c9" stopOpacity={0.55} />
              <stop offset="100%" stopColor="#3fa9c9" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#2a4536" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatTick}
            stroke="#8ba79a"
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: "#2a4536" }}
            minTickGap={40}
          />
          <YAxis stroke="#8ba79a" fontSize={11} tickLine={false} axisLine={false} width={36} />
          <Tooltip
            contentStyle={{
              background: "#16281f",
              border: "1px solid #2a4536",
              borderRadius: 8,
              fontSize: 12,
              color: "#eaf2ec",
            }}
            labelFormatter={(iso) => new Date(iso).toLocaleString()}
            formatter={(v) => [`${v.toFixed(2)} mm`, "Rainfall"]}
          />
          <Area type="monotone" dataKey="rainfall_mm" stroke="#3fa9c9" strokeWidth={1.5} fill="url(#rainfallFill)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
