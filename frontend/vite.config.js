import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  build: {
    rolldownOptions: {
      output: {
        // Rolldown requires manualChunks as a function
        manualChunks(id) {
          if (id.includes("node_modules/react-dom") || id.includes("node_modules/react/")) {
            return "vendor-react";
          }
          if (id.includes("node_modules/recharts") || id.includes("node_modules/d3-")) {
            return "vendor-recharts";
          }
          if (id.includes("node_modules/leaflet") || id.includes("node_modules/react-leaflet")) {
            return "vendor-leaflet";
          }
        },
      },
    },
  },
});
