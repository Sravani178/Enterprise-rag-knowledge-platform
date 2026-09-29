import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        changeOrigin: true,
        target: process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
      },
    },
  },
});
