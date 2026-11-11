import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server only (the skeleton has no production build target yet).
// /api is proxied to the FastAPI container so the browser stays same-origin. In compose the
// target is the container (http://api:8000); the fallback is the host port, for a local vite.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    // Bind mounts from macOS do not deliver inotify events reliably: poll instead.
    watch: { usePolling: process.env.VITE_POLL === "1", interval: 300 },
    proxy: {
      "/api": {
        target: process.env.API_PROXY_TARGET ?? "http://localhost:8010",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
