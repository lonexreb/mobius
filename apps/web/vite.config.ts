import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API stays on 127.0.0.1:8000; proxying keeps the browser same-origin (REST and SSE).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } } },
  preview: { proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } } },
});
