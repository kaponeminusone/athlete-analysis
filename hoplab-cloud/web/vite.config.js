import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // En dev local, si se pasa VITE_API_BASE, hacemos proxy a esa URL.
  // En producción (Vercel) no hay proxy: el cliente usa la URL del motor directamente.
  const apiBase = env.VITE_API_BASE || "";

  const proxyTargets = ["/api", "/frame", "/correct", "/mask", "/media", "/status", "/analysis"];
  const proxy = apiBase
    ? Object.fromEntries(proxyTargets.map((p) => [p, { target: apiBase, changeOrigin: true }]))
    : {};

  return {
    plugins: [react(), tailwindcss()],
    server: {
      host: "0.0.0.0",
      port: 5175,
      proxy,
    },
    build: {
      outDir: "dist",
      emptyOutDir: true,
    },
    define: {
      // Exponer la base de la API en tiempo de build (puede quedar vacía)
      __API_BASE__: JSON.stringify(apiBase),
    },
  };
});
