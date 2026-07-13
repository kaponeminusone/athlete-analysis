import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const outputRoot = path.resolve(repoRoot, "output");

// Solo montamos las sesiones reales que usa el prototipo + overlays de pose.
const MEDIA_MOUNTS = {
  "/media-vod2": path.join(outputRoot, "VOD2"),
  "/media-vod9": path.join(outputRoot, "VOD9"),
  "/media-overlays-vod2": path.join(outputRoot, "VOD2", "overlays"),
  "/media-overlays-vod7": path.join(outputRoot, "VOD7", "overlays"),
};

function mediaMountPlugin() {
  function attach(middlewares) {
    for (const [prefix, root] of Object.entries(MEDIA_MOUNTS)) {
      middlewares.use(prefix, (req, res, next) => {
        const rel = decodeURIComponent((req.url || "/").split("?")[0]);
        const file = path.normalize(path.join(root, rel));
        if (!file.startsWith(root) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
          return next();
        }
        const ext = path.extname(file).toLowerCase();
        const types = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp" };
        res.setHeader("Content-Type", types[ext] || "application/octet-stream");
        res.setHeader("Cache-Control", "public, max-age=3600");
        fs.createReadStream(file).pipe(res);
      });
    }
  }

  return {
    name: "hoplab-media-mounts",
    configureServer(server) {
      attach(server.middlewares);
    },
    configurePreviewServer(server) {
      attach(server.middlewares);
    },
  };
}

const API_PROXY = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss(), mediaMountPlugin()],
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
    fs: {
      allow: [repoRoot, __dirname],
    },
    proxy: {
      "/api": API_PROXY,
      "/media": API_PROXY,
      "/correct": API_PROXY,
      "/mask": API_PROXY,
      "/frame": API_PROXY,
      "/analysis": API_PROXY,
      "/status": API_PROXY,
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
    proxy: {
      "/api": API_PROXY,
      "/media": API_PROXY,
      "/correct": API_PROXY,
      "/mask": API_PROXY,
      "/frame": API_PROXY,
      "/analysis": API_PROXY,
      "/status": API_PROXY,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
