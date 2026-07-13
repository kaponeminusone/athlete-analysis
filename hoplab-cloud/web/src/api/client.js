/**
 * Cliente HTTP — HopLab Cloud.
 * La base URL del motor (tunnel Colab) viene de:
 *   1. localStorage "hoplab_motor_url"  (persistida por el usuario en la UI)
 *   2. VITE_API_BASE (variable de entorno en Vercel, opcional)
 *   3. "" → URLs relativas (solo útil en dev con proxy)
 */

export function getApiBase() {
  try {
    const saved = localStorage.getItem("hoplab_motor_url");
    if (saved && saved.startsWith("http")) return saved.replace(/\/$/, "");
  } catch {
    /* localStorage bloqueado */
  }
  const buildTime = (typeof __API_BASE__ !== "undefined" ? __API_BASE__ : "") || "";
  return buildTime.replace(/\/$/, "");
}

function absUrl(path) {
  const base = getApiBase();
  return base ? `${base}${path}` : path;
}

export function apiUrl(path, params = {}) {
  const url = new URL(absUrl(path), window.location.href);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  return url;
}

export async function fetchJson(url, options) {
  const response = await fetch(typeof url === "string" ? url : url.toString(), options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail ?? data.error;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : detail
            ? JSON.stringify(detail)
            : response.statusText;
    throw new Error(message || `HTTP ${response.status}`);
  }
  return data;
}

/** Guarda la URL del motor en localStorage. */
export function saveMotorUrl(url) {
  try {
    const clean = (url || "").trim().replace(/\/$/, "");
    if (clean) {
      localStorage.setItem("hoplab_motor_url", clean);
    } else {
      localStorage.removeItem("hoplab_motor_url");
    }
  } catch {
    /* ignorar */
  }
}

/** Comprueba si la API responde. Devuelve true/false. */
export async function checkApi() {
  try {
    const r = await fetch(absUrl("/status"), { signal: AbortSignal.timeout(5000) });
    if (r.ok) return true;
  } catch {
    /* fall through */
  }
  try {
    await fetchJson(absUrl("/api/videos"));
    return true;
  } catch {
    return false;
  }
}

export async function listVideos() {
  const data = await fetchJson(absUrl("/api/videos"));
  return data.videos || [];
}

export async function getProject(videoPath, outputDir) {
  return fetchJson(apiUrl("/api/project", { video_path: videoPath, output_dir: outputDir }));
}

export async function getSections(videoName) {
  return fetchJson(absUrl(`/api/sections/${encodeURIComponent(videoName)}`));
}

export async function getMetrics(videoName) {
  return fetchJson(absUrl(`/api/metrics/${encodeURIComponent(videoName)}`));
}

export async function computeMetrics(videoName, athleteId) {
  const qs = athleteId ? `?athlete_id=${encodeURIComponent(athleteId)}` : "";
  return fetchJson(absUrl(`/api/metrics/${encodeURIComponent(videoName)}/compute${qs}`), {
    method: "POST",
  });
}

export async function scaleMetrics(videoName, hopsCorridorM, athleteId) {
  const body = { hops_corridor_m: Number(hopsCorridorM) };
  if (athleteId) body.athlete_id = athleteId;
  return fetchJson(absUrl(`/api/metrics/${encodeURIComponent(videoName)}/scale`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function poseOverlayUrl(videoName, phase, outputDir, cacheKey) {
  if (!videoName || !phase) return "";
  return apiUrl(
    `/api/metrics/${encodeURIComponent(videoName)}/pose-overlay/${encodeURIComponent(phase)}`,
    { output_dir: outputDir, v: cacheKey },
  ).toString();
}

export function frameUrl(videoName, frameIdx, { annotated = true } = {}) {
  return absUrl(
    `/frame/${encodeURIComponent(videoName)}/${frameIdx}?annotated=${annotated ? "true" : "false"}`,
  );
}

export function mediaUrl(absPath) {
  if (!absPath) return "";
  return absUrl(`/media?path=${encodeURIComponent(String(absPath).replace(/\\/g, "/"))}`);
}

export async function getCalibration(videoName) {
  return fetchJson(absUrl(`/api/calibration/${encodeURIComponent(videoName)}`));
}

export async function analyzeVideo({ videoPath, stride = 2, startSec = 0, endSec = null }) {
  const body = {
    video_path: videoPath,
    stride: Number(stride) || 1,
    start_sec: Number(startSec) || 0,
  };
  if (endSec != null && endSec !== "") body.end_sec = Number(endSec);
  return fetchJson(absUrl("/api/analyze"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function reanalyzeVideo({
  videoPath,
  stride = 1,
  startSec = 0,
  endSec = null,
  useCnnMasks = false,
  refineV2 = false,
}) {
  const body = {
    video_path: videoPath,
    stride: Number(stride) || 1,
    start_sec: Number(startSec) || 0,
    use_cnn_masks: Boolean(useCnnMasks),
    refine_v2: Boolean(refineV2),
  };
  if (endSec != null && endSec !== "") body.end_sec = Number(endSec);
  return fetchJson(absUrl("/api/reanalyze"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function applyVenueMasks(videoName, payload = {}) {
  return fetchJson(absUrl(`/api/venue/apply/${encodeURIComponent(videoName)}`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ merge: true, prefer_propagation: false, use_masks: true, prefer_keyframes: true, ...payload }),
  });
}

export async function analyzeSections(videoName, usePose = true) {
  return fetchJson(absUrl(`/api/sections/analyze/${encodeURIComponent(videoName)}?use_pose=${usePose ? "true" : "false"}`), {
    method: "POST",
  });
}

export async function pollJob(jobId, { onProgress, intervalMs = 700 } = {}) {
  for (;;) {
    const job = await fetchJson(absUrl(`/api/jobs/${encodeURIComponent(jobId)}`));
    onProgress?.(job);
    if (job.status === "done") return job;
    if (job.status === "error" || job.status === "failed") {
      throw new Error(job.error || job.message || "El job falló");
    }
    await new Promise((r) => window.setTimeout(r, intervalMs));
  }
}

export async function markPhase(videoName, { frameIdx, phase, poseTag, athleteId }) {
  return fetchJson(absUrl(`/api/sections/mark/${encodeURIComponent(videoName)}`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frame_idx: frameIdx,
      phase,
      pose_tag: poseTag || undefined,
      athlete_id: athleteId || undefined,
    }),
  });
}

export async function movePhaseMarker(videoName, fromFrameIdx, toFrameIdx) {
  return fetchJson(absUrl(`/api/sections/mark/${encodeURIComponent(videoName)}/move`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ from_frame_idx: fromFrameIdx, to_frame_idx: toFrameIdx }),
  });
}

export async function correctFrame({
  videoPath, frameIdx, type, data, outputDir,
  propagationRadius = 15, sotBackend = "none", propagationEndFrame,
}) {
  const body = {
    video_path: videoPath, frame_idx: frameIdx, type, data,
    propagation_radius: propagationRadius, sot_backend: sotBackend,
  };
  if (outputDir) body.output_dir = outputDir;
  if (propagationEndFrame != null) body.propagation_end_frame = propagationEndFrame;
  return fetchJson(absUrl("/correct"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
