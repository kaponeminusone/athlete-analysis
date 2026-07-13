/**
 * Cliente HTTP mínimo hacia el FastAPI (proxy Vite → :8000).
 */

export function apiUrl(path, params = {}) {
  const url = new URL(path, window.location.origin);
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

/** Comprueba si la API responde (status o listado). */
export async function checkApi() {
  try {
    const r = await fetch("/status");
    if (r.ok) return true;
  } catch {
    /* fall through */
  }
  try {
    await fetchJson("/api/videos");
    return true;
  } catch {
    return false;
  }
}

export async function listVideos() {
  const data = await fetchJson("/api/videos");
  return data.videos || [];
}

export async function getProject(videoPath, outputDir) {
  return fetchJson(apiUrl("/api/project", { video_path: videoPath, output_dir: outputDir }));
}

export async function getSections(videoName) {
  return fetchJson(`/api/sections/${encodeURIComponent(videoName)}`);
}

export async function getMetrics(videoName) {
  return fetchJson(`/api/metrics/${encodeURIComponent(videoName)}`);
}

export async function computeMetrics(videoName, athleteId) {
  const qs = athleteId ? `?athlete_id=${encodeURIComponent(athleteId)}` : "";
  return fetchJson(`/api/metrics/${encodeURIComponent(videoName)}/compute${qs}`, {
    method: "POST",
  });
}

export async function scaleMetrics(videoName, hopsCorridorM, athleteId) {
  const body = { hops_corridor_m: Number(hopsCorridorM) };
  if (athleteId) body.athlete_id = athleteId;
  return fetchJson(`/api/metrics/${encodeURIComponent(videoName)}/scale`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function poseOverlayUrl(videoName, phase, outputDir, cacheKey) {
  if (!videoName || !phase) return "";
  return apiUrl(`/api/metrics/${encodeURIComponent(videoName)}/pose-overlay/${encodeURIComponent(phase)}`, {
    output_dir: outputDir,
    v: cacheKey,
  }).toString();
}

/** Frame anotado (default) o crudo vía query `annotated`. */
export function frameUrl(videoName, frameIdx, { annotated = true } = {}) {
  return `/frame/${encodeURIComponent(videoName)}/${frameIdx}?annotated=${annotated ? "true" : "false"}`;
}

export function mediaUrl(absPath) {
  if (!absPath) return "";
  return `/media?path=${encodeURIComponent(String(absPath).replace(/\\/g, "/"))}`;
}

export async function getCalibration(videoName) {
  return fetchJson(`/api/calibration/${encodeURIComponent(videoName)}`);
}

export async function analyzeVideo({ videoPath, stride = 2, startSec = 0, endSec = null }) {
  const body = {
    video_path: videoPath,
    stride: Number(stride) || 1,
    start_sec: Number(startSec) || 0,
  };
  if (endSec != null && endSec !== "") body.end_sec = Number(endSec);
  return fetchJson("/api/analyze", {
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
  return fetchJson("/api/reanalyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Aplica máscaras CNN / perfil de venue (síncrono). */
export async function applyVenueMasks(videoName, payload = {}) {
  return fetchJson(`/api/venue/apply/${encodeURIComponent(videoName)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      merge: true,
      prefer_propagation: false,
      use_masks: true,
      prefer_keyframes: true,
      ...payload,
    }),
  });
}

/** Detecta fases / contactos (síncrono). */
export async function analyzeSections(videoName, usePose = true) {
  const qs = `?use_pose=${usePose ? "true" : "false"}`;
  return fetchJson(`/api/sections/analyze/${encodeURIComponent(videoName)}${qs}`, {
    method: "POST",
  });
}

/**
 * Poll básico de job hasta done/error.
 * @param {string} jobId
 * @param {{ onProgress?: (job: object) => void, intervalMs?: number }} [opts]
 */
export async function pollJob(jobId, { onProgress, intervalMs = 700 } = {}) {
  for (;;) {
    const job = await fetchJson(`/api/jobs/${encodeURIComponent(jobId)}`);
    onProgress?.(job);
    if (job.status === "done") return job;
    if (job.status === "error" || job.status === "failed") {
      throw new Error(job.error || job.message || "El job falló");
    }
    await new Promise((r) => window.setTimeout(r, intervalMs));
  }
}

export async function markPhase(videoName, { frameIdx, phase, poseTag, athleteId }) {
  return fetchJson(`/api/sections/mark/${encodeURIComponent(videoName)}`, {
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
  return fetchJson(`/api/sections/mark/${encodeURIComponent(videoName)}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      from_frame_idx: fromFrameIdx,
      to_frame_idx: toFrameIdx,
    }),
  });
}

/**
 * Corrección manual de un frame (máscara / bbox / click).
 * Mirror del expert UI: POST /correct con `type` (alias de correction_type).
 */
export async function correctFrame({
  videoPath,
  frameIdx,
  type,
  data,
  outputDir,
  propagationRadius = 15,
  sotBackend = "none",
  propagationEndFrame,
}) {
  const body = {
    video_path: videoPath,
    frame_idx: frameIdx,
    type,
    data,
    propagation_radius: propagationRadius,
    sot_backend: sotBackend,
  };
  if (outputDir) body.output_dir = outputDir;
  if (propagationEndFrame != null) body.propagation_end_frame = propagationEndFrame;
  return fetchJson("/correct", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
