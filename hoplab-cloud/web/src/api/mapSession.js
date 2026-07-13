/**
 * Mapea respuestas de la API al shape que espera WatchPage / SidePanel / Library.
 */
import { poseLabelFromScore, successPct as mockSuccessPct } from "../mock/data";
import {
  computeMetrics,
  frameUrl,
  getCalibration,
  getMetrics,
  getProject,
  getSections,
  listVideos,
  mediaUrl,
  poseOverlayUrl,
} from "./client";

const ACCENTS = ["#ff0033", "#38bdf8", "#3dd68c", "#f5a524", "#a78bfa", "#22d3ee"];

const HOP_LABELS = {
  hop_1: "H1",
  hop_2: "H2",
  hop_3: "H3",
  hop_4: "H4",
  landing: "LA",
  final_jump: "SF",
  approach: "CAR",
};

function initialsFromName(name) {
  const parts = String(name || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (!parts.length) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function shortName(name) {
  const parts = String(name || "").trim().split(/\s+/);
  return parts[0] || name || "Atleta";
}

function formatDurationLabel(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "—";
  const total = Math.max(0, Math.round(Number(seconds)));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function indexOfFrameId(frames, frameId) {
  if (!frames?.length || frameId == null) return 0;
  let best = 0;
  let bestDist = Infinity;
  frames.forEach((f, i) => {
    const d = Math.abs(f.frameId - frameId);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  });
  return best;
}

function deltaFromVsRow(row) {
  if (!row) return 0;
  if (row.speed_delta_ms != null && Number.isFinite(Number(row.speed_delta_ms))) {
    return Number(row.speed_delta_ms);
  }
  if (row.indicator === "+") return 0.3;
  if (row.indicator === "-" || row.indicator === "−") return -0.3;
  return 0;
}

/** % éxito desde metrics (consistency → pose_quality → promedio hops). */
export function successPctFromMetrics(metrics) {
  if (!metrics) return null;
  const c = metrics.consistency?.overall;
  if (c != null && Number.isFinite(Number(c))) return Math.round(Number(c) * 100);
  const p = metrics.comparison?.pose_quality?.overall;
  if (p != null && Number.isFinite(Number(p))) return Math.round(Number(p) * 100);
  const hops = metrics.comparison?.pose_quality?.hops || [];
  if (!hops.length) return null;
  const avg = hops.reduce((a, h) => a + (Number(h.score) || 0), 0) / hops.length;
  return Math.round(avg * 100);
}

export function analysisStateFromProject(project, metrics, sections) {
  const hasAnalysis = Boolean(project?.analysis?.exists);
  if (!hasAnalysis) return "none";
  const contacts = sections?.contacts?.length || 0;
  const segs = metrics?.segments?.length || 0;
  if (contacts >= 4 && segs > 0) return "full";
  return "partial";
}

function joinOutputPath(outputPath, rel) {
  if (!outputPath || !rel) return "";
  const base = String(outputPath).replace(/\\/g, "/").replace(/\/$/, "");
  const r = String(rel).replace(/\\/g, "/").replace(/^\//, "");
  return `${base}/${r}`;
}

/**
 * Frames listos para el player (URLs /frame + máscaras vía /media).
 */
export function mapFrames(project, videoName, calibration) {
  const rawFrames = project?.analysis?.frames || [];
  const assets = project?.assets || {};
  const outPath = project?.output?.path || "";
  const maskFrames = calibration?.mask_frames || {};
  const hasMasks = Object.keys(maskFrames).length > 0;
  const fps = project?.analysis?.data?.video_info?.fps || 30;
  const n = Math.max(1, rawFrames.length);

  const frames = rawFrames.map((f, index) => {
    const frameId = f.frame_idx ?? index;
    const key = String(frameId);
    const entry = maskFrames[key] || maskFrames[String(Number(key))];
    const q = f.quality_score;
    const quality =
      q != null && Number.isFinite(Number(q))
        ? Math.round(Math.min(1, Math.max(0, Number(q))) * 100)
        : f.usable_for_analysis === false
          ? 55
          : 85;

    const rawAsset = assets.frames?.[key] || assets.frames?.[String(Number(key))];
    const annAsset = assets.annotated?.[key] || assets.annotated?.[String(Number(key))];

    return {
      index,
      frameId,
      raw: frameUrl(videoName, frameId, { annotated: false }) || rawAsset || "",
      annotated: frameUrl(videoName, frameId, { annotated: true }) || annAsset || "",
      trackMask: entry?.track ? mediaUrl(joinOutputPath(outPath, entry.track)) : null,
      sandMask: entry?.sand ? mediaUrl(joinOutputPath(outPath, entry.sand)) : null,
      quality,
      skip: f.usable_for_analysis === false,
      timeMs: Math.round((f.timestamp_s != null ? f.timestamp_s : index / fps) * 1000),
    };
  });

  const durationS =
    frames.length > 1
      ? (frames[frames.length - 1].timeMs - frames[0].timeMs) / 1000
      : n / Math.max(1, fps);
  const fpsPlayback = Math.max(4, Math.min(14, Math.round((n - 1) / Math.max(0.5, durationS))));

  return { frames, hasMasks, fpsPlayback };
}

export function mapContacts(sections, frames) {
  const contacts = sections?.contacts || [];
  if (!contacts.length) {
    // Fallback: phase_markers de hop/landing
    const markers = (sections?.phase_markers || []).filter((m) =>
      ["hop_1", "hop_2", "hop_3", "hop_4", "landing", "final_jump"].includes(m.phase),
    );
    return markers.map((m, i) => {
      const type = m.phase === "landing" || m.phase === "final_jump" ? "landing" : "hop";
      const label = HOP_LABELS[m.phase] || `H${i + 1}`;
      return {
        id: `m-${m.phase}-${m.frame_idx}`,
        label,
        type,
        frameId: m.frame_idx,
        index: indexOfFrameId(frames, m.frame_idx),
        phase: m.phase,
      };
    });
  }

  return contacts.map((c) => {
    const type = c.type === "landing" ? "landing" : c.type === "approach" ? "approach" : "hop";
    const label =
      HOP_LABELS[c.phase] ||
      (type === "landing" ? "LA" : type === "approach" ? "CAR" : `H${c.index || "?"}`);
    return {
      id: `c-${c.index ?? c.frame_idx}`,
      label,
      type,
      frameId: c.frame_idx,
      index: indexOfFrameId(frames, c.frame_idx),
      phase: c.phase,
    };
  });
}

/**
 * Metrics API → { general, hops[], finalFlightNote } para SidePanel.
 */
export function mapMetricsToPanel(metrics, videoName, outputDir, frames) {
  if (!metrics) {
    return { general: { totalTimeMs: 0, avgSpeed: 0 }, hops: [], finalFlightNote: null };
  }

  const segments = metrics.segments || [];
  const vsSegs = metrics.comparison?.vs_general?.segments || [];
  const pq = metrics.comparison?.pose_quality || {};
  const cacheKey = metrics.derived_version ?? "";

  const hopSegs = segments.filter((s) => String(s.id || "").startsWith("hop_"));
  const totalTimeMs = Math.round(hopSegs.reduce((a, s) => a + (Number(s.dt_s) || 0), 0) * 1000);
  const speeds = hopSegs.map((s) => Number(s.speed_m_s)).filter((n) => Number.isFinite(n));
  const avgSpeed = speeds.length
    ? Number((speeds.reduce((a, b) => a + b, 0) / speeds.length).toFixed(1))
    : 0;

  const hops = [];

  for (const hop of pq.hops || []) {
    const phase = hop.phase;
    const seg = segments.find((s) => s.id === phase);
    const vsRow = vsSegs.find((s) => s.id === phase);
    const frameId = hop.frame_idx ?? seg?.from_frame;
    const score = Number(hop.score);
    hops.push({
      id: phase,
      label: HOP_LABELS[phase] || phase,
      frameId,
      index: indexOfFrameId(frames, frameId),
      timeMs: Math.round((Number(seg?.dt_s) || 0) * 1000),
      speed: Number((Number(seg?.speed_m_s) || 0).toFixed(1)),
      deltaVsGeneral: deltaFromVsRow(vsRow),
      poseQuality: Number.isFinite(score) ? score : 0,
      poseLabel: hop.label || poseLabelFromScore(score),
      poseOverlayUrl: poseOverlayUrl(videoName, phase, outputDir, cacheKey),
      isFinalFlight: false,
    });
  }

  const ff = pq.final_flight;
  if (ff) {
    const hop4 = segments.find((s) => s.id === "hop_4");
    const vsRow = vsSegs.find((s) => s.id === "hop_4");
    const frameId = ff.to_frame ?? hop4?.to_frame ?? ff.from_frame;
    const score = Number(ff.score);
    hops.push({
      id: "la",
      label: "LA",
      frameId,
      index: indexOfFrameId(frames, frameId),
      timeMs: Math.round((Number(hop4?.dt_s) || 0) * 1000),
      speed: Number((Number(hop4?.speed_m_s) || 0).toFixed(1)),
      deltaVsGeneral: deltaFromVsRow(vsRow),
      poseQuality: Number.isFinite(score) ? score : 0,
      poseLabel: ff.label || poseLabelFromScore(score),
      poseOverlayUrl: poseOverlayUrl(videoName, "final_flight", outputDir, cacheKey),
      isFinalFlight: true,
    });
  }

  // Si no hay pose_quality, armar hops desde segments de contacto
  if (!hops.length && hopSegs.length) {
    hopSegs.forEach((seg, i) => {
      const vsRow = vsSegs.find((s) => s.id === seg.id);
      const frameId = seg.from_frame;
      hops.push({
        id: seg.id,
        label: HOP_LABELS[seg.id] || `H${i + 1}`,
        frameId,
        index: indexOfFrameId(frames, frameId),
        timeMs: Math.round((Number(seg.dt_s) || 0) * 1000),
        speed: Number((Number(seg.speed_m_s) || 0).toFixed(1)),
        deltaVsGeneral: deltaFromVsRow(vsRow),
        poseQuality: 0.7,
        poseLabel: "regular",
        poseOverlayUrl: poseOverlayUrl(videoName, seg.id, outputDir, cacheKey),
        isFinalFlight: seg.id === "hop_4",
      });
    });
  }

  return {
    general: { totalTimeMs, avgSpeed },
    hops,
    finalFlightNote: ff
      ? "Vuelo final H4→aterrizaje · overlay general (ámbar) vs esta toma (cian)"
      : null,
  };
}

/** Nombre de carpeta en output/ (p. ej. VOD2_refined) a partir del path. */
function videoNameFromOutputDir(outputDir) {
  if (!outputDir) return null;
  const base = String(outputDir).replace(/\\/g, "/").replace(/\/$/, "").split("/").pop();
  return base || null;
}

/**
 * Carga proyecto + sections + metrics + calibration y arma el paquete WatchPage.
 */
export async function loadWatchSession(session) {
  const videoPath = session.videoPath;
  const videoName = session.videoName || session.id;
  const outputDir = session.outputDir || undefined;

  const project = await getProject(videoPath, outputDir);
  // Preferir el basename de output_dir para *_refined (frames/sections/metrics).
  const resolvedName =
    videoNameFromOutputDir(outputDir || project?.output?.path) ||
    project.video?.video_name ||
    videoName;

  let sections = project.sections?.data || null;
  if (!sections && project.sections?.exists) {
    try {
      sections = await getSections(resolvedName);
    } catch {
      sections = null;
    }
  }

  let metrics = project.metrics?.data || null;
  const needsCompute =
    project.analysis?.exists &&
    (sections?.contacts?.length || 0) > 0 &&
    !(metrics?.segments?.length > 0);

  if (needsCompute) {
    try {
      await computeMetrics(resolvedName, metrics?.athlete_id || session.athleteId);
      metrics = await getMetrics(resolvedName);
    } catch {
      /* keep empty metrics */
    }
  } else if (project.analysis?.exists && !metrics) {
    try {
      metrics = await getMetrics(resolvedName);
      if (!(metrics?.segments?.length > 0) && (sections?.contacts?.length || 0) > 0) {
        await computeMetrics(resolvedName, metrics?.athlete_id);
        metrics = await getMetrics(resolvedName);
      }
    } catch {
      metrics = null;
    }
  }

  let calibration = null;
  try {
    calibration = await getCalibration(resolvedName);
  } catch {
    calibration = null;
  }

  const { frames, hasMasks, fpsPlayback } = mapFrames(project, resolvedName, calibration);
  const contacts = mapContacts(sections, frames);
  const panelMetrics = mapMetricsToPanel(metrics, resolvedName, outputDir || project.output?.path, frames);
  const pct = successPctFromMetrics(metrics);
  const analysis = analysisStateFromProject(project, metrics, sections);

  return {
    ...session,
    videoName: resolvedName,
    videoPath: project.video?.path || videoPath,
    outputDir: outputDir || project.output?.path,
    analysis,
    hasAnalysis: analysis !== "none",
    hasMasks,
    fpsMock: fpsPlayback,
    frames,
    frameCount: frames.length,
    firstFrameId: frames[0]?.frameId ?? 0,
    lastFrameId: frames[frames.length - 1]?.frameId ?? 0,
    contacts,
    metrics: panelMetrics,
    successPct: pct,
    overlayNote: null,
    rawMetrics: metrics,
    sections,
    project,
    athleteId: metrics?.athlete_id || session.athleteId || null,
    hopsCorridorM: metrics?.scale?.hops_corridor_m ?? metrics?.overrides?.hops_corridor_m ?? 10,
    source: "api",
  };
}

/**
 * Lista videos de la API y agrupa por atleta (metrics.athlete_id) o "Sin asignar".
 */
export async function loadLibraryFromApi() {
  const videos = await listVideos();
  if (!videos.length) {
    return { athletes: [], fromApi: true, empty: true };
  }

  const enriched = await Promise.all(
    videos.map(async (v) => {
      const apiName = v.has_refined ? `${v.video_name}_refined` : v.video_name;
      const base = {
        id: v.video_name,
        videoName: apiName,
        videoPath: v.path,
        outputDir: v.has_refined ? v.refined_output_dir : undefined,
        title: (v.name || v.video_name || "").replace(/\.[^.]+$/, ""),
        date: "",
        note: v.has_refined ? "Versión refinada disponible" : v.has_analysis ? "Con análisis" : "Pendiente de procesar",
        durationLabel: formatDurationLabel(v.duration_s),
        thumb: v.has_analysis ? frameUrl(apiName, 0, { annotated: true }) : "",
        analysis: v.has_analysis ? "partial" : "none",
        poseQualities: [],
        successPct: null,
        athleteId: null,
        athleteName: null,
        source: "api",
      };

      if (!v.has_analysis) return base;

      try {
        const project = await getProject(v.path, base.outputDir);
        const frames = project.analysis?.frames || [];
        const firstIdx = frames[0]?.frame_idx ?? 0;
        base.thumb = frameUrl(apiName, firstIdx, { annotated: true });

        let metrics = project.metrics?.data;
        const sections = project.sections?.data;
        if (!metrics && project.metrics?.exists === false && project.analysis?.exists) {
          try {
            metrics = await getMetrics(apiName);
          } catch {
            metrics = null;
          }
        }
        if (metrics && !(metrics.segments?.length > 0) && (sections?.contacts?.length || 0) > 0) {
          try {
            await computeMetrics(apiName, metrics.athlete_id);
            metrics = await getMetrics(apiName);
          } catch {
            /* ignore */
          }
        }

        base.athleteId = metrics?.athlete_id || sections?.athlete_id || null;
        base.athleteName = base.athleteId || null;
        base.successPct = successPctFromMetrics(metrics);
        const pq = metrics?.comparison?.pose_quality?.hops || [];
        base.poseQualities = pq.map((h) => h.score).filter((n) => n != null);
        base.analysis = analysisStateFromProject(project, metrics, sections);
        if (base.successPct != null) {
          base.note = base.analysis === "full" ? "Análisis completo" : base.note;
        }
        if (frames.length) {
          const last = frames[frames.length - 1];
          const first = frames[0];
          if (first?.timestamp_s != null && last?.timestamp_s != null) {
            base.durationLabel = formatDurationLabel(last.timestamp_s - first.timestamp_s);
          }
        }
      } catch {
        /* keep partial defaults */
      }

      return base;
    }),
  );

  const groups = new Map();
  for (const sess of enriched) {
    const name = sess.athleteName?.trim() || "Sin asignar";
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(sess);
  }

  const athletes = [...groups.entries()].map(([name, sessions], i) => {
    const id =
      name === "Sin asignar"
        ? "sin-asignar"
        : name
            .toLowerCase()
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "")
            .replace(/\s+/g, "-");
    return {
      id,
      name,
      short: name === "Sin asignar" ? "Sin asignar" : shortName(name),
      initials: name === "Sin asignar" ? "—" : initialsFromName(name),
      accent: ACCENTS[i % ACCENTS.length],
      note: name === "Sin asignar" ? "Videos sin atleta en metadata" : "Sesiones desde API",
      sessions: sessions.sort((a, b) => String(b.title).localeCompare(String(a.title))),
      source: "api",
    };
  });

  athletes.sort((a, b) => {
    if (a.id === "sin-asignar") return 1;
    if (b.id === "sin-asignar") return -1;
    return a.name.localeCompare(b.name, "es");
  });

  return { athletes, fromApi: true, empty: athletes.length === 0 };
}

/** successPct unificado: API session.successPct o mock poseQualities. */
export function sessionSuccessPct(session) {
  if (session?.successPct != null) return session.successPct;
  return mockSuccessPct(session);
}
