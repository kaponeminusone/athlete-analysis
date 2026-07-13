/**
 * Fixtures estáticas + rutas a frames REALES vía mounts de Vite. Sin API.
 *
 * Dos fuentes reales de frames:
 *   - VOD9  → frames + annotated + venue_masks (track_/sand_). Demuestra los DOS ojos
 *             (Seguimiento = raw↔annotated, Pista = máscaras track/sand reales).
 *   - VOD2  → frames de alta resolución + annotated (sin mapa de pista; ojo Pista off).
 *
 * Overlays de pose (general ámbar vs esta toma cian):
 *   - VOD2 → /media-overlays-vod2 (output/VOD2/overlays)
 *   - VOD9 → /media-overlays-vod7 (output/VOD7/overlays) como mock visual, etiquetado.
 */

function pad(n) {
  return String(n).padStart(6, "0");
}

function range(start, step, count) {
  return Array.from({ length: count }, (_, i) => start + i * step);
}

/** Construye la lista de frames indexada para una fuente. */
function buildFrames({ media, frameIds, hasMasks, durationMs, quality, skip }) {
  const n = frameIds.length;
  return frameIds.map((frameId, index) => ({
    index,
    frameId,
    raw: `${media}/frames/frame_${pad(frameId)}.jpg`,
    annotated: `${media}/annotated/annotated_${pad(frameId)}.jpg`,
    trackMask: hasMasks ? `${media}/venue_masks/track_${pad(frameId)}.png` : null,
    sandMask: hasMasks ? `${media}/venue_masks/sand_${pad(frameId)}.png` : null,
    quality: quality(index, n),
    skip: skip(index),
    timeMs: Math.round((index / (n - 1)) * durationMs),
  }));
}

function baseQuality(low = [], mid = []) {
  return (i) => {
    if (low.includes(i)) return 58 + ((i * 5) % 8);
    if (mid.includes(i)) return 70 + ((i * 3) % 6);
    return 82 + ((i * 7) % 15);
  };
}

/** Etiqueta Buena / Regular / Débil según score 0–1 (mismo umbral que UI experta). */
export function poseLabelFromScore(score) {
  if (score == null) return null;
  if (score >= 0.75) return "buena";
  if (score >= 0.55) return "regular";
  return "débil";
}

export const POSE_LABEL_TEXT = {
  buena: "Buena",
  regular: "Regular",
  débil: "Débil",
  debil: "Débil",
};

/** Base de overlays de pose por fuente de video. */
function overlayBase(vod) {
  if (vod === "vod2") return "/media-overlays-vod2";
  // VOD9 reutiliza overlays VOD7 como mock visual (etiquetado en UI).
  return "/media-overlays-vod7";
}

function hopOverlayUrl(vod, hopNum) {
  return `${overlayBase(vod)}/hop_${hopNum}.png`;
}

function finalFlightOverlayUrl(vod) {
  // final_flight solo existe en VOD7; VOD9 lo usa como mock.
  if (vod === "vod2") return null;
  return `${overlayBase(vod)}/final_flight.png`;
}

/* ------------------------------------------------------------------ VOD9 */
// 60 frames: 268,270,…,386 (stride 2). track_/sand_ PNG por frame.
const VOD9_IDS = range(268, 2, 60);

const VOD9 = {
  vod: "vod9",
  media: "/media-vod9",
  overlayNote: "Overlays de referencia VOD7 (mock visual)",
  hasMasks: true,
  fpsMock: 8,
  frames: buildFrames({
    media: "/media-vod9",
    frameIds: VOD9_IDS,
    hasMasks: true,
    durationMs: 5900,
    quality: baseQuality([12, 13, 14], [26, 27, 44]),
    skip: (i) => i === 12 || i === 13 || i === 33,
  }),
  // Contactos: hops 320,332,348,358 · aterrizaje ≈ final (384)
  contacts: [
    { id: "h1", label: "H1", type: "hop", frameId: 320 },
    { id: "h2", label: "H2", type: "hop", frameId: 332 },
    { id: "h3", label: "H3", type: "hop", frameId: 348 },
    { id: "h4", label: "H4", type: "hop", frameId: 358 },
    { id: "la", label: "LA", type: "landing", frameId: 384 },
  ],
  metrics: {
    general: { totalTimeMs: 4100, avgSpeed: 8.7 },
    finalFlightNote: "Vuelo final H4→aterrizaje · overlay de referencia VOD7",
    hops: [
      { id: "h1", label: "H1", frameId: 320, timeMs: 640, speed: 8.9, deltaVsGeneral: 0.2, poseQuality: 0.88, poseOverlayUrl: hopOverlayUrl("vod9", 1) },
      { id: "h2", label: "H2", frameId: 332, timeMs: 700, speed: 8.4, deltaVsGeneral: -0.3, poseQuality: 0.8, poseOverlayUrl: hopOverlayUrl("vod9", 2) },
      { id: "h3", label: "H3", frameId: 348, timeMs: 720, speed: 8.6, deltaVsGeneral: -0.1, poseQuality: 0.85, poseOverlayUrl: hopOverlayUrl("vod9", 3) },
      { id: "h4", label: "H4", frameId: 358, timeMs: 690, speed: 8.8, deltaVsGeneral: 0.1, poseQuality: 0.79, poseOverlayUrl: hopOverlayUrl("vod9", 4) },
      {
        id: "la",
        label: "LA",
        frameId: 384,
        timeMs: 940,
        speed: 7.6,
        deltaVsGeneral: -1.1,
        poseQuality: 0.9,
        poseOverlayUrl: finalFlightOverlayUrl("vod9"),
        isFinalFlight: true,
      },
    ],
  },
};

/* ------------------------------------------------------------------ VOD2 */
// 61 frames: 0,3,…,180 (stride 3). Alta resolución, sin máscaras de pista.
const VOD2_IDS = range(0, 3, 61);

const VOD2 = {
  vod: "vod2",
  media: "/media-vod2",
  overlayNote: null,
  hasMasks: false,
  fpsMock: 8,
  frames: buildFrames({
    media: "/media-vod2",
    frameIds: VOD2_IDS,
    hasMasks: false,
    durationMs: 4800,
    quality: baseQuality([12, 13, 14, 15], [28, 29, 30]),
    skip: (i) => i === 9 || i === 10 || i === 25 || i === 41,
  }),
  // Contactos: hops 60,75,90,108 · aterrizaje 144
  contacts: [
    { id: "h1", label: "H1", type: "hop", frameId: 60 },
    { id: "h2", label: "H2", type: "hop", frameId: 75 },
    { id: "h3", label: "H3", type: "hop", frameId: 90 },
    { id: "h4", label: "H4", type: "hop", frameId: 108 },
    { id: "la", label: "LA", type: "landing", frameId: 144 },
  ],
  metrics: {
    general: { totalTimeMs: 3720, avgSpeed: 8.4 },
    finalFlightNote: null,
    hops: [
      { id: "h1", label: "H1", frameId: 60, timeMs: 680, speed: 8.1, deltaVsGeneral: -0.3, poseQuality: 0.86, poseOverlayUrl: hopOverlayUrl("vod2", 1) },
      { id: "h2", label: "H2", frameId: 75, timeMs: 720, speed: 8.6, deltaVsGeneral: 0.2, poseQuality: 0.79, poseOverlayUrl: hopOverlayUrl("vod2", 2) },
      { id: "h3", label: "H3", frameId: 90, timeMs: 710, speed: 8.9, deltaVsGeneral: 0.5, poseQuality: 0.84, poseOverlayUrl: hopOverlayUrl("vod2", 3) },
      { id: "h4", label: "H4", frameId: 108, timeMs: 690, speed: 8.2, deltaVsGeneral: -0.2, poseQuality: 0.77, poseOverlayUrl: hopOverlayUrl("vod2", 4) },
      { id: "la", label: "LA", frameId: 144, timeMs: 920, speed: 7.4, deltaVsGeneral: -1.0, poseQuality: 0.88, poseOverlayUrl: null, isFinalFlight: true },
    ],
  },
};

const SOURCES = { vod9: VOD9, vod2: VOD2 };

/* ------------------------------------------------------------- Éxito (%) */
/**
 * % de éxito general de la práctica = promedio de las pose-qualities de cada
 * hop + el salto final (0–1 → 0–100). Solo existe si hay análisis.
 * Sirve para comparar la mejor marca contra días previos.
 */
export function successPct(session) {
  if (!session || session.analysis === "none") return null;
  if (session.successPct != null && Number.isFinite(Number(session.successPct))) {
    return Math.round(Number(session.successPct));
  }
  const q = session.poseQualities || [];
  if (!q.length) return null;
  const avg = q.reduce((a, b) => a + b, 0) / q.length;
  return Math.round(avg * 100);
}

/** Color semántico del % (verde alto / ámbar medio / rojo bajo). */
export function successTone(pct) {
  if (pct == null) return { text: "text-soft", ring: "ring-border", bg: "bg-elevated", hex: "#717171" };
  if (pct >= 80) return { text: "text-ok", ring: "ring-ok/40", bg: "bg-ok/15", hex: "#3dd68c" };
  if (pct >= 65) return { text: "text-warn", ring: "ring-warn/40", bg: "bg-warn/15", hex: "#f5a524" };
  return { text: "text-accent", ring: "ring-accent/40", bg: "bg-accent/15", hex: "#ff0033" };
}

export const ANALYSIS_LABEL = {
  full: "Analizado",
  partial: "Parcial",
  none: "Sin análisis",
};

/** Resuelve el índice de frame más cercano a un frameId dentro de una fuente. */
function indexOfFrameId(source, frameId) {
  let best = 0;
  let bestDist = Infinity;
  source.frames.forEach((f, i) => {
    const d = Math.abs(f.frameId - frameId);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  });
  return best;
}

/**
 * Devuelve el paquete de datos listo para el reproductor de una sesión.
 * Mapea los contactos (frameId) a índices concretos de la fuente.
 */
export function resolveSession(session) {
  const source = SOURCES[session.vod] || VOD2;
  const contacts = source.contacts.map((c) => ({
    ...c,
    index: indexOfFrameId(source, c.frameId),
  }));
  const hops = source.metrics.hops.map((h) => ({
    ...h,
    index: indexOfFrameId(source, h.frameId),
    poseLabel: poseLabelFromScore(h.poseQuality),
  }));
  return {
    ...session,
    vod: source.vod,
    media: source.media,
    overlayNote: source.overlayNote,
    hasMasks: source.hasMasks,
    fpsMock: source.fpsMock,
    frames: source.frames,
    frameCount: source.frames.length,
    firstFrameId: source.frames[0].frameId,
    lastFrameId: source.frames[source.frames.length - 1].frameId,
    contacts,
    metrics: {
      general: source.metrics.general,
      finalFlightNote: source.metrics.finalFlightNote,
      hops,
    },
    successPct: successPct(session),
    hasAnalysis: session.analysis !== "none",
  };
}

/* --------------------------------------------------------------- Biblioteca */
// Solo stills REALES de VOD9 / VOD2. Un atleta agrupa sus prácticas por día.
function still(vod, frameId, annotated = false) {
  const media = vod === "vod9" ? "/media-vod9" : "/media-vod2";
  const dir = annotated ? "annotated" : "frames";
  const prefix = annotated ? "annotated" : "frame";
  return `${media}/${dir}/${prefix}_${pad(frameId)}.jpg`;
}

export const ATHLETES = [
  {
    id: "sofia",
    name: "Sofía Reyes",
    short: "Sofía",
    initials: "SR",
    accent: "#ff0033",
    note: "Salto triple · pista al aire libre",
    sessions: [
      {
        id: "sofia-vod9-a",
        vod: "vod9",
        date: "2026-07-11",
        title: "Serie con mapa de pista",
        note: "Mapa de pista y arena aplicado · mejor marca del mes",
        durationLabel: "0:07",
        thumb: still("vod9", 320, true),
        analysis: "full",
        poseQualities: [0.88, 0.8, 0.85, 0.79, 0.9],
      },
      {
        id: "sofia-vod9-c",
        vod: "vod9",
        date: "2026-07-05",
        title: "Revisión de aterrizaje",
        note: "Análisis parcial · faltan H3–H4 por confirmar",
        durationLabel: "0:07",
        thumb: still("vod9", 358, true),
        analysis: "partial",
        poseQualities: [0.82, 0.71, 0.68],
      },
      {
        id: "sofia-vod9-b",
        vod: "vod9",
        date: "2026-06-28",
        title: "Salto sin analizar",
        note: "Pendiente de procesar",
        durationLabel: "0:07",
        thumb: still("vod9", 300),
        analysis: "none",
        poseQualities: [],
      },
    ],
  },
  {
    id: "mateo",
    name: "Mateo Ávila",
    short: "Mateo",
    initials: "MA",
    accent: "#38bdf8",
    note: "Salto triple · pista cubierta",
    sessions: [
      {
        id: "mateo-vod2-a",
        vod: "vod2",
        date: "2026-07-08",
        title: "Sesión mañana",
        note: "Buen ritmo en H2–H3",
        durationLabel: "0:06",
        thumb: still("vod2", 90),
        analysis: "full",
        poseQualities: [0.86, 0.79, 0.84, 0.77, 0.88],
      },
      {
        id: "mateo-vod2-b",
        vod: "vod2",
        date: "2026-07-03",
        title: "Técnica de hops",
        note: "Foco en el aterrizaje",
        durationLabel: "0:06",
        thumb: still("vod2", 108, true),
        analysis: "full",
        poseQualities: [0.74, 0.72, 0.78, 0.7, 0.83],
      },
      {
        id: "mateo-vod2-c",
        vod: "vod2",
        date: "2026-06-28",
        title: "Carrera + 5 hops",
        note: "Pendiente de procesar",
        durationLabel: "0:06",
        thumb: still("vod2", 45),
        analysis: "none",
        poseQualities: [],
      },
    ],
  },
];

export function formatTime(ms) {
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatDelta(v) {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(1)}`;
}

/** Chip + / − / ≈ según delta vs general (velocidad). */
export function deltaChip(v) {
  if (v == null || Math.abs(v) < 0.05) {
    return { glyph: "≈", text: "≈0.0", cls: "bg-elevated text-muted ring-border" };
  }
  if (v > 0) {
    return { glyph: "+", text: formatDelta(v), cls: "bg-ok/15 text-ok ring-ok/40" };
  }
  return { glyph: "−", text: formatDelta(v), cls: "bg-accent/15 text-accent ring-accent/40" };
}

/* ----------------------------------------------------- Colores de fase */
// H1–H4 comparten una familia (sky→violeta); LA/aterrizaje distinto (ámbar);
// carrera (approach) en verde. Marcadores de timeline en negrita y con color.
const HOP_FAMILY = ["#38bdf8", "#22d3ee", "#818cf8", "#a78bfa", "#c084fc"];

export function phaseColor(contact, hopOrder = 0) {
  if (contact.type === "landing") return "#f5a524";
  if (contact.type === "approach") return "#3dd68c";
  return HOP_FAMILY[hopOrder % HOP_FAMILY.length];
}
