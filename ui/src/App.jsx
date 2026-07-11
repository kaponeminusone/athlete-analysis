import { useEffect, useMemo, useRef, useState } from "react";
import {
  Group as PanelGroup,
  Panel,
  Separator as PanelResizeHandle,
} from "react-resizable-panels";
import {
  Brush,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  FolderOpen,
  Play,
  Scan,
  Sparkles,
  SquareDashedMousePointer,
  RefreshCw,
  Map,
  ChartColumn,
} from "lucide-react";

const angleColors = {
  LATERAL: "#f47c30",
  SEMI_BACK: "#d86b25",
  SEMI_FRONT: "#2fc66d",
  FRONTAL: "#dfc83f",
  UNKNOWN: "#64717f",
};

const athleteStateColors = {
  GROUND: "#22c55e",
  AIR: "#38bdf8",
  OFF_TRACK_NEAR: "#f59e0b",
  FINAL_FLIGHT: "#a855f7",
  LOST: "#64748b",
};

const phaseColors = {
  approach: "#3b82f6",
  hop_1: "#15803d",
  hop_2: "#16a34a",
  hop_3: "#22c55e",
  hop_4: "#4ade80",
  final_jump: "#f97316",
  landing: "#92400e",
};

const phaseLabels = {
  approach: "CARRERA",
  hop_1: "HOP 1",
  hop_2: "HOP 2",
  hop_3: "HOP 3",
  hop_4: "HOP 4",
  final_jump: "SALTO FINAL",
  landing: "ATERRIZAJE",
  final: "FINAL",
};

const segmentLabels = {
  approach: "Carrera",
  hop_1: "H1",
  hop_2: "H2",
  hop_3: "H3",
  hop_4: "H4 / final",
  final: "Final",
};

const compareSegmentLabels = {
  hop_1: "H1",
  hop_2: "H2",
  hop_3: "H3",
  hop_4: "H4→aterrizaje",
  final: "H4→aterrizaje",
};

const poseHopLabels = {
  hop_1: "H1",
  hop_2: "H2",
  hop_3: "H3",
  hop_4: "H4",
};

const poseLabelText = {
  buena: "Buena",
  regular: "Regular",
  débil: "Débil",
  debil: "Débil",
};

function poseOverlayUrl(videoName, phase, outputDir, cacheKey) {
  if (!videoName || !phase) return "";
  const params = {};
  if (outputDir) params.output_dir = outputDir;
  if (cacheKey != null && cacheKey !== "") params.v = String(cacheKey);
  return apiUrl(`/api/metrics/${encodeURIComponent(videoName)}/pose-overlay/${encodeURIComponent(phase)}`, params).toString();
}

function PoseOverlayThumb({ videoName, phase, outputDir, cacheKey, title }) {
  const [status, setStatus] = useState("loading"); // loading | ok | error
  const [enlarged, setEnlarged] = useState(false);
  const src = poseOverlayUrl(videoName, phase, outputDir, cacheKey);

  useEffect(() => {
    setStatus("loading");
  }, [src]);

  if (!videoName || !src) {
    return (
      <div className="mt-1 flex h-16 items-center justify-center border border-dashed border-editor-700 bg-editor-850 text-[9px] text-slate-500">
        Sin video
      </div>
    );
  }

  return (
    <>
      <button
        type="button"
        className="mt-1 block w-full overflow-hidden border border-editor-700 bg-editor-850 text-left"
        onClick={() => status === "ok" && setEnlarged(true)}
        title={title || "Ampliar superposición"}
      >
        {status === "loading" ? (
          <div className="flex h-16 items-center justify-center text-[9px] text-slate-500">
            Cargando overlay…
          </div>
        ) : null}
        {status === "error" ? (
          <div className="flex h-16 items-center justify-center text-[9px] text-slate-500">
            Overlay no disponible
          </div>
        ) : null}
        <img
          src={src}
          alt={title || `Overlay ${phase}`}
          className={`max-h-28 w-full object-contain ${status === "ok" ? "block" : "hidden"}`}
          onLoad={() => setStatus("ok")}
          onError={() => setStatus("error")}
        />
      </button>
      {enlarged ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setEnlarged(false)}
          onKeyDown={(e) => e.key === "Escape" && setEnlarged(false)}
          role="dialog"
          aria-modal="true"
        >
          <img
            src={src}
            alt={title || `Overlay ${phase}`}
            className="max-h-[85vh] max-w-[90vw] border border-editor-600 bg-editor-900 object-contain shadow-lg"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      ) : null}
    </>
  );
}

function formatSignedDelta(value, digits = 2, unit = "") {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const n = Number(value);
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(digits)}${unit}`;
}

function indicatorGlyph(ind) {
  if (ind === "+") return "+";
  if (ind === "-" || ind === "−") return "−";
  if (ind === "~" || ind === "≈") return "≈";
  return "·";
}

function indicatorClass(ind) {
  if (ind === "+") return "text-emerald-300";
  if (ind === "-" || ind === "−") return "text-amber-300";
  if (ind === "~" || ind === "≈") return "text-slate-400";
  return "text-slate-500";
}

function poseBarClass(label) {
  if (label === "buena") return "bg-emerald-500";
  if (label === "regular") return "bg-amber-400";
  return "bg-rose-400";
}

function poseBadgeClass(label) {
  if (label === "buena") return "border-emerald-600/60 text-emerald-200";
  if (label === "regular") return "border-amber-600/60 text-amber-200";
  return "border-rose-600/60 text-rose-200";
}

const phaseMarkerShort = {
  approach: "CR",
  hop_1: "H1",
  hop_2: "H2",
  hop_3: "H3",
  hop_4: "H4",
  final_jump: "SF",
  landing: "LA",
};

const PHASE_OPTIONS = [
  "approach",
  "hop_1",
  "hop_2",
  "hop_3",
  "hop_4",
  "final_jump",
  "landing",
];

const POSE_TAG_OPTIONS = [
  { value: "", label: "(sin etiqueta)" },
  { value: "hop_contact", label: "Hop — contacto" },
  { value: "hop_flight", label: "Hop — vuelo" },
  { value: "final_takeoff", label: "Salto final — despegue/arco" },
  { value: "feet_together", label: "Pies juntos adelante" },
];

function phaseForFrame(sections, frameIdx) {
  if (!sections?.phases || frameIdx == null) return null;
  for (const [name, bounds] of Object.entries(sections.phases)) {
    const start = bounds?.start_frame;
    const end = bounds?.end_frame;
    if (start != null && end != null && frameIdx >= start && frameIdx <= end) return name;
  }
  return null;
}

function phaseMarkerAtFrame(sections, frameIdx) {
  if (!sections?.phase_markers?.length || frameIdx == null) return null;
  return sections.phase_markers.find((m) => m.frame_idx === frameIdx) || null;
}

const VENUE_MASK_GRID = { width: 160, height: 90 };
const DEFAULT_VENUE_ID = "default";
/** Radio del pincel pista/arena en celdas de la mascara (antes ~3, muy pequeno). */
const VENUE_BRUSH_RADIUS_GRID = 5;

function paintVenueBrushStroke(mask, nx, ny, radius = VENUE_BRUSH_RADIUS_GRID) {
  const { width, height } = VENUE_MASK_GRID;
  const cx = Math.round(nx * (width - 1));
  const cy = Math.round(ny * (height - 1));
  const r2 = radius * radius;
  for (let y = Math.max(0, cy - radius); y <= Math.min(height - 1, cy + radius); y += 1) {
    for (let x = Math.max(0, cx - radius); x <= Math.min(width - 1, cx + radius); x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= r2) mask[y][x] = 1;
    }
  }
}

const CORNER_LABELS = ["corner_tl", "corner_tr", "corner_br", "corner_bl"];
const OPTIONAL_SEED_LABELS = ["foul_board", "arena_tl", "arena_tr"];

function apiUrl(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  return url;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
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

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toFixed(digits);
}

function polygonAreaNorm(points) {
  if (!points || points.length < 3) return null;
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[(i + 1) % points.length];
    area += x0 * y1 - x1 * y0;
  }
  return Math.abs(area) * 0.5;
}

function formatTime(value) {
  if (value === null || value === undefined) return "N/A";
  return `${Number(value).toFixed(3)}s`;
}

function formatDuration(value) {
  if (!value) return "00:00";
  const total = Math.round(Number(value));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function hasMaskCalibration(cal) {
  const mode = cal?.mode;
  const hasMasks = Object.keys(cal?.mask_frames || {}).length > 0;
  return hasMasks && (mode === "color_masks" || mode === "keyframe_masks" || mode === "cnn_masks");
}

function calibrationKeyframesList(calibration) {
  const raw = calibration?.keyframes;
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  return [];
}

function normalizeCalibration(calibration) {
  if (!calibration) return calibration;
  return {
    ...calibration,
    keyframes: calibrationKeyframesList(calibration),
    seeds: Array.isArray(calibration.seeds) ? calibration.seeds : [],
    mask_frames: calibration.mask_frames && typeof calibration.mask_frames === "object"
      ? calibration.mask_frames
      : {},
  };
}

function countPolygonKeyframeFrames(calibration) {
  let track = 0;
  let sand = 0;
  for (const kf of calibrationKeyframesList(calibration)) {
    if (kf.source === "venue_auto") continue;
    if ((kf.track_polygon?.length ?? 0) >= 3) track += 1;
    if ((kf.landing_zone?.length ?? 0) >= 3) sand += 1;
  }
  return { track, sand };
}

function hasKeyframePolygonPipeline(calibration) {
  return countPolygonKeyframeFrames(calibration).track >= 5;
}

function maskSourceLabel(source) {
  switch (source) {
    case "keyframe":
      return "Keyframe exacto";
    case "interpolated":
      return "Interpolado";
    case "flow_warp":
      return "Flujo optico";
    case "cnn":
      return "CNN (YOLO-seg)";
    case "color":
    default:
      return "Color (HSV)";
  }
}

function maskFrameCount(cal) {
  return Object.keys(cal?.mask_frames || {}).length;
}

function countManualPolygonFrames(calibration) {
  return countPolygonKeyframeFrames(calibration);
}

function hasLearnableVenueData(calibration, venueBrushByFrame, venueBrushPoints) {
  const brush =
    Object.keys(venueBrushByFrame || {}).length > 0 || (venueBrushPoints?.length ?? 0) > 0;
  const polys = countManualPolygonFrames(calibration);
  return brush || polys.track > 0;
}

function getFrameAsset(project, frame, version = 0) {
  if (!project || !frame) return "";
  const padded = String(frame.frame_idx ?? 0).padStart(6, "0");
  const url = (
    project.assets.annotated[padded] ||
    project.assets.frames[padded] ||
    project.assets.annotated[String(frame.frame_idx)] ||
    project.assets.frames[String(frame.frame_idx)] ||
    ""
  );
  return url && version ? `${url}${url.includes("?") ? "&" : "?"}v=${version}` : url;
}

function findCalibrationKeyframe(calibration, frameIdx) {
  return calibrationKeyframesList(calibration).find((k) => k.frame_idx === frameIdx) ?? null;
}

function upsertCalibrationKeyframe(calibration, frameIdx, patch) {
  const keyframes = [...calibrationKeyframesList(calibration)];
  const idx = keyframes.findIndex((k) => k.frame_idx === frameIdx);
  const base =
    idx >= 0
      ? { ...keyframes[idx] }
      : {
          frame_idx: frameIdx,
          track_polygon: [],
          corridor_polygon: [],
          landing_zone: [],
        };
  const next = { ...base, ...patch, frame_idx: frameIdx };
  if (idx >= 0) keyframes[idx] = next;
  else keyframes.push(next);
  keyframes.sort((a, b) => a.frame_idx - b.frame_idx);
  return { ...calibration, keyframes };
}

function interpCalibrationKeyframe(calibration, frameIdx) {
  const keyframes = calibrationKeyframesList(calibration);
  if (!keyframes.length) return null;
  const exact = findCalibrationKeyframe(calibration, frameIdx);
  if (exact) return exact;

  const before = keyframes.filter((k) => k.frame_idx <= frameIdx);
  const after = keyframes.filter((k) => k.frame_idx >= frameIdx);
  if (!before.length) return keyframes[0];
  if (!after.length) return keyframes.at(-1);

  const k0 = before.at(-1);
  const k1 = after[0];
  if (k0.frame_idx === k1.frame_idx) return k0;

  const t = (frameIdx - k0.frame_idx) / Math.max(k1.frame_idx - k0.frame_idx, 1);
  const lerpPoly = (key) => {
    const p0 = k0[key] || [];
    const p1 = k1[key] || [];
    if (!p0.length || p0.length !== p1.length) return p0.length ? p0 : p1;
    return p0.map((a, i) => [a[0] + t * (p1[i][0] - a[0]), a[1] + t * (p1[i][1] - a[1])]);
  };

  return {
    frame_idx: frameIdx,
    track_polygon: lerpPoly("track_polygon"),
    corridor_polygon: lerpPoly("corridor_polygon"),
    landing_zone: lerpPoly("landing_zone"),
  };
}

function nextSeedLabel(seedDraftPoints, seedStep) {
  if (seedStep === "corners") {
    return CORNER_LABELS[seedDraftPoints.length] || null;
  }
  const optionalCount = seedDraftPoints.filter((p) => !CORNER_LABELS.includes(p.label)).length;
  return OPTIONAL_SEED_LABELS[optionalCount] || null;
}

function normalizedToDisplay(point, metrics) {
  const [nx, ny] = point;
  return {
    x: metrics.offsetX + nx * metrics.displayWidth,
    y: metrics.offsetY + ny * metrics.displayHeight,
  };
}

function polygonToSvgPoints(points, metrics) {
  if (!points?.length || !metrics) return "";
  return points
    .map(([nx, ny]) => {
      const { x, y } = normalizedToDisplay([nx, ny], metrics);
      return `${x},${y}`;
    })
    .join(" ");
}

function ResizeHandle({ direction = "vertical" }) {
  return (
    <PanelResizeHandle
      className={
        direction === "vertical"
          ? "group relative w-1.5 bg-editor-950 outline-none transition hover:bg-accent/60 data-[resize-handle-active]:bg-accent"
          : "group relative h-1.5 bg-editor-950 outline-none transition hover:bg-accent/60 data-[resize-handle-active]:bg-accent"
      }
    >
      <div
        className={
          direction === "vertical"
            ? "absolute inset-y-0 -left-1 -right-1"
            : "absolute inset-x-0 -top-1 -bottom-1"
        }
      />
    </PanelResizeHandle>
  );
}

function App() {
  const [videoPath, setVideoPath] = useState("");
  const [videos, setVideos] = useState([]);
  const [isLoadingVideos, setIsLoadingVideos] = useState(false);
  const [loadingVideoPath, setLoadingVideoPath] = useState("");
  const [project, setProject] = useState(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [mode, setMode] = useState("video");
  const [notice, setNotice] = useState("");
  const [noticeStrong, setNoticeStrong] = useState(false);
  const [analysisLog, setAnalysisLog] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisJob, setAnalysisJob] = useState(null);
  const [isCorrecting, setIsCorrecting] = useState(false);
  const [isLoadingDetections, setIsLoadingDetections] = useState(false);
  const [correctionMode, setCorrectionMode] = useState("inspect");
  const [sotBackend, setSotBackend] = useState("none");
  const [stride, setStride] = useState(3);
  const [analysisStartSec, setAnalysisStartSec] = useState("0");
  const [analysisEndSec, setAnalysisEndSec] = useState("");
  const [propagationEndFrame, setPropagationEndFrame] = useState(null);
  const [isReanalyzing, setIsReanalyzing] = useState(false);
  const [reanalysisJob, setReanalysisJob] = useState(null);
  const [refinedOutputDir, setRefinedOutputDir] = useState(null);
  const [useCnnMasks, setUseCnnMasks] = useState(false);
  const [refineV2, setRefineV2] = useState(false);
  const [detections, setDetections] = useState([]);
  const [correctionVersion, setCorrectionVersion] = useState(0);
  const [workMode, setWorkMode] = useState("athlete");
  const [trackSection, setTrackSection] = useState("learn");
  const [trackCorrectionOp, setTrackCorrectionOp] = useState("add_track");
  const [trackCorrectionPoints, setTrackCorrectionPoints] = useState([]);
  const [showTrackMaskOverlay, setShowTrackMaskOverlay] = useState(true);
  const [showSandMaskOverlay, setShowSandMaskOverlay] = useState(true);
  const [showAthleteOverlay, setShowAthleteOverlay] = useState(true);
  const [showAdvancedTrack, setShowAdvancedTrack] = useState(false);
  const [trackLearnInput, setTrackLearnInput] = useState("brush");
  const [showPolygonOverlay, setShowPolygonOverlay] = useState(true);
  const [isCorrectingTrack, setIsCorrectingTrack] = useState(false);
  const [calibration, setCalibration] = useState(null);
  const [calibrationLayer, setCalibrationLayer] = useState("track_polygon");
  const [calibrationSubMode, setCalibrationSubMode] = useState("seed");
  const [seedDraftPoints, setSeedDraftPoints] = useState([]);
  const [seedStep, setSeedStep] = useState("corners");
  const [isPropagating, setIsPropagating] = useState(false);
  const [snapToLines, setSnapToLines] = useState(false);
  const [calDraftPoints, setCalDraftPoints] = useState([]);
  const [isSavingCalibration, setIsSavingCalibration] = useState(false);
  const [isRecomputingTracking, setIsRecomputingTracking] = useState(false);
  const [isAnalyzingSections, setIsAnalyzingSections] = useState(false);
  const [isMarkingPhase, setIsMarkingPhase] = useState(false);
  const [isPropagatingPhases, setIsPropagatingPhases] = useState(false);
  const [phaseMarkPhase, setPhaseMarkPhase] = useState("final_jump");
  const [phaseMarkTag, setPhaseMarkTag] = useState("");
  const [phasePlaceMode, setPhasePlaceMode] = useState(false);
  const [athleteId, setAthleteId] = useState("");
  const [isComputingMetrics, setIsComputingMetrics] = useState(false);
  const [isApplyingOverrides, setIsApplyingOverrides] = useState(false);
  const [corridorMetersInput, setCorridorMetersInput] = useState("10");
  const metricsAutoComputeRef = useRef("");
  const [venueProfile, setVenueProfile] = useState(null);
  const [venueModel, setVenueModel] = useState(null);
  const [venueDataset, setVenueDataset] = useState(null);
  const [isLearningVenue, setIsLearningVenue] = useState(false);
  const [isTrainingVenueCnn, setIsTrainingVenueCnn] = useState(false);
  const [isApplyingVenue, setIsApplyingVenue] = useState(false);
  const [venueBrushLayer, setVenueBrushLayer] = useState("track");
  const [venueBrushPoints, setVenueBrushPoints] = useState([]);
  const [venueBrushByFrame, setVenueBrushByFrame] = useState({});
  const videoRef = useRef(null);

  const frames = project?.analysis?.frames || [];
  const sections = project?.sections?.data || null;
  const metrics = project?.metrics?.data || null;
  const currentFrame = frames[frameIndex] || null;
  const summary = project?.analysis?.data?.summary || {};
  const analysisExists = Boolean(project?.analysis?.exists);
  const frameImage = getFrameAsset(project, currentFrame, correctionVersion);

  useEffect(() => {
    loadVideos();
    loadVenueProfile();
    loadVenueModel();
    loadVenueDataset();
    try {
      const raw = sessionStorage.getItem("prototype:lastProject");
      if (raw) {
        const { path, outputDir } = JSON.parse(raw);
        if (path) openVideo(path, outputDir || null);
      }
    } catch {
      // ignore bad session data
    }
  }, []);

  useEffect(() => {
    if (sections?.athlete_id) {
      setAthleteId(sections.athlete_id);
    }
  }, [sections?.athlete_id, project?.video?.video_name]);

  useEffect(() => {
    const corridor =
      metrics?.overrides?.hops_corridor_m
      ?? metrics?.scale?.hops_corridor_m
      ?? venueProfile?.profile?.hops_corridor_m
      ?? venueProfile?.hops_corridor_m
      ?? 10;
    setCorridorMetersInput(String(corridor));
  }, [project?.video?.video_name, metrics?.derived_version, metrics?.scale?.hops_corridor_m, venueProfile?.profile?.hops_corridor_m, venueProfile?.hops_corridor_m]);

  // Auto-compute metrics once when entering Análisis with contacts but no segments yet
  useEffect(() => {
    if (workMode !== "analisis") return;
    const videoName = project?.video?.video_name;
    if (!videoName || !analysisExists) return;
    const contactCount = sections?.contacts?.length ?? 0;
    if (contactCount < 5) return;
    const hasSegments = Boolean(metrics?.segments?.length);
    const key = `${videoName}:${metrics?.derived_version ?? "none"}:${contactCount}`;
    if (hasSegments) {
      metricsAutoComputeRef.current = key;
      return;
    }
    if (metricsAutoComputeRef.current === key || isComputingMetrics) return;
    metricsAutoComputeRef.current = key;
    recomputeMetrics();
  }, [workMode, project?.video?.video_name, analysisExists, sections?.contacts?.length, metrics?.segments?.length, metrics?.derived_version]);

  useEffect(() => {
    const cfg = project?.analysis?.data?.config;
    if (!cfg) return;
    if (cfg.start_sec != null) setAnalysisStartSec(String(cfg.start_sec));
    if (cfg.end_sec != null && cfg.end_sec > 0) setAnalysisEndSec(String(cfg.end_sec));
    else setAnalysisEndSec("");
    if (cfg.stride != null) setStride(Number(cfg.stride));
  }, [project?.video?.video_name, project?.analysis?.data?.config?.start_sec]);

  function buildAnalysisParams(path) {
    const start = Number(analysisStartSec);
    const endRaw = analysisEndSec.trim();
    const end = endRaw === "" ? null : Number(endRaw);
    if (!Number.isNaN(start) && start < 0) {
      throw new Error("El inicio debe ser >= 0 segundos.");
    }
    if (end != null && !Number.isNaN(end) && end <= (Number.isNaN(start) ? 0 : start)) {
      throw new Error("El fin debe ser mayor que el inicio.");
    }
    const body = { video_path: path, stride };
    if (!Number.isNaN(start) && start > 0) body.start_sec = start;
    if (end != null && !Number.isNaN(end) && end > 0) body.end_sec = end;
    return body;
  }

  async function loadVenueProfile() {
    try {
      const data = await fetchJson(`/api/venue/profile?venue_id=${encodeURIComponent(DEFAULT_VENUE_ID)}`);
      setVenueProfile(data);
    } catch {
      setVenueProfile({ learned: false });
    }
  }

  async function loadVenueModel(venueId = DEFAULT_VENUE_ID) {
    try {
      const params = `?venue_id=${encodeURIComponent(venueId)}`;
      const data = await fetchJson(`/api/venue/model${params}`);
      setVenueModel(data);
    } catch {
      setVenueModel({ trained: false });
    }
  }

  async function loadVenueDataset(venueId = DEFAULT_VENUE_ID) {
    try {
      const data = await fetchJson(`/api/venue/dataset?venue_id=${encodeURIComponent(venueId)}`);
      setVenueDataset(data);
    } catch {
      setVenueDataset({ videos: [], total_frames: 0, ready_to_train: false, can_train: false });
    }
  }

  useEffect(() => {
    setVenueBrushPoints([]);
    setTrackCorrectionPoints([]);
  }, [frameIndex, venueBrushLayer, trackSection, trackCorrectionOp]);

  useEffect(() => {
    if (trackLearnInput === "polygon") setShowPolygonOverlay(true);
    if (trackLearnInput === "brush") setCalDraftPoints([]);
  }, [trackLearnInput]);

  function switchWorkMode(mode) {
    setWorkMode(mode);
    setVenueBrushPoints([]);
    setTrackCorrectionPoints([]);
    setCalDraftPoints([]);
    setSeedDraftPoints([]);
    if (mode === "athlete") {
      setCorrectionMode("inspect");
    }
  }

  function brushPointsToMaskGrid(points) {
    const { height, width } = VENUE_MASK_GRID;
    const mask = Array.from({ length: height }, () => Array(width).fill(0));
    points.forEach((point) => paintVenueBrushStroke(mask, point.nx, point.ny));
    return mask;
  }

  function buildVenueLearnSamples() {
    const samples = [];
    const frames = new Set([
      ...Object.keys(venueBrushByFrame).map(Number),
      ...(currentFrame ? [currentFrame.frame_idx] : []),
    ]);
    frames.forEach((frameIdx) => {
      const stored = venueBrushByFrame[frameIdx] || {};
      const sample = { frame_idx: frameIdx, source: "manual" };
      let hasData = false;
      if (stored.track?.length) {
        sample.track_mask = stored.track;
        hasData = true;
      }
      if (stored.sand?.length) {
        sample.sand_mask = stored.sand;
        hasData = true;
      }
      if (currentFrame?.frame_idx === frameIdx && venueBrushPoints.length > 0) {
        const grid = brushPointsToMaskGrid(venueBrushPoints);
        if (venueBrushLayer === "track") sample.track_mask = grid;
        else sample.sand_mask = grid;
        hasData = true;
      }
      if (hasData) samples.push(sample);
    });
    return samples;
  }

  function saveVenueBrushForFrame(frameIdx) {
    if (!venueBrushPoints.length) return;
    const grid = brushPointsToMaskGrid(venueBrushPoints);
    setVenueBrushByFrame((prev) => {
      const next = { ...prev };
      const entry = { ...(next[frameIdx] || {}) };
      if (venueBrushLayer === "track") entry.track = grid;
      else entry.sand = grid;
      next[frameIdx] = entry;
      return next;
    });
  }

  useEffect(() => {
    if (!project?.video?.video_name) {
      setCalibration(null);
      return;
    }
    loadCalibration(project.video.video_name);
  }, [project?.video?.video_name]);

  const canUseCnnMasks = hasMaskCalibration(calibration);
  useEffect(() => {
    setUseCnnMasks(canUseCnnMasks);
  }, [canUseCnnMasks, project?.video?.video_name]);

  useEffect(() => {
    setCalDraftPoints([]);
    if (workMode !== "track" || !showAdvancedTrack || calibrationSubMode !== "seed" || !currentFrame) {
      setSeedDraftPoints([]);
      return;
    }
    const seed = calibration?.seeds?.find((s) => s.frame_idx === currentFrame.frame_idx);
    if (seed?.seed_points?.length) {
      setSeedDraftPoints(
        seed.seed_points.map(([nx, ny], index) => ({
          nx,
          ny,
          label: seed.labels?.[index] || CORNER_LABELS[index] || `pt_${index}`,
        })),
      );
      setSeedStep(seed.seed_points.length >= 4 ? "optional" : "corners");
    } else {
      setSeedDraftPoints([]);
      setSeedStep("corners");
    }
  }, [frameIndex, calibrationLayer, workMode, calibrationSubMode, calibration, currentFrame?.frame_idx]);

  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
      if (e.key === "ArrowRight") { e.preventDefault(); selectFrame(frameIndex + 1); }
      if (e.key === "ArrowLeft")  { e.preventDefault(); selectFrame(frameIndex - 1); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frameIndex, frames]);

  const sampledFrames = useMemo(
    () =>
      frames
        .map((frame, index) => ({ frame, index }))
        .filter((_, index) => index % 6 === 0)
        .slice(0, 30),
    [frames]
  );

  async function openVideo(path = videoPath, outputDir = null) {
    if (!path?.trim()) {
      setNotice("Indica la ruta del video.");
      setNoticeStrong(true);
      return;
    }

    setLoadingVideoPath(path.trim());
    setNotice("Buscando analysis.json...");
    setNoticeStrong(false);
    try {
      const params = { video_path: path.trim() };
      if (outputDir) params.output_dir = outputDir;
      const nextProject = await fetchJson(apiUrl("/api/project", params));
      setVideoPath(path.trim());
      setProject(nextProject);
      setFrameIndex(0);
      setMode(nextProject.video.exists ? "video" : "annotation");
      sessionStorage.setItem(
        "prototype:lastProject",
        JSON.stringify({
          path: path.trim(),
          outputDir: outputDir || nextProject.output?.path || null,
        }),
      );
      setNotice(nextProject.analysis.exists ? "" : "No existe analysis.json. Puedes generarlo desde Analisis.");
      setNoticeStrong(!nextProject.analysis.exists);
    } catch (error) {
      setNotice(error.message);
      setNoticeStrong(true);
    } finally {
      setLoadingVideoPath("");
    }
  }

  async function loadVideos() {
    setIsLoadingVideos(true);
    try {
      const response = await fetchJson("/api/videos");
      setVideos(response.videos || []);
    } catch (error) {
      setNotice(error.message);
      setNoticeStrong(true);
    } finally {
      setIsLoadingVideos(false);
    }
  }

  function openPickedFile(file) {
    const match = videos.find((video) => video.name.toLowerCase() === file.name.toLowerCase());
    if (match) {
      openVideo(match.path, match.refined_output_dir || null);
      return;
    }
    setNotice(`Intentando abrir "${file.name}" desde la carpeta del proyecto...`);
    setNoticeStrong(false);
    openVideo(file.name);
  }

  async function loadDemo() {
    setNotice("Cargando demo...");
    setNoticeStrong(false);
    try {
      const demo = await fetchJson(apiUrl("/api/demo"));
      setProject(demo);
      setVideoPath(demo.video.path || demo.video.name || "IMG_2048.mp4");
      setFrameIndex(Math.min(6, demo.analysis.frames?.length || 0));
      setMode(demo.video.exists ? "video" : "annotation");
      setNotice("");
    } catch (error) {
      setNotice(error.message);
      setNoticeStrong(true);
    }
  }

  async function runAnalysis() {
    const path = project?.video?.path || videoPath;
    if (!path.trim()) {
      setNotice("Indica la ruta del video antes de generar.");
      setNoticeStrong(true);
      return;
    }

    setIsAnalyzing(true);
    setAnalysisJob(null);
    setAnalysisLog("Creando job de analisis...\n");
    setNotice("Creando job de analisis...");
    setNoticeStrong(false);

    try {
      const body = buildAnalysisParams(path);
      const result = await fetchJson("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const rangeNote = body.start_sec || body.end_sec
        ? ` · recorte ${body.start_sec ?? 0}s → ${body.end_sec ?? "fin"}s`
        : "";
      setAnalysisLog(`Analisis iniciado${rangeNote}. Job: ${result.job_id}\n`);
      pollAnalysisJob(result.job_id, path);
    } catch (error) {
      setIsAnalyzing(false);
      setNotice(error.message);
      setNoticeStrong(true);
    }
  }

  async function runReanalysis() {
    const path = project?.video?.path || videoPath;
    if (!path?.trim()) return;
    setIsReanalyzing(true);
    setReanalysisJob(null);
    setRefinedOutputDir(null);

    // Use Shift+Click range as seed interval when both bounds are set
    const currentFrameIdx = currentFrame?.frame_idx ?? null;
    const seedStart = (propagationEndFrame !== null && currentFrameIdx !== null && currentFrameIdx < propagationEndFrame)
      ? currentFrameIdx : null;
    const seedEnd   = seedStart !== null ? propagationEndFrame : null;

    try {
      const body = buildAnalysisParams(path);
      const result = await fetchJson("/api/reanalyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...body,
          seed_start_frame: seedStart,
          seed_end_frame: seedEnd,
          use_cnn_masks: Boolean(useCnnMasks && canUseCnnMasks),
          refine_v2: Boolean(refineV2),
        }),
      });
      pollReanalysisJob(result.job_id, path, result.output_dir);
    } catch (error) {
      setIsReanalyzing(false);
      setNotice(error.message);
      setNoticeStrong(true);
    }
  }

  async function pollReanalysisJob(jobId, path, outputDir) {
    try {
      const job = await fetchJson(`/api/jobs/${jobId}`);
      setReanalysisJob(job);
      if (job.status === "running" || job.status === "pending") {
        setNotice(`[Refinado] ${Math.round(job.percent || 0)}% · ${job.message || "Refinando..."}`);
        window.setTimeout(() => pollReanalysisJob(jobId, path, outputDir), 600);
        return;
      }
      setIsReanalyzing(false);
      if (job.status === "done") {
        setRefinedOutputDir(outputDir || null);
        setNotice("Refinado completo. Abre el resultado con el botón de abajo.");
      } else {
        setNotice(job.error || "Refinado falló.");
        setNoticeStrong(true);
      }
    } catch {
      window.setTimeout(() => pollReanalysisJob(jobId, path, outputDir), 1000);
    }
  }

  async function openRefinedProject() {
    const path = project?.video?.path || videoPath;
    if (!path || !refinedOutputDir) return;
    const url = apiUrl("/api/project", { video_path: path, output_dir: refinedOutputDir });
    const nextProject = await fetchJson(url).catch((e) => { setNotice(e.message); setNoticeStrong(true); return null; });
    if (!nextProject) return;
    setProject(nextProject);
    setFrameIndex(0);
    setMode("annotation");
    setNotice("Proyecto refinado cargado.");
    setNoticeStrong(false);
  }

  async function refreshProject(path = project?.video?.path || videoPath, outputDir = project?.output?.path) {
    if (!path) return;
    const params = { video_path: path };
    if (outputDir) params.output_dir = outputDir;
    const nextProject = await fetchJson(apiUrl("/api/project", params));
    setProject(nextProject);
    setCorrectionVersion((version) => version + 1);
  }

  async function loadCalibration(videoName) {
    try {
      const data = await fetchJson(`/api/calibration/${encodeURIComponent(videoName)}`);
      setCalibration(normalizeCalibration(data));
    } catch (error) {
      setCalibration({ version: 1, video: `${videoName}.mp4`, keyframes: [] });
      setNotice(error.message);
      setNoticeStrong(true);
    }
  }

  async function persistCalibration(nextCalibration) {
    const videoName = project?.video?.video_name;
    if (!videoName || !nextCalibration) return;
    setIsSavingCalibration(true);
    setNotice("Guardando calibracion...");
    setNoticeStrong(false);
    try {
      const payload = {
        ...nextCalibration,
        video: project.video.name || nextCalibration.video,
      };
      const result = await fetchJson(`/api/calibration/${encodeURIComponent(videoName)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setCalibration(normalizeCalibration(result.calibration || payload));
      setNotice("Calibracion guardada.");
    } catch (error) {
      setNotice(error.message);
      setNoticeStrong(true);
    } finally {
      setIsSavingCalibration(false);
    }
  }

  async function recomputeTracking() {
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    setIsRecomputingTracking(true);
    setNotice("Recalculando tracking (pista)...");
    setNoticeStrong(false);
    try {
      const result = await fetchJson(
        `/api/recompute-tracking/${encodeURIComponent(videoName)}`,
        { method: "POST" },
      );
      const params = { video_path: project.video.path };
      if (project.output?.path) params.output_dir = project.output.path;
      const nextProject = await fetchJson(apiUrl("/api/project", params));
      setProject(nextProject);
      setNotice(`Tracking recalculado en ${result.frames_updated ?? 0} frames.`);
    } catch (error) {
      setNotice(`Error al recalcular tracking: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsRecomputingTracking(false);
    }
  }

  async function reloadProject() {
    if (!project?.video?.path) return null;
    const params = { video_path: project.video.path };
    if (project.output?.path) params.output_dir = project.output.path;
    const nextProject = await fetchJson(apiUrl("/api/project", params));
    setProject(nextProject);
    return nextProject;
  }

  async function analyzeSections() {
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    setIsAnalyzingSections(true);
    setNotice("Detectando fases y contactos...");
    setNoticeStrong(false);
    try {
      const result = await fetchJson(
        `/api/sections/analyze/${encodeURIComponent(videoName)}?use_pose=true`,
        { method: "POST" },
      );
      await reloadProject();
      const n = result.contacts_found ?? 0;
      const conf = result.confidence != null ? `${Math.round(result.confidence * 100)}%` : "N/A";
      setNotice(`Fases detectadas: ${n}/5 contactos · confianza ${conf}.`);
      if (result.sections?.notes) {
        setNoticeStrong(true);
        setNotice(`${result.sections.notes} (${n}/5 contactos, confianza ${conf})`);
      }
    } catch (error) {
      setNotice(`Error al detectar fases: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsAnalyzingSections(false);
    }
  }

  async function markPhaseAtFrame(frameIdx, phase = phaseMarkPhase, poseTag = phaseMarkTag) {
    const videoName = project?.video?.video_name;
    if (!videoName || frameIdx == null) return;
    setIsMarkingPhase(true);
    setNotice(`Asignando ${phaseLabels[phase] || phase} al frame ${frameIdx}...`);
    setNoticeStrong(false);
    try {
      const body = {
        frame_idx: frameIdx,
        phase,
        update_template: Boolean(athleteId),
      };
      if (poseTag) body.pose_tag = poseTag;
      if (athleteId) body.athlete_id = athleteId;
      const result = await fetchJson(
        `/api/sections/mark/${encodeURIComponent(videoName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      await reloadProject();
      const n = result.contacts_found ?? 0;
      setNotice(`Marcado frame ${frameIdx} como ${phaseLabels[phase] || phase}. Contactos: ${n}/5.`);
    } catch (error) {
      setNotice(`Error al marcar fase: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsMarkingPhase(false);
    }
  }

  async function markPhaseOnCurrentFrame() {
    await markPhaseAtFrame(currentFrame?.frame_idx);
  }

  async function movePhaseMarker(fromFrameIdx, toFrameIdx) {
    const videoName = project?.video?.video_name;
    if (!videoName || fromFrameIdx == null || toFrameIdx == null) return;
    if (fromFrameIdx === toFrameIdx) return;
    setIsMarkingPhase(true);
    try {
      const result = await fetchJson(
        `/api/sections/mark/${encodeURIComponent(videoName)}/move`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ from_frame_idx: fromFrameIdx, to_frame_idx: toFrameIdx }),
        },
      );
      await reloadProject();
      setNotice(`Marcador movido: frame ${fromFrameIdx} → ${toFrameIdx}. Contactos: ${result.contacts_found ?? 0}/5.`);
    } catch (error) {
      setNotice(`Error al mover marcador: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsMarkingPhase(false);
    }
  }

  async function removePhaseMarker(frameIdx) {
    const videoName = project?.video?.video_name;
    if (!videoName || frameIdx == null) return;
    setIsMarkingPhase(true);
    try {
      await fetchJson(
        `/api/sections/mark/${encodeURIComponent(videoName)}/${frameIdx}`,
        { method: "DELETE" },
      );
      await reloadProject();
      setNotice(`Marcador eliminado en frame ${frameIdx}.`);
    } catch (error) {
      setNotice(`Error al quitar marcador: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsMarkingPhase(false);
    }
  }

  async function propagatePhases() {
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    setIsPropagatingPhases(true);
    setNotice("Propagando hops desde ancla (salto final / aterrizaje)...");
    setNoticeStrong(false);
    try {
      const body = athleteId ? { athlete_id: athleteId } : {};
      const result = await fetchJson(
        `/api/sections/propagate/${encodeURIComponent(videoName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      await reloadProject();
      const n = result.contacts_found ?? 0;
      setNotice(`Propagacion completada. Contactos: ${n}/5 · marcadores: ${result.markers_count ?? 0}.`);
    } catch (error) {
      setNotice(`Error al propagar fases: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsPropagatingPhases(false);
    }
  }

  async function recomputeMetrics() {
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    setIsComputingMetrics(true);
    setNotice("Recalculando métricas...");
    setNoticeStrong(false);
    try {
      const qs = athleteId ? `?athlete_id=${encodeURIComponent(athleteId)}` : "";
      await fetchJson(`/api/metrics/${encodeURIComponent(videoName)}/compute${qs}`, {
        method: "POST",
      });
      await reloadProject();
      setNotice("Métricas actualizadas.");
    } catch (error) {
      setNotice(`Error al recalcular métricas: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsComputingMetrics(false);
    }
  }

  async function applyCorridorScale() {
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    const n = Number(String(corridorMetersInput).trim());
    if (!Number.isFinite(n) || n <= 0) {
      setNotice("Indica una longitud de pista de hops válida (metros).");
      setNoticeStrong(true);
      return;
    }
    setIsApplyingOverrides(true);
    setNotice("Actualizando escala del corredor...");
    setNoticeStrong(false);
    try {
      const body = { hops_corridor_m: n };
      if (athleteId) body.athlete_id = athleteId;
      await fetchJson(`/api/metrics/${encodeURIComponent(videoName)}/scale`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      await loadVenueProfile();
      await reloadProject();
      setNotice(`Escala actualizada: ${n} m (1er hop → aterrizaje).`);
    } catch (error) {
      setNotice(`Error al actualizar escala: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsApplyingOverrides(false);
    }
  }

  async function learnVenueProfile() {
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    if (currentFrame) saveVenueBrushForFrame(currentFrame.frame_idx);
    const samples = buildVenueLearnSamples();
    const manualPolys = countManualPolygonFrames(calibration);
    setIsLearningVenue(true);
    setNotice("Aprendiendo colores de pista del video (acumulativo)...");
    setNoticeStrong(false);
    try {
      if (calibration && (manualPolys.track > 0 || manualPolys.sand > 0)) {
        await persistCalibration(calibration);
      }
      const result = await fetchJson("/api/venue/learn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_name: videoName,
          video_path: project.video.path,
          venue_id: DEFAULT_VENUE_ID,
          accumulate: true,
          samples: samples.length ? samples : undefined,
        }),
      });
      await loadVenueProfile();
      await loadVenueDataset();
      const videosInDataset = result.videos_in_dataset ?? result.dataset_manifest?.videos?.length ?? 0;
      const totalFrames = result.total_dataset_frames ?? 0;
      if ((result.frames_exported ?? 0) > 0) {
        setNotice(
          `Video ${videoName} añadido: ${manualPolys.track} polígonos pista, ${manualPolys.sand} arena. ` +
          `Dataset total: ${videosInDataset} videos, ${totalFrames} frames. Entrena CNN cuando termines.`,
        );
      } else {
        const videos = (result.videos_contributed || []).join(", ") || videoName;
        setNotice(
          `Perfil aprendido (${result.frames_used ?? 0} keyframes, arena: ${result.sand_frames_used ?? 0}). ` +
          `Videos en perfil: ${videos}. Añade polígonos de pista para exportar al dataset CNN.`,
        );
      }
    } catch (error) {
      setNotice(`Error al aprender venue: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsLearningVenue(false);
    }
  }

  async function trainVenueCnn() {
    const videoName = project?.video?.video_name;
    const manualPolys = countManualPolygonFrames(calibration);
    setIsTrainingVenueCnn(true);
    setNotice("Iniciando entrenamiento CNN (pista/arena)...");
    setNoticeStrong(false);
    try {
      if (videoName && calibration && (manualPolys.track > 0 || manualPolys.sand > 0)) {
        await persistCalibration(calibration);
      }
      const result = await fetchJson("/api/venue/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_name: videoName || undefined,
          venue_id: DEFAULT_VENUE_ID,
          epochs: 40,
          imgsz: 640,
        }),
      });
      if (result.job_id) {
        pollVenueTrainJob(result.job_id);
        return;
      }
      await loadVenueModel();
      await loadVenueDataset();
      setNotice("Entrenamiento CNN completado.");
    } catch (error) {
      setNotice(`Error al entrenar CNN: ${error.message}`);
      setNoticeStrong(true);
      setIsTrainingVenueCnn(false);
    }
  }

  async function pollVenueTrainJob(jobId) {
    try {
      const job = await fetchJson(`/api/jobs/${jobId}`);
      if (job.status === "running" || job.status === "pending") {
        setNotice(`[CNN] ${Math.round(job.percent || 0)}% · ${job.message || job.stage || "Entrenando..."}`);
        window.setTimeout(() => pollVenueTrainJob(jobId), 800);
        return;
      }
      setIsTrainingVenueCnn(false);
      if (job.status === "done") {
        await loadVenueModel();
        await loadVenueDataset();
        setNotice("Modelo CNN entrenado. Puedes aplicar mascaras.");
      } else {
        setNotice(job.error || "Entrenamiento CNN falló.");
        setNoticeStrong(true);
      }
    } catch {
      window.setTimeout(() => pollVenueTrainJob(jobId), 1000);
    }
  }

  async function applyVenueProfile() {
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    setIsApplyingVenue(true);
    const cnnReady = venueModel?.trained;
    setNotice(
      cnnReady
        ? "Aplicando mascaras con modelo CNN..."
        : hasKeyframePolygonPipeline(calibration)
          ? "Generando mascaras desde poligonos..."
          : "Aplicando perfil de venue (mascaras de color)...",
    );
    setNoticeStrong(false);
    try {
      const result = await fetchJson(
        `/api/venue/apply/${encodeURIComponent(videoName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            video_path: project.video.path,
            venue_id: DEFAULT_VENUE_ID,
            merge: true,
            prefer_propagation: false,
            use_masks: true,
            prefer_keyframes: true,
          }),
        },
      );
      if (result.calibration) setCalibration(normalizeCalibration(result.calibration));
      await loadCalibration(videoName);
      const params = { video_path: project.video.path };
      if (project.output?.path) params.output_dir = project.output.path;
      const nextProject = await fetchJson(apiUrl("/api/project", params));
      setProject(nextProject);
      setTrackSection("apply");
      const applied = result.mask_frames_applied ?? result.keyframes_applied ?? 0;
      const total = nextProject?.analysis?.data?.frames?.length ?? applied;
      setNotice(`Mascaras aplicadas: ${applied}/${total} frames con mascara.`);
    } catch (error) {
      setNotice(`Error al aplicar venue: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsApplyingVenue(false);
    }
  }

  async function saveSeedPoints() {
    const videoName = project?.video?.video_name;
    if (!videoName || !currentFrame || seedDraftPoints.length < 4) {
      setNotice("Marca las 4 esquinas de la pista antes de guardar semillas.");
      setNoticeStrong(true);
      return;
    }
    setIsSavingCalibration(true);
    setNotice("Guardando semillas...");
    setNoticeStrong(false);
    try {
      const labels = seedDraftPoints.map((p) => p.label);
      const seedPoints = seedDraftPoints.map((p) => [p.nx, p.ny]);
      const seeds = [...(calibration?.seeds || [])].filter((s) => s.frame_idx !== currentFrame.frame_idx);
      seeds.push({
        frame_idx: currentFrame.frame_idx,
        seed_points: seedPoints,
        labels,
      });
      const result = await fetchJson(`/api/calibration/${encodeURIComponent(videoName)}/seeds`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seeds,
          video_path: project.video.path,
          mode: "seed_auto",
        }),
      });
      setCalibration(normalizeCalibration(result.calibration));
      setNotice("Semillas guardadas. Pulsa propagar para extender a todos los frames.");
    } catch (error) {
      setNotice(error.message);
      setNoticeStrong(true);
    } finally {
      setIsSavingCalibration(false);
    }
  }

  async function propagateCalibration(fromFrame = null) {
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    setIsPropagating(true);
    setNotice("Propagando calibracion (flujo optico)...");
    setNoticeStrong(false);
    try {
      const result = await fetchJson(
        `/api/calibration/${encodeURIComponent(videoName)}/propagate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            video_path: project.video.path,
            snap_to_lines: snapToLines,
            from_frame: fromFrame ?? undefined,
          }),
        },
      );
      setCalibration(normalizeCalibration(result.calibration));
      const params = { video_path: project.video.path };
      if (project.output?.path) params.output_dir = project.output.path;
      const nextProject = await fetchJson(apiUrl("/api/project", params));
      setProject(nextProject);
      setNotice(
        `Propagacion completa: ${result.frames_propagated ?? 0} keyframes` +
          (result.tracking_frames_updated ? ` · tracking en ${result.tracking_frames_updated} frames` : ""),
      );
    } catch (error) {
      setNotice(error.message);
      setNoticeStrong(true);
    } finally {
      setIsPropagating(false);
    }
  }

  function addSeedPoint(point) {
    const label = nextSeedLabel(seedDraftPoints, seedStep);
    if (!label) {
      setNotice("Semillas opcionales completas. Guarda o propaga.");
      setNoticeStrong(false);
      return;
    }
    if (seedStep === "corners" && seedDraftPoints.length >= 4) {
      setSeedStep("optional");
      return addSeedPoint(point);
    }
    setSeedDraftPoints((points) => [...points, { ...point, label }]);
    if (seedStep === "corners" && seedDraftPoints.length + 1 >= 4) {
      setSeedStep("optional");
    }
  }

  function updateKeyframeVertex(frameIdx, vertexIndex, nx, ny) {
    const kf = interpCalibrationKeyframe(calibration, frameIdx);
    if (!kf?.track_polygon?.length) return;
    const polygon = kf.track_polygon.map((p, i) =>
      i === vertexIndex ? [Math.max(0, Math.min(1, nx)), Math.max(0, Math.min(1, ny))] : p,
    );
    const next = upsertCalibrationKeyframe(calibration, frameIdx, { track_polygon: polygon });
    setCalibration(normalizeCalibration(next));
    return next;
  }

  async function handleVertexDragComplete(frameIdx, vertexIndex, nx, ny) {
    const next = updateKeyframeVertex(frameIdx, vertexIndex, nx, ny);
    if (!next) return;
    const videoName = project?.video?.video_name;
    if (!videoName) return;
    await persistCalibration(next);
    await propagateCalibration(frameIdx);
  }

  async function closeCalibrationPolygon() {
    if (!currentFrame || calDraftPoints.length < 3) {
      setNotice("Dibuja al menos 3 puntos para cerrar el poligono.");
      setNoticeStrong(true);
      return;
    }
    const polygon = calDraftPoints.map((p) => [p.nx, p.ny]);
    const base = calibration || { version: 1, video: project?.video?.name || "", keyframes: [] };
    const next = upsertCalibrationKeyframe(base, currentFrame.frame_idx, {
      [calibrationLayer]: polygon,
      source: "manual",
    });
    setCalibration(normalizeCalibration(next));
    setCalDraftPoints([]);
    await persistCalibration(next);
    setNotice("Poligono guardado. Los pixels dentro se usaran al aprender.");
    setNoticeStrong(false);
  }

  function parseTrackCorrectionOp(op) {
    if (op === "add_track") return { layer: "track", operation: "add" };
    if (op === "remove_track") return { layer: "track", operation: "remove" };
    if (op === "add_sand") return { layer: "sand", operation: "add" };
    return { layer: "sand", operation: "remove" };
  }

  async function submitTrackCorrection() {
    const videoName = project?.video?.video_name;
    if (!videoName || !currentFrame || !trackCorrectionPoints.length) {
      setNotice("Pinta una correccion en el frame antes de aplicar.");
      setNoticeStrong(true);
      return;
    }
    const { layer, operation } = parseTrackCorrectionOp(trackCorrectionOp);
    const maskGrid = brushPointsToMaskGrid(trackCorrectionPoints);
    const radius = propagationEndFrame != null && currentFrame.frame_idx < propagationEndFrame
      ? Math.abs(propagationEndFrame - currentFrame.frame_idx)
      : 15;
    setIsCorrectingTrack(true);
    setNotice("Aplicando correccion de pista y propagando...");
    setNoticeStrong(false);
    try {
      const result = await fetchJson(
        `/api/venue/correct/${encodeURIComponent(videoName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            frame_idx: currentFrame.frame_idx,
            layer,
            operation,
            mask_grid: maskGrid,
            radius,
            direction: "both",
            video_path: project.video.path,
          }),
        },
      );
      const cal = await fetchJson(`/api/calibration/${encodeURIComponent(videoName)}`);
      setCalibration(normalizeCalibration(cal));
      setTrackCorrectionPoints([]);
      setPropagationEndFrame(null);
      setNotice(`Correccion de pista aplicada: ${result.total_affected ?? 1} frames afectados.`);
    } catch (error) {
      setNotice(`Error al corregir pista: ${error.message}`);
      setNoticeStrong(true);
    } finally {
      setIsCorrectingTrack(false);
    }
  }

  async function submitCorrection(type, data) {
    if (!project?.video?.path || !currentFrame) {
      setNotice("Abre un video y selecciona un frame antes de corregir.");
      setNoticeStrong(true);
      return;
    }

    setNotice(`Aplicando correccion ${type}...`);
    setNoticeStrong(false);
    setIsCorrecting(true);
    try {
      const result = await fetchJson("/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_path: project.video.path,
          frame_idx: currentFrame.frame_idx,
          type,
          data,
          propagation_radius: 15,
          propagation_end_frame: propagationEndFrame ?? undefined,
          sot_backend: sotBackend,
          output_dir: project.output?.path || undefined,
        }),
      });
      await refreshProject(project.video.path, project.output?.path);
      setDetections([]);
      setPropagationEndFrame(null);
      const saved = result.total_affected ?? 0;
      const poseNote = result.pose_warning ? ` ${result.pose_warning}` : "";
      setNotice(`Correccion aplicada y guardada: ${saved} frames afectados.${poseNote}`);
      if (result.pose_warning) setNoticeStrong(true);
    } catch (error) {
      setNotice(error.message);
      setNoticeStrong(true);
    } finally {
      setIsCorrecting(false);
    }
  }

  async function loadDetections() {
    if (!project?.video?.video_name || !currentFrame) return;
    setNotice("Cargando detecciones del frame...");
    setNoticeStrong(false);
    setIsLoadingDetections(true);
    try {
      const response = await fetchJson(
        `/mask/${project.video.video_name}/${currentFrame.frame_idx}?video_path=${encodeURIComponent(project.video.path)}`
      );
      setDetections(response.detections || []);
      setNotice(`${response.detections?.length || 0} detecciones encontradas.`);
    } catch (error) {
      setNotice(error.message);
      setNoticeStrong(true);
    } finally {
      setIsLoadingDetections(false);
    }
  }

  async function pollAnalysisJob(jobId, path) {
    try {
      const job = await fetchJson(`/api/jobs/${jobId}`);
      setAnalysisJob(job);
      setAnalysisLog(
        [
          `${Math.round(job.percent || 0)}% · ${job.stage || "pending"}`,
          job.message,
          job.current_frame || job.total_frames ? `Frame: ${job.current_frame}/${job.total_frames}` : "",
          job.analyzed_frames ? `Frames analizados: ${job.analyzed_frames}` : "",
          job.last_log,
          job.error ? `ERROR: ${job.error}` : "",
        ]
          .filter(Boolean)
          .join("\n")
      );
      if (job.status === "running" || job.status === "pending") {
        setNotice(`${Math.round(job.percent || 0)}% · ${job.message || "Analizando..."}`);
        window.setTimeout(() => pollAnalysisJob(jobId, path), 500);
        return;
      }
      setIsAnalyzing(false);
      if (job.status === "done") {
        setNotice("Analisis terminado.");
        await openVideo(path);
        return;
      }
      setNotice(job.error || "El analisis fallo.");
      setNoticeStrong(true);
    } catch (error) {
      setAnalysisLog(`No pude consultar el job ${jobId}: ${error.message}`);
      window.setTimeout(() => pollAnalysisJob(jobId, path), 1000);
    }
  }

  function selectFrame(index, sync = true) {
    if (!frames.length) return;
    const nextIndex = Math.max(0, Math.min(index, frames.length - 1));
    setFrameIndex(nextIndex);
    if (sync && videoRef.current) {
      videoRef.current.currentTime = Number(frames[nextIndex]?.timestamp_s || 0);
    }
  }

  function syncVideoToFrame() {
    if (!currentFrame || !videoRef.current) return;
    videoRef.current.currentTime = Number(currentFrame.timestamp_s || 0);
  }

  function updateFrameFromVideo() {
    if (!frames.length || mode !== "video" || !videoRef.current) return;
    const time = videoRef.current.currentTime;
    let bestIndex = frameIndex;
    let bestDistance = Infinity;
    frames.forEach((frame, index) => {
      const distance = Math.abs((frame.timestamp_s || 0) - time);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });
    if (bestIndex !== frameIndex) setFrameIndex(bestIndex);
  }

  return (
    <div className="flex h-screen min-w-[1180px] flex-col overflow-hidden bg-editor-950 text-xs text-slate-200">
      <header className="flex h-9 shrink-0 items-center border-b border-black bg-editor-900 px-2">
        <div className="w-56 text-[13px] font-semibold text-slate-200">YOLO Review</div>
        <div className="ml-auto flex items-center gap-2">
          <span className="max-w-[460px] truncate text-[11px] text-slate-500">
            {loadingVideoPath ? "Cargando video..." : project?.video?.name || "Selecciona un video desde Medios"}
          </span>
          <button
            className="h-6 border border-editor-600 bg-editor-800 px-2 text-[11px] text-slate-200"
            type="button"
            onClick={loadDemo}
          >
            Demo
          </button>
        </div>
      </header>

      {notice ? (
        <div className={`shrink-0 border-b border-editor-700 bg-editor-850 px-3 py-1.5 ${noticeStrong ? "text-amber-200" : "text-slate-400"}`}>
          {notice}
        </div>
      ) : null}

      <PanelGroup orientation="vertical" className="min-h-0 flex-1">
        <Panel defaultSize="74%" minSize="360px">
          <PanelGroup orientation="horizontal">
            <Panel defaultSize="22%" minSize="210px" maxSize="420px" groupResizeBehavior="preserve-pixel-size">
              <MediaPanel
                sampledFrames={sampledFrames}
                project={project}
                currentFrame={currentFrame}
                videos={videos}
                activeIndex={frameIndex}
                onSelectFrame={selectFrame}
                onOpenVideo={(path, outputDir) => openVideo(path, outputDir)}
                onRefreshVideos={loadVideos}
                onPickFile={openPickedFile}
                loadingVideoPath={loadingVideoPath}
                isLoadingVideos={isLoadingVideos}
              />
            </Panel>
            <ResizeHandle />
            <Panel defaultSize="58%" minSize="420px">
              <ViewerPanel
                project={project}
                videoRef={videoRef}
                mode={mode}
                setMode={setMode}
                currentFrame={currentFrame}
                frameImage={frameImage}
                onPrev={() => selectFrame(frameIndex - 1)}
                onNext={() => selectFrame(frameIndex + 1)}
                onSync={syncVideoToFrame}
                onTimeUpdate={updateFrameFromVideo}
                workMode={workMode}
                correctionMode={correctionMode}
                detections={detections}
                correctionVersion={correctionVersion}
                isCorrecting={isCorrecting}
                isLoadingDetections={isLoadingDetections}
                onSubmitCorrection={submitCorrection}
                calibration={calibration}
                showTrackMaskOverlay={showTrackMaskOverlay}
                showSandMaskOverlay={showSandMaskOverlay}
                showAthleteOverlay={showAthleteOverlay}
                trackSection={trackSection}
                trackCorrectionOp={trackCorrectionOp}
                trackCorrectionPoints={trackCorrectionPoints}
                setTrackCorrectionPoints={setTrackCorrectionPoints}
                venueBrushLayer={venueBrushLayer}
                venueBrushPoints={venueBrushPoints}
                setVenueBrushPoints={setVenueBrushPoints}
                showAdvancedTrack={showAdvancedTrack}
                trackLearnInput={trackLearnInput}
                showPolygonOverlay={showPolygonOverlay}
                calibrationLayer={calibrationLayer}
                calibrationSubMode={calibrationSubMode}
                calDraftPoints={calDraftPoints}
                setCalDraftPoints={setCalDraftPoints}
                seedDraftPoints={seedDraftPoints}
                onAddSeedPoint={addSeedPoint}
                onCloseCalibrationPolygon={closeCalibrationPolygon}
                onVertexDragComplete={handleVertexDragComplete}
                sections={sections}
              />
            </Panel>
            <ResizeHandle />
            <Panel defaultSize="20%" minSize="230px" maxSize="420px" groupResizeBehavior="preserve-pixel-size">
              <Inspector
                currentFrame={currentFrame}
                summary={summary}
                analysisExists={analysisExists}
                onRunAnalysis={runAnalysis}
                isAnalyzing={isAnalyzing}
                analysisJob={analysisJob}
                analysisLog={analysisLog}
                onRunReanalysis={runReanalysis}
                isReanalyzing={isReanalyzing}
                reanalysisJob={reanalysisJob}
                refinedOutputDir={refinedOutputDir}
                onOpenRefined={openRefinedProject}
                useCnnMasks={useCnnMasks}
                setUseCnnMasks={setUseCnnMasks}
                canUseCnnMasks={canUseCnnMasks}
                refineV2={refineV2}
                setRefineV2={setRefineV2}
                workMode={workMode}
                onSwitchWorkMode={switchWorkMode}
                correctionMode={correctionMode}
                setCorrectionMode={setCorrectionMode}
                sotBackend={sotBackend}
                setSotBackend={setSotBackend}
                stride={stride}
                setStride={setStride}
                analysisStartSec={analysisStartSec}
                setAnalysisStartSec={setAnalysisStartSec}
                analysisEndSec={analysisEndSec}
                setAnalysisEndSec={setAnalysisEndSec}
                videoDuration={
                  project?.analysis?.data?.video_info?.duration_s
                  ?? project?.video?.duration_s
                  ?? null
                }
                propagationEndFrame={propagationEndFrame}
                setPropagationEndFrame={setPropagationEndFrame}
                currentFrameIdx={currentFrame?.frame_idx ?? null}
                onLoadDetections={loadDetections}
                isCorrecting={isCorrecting}
                isLoadingDetections={isLoadingDetections}
                calibration={calibration}
                trackSection={trackSection}
                setTrackSection={setTrackSection}
                trackCorrectionOp={trackCorrectionOp}
                setTrackCorrectionOp={setTrackCorrectionOp}
                onSubmitTrackCorrection={submitTrackCorrection}
                isCorrectingTrack={isCorrectingTrack}
                showTrackMaskOverlay={showTrackMaskOverlay}
                setShowTrackMaskOverlay={setShowTrackMaskOverlay}
                showSandMaskOverlay={showSandMaskOverlay}
                setShowSandMaskOverlay={setShowSandMaskOverlay}
                showAthleteOverlay={showAthleteOverlay}
                setShowAthleteOverlay={setShowAthleteOverlay}
                showAdvancedTrack={showAdvancedTrack}
                setShowAdvancedTrack={setShowAdvancedTrack}
                trackLearnInput={trackLearnInput}
                setTrackLearnInput={setTrackLearnInput}
                showPolygonOverlay={showPolygonOverlay}
                setShowPolygonOverlay={setShowPolygonOverlay}
                calibrationLayer={calibrationLayer}
                setCalibrationLayer={setCalibrationLayer}
                calibrationSubMode={calibrationSubMode}
                setCalibrationSubMode={setCalibrationSubMode}
                seedDraftPoints={seedDraftPoints}
                setSeedDraftPoints={setSeedDraftPoints}
                seedStep={seedStep}
                setSeedStep={setSeedStep}
                calDraftPoints={calDraftPoints}
                setCalDraftPoints={setCalDraftPoints}
                onCloseCalibrationPolygon={closeCalibrationPolygon}
                onSaveCalibration={() => persistCalibration(calibration)}
                onSaveSeedPoints={saveSeedPoints}
                onPropagateCalibration={() => propagateCalibration()}
                isPropagating={isPropagating}
                snapToLines={snapToLines}
                setSnapToLines={setSnapToLines}
                isSavingCalibration={isSavingCalibration}
                onRecomputeTracking={recomputeTracking}
                isRecomputingTracking={isRecomputingTracking}
                venueProfile={venueProfile}
                venueModel={venueModel}
                venueDataset={venueDataset}
                onLearnVenueProfile={learnVenueProfile}
                onTrainVenueCnn={trainVenueCnn}
                onApplyVenueProfile={applyVenueProfile}
                isLearningVenue={isLearningVenue}
                isTrainingVenueCnn={isTrainingVenueCnn}
                isApplyingVenue={isApplyingVenue}
                venueBrushLayer={venueBrushLayer}
                setVenueBrushLayer={setVenueBrushLayer}
                venueBrushPoints={venueBrushPoints}
                setVenueBrushPoints={setVenueBrushPoints}
                venueBrushByFrame={venueBrushByFrame}
                onSaveVenueBrush={() => {
                  if (currentFrame) saveVenueBrushForFrame(currentFrame.frame_idx);
                  setVenueBrushPoints([]);
                }}
                sections={sections}
                onAnalyzeSections={analyzeSections}
                isAnalyzingSections={isAnalyzingSections}
                athleteId={athleteId}
                setAthleteId={setAthleteId}
                phaseMarkPhase={phaseMarkPhase}
                setPhaseMarkPhase={setPhaseMarkPhase}
                phaseMarkTag={phaseMarkTag}
                setPhaseMarkTag={setPhaseMarkTag}
                onMarkPhase={markPhaseOnCurrentFrame}
                onMarkPhaseAtFrame={markPhaseAtFrame}
                onRemovePhaseMarker={removePhaseMarker}
                onPropagatePhases={propagatePhases}
                isMarkingPhase={isMarkingPhase}
                isPropagatingPhases={isPropagatingPhases}
                phasePlaceMode={phasePlaceMode}
                setPhasePlaceMode={setPhasePlaceMode}
                onSelectFrame={selectFrame}
                frames={frames}
                metrics={metrics}
                videoName={project?.video?.video_name || null}
                outputDir={project?.output?.path || null}
                onRecomputeMetrics={recomputeMetrics}
                isComputingMetrics={isComputingMetrics}
                corridorMetersInput={corridorMetersInput}
                setCorridorMetersInput={setCorridorMetersInput}
                onApplyCorridorScale={applyCorridorScale}
                isApplyingOverrides={isApplyingOverrides}
              />
            </Panel>
          </PanelGroup>
        </Panel>
        <ResizeHandle direction="horizontal" />
        <Panel defaultSize="26%" minSize="150px" maxSize="45%">
          <BottomPanel
            project={project}
            summary={summary}
            frames={frames}
            currentFrame={currentFrame}
            currentIndex={frameIndex}
            onSelectFrame={selectFrame}
            propagationEndFrame={propagationEndFrame}
            workMode={workMode}
            trackSection={trackSection}
            calibration={calibration}
            sections={sections}
            phasePlaceMode={phasePlaceMode}
            onMarkPhaseAtFrame={markPhaseAtFrame}
            onMovePhaseMarker={movePhaseMarker}
            phaseMarkPhase={phaseMarkPhase}
            isMarkingPhase={isMarkingPhase}
            onShiftSelect={(idx) => {
              const f = frames[idx];
              if (!f || !currentFrame) return;
              if (f.frame_idx !== currentFrame.frame_idx)
                setPropagationEndFrame(f.frame_idx > currentFrame.frame_idx ? f.frame_idx : null);
            }}
          />
        </Panel>
      </PanelGroup>
    </div>
  );
}

function MediaPanel({
  sampledFrames,
  project,
  currentFrame,
  videos,
  activeIndex,
  onSelectFrame,
  onOpenVideo,
  onRefreshVideos,
  onPickFile,
  loadingVideoPath,
  isLoadingVideos,
}) {
  const [activeTab, setActiveTab] = useState("media");
  const fileInputRef = useRef(null);

  function handleFiles(files) {
    const [file] = files;
    if (file) onPickFile(file);
  }

  return (
    <aside className="h-full overflow-hidden border-r border-black bg-editor-800">
      <div className="flex h-9 border-b border-editor-700 bg-editor-850">
        <button
          className={`flex-1 text-[11px] ${activeTab === "media" ? "border-b-2 border-accent text-accent" : "text-slate-500"}`}
          type="button"
          onClick={() => setActiveTab("media")}
        >
          Medios
        </button>
        <button
          className={`flex-1 text-[11px] ${activeTab === "frames" ? "border-b-2 border-accent text-accent" : "text-slate-500"}`}
          type="button"
          onClick={() => setActiveTab("frames")}
        >
          Frames
        </button>
      </div>
      {activeTab === "media" ? (
        <>
          <button
            className="m-2 grid h-20 w-[calc(100%-16px)] place-items-center border border-dashed border-editor-500 bg-editor-850 text-slate-500"
            type="button"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              handleFiles([...event.dataTransfer.files]);
            }}
          >
            <FolderOpen className="mb-1 h-5 w-5 text-accent" />
            <span className="text-[11px]">Abrir o arrastrar video</span>
          </button>
          <input
            ref={fileInputRef}
            className="hidden"
            type="file"
            accept="video/*"
            onChange={(event) => handleFiles([...event.target.files])}
          />
          <div className="flex items-center justify-between px-2 pb-2 text-[10px] uppercase text-slate-500">
            <span>Videos locales</span>
            <button
              className="inline-flex items-center gap-1 text-accent disabled:opacity-60"
              type="button"
              disabled={isLoadingVideos}
              onClick={onRefreshVideos}
            >
              <RefreshCw className={`h-3 w-3 ${isLoadingVideos ? "animate-spin" : ""}`} />
              {isLoadingVideos ? "buscando" : "refrescar"}
            </button>
          </div>
          <div className="grid max-h-[calc(100%-142px)] gap-2 overflow-auto px-2 pb-2">
            {isLoadingVideos ? (
              <div className="border border-accent/40 bg-accent/10 px-2 py-3 text-[11px] text-slate-200">
                <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-accent" />
                Buscando videos locales...
              </div>
            ) : videos.length ? (
              videos.map((video) => {
                const isActive   = project?.video?.path === video.path;
                const isLoading  = loadingVideoPath === video.path;
                return (
                  <div key={video.path} className="grid gap-1">
                    <button
                      className={`relative grid grid-cols-[96px_1fr] gap-2 border bg-editor-850 p-1 text-left transition ${
                        isActive ? "border-accent" : "border-editor-700 hover:border-editor-500"
                      } ${isLoading ? "opacity-80" : ""}`}
                      type="button"
                      disabled={Boolean(loadingVideoPath)}
                      onClick={() => onOpenVideo(video.path)}
                    >
                      <div className="relative h-14 overflow-hidden bg-black">
                        <video className="h-full w-full object-cover" src={video.url} muted preload="metadata" />
                        <span className="absolute bottom-1 right-1 bg-black/70 px-1 text-[10px] text-slate-200">
                          {formatDuration(video.duration_s)}
                        </span>
                      </div>
                      <div className="min-w-0">
                        <div className="truncate text-[11px] font-semibold text-slate-200">{video.name}</div>
                        <div className="mt-1 text-[10px] text-slate-500">
                          {isLoading
                            ? "cargando..."
                            : isActive
                              ? "video activo"
                              : video.has_analysis
                                ? "analysis.json listo"
                                : "sin analisis"}
                        </div>
                        {isLoading ? (
                          <div className="mt-2 h-1 overflow-hidden bg-editor-950">
                            <div className="h-full w-1/2 animate-pulse bg-accent" />
                          </div>
                        ) : null}
                      </div>
                    </button>

                    {video.has_refined ? (
                      <button
                        className="flex items-center gap-2 border border-purple-700/50 bg-purple-950/30 px-2 py-1 text-left text-[11px] text-purple-300 hover:border-purple-500 disabled:opacity-50"
                        type="button"
                        disabled={Boolean(loadingVideoPath)}
                        onClick={() => onOpenVideo(video.path, video.refined_output_dir)}
                      >
                        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-purple-400" />
                        <span className="truncate">{video.name} — refinado</span>
                      </button>
                    ) : null}
                  </div>
                );
              })
            ) : (
              <div className="py-4 text-center text-[11px] text-slate-500">No hay videos en la carpeta del proyecto.</div>
            )}
          </div>
        </>
      ) : (
        <div className="h-[calc(100%-36px)] overflow-auto">
          <DetectionDataPanel frame={currentFrame} compact />
        </div>
      )}
    </aside>
  );
}

function ViewerPanel({
  project,
  videoRef,
  mode,
  setMode,
  currentFrame,
  frameImage,
  onPrev,
  onNext,
  onSync,
  onTimeUpdate,
  workMode,
  correctionMode,
  detections,
  isCorrecting,
  isLoadingDetections,
  onSubmitCorrection,
  calibration,
  showTrackMaskOverlay,
  showSandMaskOverlay,
  showAthleteOverlay,
  trackSection,
  trackCorrectionOp,
  trackCorrectionPoints,
  setTrackCorrectionPoints,
  venueBrushLayer,
  venueBrushPoints,
  setVenueBrushPoints,
  showAdvancedTrack,
  trackLearnInput,
  showPolygonOverlay,
  calibrationLayer,
  calibrationSubMode,
  calDraftPoints,
  setCalDraftPoints,
  seedDraftPoints,
  onAddSeedPoint,
  onCloseCalibrationPolygon,
  onVertexDragComplete,
  sections,
}) {
  const [dragStart, setDragStart] = useState(null);
  const [dragBox, setDragBox] = useState(null);
  const [maskPoints, setMaskPoints] = useState([]);
  const [isPainting, setIsPainting] = useState(false);
  const [isVenuePainting, setIsVenuePainting] = useState(false);
  const [venueMaskInfo, setVenueMaskInfo] = useState(null);
  const [debugUrl, setDebugUrl] = useState(null);
  const [layoutVersion, setLayoutVersion] = useState(0);  // bumped on resize → forces re-render
  const [draggingVertex, setDraggingVertex] = useState(null);
  const [livePolygon, setLivePolygon] = useState(null);
  const stageRef = useRef(null);

  useEffect(() => {
    setDragStart(null);
    setDragBox(null);
    setMaskPoints([]);
    setIsPainting(false);
    setDebugUrl(null);
    setDraggingVertex(null);
    setLivePolygon(null);
  }, [currentFrame?.frame_idx, correctionMode, workMode, calibrationLayer, calibrationSubMode, trackSection, trackCorrectionOp, trackLearnInput]);

  useEffect(() => {
    setVenueMaskInfo(null);
    const videoName = project?.video?.video_name;
    if (!videoName || !hasMaskCalibration(calibration) || !currentFrame || workMode !== "track") return;
    const frameKey = String(currentFrame.frame_idx);
    if (!calibration.mask_frames?.[frameKey]) {
      setVenueMaskInfo({ hasMask: false, frame_idx: currentFrame.frame_idx });
      return;
    }
    const params = new URLSearchParams();
    if (project?.video?.path) params.set("video_path", project.video.path);
    fetch(`/api/venue/masks/${encodeURIComponent(videoName)}/${currentFrame.frame_idx}?${params}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setVenueMaskInfo({ ...data, hasMask: true });
        else setVenueMaskInfo({ hasMask: false, frame_idx: currentFrame.frame_idx });
      })
      .catch(() => setVenueMaskInfo({ hasMask: false, frame_idx: currentFrame.frame_idx }));
  }, [currentFrame?.frame_idx, calibration, project?.video?.video_name, project?.video?.path, workMode]);

  const isTrackPainting =
    workMode === "track" &&
    ((trackSection === "learn" && trackLearnInput === "brush") || trackSection === "correct");
  const isPolygonDrawing =
    workMode === "track" && trackSection === "learn" && trackLearnInput === "polygon";
  const activeBrushPoints = trackSection === "correct" ? trackCorrectionPoints : venueBrushPoints;
  const setActiveBrushPoints = trackSection === "correct" ? setTrackCorrectionPoints : setVenueBrushPoints;
  const activeBrushLayer = trackSection === "correct"
    ? (trackCorrectionOp.includes("sand") ? "sand" : "track")
    : venueBrushLayer;

  function findNearestVertex(point, polygon, metrics, radius = 14) {
    if (!polygon?.length || !metrics) return -1;
    let best = -1;
    let bestDist = radius;
    polygon.forEach(([nx, ny], index) => {
      const { x, y } = normalizedToDisplay([nx, ny], metrics);
      const dist = Math.hypot(x - point.px, y - point.py);
      if (dist <= bestDist) {
        bestDist = dist;
        best = index;
      }
    });
    return best;
  }

  // Re-render overlay positions whenever the stage container is resized
  // (e.g. dragging the panel split handle)
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setLayoutVersion(v => v + 1));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  function mediaMetrics() {
    const stage = stageRef.current;
    if (!stage) return null;
    const stageRect = stage.getBoundingClientRect();

    const mediaEl = stage.querySelector("[data-media]");
    if (!mediaEl) return null;

    const mediaRect = mediaEl.getBoundingClientRect();

    // Natural dimensions: video_info from analysis is the most reliable
    // source because it's always set after analysis. Element properties
    // (naturalWidth / videoWidth) may still be 0 while the media is loading.
    const infoW = project?.analysis?.data?.video_info?.width;
    const infoH = project?.analysis?.data?.video_info?.height;
    const naturalWidth  = infoW
                       || (mediaEl.naturalWidth  > 0 ? mediaEl.naturalWidth  : 0)
                       || (mediaEl.videoWidth    > 0 ? mediaEl.videoWidth    : 0)
                       || 1280;
    const naturalHeight = infoH
                       || (mediaEl.naturalHeight > 0 ? mediaEl.naturalHeight : 0)
                       || (mediaEl.videoHeight   > 0 ? mediaEl.videoHeight   : 0)
                       || 720;

    // object-contain letterbox within the element's rendered box
    const elW   = mediaRect.width;
    const elH   = mediaRect.height;
    const scale = Math.min(elW / naturalWidth, elH / naturalHeight);
    const displayWidth  = naturalWidth  * scale;
    const displayHeight = naturalHeight * scale;
    const lbX = (elW - displayWidth)  / 2;
    const lbY = (elH - displayHeight) / 2;

    // Offset from stage top-left to the image's pixel (0,0)
    const offsetX = (mediaRect.left - stageRect.left) + lbX;
    const offsetY = (mediaRect.top  - stageRect.top)  + lbY;

    return { rect: stageRect, naturalWidth, naturalHeight, scale, displayWidth, displayHeight, offsetX, offsetY };
  }

  function eventToFramePoint(event) {
    const metrics = mediaMetrics();
    if (!metrics) return null;
    const localX = event.clientX - metrics.rect.left - metrics.offsetX;
    const localY = event.clientY - metrics.rect.top - metrics.offsetY;
    if (localX < 0 || localY < 0 || localX > metrics.displayWidth || localY > metrics.displayHeight) return null;
    return {
      x: Math.round(localX / metrics.scale),
      y: Math.round(localY / metrics.scale),
      nx: localX / metrics.displayWidth,
      ny: localY / metrics.displayHeight,
      px: metrics.offsetX + localX,
      py: metrics.offsetY + localY,
    };
  }

  function detectionStyle(bbox) {
    const metrics = mediaMetrics();
    if (!metrics) return {};
    const [x1, y1, x2, y2] = bbox;
    return {
      left: `${metrics.offsetX + x1 * metrics.scale}px`,
      top: `${metrics.offsetY + y1 * metrics.scale}px`,
      width: `${Math.max(1, (x2 - x1) * metrics.scale)}px`,
      height: `${Math.max(1, (y2 - y1) * metrics.scale)}px`,
    };
  }

  function boxStyle(box) {
    if (!box) return {};
    const metrics = mediaMetrics();
    if (!metrics) return {};
    const x1 = Math.min(box.start.x, box.end.x);
    const y1 = Math.min(box.start.y, box.end.y);
    const x2 = Math.max(box.start.x, box.end.x);
    const y2 = Math.max(box.start.y, box.end.y);
    return {
      left: `${metrics.offsetX + x1 * metrics.scale}px`,
      top: `${metrics.offsetY + y1 * metrics.scale}px`,
      width: `${(x2 - x1) * metrics.scale}px`,
      height: `${(y2 - y1) * metrics.scale}px`,
    };
  }

  function submitMask() {
    const width = 160;
    const height = 90;
    const mask = Array.from({ length: height }, () => Array(width).fill(0));
    maskPoints.forEach((point) => {
      const cx = Math.round(point.nx * (width - 1));
      const cy = Math.round(point.ny * (height - 1));
      for (let y = Math.max(0, cy - 3); y <= Math.min(height - 1, cy + 3); y += 1) {
        for (let x = Math.max(0, cx - 3); x <= Math.min(width - 1, cx + 3); x += 1) {
          mask[y][x] = 1;
        }
      }
    });
    onSubmitCorrection("mask_correction", { mask });
    setMaskPoints([]);
    setIsPainting(false);
  }

  function validateBox() {
    if (!dragBox || !currentFrame || !project?.video?.path) return;
    const x1 = Math.round(Math.min(dragBox.start.x, dragBox.end.x));
    const y1 = Math.round(Math.min(dragBox.start.y, dragBox.end.y));
    const x2 = Math.round(Math.max(dragBox.start.x, dragBox.end.x));
    const y2 = Math.round(Math.max(dragBox.start.y, dragBox.end.y));
    const url = apiUrl("/api/debug_coords", {
      video_path: project.video.path,
      frame_idx: currentFrame.frame_idx,
      x1, y1, x2, y2,
    });
    setDebugUrl(url.toString() + "&t=" + Date.now());
  }

  function handlePointerDown(event) {
    if (workMode === "track") {
      const point = eventToFramePoint(event);
      if (!point) return;

      if (showAdvancedTrack && calibrationSubMode === "seed") {
        const metrics = mediaMetrics();
        const kf = currentFrame && calibration ? interpCalibrationKeyframe(calibration, currentFrame.frame_idx) : null;
        const vertexIndex = findNearestVertex(point, kf?.track_polygon, metrics);
        if (vertexIndex >= 0 && kf?.track_polygon) {
          setDraggingVertex({ index: vertexIndex, frameIdx: currentFrame.frame_idx });
          return;
        }
        onAddSeedPoint?.(point);
        return;
      }

      if (isPolygonDrawing) {
        setCalDraftPoints((points) => [...points, point]);
        return;
      }

      if (isTrackPainting) {
        setIsVenuePainting(true);
        setActiveBrushPoints((points) => [...points, point]);
      }
      return;
    }
    if (correctionMode === "inspect") return;
    const point = eventToFramePoint(event);
    if (!point) return;

    if (correctionMode === "click_selection") {
      onSubmitCorrection("click_selection", { x: point.x, y: point.y });
      setMaskPoints([]);
      setDragBox(null);
      return;
    }

    if (correctionMode === "bbox_correction") {
      setDragStart(point);
      setDragBox({ start: point, end: point });
      return;
    }

    if (correctionMode === "mask_correction") {
      setIsPainting(true);
      setMaskPoints((points) => [...points, point]);
    }
  }

  function handlePointerMove(event) {
    const point = eventToFramePoint(event);
    if (!point) return;
    if (draggingVertex != null && currentFrame) {
      const kf = interpCalibrationKeyframe(calibration, draggingVertex.frameIdx);
      if (kf?.track_polygon) {
        const polygon = kf.track_polygon.map((p, i) =>
          i === draggingVertex.index ? [point.nx, point.ny] : p,
        );
        setLivePolygon(polygon);
      }
      return;
    }
    if (correctionMode === "bbox_correction" && dragStart) {
      setDragBox({ start: dragStart, end: point });
    }
    if (correctionMode === "mask_correction" && isPainting) {
      setMaskPoints((points) => [...points, point]);
    }
    if (workMode === "track" && isTrackPainting && isVenuePainting) {
      setActiveBrushPoints((points) => [...points, point]);
    }
  }

  function handlePointerUp(event) {
    if (draggingVertex != null) {
      const point = eventToFramePoint(event);
      if (point && onVertexDragComplete) {
        onVertexDragComplete(draggingVertex.frameIdx, draggingVertex.index, point.nx, point.ny);
      }
      setDraggingVertex(null);
      setLivePolygon(null);
      return;
    }
    if (correctionMode === "bbox_correction" && dragStart) {
      // Freeze the box — don't submit yet. Let user Validar or Aplicar.
      const point = eventToFramePoint(event) || dragBox?.end;
      if (point) setDragBox({ start: dragStart, end: point });
      setDragStart(null);
    }
    if (correctionMode === "mask_correction") {
      setIsPainting(false);
    }
    if (workMode === "track" && isTrackPainting) {
      setIsVenuePainting(false);
    }
  }

  function submitBbox() {
    if (!dragBox) return;
    const x1 = Math.min(dragBox.start.x, dragBox.end.x);
    const y1 = Math.min(dragBox.start.y, dragBox.end.y);
    const x2 = Math.max(dragBox.start.x, dragBox.end.x);
    const y2 = Math.max(dragBox.start.y, dragBox.end.y);
    if (Math.abs(x2 - x1) < 5 || Math.abs(y2 - y1) < 5) return;
    setDragBox(null);
    setDebugUrl(null);
    onSubmitCorrection("bbox_correction", { x1, y1, x2, y2 });
  }

  const currentKeyframe =
    currentFrame && calibration ? interpCalibrationKeyframe(calibration, currentFrame.frame_idx) : null;
  const overlayKeyframe =
    currentKeyframe && livePolygon
      ? { ...currentKeyframe, track_polygon: livePolygon }
      : currentKeyframe;
  const metrics = mediaMetrics();
  const useMaskOverlay = hasMaskCalibration(calibration);
  const inPolygonLearnMode = isPolygonDrawing;
  const showCalibrationOverlay =
    workMode === "track" &&
    calibration &&
    (inPolygonLearnMode || showPolygonOverlay || (showAdvancedTrack && calibrationSubMode === "seed"));
  const interactiveLayer =
    workMode === "track"
      ? (isTrackPainting || isPolygonDrawing || (showAdvancedTrack && calibrationSubMode !== "color"))
      : correctionMode !== "inspect";

  return (
    <section className="grid h-full grid-rows-[32px_minmax(0,1fr)_30px] bg-editor-950">
      <div className="flex items-center justify-between border-b border-editor-700 bg-editor-850 px-2">
        <div className="truncate">
          <span className="mr-2 font-semibold text-accent">{project?.video?.name || "Sin video"}</span>
          <span className="text-slate-500">
            {currentFrame
              ? `Frame ${currentFrame.frame_idx} · ${formatTime(currentFrame.timestamp_s)} · ${currentFrame.camera_angle}`
              : "Abre un video para revisar"}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <ToolButton onClick={onPrev}>
            <ChevronLeft className="h-3.5 w-3.5" />
          </ToolButton>
          <button className="h-6 border border-editor-600 bg-editor-800 px-2 text-[11px]" type="button" onClick={onSync}>
            Ir al frame
          </button>
          <ToolButton onClick={onNext}>
            <ChevronRight className="h-3.5 w-3.5" />
          </ToolButton>
        </div>
      </div>

      <div
        ref={stageRef}
        className="relative grid min-h-0 place-items-center overflow-hidden bg-black p-2"
      >
        {mode === "video" && project?.video.exists ? (
          <video
            ref={videoRef}
            data-media
            className="h-full w-full select-none object-contain"
            src={project.video.url}
            controls
            draggable={false}
            onTimeUpdate={onTimeUpdate}
            onDragStart={(event) => event.preventDefault()}
          />
        ) : frameImage ? (
          <img
            data-media
            className="h-full w-full select-none object-contain"
            src={frameImage}
            alt="Frame YOLO"
            draggable={false}
            onDragStart={(event) => event.preventDefault()}
          />
        ) : (
          <div className="grid h-full w-full place-items-center border border-dashed border-editor-600 text-slate-500">
            Indica la ruta del video. Si ya existe el analisis, se cargara automaticamente.
          </div>
        )}
        {interactiveLayer ? (
          <div
            className="absolute inset-0 z-30 cursor-crosshair"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={() => {
              setIsPainting(false);
              setIsVenuePainting(false);
            }}
          />
        ) : null}
        {workMode === "track" && useMaskOverlay && venueMaskInfo?.hasMask ? (
          <VenueMaskOverlay
            metrics={metrics}
            maskInfo={venueMaskInfo}
            showTrack={showTrackMaskOverlay}
            showSand={showSandMaskOverlay}
          />
        ) : null}
        {showCalibrationOverlay ? (
          <CalibrationOverlay
            metrics={metrics}
            keyframe={overlayKeyframe}
            draftPoints={isPolygonDrawing ? calDraftPoints : []}
            seedDraftPoints={showAdvancedTrack && calibrationSubMode === "seed" ? seedDraftPoints : []}
            activeLayer={calibrationLayer}
            draggableVertices={showAdvancedTrack && calibrationSubMode === "seed"}
            isDrawing={isPolygonDrawing && calDraftPoints.length > 0}
          />
        ) : null}
        {workMode === "athlete" ? (
        <CorrectionOverlay
          detections={detections}
          detectionStyle={detectionStyle}
          dragBox={dragBox}
          boxStyle={boxStyle}
          maskPoints={maskPoints}
          correctionMode={correctionMode}
          isBusy={isCorrecting || isLoadingDetections}
          busyLabel={isCorrecting ? "Aplicando correccion..." : "Cargando detecciones..."}
          onSubmitMask={submitMask}
          onClearMask={() => setMaskPoints([])}
        />
        ) : null}
        {workMode === "track" && isTrackPainting ? (
          <VenueBrushOverlay
            metrics={metrics}
            brushPoints={activeBrushPoints}
            layer={activeBrushLayer}
            correctionOp={trackSection === "correct" ? trackCorrectionOp : null}
          />
        ) : null}
        {workMode === "athlete" && showAthleteOverlay && currentFrame ? (
          <TrackingOverlay
            frame={currentFrame}
            detectionStyle={detectionStyle}
          />
        ) : null}
        {sections && currentFrame ? (
          <PhaseBadge frame={currentFrame} sections={sections} />
        ) : null}

        {/* Bbox action bar — shown while a box is drawn, before submitting */}
        {correctionMode === "bbox_correction" && dragBox && !dragStart && !isCorrecting ? (
          <div className="absolute bottom-2 right-2 z-50 flex gap-1">
            <button
              className="border border-yellow-500 bg-yellow-900/90 px-3 py-1 text-[11px] text-yellow-200 hover:bg-yellow-800"
              type="button"
              onClick={validateBox}
            >
              Validar
            </button>
            <button
              className="border border-accent bg-accent/20 px-3 py-1 text-[11px] text-accent hover:bg-accent/30"
              type="button"
              onClick={submitBbox}
            >
              Aplicar
            </button>
            <button
              className="border border-editor-600 bg-editor-900/90 px-2 py-1 text-[11px] text-slate-400 hover:bg-editor-800"
              type="button"
              onClick={() => { setDragBox(null); setDebugUrl(null); }}
            >
              ✕
            </button>
          </div>
        ) : null}

        {isPolygonDrawing && calDraftPoints.length > 0 ? (
          <div className="pointer-events-auto absolute bottom-2 left-2 z-50 flex gap-1">
            <button
              className="border border-emerald-500 bg-emerald-900/90 px-3 py-1 text-[11px] text-emerald-100 hover:bg-emerald-800"
              type="button"
              onClick={onCloseCalibrationPolygon}
            >
              Cerrar poligono ({calDraftPoints.length})
            </button>
            <button
              className="border border-editor-600 bg-editor-900/90 px-2 py-1 text-[11px] text-slate-400 hover:bg-editor-800"
              type="button"
              onClick={() => setCalDraftPoints([])}
            >
              Limpiar
            </button>
          </div>
        ) : null}

        {/* Debug overlay — full-frame image with server-drawn bbox */}
        {debugUrl ? (
          <div
            className="absolute inset-0 z-50 flex flex-col bg-black/90"
            onClick={() => setDebugUrl(null)}
          >
            <div className="shrink-0 bg-yellow-900/80 px-3 py-1 text-[11px] text-yellow-200">
              Validacion del servidor — las coords que recibio dibujadas sobre el frame real.
              Si el recuadro rojo coincide con tu seleccion, el mapeo es correcto.
              Click para cerrar.
            </div>
            <img
              className="min-h-0 flex-1 object-contain"
              src={debugUrl}
              alt="debug coords"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        ) : null}
      </div>

      <div className="flex items-center gap-1.5 border-y border-editor-700 bg-editor-850 px-2">
        <ModeButton active={mode === "video"} onClick={() => setMode("video")}>
          <Play className="h-3 w-3" /> Video
        </ModeButton>
        <ModeButton active={mode === "annotation"} onClick={() => setMode("annotation")}>
          <Sparkles className="h-3 w-3" /> Frame YOLO
        </ModeButton>
      </div>
    </section>
  );
}

function ToolButton({ children, onClick }) {
  return (
    <button className="grid h-6 w-6 place-items-center border border-editor-600 bg-editor-800" type="button" onClick={onClick}>
      {children}
    </button>
  );
}

function ModeButton({ active, children, onClick }) {
  return (
    <button
      className={`inline-flex h-6 items-center gap-1 border px-2 text-[11px] ${
        active ? "border-accent bg-accent/15 text-slate-100" : "border-editor-600 bg-transparent text-slate-500"
      }`}
      type="button"
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function CorrectionOverlay({
  detections,
  detectionStyle,
  dragBox,
  boxStyle,
  maskPoints,
  correctionMode,
  isBusy,
  busyLabel,
  onSubmitMask,
  onClearMask,
}) {
  return (
    <div className="pointer-events-none absolute inset-0 z-40">
      {isBusy ? (
        <div className="absolute left-3 top-3 z-50 border border-accent bg-black/80 px-3 py-2 text-[11px] text-slate-100">
          <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-accent" />
          {busyLabel}
        </div>
      ) : null}
      {detections.map((det) => (
        <div
          className="absolute border border-accent bg-accent/10"
          key={det.detection_idx}
          style={detectionStyle(det.bbox)}
        >
          <span className="absolute left-0 top-0 bg-accent px-1 text-[10px] text-black">
            ID {det.track_id ?? det.detection_idx} · {det.conf}
          </span>
        </div>
      ))}
      {dragBox ? <div className="absolute border border-amber-300 bg-amber-300/10" style={boxStyle(dragBox)} /> : null}
      {maskPoints.map((point, index) => (
        <span
          className="absolute h-3 w-3 rounded-full bg-sky-400/60"
          key={`${point.nx}-${point.ny}-${index}`}
          style={{
            left: `${point.px}px`,
            top: `${point.py}px`,
            transform: "translate(-50%, -50%)",
          }}
        />
      ))}
      {correctionMode === "mask_correction" && maskPoints.length > 0 ? (
        <div className="pointer-events-auto absolute bottom-3 left-3 flex gap-2">
          <button className="border border-accent bg-accent px-2 py-1 text-[11px] text-black" type="button" onClick={onSubmitMask}>
            Aplicar mascara
          </button>
          <button className="border border-editor-600 bg-editor-850 px-2 py-1 text-[11px]" type="button" onClick={onClearMask}>
            Limpiar
          </button>
        </div>
      ) : null}
    </div>
  );
}

function PhaseBadge({ frame, sections }) {
  const phase = frame.phase || phaseForFrame(sections, frame.frame_idx);
  if (!phase) return null;
  const label = phaseLabels[phase] || phase.toUpperCase();
  const color = phaseColors[phase] || "#64748b";
  const usable = frame.usable_for_analysis
    || frame.camera_angle === "LATERAL"
    || frame.camera_angle === "SEMI_BACK";
  return (
    <div className="pointer-events-none absolute inset-0 z-[39]">
      <div
        className="absolute right-2 top-2 border px-2 py-0.5 text-[10px] font-bold uppercase text-white"
        style={{ background: color }}
      >
        {label}
      </div>
      {!usable ? (
        <div className="absolute left-2 bottom-2 border border-amber-500/60 bg-amber-900/80 px-2 py-0.5 text-[10px] text-amber-100">
          Angulo no ideal para analisis
        </div>
      ) : null}
    </div>
  );
}

function TrackingOverlay({ frame, detectionStyle }) {
  const bbox = frame?.person_bbox;
  const predicted = frame?.predicted_bbox;
  const overlap = frame?.track_overlap;
  const state = frame?.athlete_state;
  const highOverlap = overlap != null && overlap > 0.5;

  const stateLabel = {
    GROUND: "Suelo",
    AIR: "Vuelo",
    OFF_TRACK_NEAR: "Fuera cerca",
    FINAL_FLIGHT: "Salto final",
    LOST: "Perdido",
  };

  return (
    <div className="pointer-events-none absolute inset-0 z-[38]">
      {bbox && bbox.length >= 4 ? (
        <div
          className={`absolute ${highOverlap ? "border-2 border-emerald-400" : "border border-slate-500/60"}`}
          style={detectionStyle(bbox)}
        />
      ) : null}
      {predicted && predicted.length >= 4 ? (
        <div
          className="absolute border border-dashed border-sky-400/80 bg-sky-400/5"
          style={detectionStyle(predicted)}
        />
      ) : null}
      {state ? (
        <div
          className="absolute left-2 top-2 border px-2 py-0.5 text-[10px] font-semibold uppercase text-black"
          style={{ background: athleteStateColors[state] || athleteStateColors.LOST }}
        >
          {stateLabel[state] || state}
          {overlap != null ? ` · ${Math.round(overlap * 100)}%` : ""}
        </div>
      ) : null}
    </div>
  );
}

function VenueBrushOverlay({ metrics, brushPoints, layer, correctionOp = null }) {
  if (!metrics || !brushPoints?.length) return null;
  let color = layer === "track" ? "#34d399" : "#fbbf24";
  if (correctionOp?.startsWith("remove")) color = "#f87171";
  const scale = metrics.displayWidth / VENUE_MASK_GRID.width;
  const displayRadius = Math.max(12, VENUE_BRUSH_RADIUS_GRID * scale);
  return (
    <svg className="pointer-events-none absolute inset-0 z-[36] h-full w-full overflow-visible">
      {brushPoints.map((point, index) => (
        <circle key={`vbrush-${index}`} cx={point.px} cy={point.py} r={displayRadius} fill={color} opacity={0.45} />
      ))}
    </svg>
  );
}

function TintedMaskLayer({ url, rgb, opacity, outlineRgb }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !url) return;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const tinted = ctx.createImageData(width, height);
      const [r, g, b] = rgb;
      const alpha = Math.round(255 * opacity);
      for (let i = 0; i < data.length; i += 4) {
        const lum = Math.max(data[i], data[i + 1], data[i + 2]);
        if (lum > 32) {
          tinted.data[i] = r;
          tinted.data[i + 1] = g;
          tinted.data[i + 2] = b;
          tinted.data[i + 3] = alpha;
        }
      }
      ctx.putImageData(tinted, 0, 0);
    };
    img.src = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
  }, [url, rgb, opacity]);
  const [or, og, ob] = outlineRgb || rgb;
  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full"
      style={{
        filter: `drop-shadow(0 0 1px rgb(${or},${og},${ob})) drop-shadow(0 0 2px rgb(${or},${og},${ob}))`,
      }}
    />
  );
}

function VenueMaskOverlay({ metrics, maskInfo, showTrack = true, showSand = true }) {
  if (!metrics || !maskInfo?.hasMask || (!showTrack && !showSand)) return null;
  const { offsetX, offsetY, displayWidth, displayHeight } = metrics;
  return (
    <div
      className="pointer-events-none absolute z-[34] overflow-hidden"
      style={{
        left: `${offsetX}px`,
        top: `${offsetY}px`,
        width: `${displayWidth}px`,
        height: `${displayHeight}px`,
      }}
    >
      {showTrack && maskInfo.track_url ? (
        <TintedMaskLayer url={maskInfo.track_url} rgb={[0, 255, 100]} opacity={0.6} outlineRgb={[0, 255, 100]} />
      ) : null}
      {showSand && maskInfo.sand_url ? (
        <TintedMaskLayer url={maskInfo.sand_url} rgb={[255, 220, 0]} opacity={0.5} outlineRgb={[255, 220, 0]} />
      ) : null}
    </div>
  );
}

function CalibrationOverlay({ metrics, keyframe, draftPoints, seedDraftPoints, activeLayer, draggableVertices, isDrawing = false }) {
  if (!metrics) return null;

  const layers = [
    { id: "track_polygon", points: keyframe?.track_polygon, stroke: "#34d399", fill: "rgba(52,211,153,0.35)", label: "Pista" },
    { id: "landing_zone", points: keyframe?.landing_zone, stroke: "#fbbf24", fill: "rgba(251,191,36,0.35)", label: "Arena" },
    { id: "corridor_polygon", points: keyframe?.corridor_polygon, stroke: "#60a5fa", fill: "rgba(96,165,250,0.25)", label: "Corredor" },
  ];

  const draftPolygon = draftPoints.map((p) => [p.nx, p.ny]);
  const seedPoints = seedDraftPoints || [];

  return (
    <svg className="pointer-events-none absolute inset-0 z-[35] h-full w-full overflow-visible">
      {layers.map((layer) =>
        layer.points?.length >= 3 ? (
          <g key={layer.id}>
            <polygon
              points={polygonToSvgPoints(layer.points, metrics)}
              fill={layer.fill}
              stroke={layer.stroke}
              strokeWidth={activeLayer === layer.id ? 3 : 2}
              opacity={activeLayer === layer.id ? 1 : 0.9}
            />
            {draggableVertices && layer.id === "track_polygon"
              ? layer.points.map(([nx, ny], index) => {
                  const { x, y } = normalizedToDisplay([nx, ny], metrics);
                  return (
                    <circle
                      key={`vertex-${index}`}
                      cx={x}
                      cy={y}
                      r={6}
                      fill="#34d399"
                      stroke="#064e3b"
                      strokeWidth={1.5}
                    />
                  );
                })
              : null}
            {layer.points[0] ? (
              <text
                x={normalizedToDisplay(layer.points[0], metrics).x + 4}
                y={normalizedToDisplay(layer.points[0], metrics).y - 4}
                fill={layer.stroke}
                fontSize="11"
                fontWeight="600"
              >
                {layer.label}
              </text>
            ) : null}
          </g>
        ) : null
      )}
      {seedPoints.map((point, index) => {
        const { x, y } = normalizedToDisplay([point.nx, point.ny], metrics);
        const isCorner = CORNER_LABELS.includes(point.label);
        return (
          <g key={`seed-${index}`}>
            <circle cx={x} cy={y} r={isCorner ? 7 : 5} fill={isCorner ? "#22c55e" : "#fbbf24"} stroke="#0f172a" strokeWidth={1.5} />
            <text x={x + 8} y={y - 8} fill="#ecfdf5" fontSize="12" fontWeight="700">
              {isCorner ? index + 1 : point.label?.replace("arena_", "") || "?"}
            </text>
          </g>
        );
      })}
      {seedPoints.length >= 4 ? (
        <polygon
          points={polygonToSvgPoints(seedPoints.slice(0, 4).map((p) => [p.nx, p.ny]), metrics)}
          fill="rgba(34,197,94,0.12)"
          stroke="#22c55e"
          strokeWidth={2}
          strokeDasharray="5 4"
        />
      ) : null}
      {draftPolygon.length >= 2 ? (
        <polyline
          points={polygonToSvgPoints(draftPolygon, metrics)}
          fill="none"
          stroke="#f8fafc"
          strokeWidth={3}
          strokeDasharray="6 4"
        />
      ) : null}
      {draftPolygon.length >= 3 ? (
        <polygon
          points={polygonToSvgPoints(draftPolygon, metrics)}
          fill="rgba(248,250,252,0.35)"
          stroke="#f8fafc"
          strokeWidth={3}
          strokeDasharray="6 4"
        />
      ) : null}
      {draftPoints.map((point, index) => (
        <g key={`draft-${index}`}>
          <circle
            cx={point.px}
            cy={point.py}
            r={isDrawing ? 8 : 5}
            fill="#f8fafc"
            stroke="#0f172a"
            strokeWidth={isDrawing ? 2 : 1.5}
          />
          <text
            x={point.px + 10}
            y={point.py - 10}
            fill="#f8fafc"
            fontSize={isDrawing ? "13" : "11"}
            fontWeight="700"
          >
            {index + 1}
          </text>
        </g>
      ))}
    </svg>
  );
}

function frameIndexFromClientX(clientX, trackEl, frames) {
  if (!trackEl || !frames.length) return 0;
  const rect = trackEl.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  return Math.min(frames.length - 1, Math.max(0, Math.floor(pct * frames.length)));
}

function markerLeftPercent(marker, frames) {
  const idx = frames.findIndex((f) => f.frame_idx === marker.frame_idx);
  if (idx < 0) return 0;
  return ((idx + 0.5) / frames.length) * 100;
}

function frameSlotStyle(index, totalFrames) {
  const n = Math.max(totalFrames, 1);
  return {
    left: `${(index / n) * 100}%`,
    width: `${(1 / n) * 100}%`,
  };
}

function Timeline({
  frames,
  currentIndex,
  onSelect,
  duration,
  propagationEndFrame,
  onShiftSelect,
  workMode = "athlete",
  trackSection = "learn",
  calibration = null,
  sections = null,
  phasePlaceMode = false,
  onMarkPhaseAtFrame,
  onMovePhaseMarker,
  phaseMarkPhase = "final_jump",
  isMarkingPhase = false,
}) {
  const trackRef = useRef(null);
  const [dragMarker, setDragMarker] = useState(null);
  const [dragGhostX, setDragGhostX] = useState(null);

  const phaseEditMode = workMode === "track" && trackSection === "fases";

  useEffect(() => {
    if (!dragMarker) return undefined;

    function onMove(e) {
      setDragGhostX(e.clientX);
    }

    async function onUp(e) {
      const fromFrame = dragMarker.frame_idx;
      setDragMarker(null);
      setDragGhostX(null);
      if (!trackRef.current || !onMovePhaseMarker) return;
      const idx = frameIndexFromClientX(e.clientX, trackRef.current, frames);
      const toFrame = frames[idx]?.frame_idx;
      if (toFrame != null && toFrame !== fromFrame) {
        await onMovePhaseMarker(fromFrame, toFrame);
      }
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragMarker, frames, onMovePhaseMarker]);

  if (!frames.length) {
    return <div className="m-2 border border-editor-700 bg-editor-850 px-2 py-2 text-[10px] uppercase text-slate-500">Timeline</div>;
  }

  const currentFrame = frames[currentIndex];
  const phaseMarkers = [...(sections?.phase_markers || [])].sort((a, b) => a.frame_idx - b.frame_idx);
  const showMarkerLane = phaseEditMode || phaseMarkers.length > 0;

  const rangeStart = currentFrame?.frame_idx ?? null;
  const rangeEnd   = propagationEndFrame;
  const hasRange   = rangeStart !== null && rangeEnd !== null && rangeEnd > rangeStart;
  const maskFrames = calibration?.mask_frames || {};
  const maskCount = Object.keys(maskFrames).length;
  const hasPhases = Boolean(
    sections?.phases && Object.values(sections.phases).some((p) => p?.start_frame != null),
  ) || phaseMarkers.length > 0;

  function handleWheel(e) {
    e.preventDefault();
    onSelect(currentIndex + (e.deltaY > 0 ? 1 : -1));
  }

  function handleClick(e, index) {
    if (phaseEditMode && phasePlaceMode && onMarkPhaseAtFrame) {
      const fidx = frames[index]?.frame_idx;
      if (fidx != null) {
        onMarkPhaseAtFrame(fidx);
        return;
      }
    }
    if (e.shiftKey) { onShiftSelect?.(index); }
    else            { onSelect(index); }
  }

  return (
    <div className="relative m-2 overflow-hidden border border-editor-600 bg-editor-850" onWheel={handleWheel}>
      <div className="absolute left-2 top-1 z-10 flex flex-wrap items-center gap-3 text-[10px] text-slate-500">
        <span className="uppercase">Timeline</span>
        {phaseEditMode ? (
          <span className={`rounded border px-1.5 py-0.5 ${phasePlaceMode ? "border-violet-400 bg-violet-500/20 text-violet-200" : "border-slate-600 text-slate-400"}`}>
            {phasePlaceMode ? "Modo colocar activo" : "Arrastra pinos para mover"}
          </span>
        ) : null}
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm bg-emerald-500" />
          Deteccion
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm bg-slate-600" />
          Sin deteccion
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm" style={{ background: angleColors.LATERAL }} />
          Angulo
        </span>
        {workMode === "track" ? (
          <>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-sm bg-emerald-400" />
              Con mascara
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-sm bg-red-900" />
              Sin mascara
            </span>
            {maskCount > 0 ? (
              <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-emerald-300">
                {maskCount}/{frames.length} frames con mascara
              </span>
            ) : null}
          </>
        ) : (
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-sm bg-emerald-400" />
            Colision pista
          </span>
        )}
        {hasPhases ? (
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: phaseColors.approach }} />
            Fases
          </span>
        ) : null}
      </div>

      {showMarkerLane ? (
        <div
          ref={trackRef}
          className="relative mx-0 mt-5 h-7 border-b border-editor-700 bg-editor-950/80"
        >
          {phaseMarkers.map((marker) => {
            const left = markerLeftPercent(marker, frames);
            const color = phaseColors[marker.phase] || "#64748b";
            const short = phaseMarkerShort[marker.phase] || marker.phase;
            const isManual = marker.source === "manual";
            const isDragging = dragMarker?.frame_idx === marker.frame_idx;
            return (
              <button
                key={`pin-${marker.frame_idx}-${marker.phase}`}
                type="button"
                className={`absolute top-0 z-30 -translate-x-1/2 cursor-grab active:cursor-grabbing ${
                  isDragging ? "opacity-40" : ""
                } ${isMarkingPhase ? "pointer-events-none" : ""}`}
                style={{ left: `${left}%` }}
                title={`${phaseLabels[marker.phase] || marker.phase} · frame ${marker.frame_idx}${
                  marker.pose_tag ? ` · ${marker.pose_tag}` : ""
                }${marker.source ? ` · ${marker.source}` : ""}\nArrastra para mover · suelta sobre otro pin para intercambiar`}
                onMouseDown={(e) => {
                  if (!phaseEditMode || isMarkingPhase) return;
                  e.preventDefault();
                  e.stopPropagation();
                  setDragMarker(marker);
                  setDragGhostX(e.clientX);
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  const idx = frames.findIndex((f) => f.frame_idx === marker.frame_idx);
                  if (idx >= 0) onSelect(idx);
                }}
              >
                <div
                  className="flex min-w-[22px] flex-col items-center"
                  style={{ filter: isManual ? undefined : "brightness(0.85)" }}
                >
                  <div
                    className="h-0 w-0 border-x-[5px] border-b-[7px] border-x-transparent"
                    style={{ borderBottomColor: color }}
                  />
                  <span
                    className="mt-0.5 rounded px-1 py-0.5 text-[9px] font-bold leading-none text-white shadow"
                    style={{
                      background: color,
                      border: isManual ? "1px solid white" : "1px dashed rgba(255,255,255,0.5)",
                    }}
                  >
                    {short}
                  </span>
                </div>
              </button>
            );
          })}
          {dragMarker && dragGhostX != null && trackRef.current ? (
            <div
              className="pointer-events-none absolute top-0 z-40 -translate-x-1/2"
              style={{
                left: `${Math.max(0, Math.min(100, ((dragGhostX - trackRef.current.getBoundingClientRect().left) / trackRef.current.getBoundingClientRect().width) * 100))}%`,
              }}
            >
              <div
                className="h-0 w-0 border-x-[6px] border-b-[8px] border-x-transparent opacity-90"
                style={{ borderBottomColor: phaseColors[dragMarker.phase] || "#fff" }}
              />
            </div>
          ) : null}
        </div>
      ) : null}

      <div
        className={`relative h-full ${
          showMarkerLane ? "min-h-[88px]" : hasPhases ? "min-h-[88px]" : "min-h-[72px]"
        } ${showMarkerLane ? "pt-0" : "pt-5"}`}
      >
        {frames.map((frame, index) => {
          const slot = frameSlotStyle(index, frames.length);
          const detected = frame.person_detected;
          const angleColor = detected
            ? (angleColors[frame.camera_angle] || angleColors.UNKNOWN)
            : "#1e2433";
          const isActive = index === currentIndex;
          const inRange  = hasRange && frame.frame_idx > rangeStart && frame.frame_idx <= rangeEnd;
          const maskEntry = maskFrames[String(frame.frame_idx)];
          const hasMask = maskEntry != null;
          const trackCov = workMode === "track"
            ? (maskEntry?.track_area_norm ?? 0)
            : (frame.track_overlap ?? 0);
          const trackColor = workMode === "track"
            ? (hasMask ? "#22c55e" : "#7f1d1d")
            : (frame.athlete_state
              ? (athleteStateColors[frame.athlete_state] || athleteStateColors.LOST)
              : "#1e2433");
          const overlapAlpha = workMode === "track"
            ? (hasMask ? Math.min(1, 0.55 + trackCov * 0.45) : 0.9)
            : (frame.track_overlap != null
              ? Math.min(1, 0.25 + frame.track_overlap * 0.75)
              : 0.15);

          const phaseName = frame.phase || phaseForFrame(sections, frame.frame_idx);
          const phaseColor = phaseName ? (phaseColors[phaseName] || "#334155") : "#1e2433";
          const marker = phaseMarkerAtFrame(sections, frame.frame_idx);
          const laneH = hasPhases
            ? { det: "22%", ang: "28%", trk: "25%", ph: "25%" }
            : { det: "28%", ang: "36%", trk: "36%", ph: "0%" };

          return (
            <button
              key={`${frame.frame_idx}-${index}`}
              type="button"
              className={`absolute bottom-0 top-5 overflow-hidden border-r border-black/30 p-0 ${
                isActive ? "ring-2 ring-white ring-inset z-10" : ""
              } ${inRange ? "brightness-125" : ""}`}
              style={{
                left: slot.left,
                width: slot.width,
                outline: inRange ? "1px solid rgba(250,204,21,0.6)" : undefined,
              }}
              title={`Frame ${frame.frame_idx} · ${frame.timestamp_s?.toFixed(2)}s\n${
                detected ? `Deteccion: SI · ${frame.camera_angle}` : "Deteccion: NO"
              }${frame.quality_score != null ? ` · Q=${frame.quality_score}` : ""}${
                workMode === "track"
                  ? `\nMascara pista: ${hasMask ? `SI (${(trackCov * 100).toFixed(1)}%)` : "NO"}`
                  : (frame.athlete_state ? `\nEstado: ${frame.athlete_state}` : "")
              }${workMode !== "track" && frame.track_overlap != null ? `\nColision: ${(frame.track_overlap * 100).toFixed(0)}%` : ""}${
                phaseName ? `\nFase: ${phaseLabels[phaseName] || phaseName}` : ""
              }${marker ? `\nMarcador: ${phaseLabels[marker.phase] || marker.phase}${marker.pose_tag ? ` (${marker.pose_tag})` : ""}` : ""}${
                inRange ? "\n[en rango de propagacion]" : ""
              }`}
              onClick={(e) => handleClick(e, index)}
            >
              <div
                className="absolute left-0 right-0 top-0"
                style={{
                  height: laneH.det,
                  background: detected ? "#22c55e" : "#1e2433",
                  opacity: detected ? 0.85 : 1,
                }}
              />
              <div
                className="absolute left-0 right-0"
                style={{
                  top: laneH.det,
                  height: laneH.ang,
                  background: angleColor,
                  opacity: detected ? 1 : 0.25,
                }}
              />
              <div
                className="absolute left-0 right-0"
                style={{
                  top: `calc(${laneH.det} + ${laneH.ang})`,
                  height: laneH.trk,
                  background: trackColor,
                  opacity: overlapAlpha,
                }}
              />
              {hasPhases ? (
                <div
                  className="absolute bottom-0 left-0 right-0"
                  style={{
                    height: laneH.ph,
                    background: phaseColor,
                    opacity: phaseName ? 0.95 : 0.2,
                  }}
                >
                  {marker ? (
                    <div
                      className="absolute left-1/2 top-0 z-10 h-full w-0.5 -translate-x-1/2 bg-white shadow-[0_0_4px_rgba(0,0,0,0.8)]"
                      title={`Marcador: ${phaseLabels[marker.phase] || marker.phase}`}
                    />
                  ) : null}
                </div>
              ) : null}
            </button>
          );
        })}

        <div
          className="absolute bottom-0 top-0 z-20 w-0.5 bg-red-400 pointer-events-none"
          style={{ left: `${((currentIndex + 0.5) / frames.length) * 100}%` }}
        />
      </div>
    </div>
  );
}

function Inspector({
  currentFrame,
  summary,
  analysisExists,
  onRunAnalysis,
  isAnalyzing,
  analysisJob,
  analysisLog,
  onRunReanalysis,
  isReanalyzing,
  reanalysisJob,
  refinedOutputDir,
  onOpenRefined,
  useCnnMasks,
  setUseCnnMasks,
  canUseCnnMasks,
  refineV2,
  setRefineV2,
  workMode,
  onSwitchWorkMode,
  correctionMode,
  setCorrectionMode,
  sotBackend,
  setSotBackend,
  stride,
  setStride,
  analysisStartSec,
  setAnalysisStartSec,
  analysisEndSec,
  setAnalysisEndSec,
  videoDuration,
  propagationEndFrame,
  setPropagationEndFrame,
  currentFrameIdx,
  onLoadDetections,
  isCorrecting,
  isLoadingDetections,
  calibration,
  trackSection,
  setTrackSection,
  trackCorrectionOp,
  setTrackCorrectionOp,
  onSubmitTrackCorrection,
  isCorrectingTrack,
  showTrackMaskOverlay,
  setShowTrackMaskOverlay,
  showSandMaskOverlay,
  setShowSandMaskOverlay,
  showAthleteOverlay,
  setShowAthleteOverlay,
  showAdvancedTrack,
  setShowAdvancedTrack,
  trackLearnInput,
  setTrackLearnInput,
  showPolygonOverlay,
  setShowPolygonOverlay,
  calibrationLayer,
  setCalibrationLayer,
  calibrationSubMode,
  setCalibrationSubMode,
  seedDraftPoints,
  setSeedDraftPoints,
  seedStep,
  setSeedStep,
  calDraftPoints,
  setCalDraftPoints,
  onCloseCalibrationPolygon,
  onSaveCalibration,
  onSaveSeedPoints,
  onPropagateCalibration,
  isPropagating,
  snapToLines,
  setSnapToLines,
  isSavingCalibration,
  onRecomputeTracking,
  isRecomputingTracking,
  venueProfile,
  venueModel,
  venueDataset,
  onLearnVenueProfile,
  onTrainVenueCnn,
  onApplyVenueProfile,
  isLearningVenue,
  isTrainingVenueCnn,
  isApplyingVenue,
  venueBrushLayer,
  setVenueBrushLayer,
  venueBrushPoints,
  setVenueBrushPoints,
  venueBrushByFrame,
  onSaveVenueBrush,
  sections,
  onAnalyzeSections,
  isAnalyzingSections,
  athleteId,
  setAthleteId,
  phaseMarkPhase,
  setPhaseMarkPhase,
  phaseMarkTag,
  setPhaseMarkTag,
  onMarkPhase,
  onMarkPhaseAtFrame,
  onRemovePhaseMarker,
  onPropagatePhases,
  isMarkingPhase,
  isPropagatingPhases,
  phasePlaceMode,
  setPhasePlaceMode,
  onSelectFrame,
  frames = [],
  metrics = null,
  videoName = null,
  outputDir = null,
  onRecomputeMetrics,
  isComputingMetrics = false,
  corridorMetersInput = "10",
  setCorridorMetersInput,
  onApplyCorridorScale,
  isApplyingOverrides = false,
}) {
  const currentPhase = currentFrame
    ? (currentFrame.phase || phaseForFrame(sections, currentFrame.frame_idx))
    : null;
  const contactCount = sections?.contacts?.length ?? 0;
  const currentKeyframe = currentFrame
    ? interpCalibrationKeyframe(calibration, currentFrame.frame_idx)
    : null;
  const seedCount = calibration?.seeds?.length ?? 0;
  const propagatedCount = calibrationKeyframesList(calibration).length;
  const maskFrameEntry = calibration?.mask_frames?.[String(currentFrame?.frame_idx ?? "")];
  const useMaskOverlay = hasMaskCalibration(calibration);
  const totalMaskFrames = maskFrameCount(calibration);
  const totalAnalysisFrames = summary?.total_frames_analyzed ?? 0;
  const hasCurrentMask = maskFrameEntry != null;
  const trackAreaNorm = useMaskOverlay
    ? maskFrameEntry?.track_area_norm
    : currentKeyframe?.venue_area_norm ?? polygonAreaNorm(currentKeyframe?.track_polygon);
  const sandAreaNorm = maskFrameEntry?.sand_area_norm;
  const venueConfidence = useMaskOverlay
    ? maskFrameEntry?.confidence
    : currentKeyframe?.venue_confidence;
  const trackAreaPct =
    trackAreaNorm != null ? (trackAreaNorm * 100).toFixed(1) : null;
  const sandAreaPct =
    sandAreaNorm != null ? (sandAreaNorm * 100).toFixed(1) : null;
  const trackAreaWarning = trackAreaNorm != null && trackAreaNorm > 0.3;
  const manualPolygons = countManualPolygonFrames(calibration);
  const canLearnVenue = hasLearnableVenueData(calibration, venueBrushByFrame, venueBrushPoints);
  const datasetTotalFrames = venueDataset?.total_frames ?? 0;
  const datasetVideoCount = venueDataset?.video_count ?? venueDataset?.videos?.length ?? 0;
  const canTrainVenueCnn = datasetTotalFrames >= 5;
  const datasetReadyHint = venueDataset?.ready_to_train ?? (datasetTotalFrames >= 10 && datasetVideoCount >= 1);
  const useKeyframePipeline = hasKeyframePolygonPipeline(calibration);
  const useCnnPipeline = Boolean(venueModel?.trained);
  const canApplyVenue = useCnnPipeline || useKeyframePipeline || venueProfile?.learned;
  const maskSource = maskFrameEntry?.source
    ?? (calibration?.mode === "color_masks" ? "color" : calibration?.mode === "cnn_masks" ? "cnn" : null);

  return (
    <aside className="h-full overflow-auto bg-editor-800">
      <PanelTitle>Modo de trabajo</PanelTitle>
      <div className="grid grid-cols-3 gap-1.5 px-2 pb-2">
        <WorkModeButton active={workMode === "athlete"} label="Atleta" onClick={() => onSwitchWorkMode("athlete")} />
        <WorkModeButton active={workMode === "track"} label="Pista" icon={<Map className="h-3 w-3" />} onClick={() => onSwitchWorkMode("track")} />
        <WorkModeButton active={workMode === "analisis"} label="Análisis" icon={<ChartColumn className="h-3 w-3" />} onClick={() => onSwitchWorkMode("analisis")} />
      </div>

      {workMode === "athlete" ? (
        <>
      <PanelTitle>Frame actual</PanelTitle>
      <div className="grid border-y border-editor-700">
        {currentFrame ? (
          <>
            <Stat label="Angulo" value={currentFrame.camera_angle} />
            <Stat label="Confianza" value={formatNumber(currentFrame.angle_confidence, 2)} />
            <Stat label="Ratio" value={formatNumber(currentFrame.shoulder_ratio, 3)} />
            <Stat label="Keypoints" value={`${currentFrame.keypoints_valid ?? "N/A"}/11`} />
            <Stat label="Tiempo" value={formatTime(currentFrame.timestamp_s)} />
            <Stat label="Persona" value={currentFrame.person_detected ? "Si" : "No"} />
            {currentFrame.track_overlap != null ? (
              <Stat label="Colision pista" value={`${Math.round(currentFrame.track_overlap * 100)}%`} />
            ) : null}
            {currentFrame.athlete_state ? (
              <Stat label="Estado atleta" value={currentFrame.athlete_state} />
            ) : null}
            {currentPhase ? (
              <Stat label="Fase" value={phaseLabels[currentPhase] || currentPhase} />
            ) : null}
            {sections ? (
              <Stat label="Contactos" value={`${contactCount}/5`} />
            ) : null}
          </>
        ) : null}
      </div>
        </>
      ) : null}

      {workMode === "track" ? (
        <>
      <PanelTitle>Pista — frame actual</PanelTitle>
      {hasMaskCalibration(calibration) ? (
        <div
          className={`mx-2 mb-2 border px-2 py-1.5 text-[12px] font-semibold ${
            hasCurrentMask
              ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-300"
              : "border-red-500/50 bg-red-500/10 text-red-300"
          }`}
        >
          {hasCurrentMask ? "✓ Mascara aplicada" : "✗ Sin mascara"}
          {hasCurrentMask && maskSource ? (
            <span className="ml-1 font-normal text-slate-300">· {maskSourceLabel(maskSource)}</span>
          ) : null}
        </div>
      ) : null}
      <div className="grid border-y border-editor-700">
        {currentFrame ? (
          <>
            <Stat label="Tiempo" value={formatTime(currentFrame.timestamp_s)} />
            {hasMaskCalibration(calibration) ? (
              <>
                <Stat label="Cobertura pista" value={trackAreaPct != null ? `${trackAreaPct}%` : "N/A"} />
                <Stat label="Cobertura arena" value={sandAreaPct != null ? `${sandAreaPct}%` : "N/A"} />
                <Stat label="Confianza mascara" value={venueConfidence != null ? formatNumber(venueConfidence, 2) : "N/A"} />
                <Stat label="Total mascaras" value={`${totalMaskFrames}/${totalAnalysisFrames || totalMaskFrames} frames`} />
              </>
            ) : (
              <>
                <Stat label="Cobertura pista" value={trackAreaPct != null ? `${trackAreaPct}%` : "N/A"} />
                <Stat label="Confianza mascara" value={venueConfidence != null ? formatNumber(venueConfidence, 2) : "N/A"} />
              </>
            )}
            <Stat label="Perfil aprendido" value={venueProfile?.learned ? "Si" : "No"} />
          </>
        ) : null}
      </div>
        </>
      ) : null}

      {workMode === "analisis" ? (
        <div className="mt-2 px-2 pb-3">
          <div className="mb-2 text-[11px] font-bold uppercase text-slate-300">Análisis — métricas</div>
          {!analysisExists ? (
            <div className="mb-2 border border-amber-500/50 bg-amber-500/10 px-2 py-2 text-[11px] leading-4 text-amber-100">
              Analiza el video primero
            </div>
          ) : contactCount < 5 ? (
            <div className="mb-2 border border-amber-500/50 bg-amber-500/10 px-2 py-2 text-[11px] leading-4 text-amber-100">
              Detecta o marca los 5 hops en Fases ({contactCount}/5). Sin 5 contactos no hay estadísticas completas.
            </div>
          ) : null}
          {analysisExists && contactCount >= 5 && !athleteId ? (
            <div className="mb-2 border border-slate-600/60 bg-editor-850 px-2 py-1.5 text-[10px] leading-4 text-slate-400">
              Opcional: define ID atleta en Fases para comparar consistencia vs historial.
            </div>
          ) : null}

          {analysisExists && contactCount >= 5 ? (
            <>
              <label className="mb-2 block text-[10px] text-slate-400">
                Longitud pista de hops (m)
                <input
                  className="mt-0.5 h-7 w-full border border-editor-600 bg-editor-900 px-2 text-[11px] text-slate-200"
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={corridorMetersInput}
                  onChange={(e) => setCorridorMetersInput(e.target.value)}
                  onBlur={() => {
                    const n = Number(String(corridorMetersInput).trim());
                    const current = metrics?.scale?.hops_corridor_m ?? metrics?.overrides?.hops_corridor_m;
                    if (Number.isFinite(n) && n > 0 && current != null && Math.abs(n - Number(current)) > 1e-6) {
                      onApplyCorridorScale?.();
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") onApplyCorridorScale?.();
                  }}
                />
              </label>
              <p className="mb-2 text-[10px] leading-4 text-slate-500">
                Escala: {metrics?.scale?.hops_corridor_m ?? corridorMetersInput} m de 1er hop → aterrizaje (por defecto 10 m)
              </p>
              {metrics?.segments?.length ? (
                <div className="mb-2 overflow-x-auto border border-editor-700 bg-editor-900">
                  <table className="w-full text-left text-[10px] text-slate-300">
                    <thead className="bg-editor-850 text-slate-400">
                      <tr>
                        <th className="px-1.5 py-1 font-medium">Segmento</th>
                        <th className="px-1.5 py-1 font-medium">t (s)</th>
                        <th className="px-1.5 py-1 font-medium">dist (m)</th>
                        <th className="px-1.5 py-1 font-medium">vel (m/s)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.segments.map((seg) => {
                        const dist = seg.length_m != null
                          ? seg.length_m.toFixed(2)
                          : "—";
                        const vel = seg.speed_m_s != null
                          ? seg.speed_m_s.toFixed(2)
                          : "—";
                        return (
                          <tr key={`${seg.id}-${seg.from_frame}-${seg.to_frame}`} className="border-t border-editor-800">
                            <td className="px-1.5 py-1">{segmentLabels[seg.id] || seg.label || seg.id}</td>
                            <td className="px-1.5 py-1">{seg.dt_s != null ? seg.dt_s.toFixed(3) : "—"}</td>
                            <td className="px-1.5 py-1">{dist}</td>
                            <td className="px-1.5 py-1">{vel}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="mb-2 text-[10px] text-slate-500">
                  {isComputingMetrics ? "Calculando métricas..." : "Sin segmentos aún. Pulsa recalcular si hace falta."}
                </div>
              )}
              <div className="mb-2 grid gap-0.5 border border-editor-700 bg-editor-850 px-2 py-1.5 text-[10px] text-slate-300">
                <div>
                  Total corredor hops:{" "}
                  {metrics?.total_hops_m != null
                    ? `${metrics.total_hops_m.toFixed(2)} m`
                    : "—"}
                </div>
                {metrics?.consistency?.overall != null ? (
                  <div className="text-slate-400">
                    vs Atleta / historial: {Math.round(metrics.consistency.overall * 100)}%
                    {metrics.consistency.sessions_compared
                      ? ` (${metrics.consistency.sessions_compared} sesión${metrics.consistency.sessions_compared === 1 ? "" : "es"})`
                      : ""}
                  </div>
                ) : athleteId ? (
                  <div className="text-slate-500">vs Atleta: sin historial/plantilla aún</div>
                ) : null}
              </div>

              {/* vs General */}
              <div className="mb-2 border border-editor-700 bg-editor-900">
                <div className="flex items-center justify-between border-b border-editor-800 bg-editor-850 px-2 py-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-300">
                    vs General
                  </span>
                  {metrics?.comparison?.vs_general?.overall != null ? (
                    <span className="text-[10px] text-slate-400">
                      Técnica {Math.round(metrics.comparison.vs_general.overall * 100)}%
                    </span>
                  ) : null}
                </div>
                {metrics?.comparison?.vs_general?.limited ||
                (metrics?.comparison?.vs_general?.notes || []).includes("Baseline general limitado") ? (
                  <div className="border-b border-editor-800 px-2 py-1 text-[10px] text-amber-200/90">
                    Baseline general limitado
                  </div>
                ) : null}
                {metrics?.comparison?.vs_general?.segments?.length ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-[10px] text-slate-300">
                      <thead className="text-slate-500">
                        <tr>
                          <th className="px-1.5 py-1 font-medium">Seg</th>
                          <th className="px-1.5 py-1 font-medium">Δt</th>
                          <th className="px-1.5 py-1 font-medium">Δv</th>
                          <th className="px-1.5 py-1 font-medium">Δd</th>
                          <th className="px-1.5 py-1 font-medium text-center">vs</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.comparison.vs_general.segments.map((row) => (
                          <tr key={`vg-${row.id}`} className="border-t border-editor-800">
                            <td className="px-1.5 py-1">
                              {compareSegmentLabels[row.id] || row.id}
                            </td>
                            <td className="px-1.5 py-1 tabular-nums">
                              {formatSignedDelta(row.dt_delta_s, 3, " s")}
                            </td>
                            <td className="px-1.5 py-1 tabular-nums">
                              {formatSignedDelta(row.speed_delta_ms, 2, " m/s")}
                            </td>
                            <td className="px-1.5 py-1 tabular-nums">
                              {formatSignedDelta(row.length_delta_m, 2, " m")}
                            </td>
                            <td className={`px-1.5 py-1 text-center font-semibold ${indicatorClass(row.indicator)}`}>
                              {indicatorGlyph(row.indicator)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="px-2 py-1.5 text-[10px] text-slate-500">
                    Sin comparación general aún. Recalcula métricas.
                  </div>
                )}
              </div>

              {/* Calidad de pose */}
              <div className="mb-2 border border-editor-700 bg-editor-900">
                <div className="flex items-center justify-between border-b border-editor-800 bg-editor-850 px-2 py-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-300">
                    Calidad de pose
                  </span>
                  {metrics?.comparison?.pose_quality?.overall != null ? (
                    <span className="text-[10px] text-slate-400">
                      {Math.round(metrics.comparison.pose_quality.overall * 100)}%
                    </span>
                  ) : null}
                </div>
                <div className="space-y-1.5 px-2 py-1.5">
                  {(metrics?.comparison?.pose_quality?.hops || []).map((hop) => {
                    const pct = hop.score != null ? Math.round(hop.score * 100) : null;
                    const label = hop.label || null;
                    return (
                      <div key={`pq-${hop.phase}-${hop.frame_idx}`}>
                        <div className="mb-0.5 flex items-center justify-between gap-1">
                          <span className="text-[10px] text-slate-300">
                            {poseHopLabels[hop.phase] || hop.phase}
                          </span>
                          <span
                            className={`rounded border px-1 py-px text-[9px] uppercase tracking-wide ${poseBadgeClass(label)}`}
                          >
                            {poseLabelText[label] || "—"}
                            {pct != null ? ` ${pct}` : ""}
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-sm bg-editor-800">
                          <div
                            className={`h-full ${poseBarClass(label)}`}
                            style={{ width: `${pct != null ? pct : 0}%` }}
                          />
                        </div>
                        <PoseOverlayThumb
                          videoName={videoName}
                          phase={hop.phase}
                          outputDir={outputDir}
                          cacheKey={metrics?.derived_version ?? hop.frame_idx}
                          title={`${poseHopLabels[hop.phase] || hop.phase} — General vs esta toma`}
                        />
                      </div>
                    );
                  })}
                  {(() => {
                    const ff = metrics?.comparison?.pose_quality?.final_flight;
                    if (!ff || (ff.score == null && !ff.from_frame)) {
                      return metrics?.comparison?.pose_quality?.hops?.length ? null : (
                        <div className="text-[10px] text-slate-500">
                          Sin scores de pose. Recalcula métricas.
                        </div>
                      );
                    }
                    const pct = ff.score != null ? Math.round(ff.score * 100) : null;
                    const label = ff.label || null;
                    const spark = (ff.samples || []).map((s) => s.score).filter((v) => v != null);
                    return (
                      <div className="border-t border-editor-800 pt-1.5">
                        <div className="mb-0.5 flex items-center justify-between gap-1">
                          <span className="text-[10px] text-slate-300">
                            Vuelo final (H4→aterrizaje)
                          </span>
                          <span
                            className={`rounded border px-1 py-px text-[9px] uppercase tracking-wide ${poseBadgeClass(label)}`}
                          >
                            {poseLabelText[label] || "—"}
                            {pct != null ? ` ${pct}` : ""}
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-sm bg-editor-800">
                          <div
                            className={`h-full ${poseBarClass(label)}`}
                            style={{ width: `${pct != null ? pct : 0}%` }}
                          />
                        </div>
                        {spark.length > 1 ? (
                          <div className="mt-1 flex h-5 items-end gap-px">
                            {spark.map((sc, i) => (
                              <div
                                key={`spark-${i}`}
                                className={`min-w-[3px] flex-1 rounded-sm ${poseBarClass(
                                  sc >= 0.75 ? "buena" : sc >= 0.55 ? "regular" : "débil",
                                )}`}
                                style={{ height: `${Math.max(12, Math.round(sc * 100))}%` }}
                                title={`${Math.round(sc * 100)}%`}
                              />
                            ))}
                          </div>
                        ) : null}
                        <PoseOverlayThumb
                          videoName={videoName}
                          phase="final_flight"
                          outputDir={outputDir}
                          cacheKey={metrics?.derived_version ?? ff.from_frame}
                          title="Vuelo final — General vs esta toma"
                        />
                      </div>
                    );
                  })()}
                </div>
              </div>

              <div className="grid gap-1.5">
                <button
                  className="h-7 w-full border border-emerald-600 bg-emerald-900/30 px-2 text-left text-[11px] text-emerald-100 disabled:opacity-50"
                  type="button"
                  disabled={isApplyingOverrides}
                  onClick={onApplyCorridorScale}
                >
                  {isApplyingOverrides ? "Aplicando..." : "Aplicar escala"}
                </button>
                <button
                  className="h-7 w-full border border-slate-500 bg-editor-850 px-2 text-left text-[11px] text-slate-200 disabled:opacity-50"
                  type="button"
                  disabled={isComputingMetrics}
                  onClick={onRecomputeMetrics}
                >
                  {isComputingMetrics ? "Recalculando..." : "Recalcular métricas"}
                </button>
              </div>
            </>
          ) : null}
        </div>
      ) : null}

      {workMode === "track" ? (
        <div className="mt-2 px-2">
          <div className="mb-2 grid grid-cols-4 gap-1.5">
            <WorkModeButton active={trackSection === "learn"} label="Aprender" onClick={() => setTrackSection("learn")} />
            <WorkModeButton active={trackSection === "apply"} label="Aplicar" onClick={() => setTrackSection("apply")} />
            <WorkModeButton active={trackSection === "correct"} label="Corregir" onClick={() => setTrackSection("correct")} />
            <WorkModeButton active={trackSection === "fases"} label="Fases" onClick={() => setTrackSection("fases")} />
          </div>

          {trackSection === "learn" ? (
            <>
              <div className="mb-2 grid grid-cols-2 gap-1.5">
                <WorkModeButton active={trackLearnInput === "brush"} label="Pincel" onClick={() => setTrackLearnInput("brush")} />
                <WorkModeButton active={trackLearnInput === "polygon"} label="Poligono" onClick={() => setTrackLearnInput("polygon")} />
              </div>

              {trackLearnInput === "brush" ? (
                <>
                  <div className="mb-2 text-[11px] text-slate-400">
                    Pinta muestras de color de pista (verde) y arena (amarillo) en el video.
                  </div>
                  <div className="mb-2 grid grid-cols-2 gap-1.5">
                    <CorrectionButton active={venueBrushLayer === "track"} icon={<Brush className="h-3.5 w-3.5" />} label="Pincel pista" onClick={() => setVenueBrushLayer("track")} />
                    <CorrectionButton active={venueBrushLayer === "sand"} icon={<Brush className="h-3.5 w-3.5" />} label="Pincel arena" onClick={() => setVenueBrushLayer("sand")} />
                  </div>
                  <div className="grid gap-1 border border-editor-700 bg-editor-850 px-2 py-1.5 text-[11px] text-slate-300">
                    <div>Capa: {venueBrushLayer === "sand" ? "Arena" : "Pista"}</div>
                    <div>Trazos en frame: {venueBrushPoints.length}</div>
                    <div>Frames con muestras: {Object.keys(venueBrushByFrame || {}).length}</div>
                    <div>Aprendido: {venueProfile?.learned ? "Si" : "No"}</div>
                    <div>Muestras: {venueProfile?.sample_count ?? "N/A"}</div>
                    <div>Videos en perfil: {(venueProfile?.videos_contributed || []).join(", ") || venueProfile?.source_video || "N/A"}</div>
                  </div>
                  <div className="mt-2 grid gap-1.5">
                    <button className="h-7 border border-editor-600 bg-editor-850 px-2 text-left text-[11px] text-slate-300" type="button" disabled={!venueBrushPoints.length} onClick={onSaveVenueBrush}>Guardar muestra del frame</button>
                    <button className="h-7 border border-editor-600 bg-editor-850 px-2 text-left text-[11px] text-slate-300" type="button" disabled={!venueBrushPoints.length} onClick={() => setVenueBrushPoints([])}>Limpiar pincel</button>
                  </div>
                </>
              ) : (
                <>
                  <div className="mb-2 text-[11px] leading-4 text-slate-400">
                    Los poligonos son la verdad de referencia (ground truth) para pista y arena.
                    Haz clic en el video para anadir vertices. Minimo 3 puntos. Cierra el poligono para guardar.
                    Al aplicar, las mascaras se generan desde estos poligonos (no solo color HSV).
                  </div>
                  <div className="mb-2 grid grid-cols-2 gap-1.5">
                    <CorrectionButton active={calibrationLayer === "track_polygon"} icon={<Map className="h-3.5 w-3.5" />} label="Poligono pista" onClick={() => setCalibrationLayer("track_polygon")} />
                    <CorrectionButton active={calibrationLayer === "landing_zone"} icon={<Map className="h-3.5 w-3.5" />} label="Poligono arena" onClick={() => setCalibrationLayer("landing_zone")} />
                  </div>
                  <div className="grid gap-1 border border-editor-700 bg-editor-850 px-2 py-1.5 text-[11px] text-slate-300">
                    <div>Capa activa: {calibrationLayer === "landing_zone" ? "Arena" : "Pista"}</div>
                    <div>Vertices en borrador: {calDraftPoints.length}</div>
                    <div>Poligonos manuales: {manualPolygons.track} frames (pista), {manualPolygons.sand} frames (arena)</div>
                    <div>Aprendido: {venueProfile?.learned ? "Si" : "No"}</div>
                  </div>
                  <div className="mt-2 grid gap-1.5">
                    <button className="h-7 border border-emerald-600 bg-emerald-900/30 px-2 text-left text-[11px] text-emerald-100 disabled:opacity-50" type="button" disabled={calDraftPoints.length < 3} onClick={onCloseCalibrationPolygon}>Cerrar poligono</button>
                    <button className="h-7 border border-accent bg-accent/20 px-2 text-left text-[11px] text-accent disabled:opacity-50" type="button" disabled={calDraftPoints.length < 3 || isSavingCalibration} onClick={onCloseCalibrationPolygon}>
                      {isSavingCalibration ? "Guardando..." : "Guardar poligono en frame"}
                    </button>
                    <button className="h-7 border border-editor-600 bg-editor-850 px-2 text-left text-[11px] text-slate-300" type="button" disabled={!calDraftPoints.length} onClick={() => setCalDraftPoints([])}>Limpiar borrador</button>
                  </div>
                </>
              )}

              <div className="mt-2 grid gap-1.5">
                <p className="text-[10px] leading-4 text-slate-500">
                  Flujo multi-video: dibuja polígonos en cada video, pulsa Aprender de este video,
                  repite en otros videos y al final Entrenar CNN una sola vez con todos los polígonos.
                </p>
                <button className="h-7 border border-violet-600 bg-violet-900/30 px-2 text-left text-[11px] text-violet-100 disabled:opacity-50" type="button" disabled={isLearningVenue || !canLearnVenue} onClick={onLearnVenueProfile}>
                  {isLearningVenue ? "Aprendiendo..." : "Aprender de este video"}
                </button>
                <button
                  className="h-7 border border-amber-600 bg-amber-900/30 px-2 text-left text-[11px] text-amber-100 disabled:opacity-50"
                  type="button"
                  disabled={isTrainingVenueCnn || !canTrainVenueCnn}
                  onClick={onTrainVenueCnn}
                >
                  {isTrainingVenueCnn ? "Entrenando CNN..." : "Entrenar CNN (pista/arena)"}
                </button>
                <div className="grid gap-1 border border-editor-700 bg-editor-850 px-2 py-1.5 text-[11px] text-slate-300">
                  <div className="font-semibold text-slate-200">Dataset CNN</div>
                  <div>Videos: {(venueDataset?.videos || []).map((v) => `${v.video_name} (${v.frames_exported})`).join(", ") || "ninguno"}</div>
                  <div>Frames totales: {datasetTotalFrames}</div>
                  <div>Listo para entrenar: {datasetReadyHint ? "Sí" : "No"} (mín. 10 frames)</div>
                </div>
                <div className="text-[10px] leading-4 text-slate-500">
                  Modelo CNN: {venueModel?.trained ? "entrenado" : "no entrenado"}
                  {venueModel?.trained_at ? ` · ${venueModel.trained_at.slice(0, 10)}` : ""}
                </div>
              </div>
            </>
          ) : null}

          {trackSection === "apply" ? (
            <>
              <div className="mb-2 text-[11px] text-slate-400">
                {useCnnPipeline
                  ? "Prioridad: modelo CNN entrenado (YOLO-seg) para pista y arena en cada frame."
                  : useKeyframePipeline
                    ? "Genera mascaras rasterizando los poligonos manuales (exactos, interpolados o con flujo optico entre keyframes)."
                    : "Genera mascaras de pista/arena para todos los frames analizados usando el perfil de color aprendido."}
              </div>
              <div className="grid gap-1 border border-editor-700 bg-editor-850 px-2 py-1.5 text-[11px] text-slate-300">
                <div>Modelo CNN: {venueModel?.trained ? "entrenado" : "no entrenado"}</div>
                <div>Poligonos manuales: {manualPolygons.track} frames (pista)</div>
                <div>Perfil color: {venueProfile?.learned ? "listo" : "sin aprender"}</div>
                <div>Mascaras en video: {useMaskOverlay ? `${totalMaskFrames}/${totalAnalysisFrames || totalMaskFrames}` : "0"} frames</div>
                <div>Cobertura frame actual: {trackAreaPct != null ? `${trackAreaPct}%` : "N/A"}</div>
                {maskSource ? <div>Origen frame actual: {maskSourceLabel(maskSource)}</div> : null}
              </div>
              <button
                className="mt-2 h-7 w-full border border-violet-600/60 bg-violet-900/15 px-2 text-left text-[11px] text-violet-200 disabled:opacity-50"
                type="button"
                disabled={isApplyingVenue || !canApplyVenue || !analysisExists}
                onClick={onApplyVenueProfile}
              >
                {isApplyingVenue
                  ? "Aplicando..."
                  : useCnnPipeline
                    ? "Aplicar mascaras CNN"
                    : useKeyframePipeline
                      ? "Generar mascaras desde poligonos"
                      : "Aplicar perfil de venue"}
              </button>
              <button className="mt-2 h-7 w-full border border-sky-600 bg-sky-900/30 px-2 text-left text-[11px] text-sky-100 disabled:opacity-50" type="button" disabled={isRecomputingTracking || !analysisExists || !useMaskOverlay} onClick={onRecomputeTracking}>
                {isRecomputingTracking ? "Recalculando..." : "Recalcular tracking atleta"}
              </button>
              <p className="mt-1 text-[10px] leading-4 text-slate-500">Recalcula colision pista/atleta tras aplicar mascaras.</p>
            </>
          ) : null}

          {trackSection === "correct" ? (
            <>
              <div className="mb-2 text-[11px] text-slate-400">Corrige mascaras aplicadas. Shift+Click en timeline fija el radio de propagacion.</div>
              <div className="mb-2 grid grid-cols-2 gap-1.5">
                <CorrectionButton active={trackCorrectionOp === "add_track"} icon={<Brush className="h-3.5 w-3.5" />} label="Anadir pista" onClick={() => setTrackCorrectionOp("add_track")} />
                <CorrectionButton active={trackCorrectionOp === "remove_track"} icon={<Brush className="h-3.5 w-3.5" />} label="Quitar pista" onClick={() => setTrackCorrectionOp("remove_track")} />
                <CorrectionButton active={trackCorrectionOp === "add_sand"} icon={<Brush className="h-3.5 w-3.5" />} label="Anadir arena" onClick={() => setTrackCorrectionOp("add_sand")} />
                <CorrectionButton active={trackCorrectionOp === "remove_sand"} icon={<Brush className="h-3.5 w-3.5" />} label="Quitar arena" onClick={() => setTrackCorrectionOp("remove_sand")} />
              </div>
              {propagationEndFrame !== null && currentFrameIdx !== null ? (
                <div className="mb-2 flex items-center gap-1 text-[11px] text-yellow-300">
                  <span className="flex-1 border border-yellow-500/40 bg-yellow-500/10 px-2 py-1">Propagacion: frame {currentFrameIdx} → {propagationEndFrame}</span>
                  <button className="h-[26px] border border-editor-600 bg-editor-850 px-2 text-slate-400" type="button" onClick={() => setPropagationEndFrame(null)}>✕</button>
                </div>
              ) : (
                <p className="mb-2 text-[10px] text-slate-500">Radio por defecto: ±15 frames. Shift+Click en timeline para cambiar.</p>
              )}
              <button className="h-7 w-full border border-accent bg-accent/20 px-2 text-left text-[11px] text-accent disabled:opacity-50" type="button" disabled={isCorrectingTrack || !useMaskOverlay} onClick={onSubmitTrackCorrection}>
                {isCorrectingTrack ? "Propagando..." : "Aplicar correccion y propagar"}
              </button>
            </>
          ) : null}

          {trackSection === "fases" ? (
            <>
              <div className="mb-2 text-[11px] leading-4 text-slate-400">
                Arrastra los pinos del timeline para corregir hops/salto/aterrizaje.
                Activa &quot;Colocar en timeline&quot; y haz clic en un frame para marcar la fase seleccionada.
              </div>
              <label className="mb-2 flex items-center gap-2 text-[11px] text-slate-300">
                <input
                  type="checkbox"
                  checked={phasePlaceMode}
                  onChange={(e) => setPhasePlaceMode(e.target.checked)}
                />
                Colocar en timeline (clic = marcar fase seleccionada)
              </label>
              <label className="mb-2 block text-[10px] text-slate-400">
                ID atleta (opcional, para comparar repetibilidad)
                <input
                  className="mt-0.5 h-7 w-full border border-editor-600 bg-editor-900 px-2 text-[11px] text-slate-200"
                  type="text"
                  value={athleteId}
                  placeholder="ej. mateo"
                  onChange={(e) => setAthleteId(e.target.value.trim())}
                />
              </label>
              <div className="mb-2 grid grid-cols-2 gap-1">
                <label className="text-[10px] text-slate-400">
                  Fase
                  <select
                    className="mt-0.5 h-7 w-full border border-editor-600 bg-editor-900 px-1 text-[11px] text-slate-200"
                    value={phaseMarkPhase}
                    onChange={(e) => setPhaseMarkPhase(e.target.value)}
                  >
                    {PHASE_OPTIONS.map((p) => (
                      <option key={p} value={p}>{phaseLabels[p] || p}</option>
                    ))}
                  </select>
                </label>
                <label className="text-[10px] text-slate-400">
                  Pose (opcional)
                  <select
                    className="mt-0.5 h-7 w-full border border-editor-600 bg-editor-900 px-1 text-[11px] text-slate-200"
                    value={phaseMarkTag}
                    onChange={(e) => setPhaseMarkTag(e.target.value)}
                  >
                    {POSE_TAG_OPTIONS.map((o) => (
                      <option key={o.value || "none"} value={o.value}>{o.label}</option>
                    ))}
                  </select>
                </label>
              </div>
              <button
                className="mb-1 h-8 w-full border border-violet-600 bg-violet-900/30 px-2 text-left text-[11px] font-semibold text-violet-100 disabled:opacity-50"
                type="button"
                disabled={isMarkingPhase || !currentFrame}
                onClick={onMarkPhase}
              >
                {isMarkingPhase ? "Guardando..." : `Marcar frame ${currentFrame?.frame_idx ?? "?"} como ${phaseLabels[phaseMarkPhase] || phaseMarkPhase}`}
              </button>
              <button
                className="mb-2 h-7 w-full border border-blue-600 bg-blue-900/30 px-2 text-left text-[11px] text-blue-100 disabled:opacity-50"
                type="button"
                disabled={isAnalyzingSections || !analysisExists}
                onClick={onAnalyzeSections}
              >
                {isAnalyzingSections ? "Detectando fases..." : "Detectar fases automaticamente"}
              </button>
              <button
                className="mb-2 h-7 w-full border border-amber-600 bg-amber-900/30 px-2 text-left text-[11px] text-amber-100 disabled:opacity-50"
                type="button"
                disabled={isPropagatingPhases || !sections?.phase_markers?.length}
                onClick={onPropagatePhases}
              >
                {isPropagatingPhases ? "Propagando..." : "Propag hops desde ancla (retroceder)"}
              </button>
              {sections?.notes ? (
                <div className="mb-2 border border-amber-500/40 bg-amber-500/10 px-2 py-1.5 text-[10px] leading-4 text-amber-200">
                  {sections.notes}
                </div>
              ) : null}
              <div className="grid gap-1 border border-editor-700 bg-editor-850 px-2 py-1.5 text-[11px] text-slate-300">
                <div>Fase actual: {currentPhase ? (phaseLabels[currentPhase] || currentPhase) : "N/A"}</div>
                <div>Contactos: {contactCount}/5</div>
                <div>Marcadores: {sections?.phase_markers?.length ?? 0}</div>
                <div>Confianza: {sections?.confidence != null ? `${Math.round(sections.confidence * 100)}%` : "N/A"}</div>
                <div className="text-[10px] text-slate-500">
                  Los marcadores manuales alimentan el GT de contactos (prototipos).
                </div>
              </div>
              {sections?.phase_markers?.length ? (
                <div className="mt-2 grid max-h-36 gap-1 overflow-y-auto border border-editor-700 bg-editor-850 px-2 py-1.5 text-[10px] text-slate-300">
                  <div className="font-semibold text-slate-200">Marcadores de fase</div>
                  {[...sections.phase_markers]
                    .sort((a, b) => a.frame_idx - b.frame_idx)
                    .map((m) => (
                      <div key={`${m.frame_idx}-${m.phase}`} className="flex items-center justify-between gap-1">
                        <button
                          type="button"
                          className="flex-1 text-left hover:text-white"
                          onClick={() => {
                            const idx = frames.findIndex((f) => f.frame_idx === m.frame_idx);
                            if (idx >= 0) onSelectFrame(idx);
                          }}
                        >
                          <span
                            className="mr-1 inline-block h-2 w-2 rounded-sm"
                            style={{ background: phaseColors[m.phase] || "#64748b" }}
                          />
                          f{m.frame_idx} · {phaseLabels[m.phase] || m.phase}
                          {m.pose_tag ? ` · ${m.pose_tag}` : ""}
                          {m.source ? ` · ${m.source}` : ""}
                        </button>
                        <button
                          type="button"
                          className="shrink-0 px-1 text-red-400 hover:text-red-300"
                          title="Quitar marcador"
                          onClick={() => onRemovePhaseMarker(m.frame_idx)}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                </div>
              ) : null}
              {sections?.contacts?.length ? (
                <div className="mt-2 grid gap-1 border border-editor-700 bg-editor-850 px-2 py-1.5 text-[10px] text-slate-300">
                  <div className="font-semibold text-slate-200">Contactos detectados</div>
                  {sections.contacts.map((c) => (
                    <div key={`${c.index}-${c.frame_idx}`}>
                      {c.index}. frame {c.frame_idx} · {c.surface} · {c.type}
                      {c.confidence != null ? ` · ${Math.round(c.confidence * 100)}%` : ""}
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          ) : null}

          <div className="mt-3 border-t border-editor-700 pt-2">
            <div className="mb-1 text-[11px] font-bold uppercase text-slate-300">Overlays</div>
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              <input type="checkbox" checked={showTrackMaskOverlay} onChange={(e) => setShowTrackMaskOverlay(e.target.checked)} />
              Mascara pista
            </label>
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              <input type="checkbox" checked={showSandMaskOverlay} onChange={(e) => setShowSandMaskOverlay(e.target.checked)} />
              Mascara arena
            </label>
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              <input type="checkbox" checked={showPolygonOverlay} onChange={(e) => setShowPolygonOverlay(e.target.checked)} />
              Poligonos manuales
            </label>
          </div>

          <div className="mt-2 border-t border-editor-700 pt-2">
            <button className="mb-2 w-full text-left text-[11px] text-slate-400" type="button" onClick={() => setShowAdvancedTrack((v) => !v)}>
              {showAdvancedTrack ? "▼ Ocultar avanzado" : "▶ Avanzado (semillas)"}
            </button>
            {showAdvancedTrack ? (
              <div className="grid gap-1.5">
                <div className="mb-1 text-[10px] leading-4 text-slate-500">
                  Marca 4 esquinas de la pista y propaga con flujo optico.
                </div>
                <button className="h-7 border border-emerald-600 bg-emerald-900/30 px-2 text-left text-[11px] text-emerald-100 disabled:opacity-50" type="button" disabled={seedDraftPoints.length < 4 || isSavingCalibration} onClick={onSaveSeedPoints}>Guardar semillas</button>
                <button className="h-7 border border-accent bg-accent/20 px-2 text-left text-[11px] text-accent disabled:opacity-50" type="button" disabled={isPropagating || seedCount < 1} onClick={onPropagateCalibration}>{isPropagating ? "Propagando..." : "Propagar semillas"}</button>
              </div>
            ) : null}
          </div>
        </div>
      ) : workMode === "athlete" ? (
      <div className="mt-2 px-2">
        <div className="mb-2 text-[11px] font-bold uppercase text-slate-300">Correccion</div>
        <div className="grid gap-1.5">
          <CorrectionButton
            active={correctionMode === "inspect"}
            icon={<Crosshair className="h-3.5 w-3.5" />}
            label="Inspeccionar"
            onClick={() => setCorrectionMode("inspect")}
          />
          <CorrectionButton
            active={correctionMode === "click_selection"}
            icon={<Scan className="h-3.5 w-3.5" />}
            label="Click-to-select"
            onClick={() => setCorrectionMode("click_selection")}
          />
          <CorrectionButton
            active={correctionMode === "bbox_correction"}
            icon={<SquareDashedMousePointer className="h-3.5 w-3.5" />}
            label="Bounding box"
            onClick={() => setCorrectionMode("bbox_correction")}
          />
          <CorrectionButton
            active={correctionMode === "mask_correction"}
            icon={<Brush className="h-3.5 w-3.5" />}
            label="Mask brush"
            onClick={() => setCorrectionMode("mask_correction")}
          />
          <button
            className="mt-1 h-6 border border-editor-600 bg-editor-850 px-2 text-left text-[11px] text-slate-300 disabled:opacity-60"
            type="button"
            disabled={isLoadingDetections || isCorrecting}
            onClick={onLoadDetections}
          >
            {isLoadingDetections ? "Cargando detecciones..." : "Ver detecciones del frame"}
          </button>
        </div>
        {isCorrecting ? (
          <div className="mt-2 border border-accent/40 bg-accent/10 px-2 py-1 text-[11px] text-slate-200">
            Aplicando correccion y actualizando frames afectados...
          </div>
        ) : null}
        <p className="mt-2 text-[11px] leading-4 text-slate-500">
          Click manda coordenadas. Bounding box arrastra un rectangulo. Mask brush pinta y luego aplica.
        </p>
        <div className="mt-3">
          <label className="mb-1 block text-[10px] font-bold uppercase text-slate-500">
            Propagacion despues de corregir
          </label>
          <select
            className="h-7 w-full border border-editor-600 bg-editor-850 px-2 text-[11px] text-slate-200 outline-none focus:border-accent"
            value={sotBackend}
            onChange={(event) => setSotBackend(event.target.value)}
            disabled={isCorrecting}
          >
            <option value="none">ByteTrack + appearance</option>
            <option value="csrt">SOT CSRT/MIL</option>
            <option value="sam2">SOT SAM2</option>
          </select>
          <p className="mt-1 text-[10px] leading-4 text-slate-500">
            CSRT usa fallback MIL si no hay opencv-contrib. SAM2 requiere checkpoint instalado.
          </p>
        </div>

        {/* Propagation range indicator */}
        <div className="mt-2">
          <label className="mb-1 block text-[10px] font-bold uppercase text-slate-500">
            Rango de propagacion
          </label>
          {propagationEndFrame !== null && currentFrameIdx !== null ? (
            <div className="flex items-center gap-1">
              <span className="flex-1 border border-yellow-500/40 bg-yellow-500/10 px-2 py-1 text-[11px] text-yellow-300">
                Frame {currentFrameIdx} → {propagationEndFrame}
              </span>
              <button
                className="h-[26px] border border-editor-600 bg-editor-850 px-2 text-[11px] text-slate-400"
                type="button"
                onClick={() => setPropagationEndFrame(null)}
              >
                ✕
              </button>
            </div>
          ) : (
            <p className="text-[10px] leading-4 text-slate-500">
              Shift+Click en el timeline para fijar el frame final de propagacion.
              Sin seleccion usa radio ±15.
            </p>
          )}
        </div>
      </div>
      ) : null}

      {workMode === "athlete" ? (
      <div className="mt-2 px-2 grid gap-1">
        <label className="flex items-center gap-2 text-[11px] text-slate-400">
          <input
            type="checkbox"
            checked={showAthleteOverlay}
            onChange={(event) => setShowAthleteOverlay(event.target.checked)}
          />
          Overlay atleta (bbox / estado)
        </label>
      </div>
      ) : null}

      <div className="mt-2 px-2">
        <div className="mb-1 text-[11px] font-bold uppercase text-slate-300">Analisis</div>
        <div className="mb-2 text-[10px] leading-4 text-slate-500">
          Recorta el tramo a analizar (salta relleno al inicio/fin del video).
          {videoDuration != null ? ` Duracion total: ${formatDuration(videoDuration)}.` : ""}
        </div>
        <div className="mb-2 grid grid-cols-2 gap-1.5">
          <label className="text-[10px] text-slate-400">
            Inicio (s)
            <input
              className="mt-0.5 h-6 w-full border border-editor-600 bg-editor-850 px-1.5 text-[11px] text-slate-200"
              type="number"
              min="0"
              step="0.1"
              value={analysisStartSec}
              disabled={isAnalyzing}
              onChange={(e) => setAnalysisStartSec(e.target.value)}
            />
          </label>
          <label className="text-[10px] text-slate-400">
            Fin (s)
            <input
              className="mt-0.5 h-6 w-full border border-editor-600 bg-editor-850 px-1.5 text-[11px] text-slate-200"
              type="number"
              min="0"
              step="0.1"
              placeholder="fin"
              value={analysisEndSec}
              disabled={isAnalyzing}
              onChange={(e) => setAnalysisEndSec(e.target.value)}
            />
          </label>
        </div>
        <div className="mb-2 flex flex-wrap gap-1">
          <button
            className="h-6 border border-editor-600 bg-editor-850 px-2 text-[10px] text-slate-300 disabled:opacity-50"
            type="button"
            disabled={!currentFrame || isAnalyzing}
            onClick={() => {
              if (currentFrame?.timestamp_s != null) {
                setAnalysisStartSec(Number(currentFrame.timestamp_s).toFixed(2));
              }
            }}
          >
            Inicio = playhead
          </button>
          <button
            className="h-6 border border-editor-600 bg-editor-850 px-2 text-[10px] text-slate-300 disabled:opacity-50"
            type="button"
            disabled={!currentFrame || isAnalyzing}
            onClick={() => {
              if (currentFrame?.timestamp_s != null) {
                setAnalysisEndSec(Number(currentFrame.timestamp_s).toFixed(2));
              }
            }}
          >
            Fin = playhead
          </button>
          <button
            className="h-6 border border-editor-600 bg-editor-850 px-2 text-[10px] text-slate-400 disabled:opacity-50"
            type="button"
            disabled={isAnalyzing}
            onClick={() => {
              setAnalysisStartSec("0");
              setAnalysisEndSec("");
            }}
          >
            Video completo
          </button>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[10px] uppercase text-slate-500 shrink-0">Stride</label>
          <select
            className="h-6 flex-1 border border-editor-600 bg-editor-850 px-1 text-[11px] text-slate-200 outline-none focus:border-accent"
            value={stride}
            onChange={(e) => setStride(Number(e.target.value))}
            disabled={isAnalyzing}
          >
            <option value={1}>1 — todos los frames</option>
            <option value={2}>2 — cada 2</option>
            <option value={3}>3 — cada 3 (rápido)</option>
            <option value={5}>5 — cada 5</option>
          </select>
          <button
            className="h-6 shrink-0 border border-editor-600 bg-editor-850 px-2 text-[11px] disabled:opacity-50"
            type="button"
            disabled={isAnalyzing}
            onClick={onRunAnalysis}
          >
            {isAnalyzing ? "Generando…" : "Generar"}
          </button>
        </div>
      </div>
      {isAnalyzing ? <AnalysisProgress job={analysisJob} /> : null}
      <div className="mt-2 grid border-y border-editor-700">
        {analysisExists ? (
          <>
            <Stat label="Frames analizados" value={summary.total_frames_analyzed} />
            <Stat label="Persona detectada" value={`${summary.detection_rate_pct ?? "N/A"}%`} />
            <Stat label="Angulo dominante" value={summary.dominant_angle} />
            <Stat label="Laterales utiles" value={`${summary.lateral_frames_pct ?? 0}%`} />
          </>
        ) : (
          <div className="p-2 text-slate-500">No hay analysis.json para este video.</div>
        )}
      </div>
      {analysisLog ? <pre className="m-2 max-h-32 overflow-auto bg-black p-2 text-[11px] text-slate-300">{analysisLog}</pre> : null}

      {/* Refined reanalysis */}
      {analysisExists ? (
        <div className="mt-2 px-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase text-slate-300">Segunda pasada</span>
            <div className="flex gap-1">
              {refinedOutputDir ? (
                <button
                  className="h-6 border border-purple-400 bg-purple-800/40 px-2 text-[11px] text-purple-200"
                  type="button"
                  onClick={onOpenRefined}
                >
                  Abrir refinado
                </button>
              ) : null}
              <button
                className="h-6 border border-purple-600 bg-purple-900/30 px-2 text-[11px] text-purple-300 disabled:opacity-40"
                type="button"
                disabled={isReanalyzing || isAnalyzing}
                onClick={onRunReanalysis}
                title="Re-analiza usando el modelo de apariencia calibrado (sin ByteTrack)"
              >
                {isReanalyzing ? "Refinando…" : "Refinar"}
              </button>
            </div>
          </div>
          <p className="text-[10px] leading-4 text-slate-500">
            {propagationEndFrame !== null && currentFrameIdx !== null && currentFrameIdx < propagationEndFrame
              ? `Semilla: frames ${currentFrameIdx}–${propagationEndFrame} (rango Shift+Click). `
              : "Sin rango: usa los mejores frames de todo el video. "}
            Shift+Click en el timeline para definir el intervalo de referencia.
          </p>
          <label
            className={`mt-1.5 flex items-start gap-1.5 text-[10px] leading-4 ${
              canUseCnnMasks ? "text-slate-300 cursor-pointer" : "text-slate-600 cursor-not-allowed"
            }`}
            title={
              canUseCnnMasks
                ? "Prefiere atletas sobre pista/arena usando máscaras de calibración"
                : "Aplica un perfil de venue (CNN / keyframes / color) con mask_frames primero"
            }
          >
            <input
              type="checkbox"
              className="mt-0.5"
              checked={Boolean(useCnnMasks && canUseCnnMasks)}
              disabled={!canUseCnnMasks || isReanalyzing || isAnalyzing}
              onChange={(e) => setUseCnnMasks(e.target.checked)}
            />
            <span>
              Usar máscaras CNN (pista/arena)
              {!canUseCnnMasks ? (
                <span className="block text-slate-600">
                  Sin mask_frames en calibración — aplica venue primero.
                </span>
              ) : null}
            </span>
          </label>
          <label
            className="mt-1.5 flex items-start gap-1.5 text-[10px] leading-4 text-slate-300 cursor-pointer"
            title="Consistencia temporal, semillas más seguras, update conservador y análisis de secciones tras el refinado"
          >
            <input
              type="checkbox"
              className="mt-0.5"
              checked={Boolean(refineV2)}
              disabled={isReanalyzing || isAnalyzing}
              onChange={(e) => setRefineV2(e.target.checked)}
            />
            <span>
              Refinado v2 (experimental)
              <span className="block text-slate-500">
                Temporal + semillas seguras + secciones. Desactívalo si empeora.
              </span>
            </span>
          </label>
          {isReanalyzing ? <AnalysisProgress job={reanalysisJob} /> : null}
        </div>
      ) : null}
    </aside>
  );
}

function PanelTitle({ children }) {
  return <div className="px-2 py-2 text-[11px] font-bold uppercase text-slate-300">{children}</div>;
}

function WorkModeButton({ active, label, icon, onClick }) {
  return (
    <button
      className={`inline-flex h-7 items-center justify-center gap-1 border px-2 text-[11px] ${
        active ? "border-accent bg-accent/15 text-slate-100" : "border-editor-600 bg-editor-850 text-slate-400"
      }`}
      type="button"
      onClick={onClick}
    >
      {icon}
      {label}
    </button>
  );
}

function CorrectionButton({ active, icon, label, onClick }) {
  return (
    <button
      className={`inline-flex h-7 items-center gap-2 border px-2 text-left text-[11px] ${
        active ? "border-accent bg-accent/15 text-slate-100" : "border-editor-600 bg-editor-850 text-slate-400"
      }`}
      type="button"
      onClick={onClick}
    >
      {icon}
      {label}
    </button>
  );
}

function AnalysisProgress({ job }) {
  const percent = Math.max(0, Math.min(100, Number(job?.percent || 0)));
  return (
    <div className="mx-2 mt-2 border border-accent/40 bg-accent/10 p-2 text-[11px] text-slate-200">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="truncate font-semibold">{job?.message || "Preparando analisis..."}</span>
        <span className="shrink-0 text-accent">{Math.round(percent)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden bg-editor-950">
        <div className="h-full bg-accent transition-all" style={{ width: `${percent}%` }} />
      </div>
      <div className="mt-2 grid gap-1 text-slate-400">
        <div>
          Etapa: <span className="text-slate-200">{job?.stage || "pending"}</span>
        </div>
        {job?.total_frames ? (
          <div>
            Frame: <span className="text-slate-200">{job.current_frame}/{job.total_frames}</span>
          </div>
        ) : null}
        {job?.analyzed_frames ? (
          <div>
            Frames analizados: <span className="text-slate-200">{job.analyzed_frames}</span>
          </div>
        ) : null}
        {job?.last_log ? <div className="truncate text-slate-500">{job.last_log}</div> : null}
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="grid grid-cols-[1fr_auto] gap-2 border-b border-editor-700 bg-editor-850 px-2 py-1.5 last:border-b-0">
      <span className="text-slate-500">{label}</span>
      <strong className="font-semibold text-slate-200">{value ?? "N/A"}</strong>
    </div>
  );
}

function DetectionDataPanel({ frame, compact = false }) {
  const bbox = frame?.person_bbox || frame?.bbox || null;
  const bboxText = Array.isArray(bbox) ? bbox.map((value) => Math.round(Number(value))).join(", ") : "N/A";
  const data = [
    ["Frame", frame?.frame_idx],
    ["Tiempo", formatTime(frame?.timestamp_s)],
    ["Persona detectada", frame?.person_detected ? "Si" : "No"],
    ["Track ID", frame?.track_id ?? "N/A"],
    ["Tracking source", frame?.tracking_source || "bytetrack"],
    ["BBox", bboxText],
    ["Area bbox", frame?.person_bbox_area ? `${Math.round(frame.person_bbox_area)} px` : "N/A"],
    ["Mascara", frame?.has_mask ? "Si" : "No"],
    ["Area mascara", frame?.mask_area_px ? `${frame.mask_area_px} px` : "N/A"],
    ["Keypoints validos", frame?.keypoints_valid ?? "N/A"],
    ["Quality score", formatNumber(frame?.quality_score, 3)],
    ["Usable", frame?.usable_for_analysis ? "Si" : "No"],
    ["Corregido manualmente", frame?.manually_corrected ? "Si" : "No"],
    ["Fuente correccion", frame?.correction_source || "auto"],
    ["Shoulder ratio", formatNumber(frame?.shoulder_ratio, 4)],
    ["Shoulder width", frame?.shoulder_width_px ? `${formatNumber(frame.shoulder_width_px, 1)} px` : "N/A"],
    ["Torso height", frame?.torso_height_px ? `${formatNumber(frame.torso_height_px, 1)} px` : "N/A"],
    ["Body height", frame?.body_height_px ? `${formatNumber(frame.body_height_px, 1)} px` : "N/A"],
    ["Confianza angulo", formatNumber(frame?.angle_confidence, 3)],
  ];

  return (
    <div className="h-full overflow-auto border-r border-black bg-editor-850 p-2">
      <div className="mb-2 text-[11px] font-bold uppercase text-slate-300">Datos de deteccion del frame</div>
      {frame ? (
        <div className={`grid ${compact ? "grid-cols-1" : "grid-cols-3"} gap-px overflow-hidden border border-editor-700 bg-editor-700`}>
          {data.map(([label, value]) => (
            <div className="grid min-h-9 grid-rows-[auto_1fr] bg-editor-900 px-2 py-1" key={label}>
              <span className="text-[10px] uppercase text-slate-500">{label}</span>
              <strong className="truncate text-[12px] font-semibold text-slate-200" title={String(value ?? "N/A")}>
                {value ?? "N/A"}
              </strong>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid h-[calc(100%-24px)] place-items-center border border-dashed border-editor-600 text-slate-500">
          Selecciona un frame para ver datos de deteccion.
        </div>
      )}
    </div>
  );
}

function BottomPanel({
  project,
  summary,
  frames,
  currentIndex,
  onSelectFrame,
  propagationEndFrame,
  onShiftSelect,
  workMode,
  trackSection,
  calibration,
  sections,
  phasePlaceMode,
  onMarkPhaseAtFrame,
  onMovePhaseMarker,
  phaseMarkPhase,
  isMarkingPhase,
}) {
  const distribution = Object.entries(summary.camera_angle_distribution || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
  const duration = project?.analysis?.data?.video_info?.duration_s;
  const markerCount = sections?.phase_markers?.length ?? 0;
  const timelineRows = markerCount > 0 || (workMode === "track" && trackSection === "fases") ? "116px" : "88px";

  return (
    <div className="grid h-full bg-editor-850" style={{ gridTemplateRows: `${timelineRows} 32px` }}>
      <div className="border-b border-black bg-editor-900">
        <Timeline
          frames={frames}
          currentIndex={currentIndex}
          onSelect={onSelectFrame}
          duration={duration}
          propagationEndFrame={propagationEndFrame}
          onShiftSelect={onShiftSelect}
          workMode={workMode}
          trackSection={trackSection}
          calibration={calibration}
          sections={sections}
          phasePlaceMode={phasePlaceMode}
          onMarkPhaseAtFrame={onMarkPhaseAtFrame}
          onMovePhaseMarker={onMovePhaseMarker}
          phaseMarkPhase={phaseMarkPhase}
          isMarkingPhase={isMarkingPhase}
        />
      </div>
      <div className="flex items-center gap-3 overflow-x-auto border-t border-editor-700 bg-editor-850 px-2 text-[11px] text-slate-400">
        <span className="shrink-0 font-bold uppercase text-slate-300">Distribucion</span>
        {distribution.length ? (
          distribution.map(([angle, pct]) => (
            <span className="inline-flex shrink-0 items-center gap-1" key={angle}>
              <span
                className="h-2 w-2 rounded-full"
                style={{ background: angleColors[angle] || angleColors.UNKNOWN }}
              />
              <strong className="font-semibold text-slate-200">{angle}</strong>
              <span>{pct}%</span>
            </span>
          ))
        ) : (
          <span className="text-slate-500">Sin distribucion disponible</span>
        )}
      </div>
    </div>
  );
}

export default App;
