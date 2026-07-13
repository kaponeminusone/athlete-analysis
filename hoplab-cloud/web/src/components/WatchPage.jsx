import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  BarChart3,
  Check,
  ChevronLeft,
  Eraser,
  Eye,
  Footprints,
  Home,
  PanelLeft,
  Paintbrush,
  PenTool,
  Settings,
  SkipBack,
  SkipForward,
  Undo2,
} from "lucide-react";
import AnnotationHUD from "./AnnotationHUD";
import AthleteRail from "./AthleteRail";
import BrushLayer from "./BrushLayer";
import PistaDraw from "./PistaDraw";
import SidePanel from "./SidePanel";
import Timeline from "./Timeline";
import {
  absUrl,
  analyzeSections,
  analyzeVideo,
  applyVenueMasks,
  computeMetrics,
  correctFrame,
  getCalibration,
  getSections,
  mediaUrl,
  movePhaseMarker,
  markPhase,
  pollJob,
  reanalyzeVideo,
  saveCalibration,
  scaleMetrics,
} from "../api/client";
import { loadWatchSession, sessionSuccessPct } from "../api/mapSession";
import { resolveSession, successTone } from "../mock/data";

/** Quita el sufijo _refined para operar sobre el primer pase. */
function baseVideoStem(name) {
  if (!name) return null;
  const s = String(name);
  return s.endsWith("_refined") ? s.slice(0, -"_refined".length) : s;
}

/**
 * Escenario de medios: object-contain + caja de overlays alineada al letterbox
 * (naturalWidth/Height → región visible). Brush/pista/máscaras usan esa caja.
 */
function MediaStage({ src, children }) {
  const wrapRef = useRef(null);
  const imgRef = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const [box, setBox] = useState(null);

  const measure = useCallback(() => {
    const wrap = wrapRef.current;
    const img = imgRef.current;
    if (!wrap || !img?.naturalWidth) return;
    const cw = wrap.clientWidth;
    const ch = wrap.clientHeight;
    if (cw <= 0 || ch <= 0) return;
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
    const width = img.naturalWidth * scale;
    const height = img.naturalHeight * scale;
    setBox({
      left: (cw - width) / 2,
      top: (ch - height) / 2,
      width,
      height,
    });
  }, []);

  useEffect(() => setLoaded(false), [src]);

  useLayoutEffect(() => {
    measure();
    const wrap = wrapRef.current;
    if (!wrap || typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(() => measure());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [measure, src, loaded]);

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden rounded-lg bg-black ring-1 ring-white/10">
      {!loaded && <div className="absolute inset-0 skeleton" />}
      <img
        ref={imgRef}
        src={src}
        alt=""
        className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-200 ${
          loaded ? "opacity-100" : "opacity-0"
        }`}
        onLoad={() => {
          setLoaded(true);
          measure();
        }}
        draggable={false}
      />
      {box && (
        <div
          className="absolute overflow-hidden"
          style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
        >
          {children}
        </div>
      )}
    </div>
  );
}

/** Capa de máscara real (PNG grises) tintada por luminancia — llena la caja letterbox. */
function MaskLayer({ src, color, opacity }) {
  return (
    <div
      className="absolute inset-0"
      style={{
        backgroundColor: color,
        opacity,
        maskImage: `url(${src})`,
        WebkitMaskImage: `url(${src})`,
        maskMode: "luminance",
        WebkitMaskMode: "luminance",
        maskSize: "100% 100%",
        WebkitMaskSize: "100% 100%",
        maskRepeat: "no-repeat",
        WebkitMaskRepeat: "no-repeat",
        maskPosition: "center",
        WebkitMaskPosition: "center",
        mixBlendMode: "screen",
      }}
    />
  );
}

/** Botón de acción del mini-header (estado activo/inactivo/deshabilitado). */
function ChromeButton({ icon: Icon, label, active, disabled, disabledHint, tone = "default", onClick }) {
  const activeCls =
    tone === "cyan"
      ? "bg-cyan-400/90 text-black ring-cyan-300"
      : tone === "ok"
        ? "bg-ok/90 text-black ring-ok"
        : "bg-accent/90 text-white ring-accent";
  return (
    <button
      type="button"
      data-chrome
      disabled={disabled}
      onClick={onClick}
      title={disabled ? disabledHint : label}
      className={`flex items-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] font-semibold ring-1 transition ${
        disabled
          ? "cursor-not-allowed bg-elevated/60 text-soft ring-border opacity-60"
          : active
            ? activeCls
            : "bg-elevated text-muted ring-border hover:text-text hover:ring-muted"
      }`}
    >
      <Icon className="h-3.5 w-3.5 shrink-0" />
      {label && <span className="hidden md:inline">{label}</span>}
    </button>
  );
}

export default function WatchPage({ athlete, session, autoAnalyze = false, onBack, onSelectSession, onSessionPatched, toast }) {
  const isApi = Boolean(session?.videoPath || session?.source === "api");

  const mockData = useMemo(
    () => (!isApi && session?.vod ? resolveSession(session) : null),
    [isApi, session],
  );

  const [data, setData] = useState(() => mockData);
  const [loadState, setLoadState] = useState(isApi ? "loading" : "ready"); // loading | ready | error
  const [reloadToken, setReloadToken] = useState(0);
  const outputDirRef = useRef(session.outputDir || null);

  const frames = data?.frames || [];
  const frameCount = frames.length;
  const hasMasks = Boolean(data?.hasMasks);
  const fpsMock = data?.fpsMock || 8;
  const successPct = data?.successPct ?? sessionSuccessPct(session);
  const tone = successTone(successPct);

  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [trackingOn, setTrackingOn] = useState(true);
  const [pistaOn, setPistaOn] = useState(false);
  const [activeTool, setActiveTool] = useState(null); // null | 'brush' | 'pista'
  const [railOpen, setRailOpen] = useState(false);
  const [sideOpen, setSideOpen] = useState(false);
  const [sideTab, setSideTab] = useState("config");
  const [scrubPreview, setScrubPreview] = useState(null);
  const [contacts, setContacts] = useState(mockData?.contacts || []);
  const [metrics, setMetrics] = useState(mockData?.metrics || { general: { totalTimeMs: 0, avgSpeed: 0 }, hops: [] });
  const [brushCount, setBrushCount] = useState(0);
  const [pistaCount, setPistaCount] = useState(0);
  const [correcting, setCorrecting] = useState(false);
  const [autoStart, setAutoStart] = useState(false);
  const autoStartDone = useRef(false);
  const [config, setConfig] = useState({
    stride: 2,
    range: "all",
    startSec: 0,
    endSec: null,
    refine: true,
    useVenueMap: true,
    hopsCorridorM: 10,
  });

  const pointerRef = useRef(null);
  const brushRef = useRef(null);
  const pistaRef = useRef(null);
  const rawVideoRef = useRef(null);
  /** Conserva el frame actual al recargar tras una corrección. */
  const keepFrameIndexRef = useRef(null);
  const [previewTimeSec, setPreviewTimeSec] = useState(0);
  const [previewDurationSec, setPreviewDurationSec] = useState(null);
  const [rawVideoError, setRawVideoError] = useState(false);

  // Nueva sesión → permitir otro autoAnalyze / autoStart.
  useEffect(() => {
    autoStartDone.current = false;
    setAutoStart(false);
    setPreviewTimeSec(0);
    setPreviewDurationSec(null);
    setRawVideoError(false);
  }, [session.id]);

  const applyWatchData = useCallback((next) => {
    setData(next);
    setContacts(next.contacts || []);
    setMetrics(next.metrics || { general: { totalTimeMs: 0, avgSpeed: 0 }, hops: [] });
    setConfig((c) => ({
      ...c,
      hopsCorridorM: next.hopsCorridorM ?? c.hopsCorridorM ?? 10,
    }));
    setFrameIndex(0);
    setPlaying(false);
    setActiveTool(null);
    setTrackingOn(true);
    setPistaOn(false);
    setRailOpen(false);
    setBrushCount(0);
    setPistaCount(0);
    if (!next.hasAnalysis) {
      setSideOpen(true);
      setSideTab("config");
    } else {
      setSideOpen(false);
    }
  }, []);

  // Al cambiar de sesión, reinicia output_dir preferido.
  useEffect(() => {
    outputDirRef.current = session.outputDir || null;
  }, [session.id, session.outputDir]);

  // Carga API o fixtures mock al cambiar de sesión / tras analizar.
  useEffect(() => {
    let cancelled = false;

    if (!isApi) {
      const resolved = session?.vod ? resolveSession(session) : null;
      if (resolved) {
        applyWatchData(resolved);
        setLoadState("ready");
      } else {
        setLoadState("error");
        toast("Sesión sin datos de video");
      }
      return undefined;
    }

    setLoadState("loading");
    (async () => {
      try {
        const next = await loadWatchSession({
          ...session,
          outputDir: outputDirRef.current || session.outputDir,
        });
        if (cancelled) return;
        applyWatchData(next);
        const keepIdx = keepFrameIndexRef.current;
        keepFrameIndexRef.current = null;
        if (keepIdx != null && next.frames?.length) {
          setFrameIndex(Math.max(0, Math.min(keepIdx, next.frames.length - 1)));
        }
        setLoadState("ready");
        onSessionPatched?.(session.id, {
          analysis: next.analysis,
          successPct: next.successPct,
          outputDir: next.outputDir,
          thumb: next.frames?.[0]?.annotated || session.thumb,
        });
      } catch (err) {
        if (cancelled) return;
        if (session.vod) {
          applyWatchData(resolveSession(session));
          setLoadState("ready");
          toast(`API falló — demo local (${err.message || "error"})`);
        } else {
          setLoadState("error");
          toast(`No se pudo cargar el proyecto: ${err.message || "error"}`);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.id, reloadToken]);

  const seek = useCallback(
    (i) => setFrameIndex(Math.max(0, Math.min(Math.max(frameCount - 1, 0), i))),
    [frameCount],
  );

  const step = useCallback(
    (dir) => {
      setPlaying(false);
      seek(frameIndex + dir);
    },
    [frameIndex, seek],
  );

  // Prefetch ±8 vecinos (raw + annotated si tracking). Ignora generaciones stale al scrubear rápido.
  const prefetchGen = useRef(0);
  useEffect(() => {
    if (!frames.length || frameCount < 2) return undefined;
    const gen = ++prefetchGen.current;
    const timer = window.setTimeout(() => {
      if (gen !== prefetchGen.current) return;
      for (let d = -8; d <= 8; d++) {
        if (d === 0) continue;
        const i = frameIndex + d;
        if (i < 0 || i >= frameCount) continue;
        const f = frames[i];
        if (!f) continue;
        const urls = [f.raw];
        if (trackingOn && f.annotated) urls.push(f.annotated);
        for (const u of urls) {
          if (!u) continue;
          const src = `${u}${u.includes("?") ? "&" : "?"}v=${reloadToken}`;
          const img = new Image();
          img.decoding = "async";
          img.src = src;
        }
      }
    }, 50);
    return () => {
      window.clearTimeout(timer);
    };
  }, [frameIndex, frameCount, frames, trackingOn, reloadToken]);

  // Reproducción ~fps de la secuencia de frames.
  useEffect(() => {
    if (!playing || frameCount < 2) return undefined;
    const id = window.setInterval(() => {
      setFrameIndex((i) => {
        if (i >= frameCount - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, Math.round(1000 / fpsMock));
    return () => window.clearInterval(id);
  }, [playing, frameCount, fpsMock]);

  // "Analizar" desde la biblioteca: abre panel y arranca el pipeline real una vez.
  useEffect(() => {
    if (loadState !== "ready" || !autoAnalyze || !isApi || autoStartDone.current) return;
    const path = session?.videoPath || data?.videoPath;
    if (!path) return;
    autoStartDone.current = true;
    if (data?.hasAnalysis) {
      setSideTab("stats");
      setSideOpen(true);
    } else {
      setSideTab("config");
      setSideOpen(true);
      setAutoStart(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadState, autoAnalyze, isApi, session?.videoPath, data?.videoPath, data?.hasAnalysis]);

  const handleAnalyze = useCallback(async (cfg, onProgress) => {
    const videoPath = session.videoPath || data?.videoPath;
    if (!videoPath) {
      toast("Sin ruta de video para analizar");
      throw new Error("Sin video_path");
    }
    const startSec = Number(cfg.startSec) || 0;
    const endSec =
      cfg.endSec != null && cfg.endSec !== "" ? Number(cfg.endSec) : null;
    if (endSec != null && Number.isFinite(endSec) && endSec <= startSec) {
      toast("El fin debe ser mayor que el inicio");
      throw new Error("Rango inválido");
    }

    const report = (pct, message) => onProgress?.(Math.round(pct), message);
    const wantsVenue = cfg.useVenueMap !== false;
    const wantsRefine = cfg.refine !== false;
    const fromScratch = !data?.hasAnalysis;
    // Desde cero siempre cadena completa; con análisis, refine según toggle.
    const runFullReady = fromScratch || wantsRefine;
    const skipFirstAnalyze = Boolean(data?.hasAnalysis && wantsRefine);

    let firstPassName =
      baseVideoStem(data?.videoName || session.videoName) ||
      baseVideoStem(session.id);
    let finalName = firstPassName;
    let finalOutputDir = outputDirRef.current || session.outputDir || null;

    // —— 0–40% análisis inicial ——
    if (!skipFirstAnalyze) {
      report(0, "Análisis inicial…");
      const start = await analyzeVideo({
        videoPath,
        stride: cfg.stride,
        startSec,
        endSec,
      });
      firstPassName = start.video_name || firstPassName;
      finalName = firstPassName;
      const job = await pollJob(start.job_id, {
        onProgress: (j) =>
          report(((j.percent || 0) / 100) * 40, j.message || "Análisis inicial…"),
      });
      firstPassName = job.result_video_name || firstPassName;
      finalName = firstPassName;
      // Tras primer pase, el output por defecto es output/<name>/ (sin _refined).
      finalOutputDir = null;
    } else {
      report(40, "Usando análisis existente…");
    }

    if (runFullReady) {
      // —— 40–50% mapa de pista (CNN / venue) ——
      if (wantsVenue) {
        report(42, "Mapeando pista con CNN…");
        try {
          await applyVenueMasks(firstPassName, { video_path: videoPath });
          report(50, "Mapa de pista aplicado");
        } catch (err) {
          toast(`Mapa de pista omitido: ${err.message || "sin CNN entrenada"}`);
          report(50, "Mapa omitido — continuo");
        }
      } else {
        report(50, "Sin mapa de pista");
      }

      // —— 50–90% refine_v2 (fases automáticas incluidas) ——
      report(52, "Refinando seguimiento…");
      const refineStart = await reanalyzeVideo({
        videoPath,
        stride: cfg.stride,
        startSec,
        endSec,
        useCnnMasks: wantsVenue,
        refineV2: true,
      });
      finalName = refineStart.video_name || `${firstPassName}_refined`;
      finalOutputDir = refineStart.output_dir || null;
      const refineJob = await pollJob(refineStart.job_id, {
        onProgress: (j) =>
          report(50 + ((j.percent || 0) / 100) * 40, j.message || "Refinando…"),
      });
      finalName = refineJob.result_video_name || finalName;
      report(90, "Refinado listo");

      // Re-aplicar máscaras sobre el output refinado (stride puede diferir del 1er pase)
      if (wantsVenue) {
        report(90, "Actualizando mapa de pista (refinado)…");
        try {
          await applyVenueMasks(finalName, { video_path: videoPath });
          report(91, "Mapa de pista actualizado");
        } catch (err) {
          toast(`Mapa refinado omitido: ${err.message || "sin CNN entrenada"}`);
          report(91, "Mapa refinado omitido — continuo");
        }
      }

      // Fallback fases si refine_v2 no escribió sections
      try {
        const secs = await getSections(finalName).catch(() => null);
        const hasContacts =
          (secs?.contacts?.length || 0) > 0 || (secs?.phase_markers?.length || 0) > 0;
        if (!hasContacts) {
          report(92, "Detectando fases…");
          await analyzeSections(finalName, true);
        }
      } catch (secErr) {
        toast(`Fases: ${secErr.message || "no detectadas"}`);
      }
    } else {
      // Solo primer pase: intentar fases + métricas
      report(70, "Detectando fases…");
      try {
        await analyzeSections(finalName, true);
      } catch (secErr) {
        toast(`Fases: ${secErr.message || "no detectadas"}`);
      }
    }

    // —— 90–100% métricas ——
    report(92, "Calculando métricas…");
    try {
      await computeMetrics(finalName, data?.athleteId || session.athleteId);
    } catch (err) {
      toast(`Métricas: ${err.message || "sin secciones aún"}`);
    }
    report(100, "Listo");

    outputDirRef.current = finalOutputDir;
    onSessionPatched?.(session.id, {
      analysis: "partial",
      videoName: finalName,
      outputDir: finalOutputDir || undefined,
      note: runFullReady ? "Análisis + refine listo" : "Análisis listo",
    });
    setReloadToken((t) => t + 1);
  }, [data, session, toast, onSessionPatched]);

  async function handleApplyScale(meters) {
    const videoName = data?.videoName || session.videoName;
    if (!videoName) {
      toast("Sin video para escalar");
      return;
    }
    await scaleMetrics(videoName, meters, data?.athleteId);
    const next = await loadWatchSession({
      ...session,
      videoName,
      videoPath: data?.videoPath || session.videoPath,
      outputDir: data?.outputDir || session.outputDir,
    });
    applyWatchData(next);
    onSessionPatched?.(session.id, { successPct: next.successPct });
    toast(`Escala actualizada: ${meters} m`);
  }

  const togglePlay = useCallback(() => setPlaying((p) => !p), []);

  function toggleTool(tool) {
    setActiveTool((prev) => (prev === tool ? null : tool));
    setPlaying(false);
    setSideOpen(false);
  }

  async function acceptTool() {
    if (activeTool === "pista") {
      const ok = pistaRef.current?.accept();
      if (ok) setActiveTool(null);
      return;
    }
    if (activeTool !== "brush") return;

    const payload = brushRef.current?.accept();
    if (!payload?.mask) return;

    if (!isApi) {
      toast("Corrección del atleta aplicada (demo local)");
      brushRef.current?.clear();
      setActiveTool(null);
      return;
    }

    const videoPath = data?.videoPath || session.videoPath;
    if (!videoPath || frame == null) {
      toast("Sin video o frame para corregir");
      return;
    }

    setCorrecting(true);
    try {
      const result = await correctFrame({
        videoPath,
        frameIdx: frame.frameId,
        type: "mask_correction",
        data: { mask: payload.mask },
        outputDir: data?.outputDir || outputDirRef.current || undefined,
        propagationRadius: 15,
        sotBackend: "none",
      });
      if (result.pose_warning) {
        toast(`Sin pose en este frame: ${result.pose_warning}`);
      } else {
        toast("Corrección del atleta aplicada");
      }
      brushRef.current?.clear();
      setActiveTool(null);
      keepFrameIndexRef.current = frameIndex;
      setReloadToken((t) => t + 1);
    } catch (err) {
      toast(err.message || "Error al corregir el atleta");
    } finally {
      setCorrecting(false);
    }
  }
  function clearTool() {
    if (activeTool === "brush") brushRef.current?.clear();
    else pistaRef.current?.clear();
  }

  /**
   * Guarda el polígono de pista dibujado como keyframe de calibración.
   * PistaDraw entrega puntos en % (0..100); la calibración usa coords
   * normalizadas 0..1. TODO(payload): asumimos que el polígono es `track_polygon`
   * del frame actual; se hace merge con los keyframes existentes.
   */
  async function submitPista(points) {
    if (!isApi) {
      toast("Editar pista: disponible solo con motor (API)");
      return;
    }
    const videoName = data?.videoName || session.videoName;
    if (!videoName || frame == null) {
      toast("Sin video o frame para calibrar la pista");
      return;
    }
    const poly = points.map((p) => [
      Number((p.x / 100).toFixed(5)),
      Number((p.y / 100).toFixed(5)),
    ]);
    const keyframe = { frame_idx: frame.frameId, track_polygon: poly, source: "manual" };

    setCorrecting(true);
    try {
      let existing = null;
      try {
        existing = await getCalibration(videoName);
      } catch {
        existing = null;
      }
      const keyframes = Array.isArray(existing?.keyframes) ? [...existing.keyframes] : [];
      const at = keyframes.findIndex((k) => k.frame_idx === frame.frameId);
      if (at >= 0) keyframes[at] = { ...keyframes[at], ...keyframe };
      else keyframes.push(keyframe);

      await saveCalibration(videoName, {
        version: existing?.version ?? 2,
        video: existing?.video || `${baseVideoStem(videoName)}.mp4`,
        keyframes,
        mode: existing?.mode,
        seeds: existing?.seeds,
        propagation: existing?.propagation,
      });
      toast(`Pista guardada · frame ${frame.frameId} (${points.length} puntos)`);
    } catch (err) {
      toast(err.message || "Error al guardar la pista");
    } finally {
      setCorrecting(false);
    }
  }

  function stepBack() {
    if (activeTool) return setActiveTool(null);
    if (sideOpen) return setSideOpen(false);
    if (railOpen) return setRailOpen(false);
    onBack();
  }

  const openPanel = useCallback((tab) => {
    setSideTab(tab);
    setSideOpen(true);
    setActiveTool(null);
  }, []);

  // Scrub horizontal + tap-para-play/pausa — SOLO sobre el área de medios.
  function onStagePointerDown(e) {
    if (activeTool) return;
    if (e.button !== 0 && e.pointerType === "mouse") return;
    if (e.target.closest?.("[data-chrome]")) return;
    pointerRef.current = { x0: e.clientX, y0: e.clientY, i0: frameIndex, moved: false };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
  function onStagePointerMove(e) {
    const p = pointerRef.current;
    if (!p) return;
    const dx = e.clientX - p.x0;
    const dy = e.clientY - p.y0;
    if (Math.abs(dx) > 6 || Math.abs(dy) > 6) p.moved = true;
    if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 4) {
      const next = Math.max(0, Math.min(frameCount - 1, p.i0 + Math.round(-dx / 12)));
      setScrubPreview(next);
      setPlaying(false);
    }
  }
  function onStagePointerUp(e) {
    const p = pointerRef.current;
    pointerRef.current = null;
    if (!p) return;
    if (scrubPreview != null) {
      seek(scrubPreview);
      setScrubPreview(null);
      return;
    }
    if (!p.moved) {
      const dx = Math.abs(e.clientX - p.x0);
      const dy = Math.abs(e.clientY - p.y0);
      if (dx < 5 && dy < 5) togglePlay();
    }
  }

  function addPhase(index, type) {
    const frame = frames[index];
    if (!frame) return;
    const hopCount = contacts.filter((c) => c.type === "hop").length;
    const phase =
      type === "landing"
        ? "landing"
        : type === "approach"
          ? "approach"
          : `hop_${Math.min(4, hopCount + 1)}`;
    const label =
      type === "landing" ? "LA" : type === "approach" ? "CAR" : `H${Math.min(4, hopCount + 1)}`;

    setContacts((prev) => [
      ...prev,
      { id: `m-${Date.now()}`, label, type, frameId: frame.frameId, index, phase },
    ]);
    toast(`Fase “${label}” creada en frame ${frame.frameId}`);

    const videoName = data?.videoName || session.videoName;
    if (isApi && videoName) {
      markPhase(videoName, {
        frameIdx: frame.frameId,
        phase,
        athleteId: data?.athleteId,
      }).catch((err) => toast(`Marcador local · API: ${err.message}`));
    }
  }

  function moveContact(id, index) {
    const frame = frames[index];
    if (!frame) return;
    let label = id;
    let fromFrameId = null;
    setContacts((prev) =>
      prev.map((c) => {
        if (c.id !== id) return c;
        label = c.label;
        fromFrameId = c.frameId;
        return { ...c, index, frameId: frame.frameId };
      }),
    );
    setMetrics((prev) => ({
      ...prev,
      hops: (prev.hops || []).map((h) => (h.id === id ? { ...h, index, frameId: frame.frameId } : h)),
    }));
    toast(`${label} movido a frame ${frame.frameId}`);

    const videoName = data?.videoName || session.videoName;
    if (isApi && videoName && fromFrameId != null && fromFrameId !== frame.frameId) {
      movePhaseMarker(videoName, fromFrameId, frame.frameId).catch((err) =>
        toast(`Movido en UI · API: ${err.message}`),
      );
    }
  }

  const displayIndex = scrubPreview ?? frameIndex;
  const frame = frames[displayIndex];
  const rawVideoSrc = useMemo(() => {
    const projectUrl = data?.project?.video?.url;
    if (projectUrl) {
      const u = String(projectUrl);
      if (u.startsWith("http://") || u.startsWith("https://")) return u;
      return absUrl(u.startsWith("/") ? u : `/${u}`);
    }
    const path = data?.videoPath || session?.videoPath;
    return path ? mediaUrl(path) : "";
  }, [data?.project?.video?.url, data?.videoPath, session?.videoPath]);

  useEffect(() => {
    setRawVideoError(false);
    setPreviewTimeSec(0);
    setPreviewDurationSec(null);
  }, [rawVideoSrc]);

  const currentTimeSec = frame?.timeMs != null ? frame.timeMs / 1000 : previewTimeSec;
  const durationSec =
    frames.length > 0 && frames[frames.length - 1]?.timeMs != null
      ? frames[frames.length - 1].timeMs / 1000
      : data?.project?.video?.duration_s ?? previewDurationSec;
  const frameSrcRaw = frame ? (trackingOn ? frame.annotated : frame.raw) : "";
  const frameSrc = frameSrcRaw
    ? `${frameSrcRaw}${frameSrcRaw.includes("?") ? "&" : "?"}v=${reloadToken}`
    : "";
  const toolActive = activeTool != null;
  const canAccept = activeTool === "brush" ? brushCount > 0 : pistaCount > 0;
  const canClear = canAccept && !correcting;
  const hasAnalysis = Boolean(data?.hasAnalysis);
  // Videos recién subidos: GET /api/project OK pero sin analysis.json → 0 frames.
  // Hay que montar SidePanel para Analizar / autoAnalyze (no bloquear con empty-state).
  const canAnalyzePending =
    isApi && Boolean(session?.videoPath || data?.videoPath) && frameCount === 0;

  if (loadState === "loading") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 bg-bg text-muted">
        <div className="h-10 w-10 animate-pulse rounded-full bg-elevated ring-1 ring-border" />
        <p className="text-sm">Cargando proyecto…</p>
      </div>
    );
  }

  if (loadState === "error" || (!frame && !canAnalyzePending)) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 bg-bg px-6 text-center">
        <p className="text-sm text-muted">No hay frames para esta sesión.</p>
        <button
          type="button"
          onClick={onBack}
          className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white"
        >
          Volver a la biblioteca
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-bg">
      {/* Mini-header: navegación + grupo de acciones */}
      <header className="no-select flex h-12 shrink-0 items-center gap-2 border-b border-border bg-surface px-3">
        <button
          type="button"
          onClick={onBack}
          title="Inicio (biblioteca)"
          className="flex h-8 w-8 items-center justify-center rounded-md text-muted transition hover:bg-elevated hover:text-text"
        >
          <Home className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={stepBack}
          title="Atrás"
          className="flex h-8 w-8 items-center justify-center rounded-md text-muted transition hover:bg-elevated hover:text-text"
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={() => setRailOpen((v) => !v)}
          title="Sesiones del atleta"
          className={`flex h-8 w-8 items-center justify-center rounded-md transition ${
            railOpen ? "bg-elevated text-text" : "text-muted hover:bg-elevated hover:text-text"
          }`}
        >
          <PanelLeft className="h-4 w-4" />
        </button>

        <div className="ml-1 hidden min-w-0 items-center gap-2 sm:flex">
          <span
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full font-display text-[10px] font-bold text-white"
            style={{ backgroundColor: athlete.accent }}
          >
            {athlete.initials}
          </span>
          <div className="min-w-0">
            <p className="truncate text-xs font-semibold leading-tight text-text">{session.title}</p>
            <p className="truncate text-[10px] leading-tight text-soft">
              {athlete.short} · {session.date || session.videoName || "—"}
            </p>
          </div>
          {successPct != null && (
            <span className={`rounded-md px-1.5 py-0.5 text-[11px] font-bold tabular-nums ring-1 ${tone.bg} ${tone.text} ${tone.ring}`}>
              {successPct}%
            </span>
          )}
        </div>

        {/* Grupo de acciones (derecha) */}
        <div className="ml-auto flex items-center gap-1.5">
          {toolActive && (
            <div className="mr-1 flex items-center gap-1.5 rounded-md bg-black/30 p-1">
              {activeTool === "pista" && (
                <ChromeButton icon={Undo2} label="Deshacer" disabled={pistaCount === 0} onClick={() => pistaRef.current?.undo()} />
              )}
              <ChromeButton icon={Eraser} label="Limpiar" disabled={!canClear} onClick={clearTool} />
              <ChromeButton
                icon={Check}
                label={correcting ? "Aplicando…" : "Aceptar"}
                tone="ok"
                active={canAccept && !correcting}
                disabled={!canAccept || correcting}
                onClick={acceptTool}
              />
            </div>
          )}

          <ChromeButton
            icon={Footprints}
            label="Seguimiento"
            active={trackingOn}
            disabled={!frame}
            disabledHint="Sin frames — analiza primero"
            onClick={() => setTrackingOn((v) => !v)}
          />
          <ChromeButton
            icon={Eye}
            label="Pista"
            active={pistaOn}
            disabled={!hasMasks || !frame}
            disabledHint={!frame ? "Sin frames — analiza primero" : "Sin mapa de pista"}
            onClick={() => setPistaOn((v) => !v)}
          />
          <span className="mx-0.5 h-5 w-px bg-border" />
          <ChromeButton
            icon={Paintbrush}
            label="Corregir atleta"
            tone="cyan"
            active={activeTool === "brush"}
            disabled={!frame}
            disabledHint="Sin frames — analiza primero"
            onClick={() => toggleTool("brush")}
          />
          <ChromeButton
            icon={PenTool}
            label="Editar pista"
            tone="cyan"
            active={activeTool === "pista"}
            disabled={!frame}
            disabledHint="Sin frames — analiza primero"
            onClick={() => toggleTool("pista")}
          />
          <span className="mx-0.5 h-5 w-px bg-border" />
          <ChromeButton
            icon={Settings}
            label="Config"
            active={sideOpen && sideTab === "config"}
            onClick={() => (sideOpen && sideTab === "config" ? setSideOpen(false) : openPanel("config"))}
          />
          <ChromeButton
            icon={BarChart3}
            label="Datos"
            active={sideOpen && sideTab === "stats"}
            onClick={() =>
              sideOpen && sideTab === "stats"
                ? setSideOpen(false)
                : openPanel(hasAnalysis ? "stats" : "config")
            }
          />
        </div>
      </header>

      {/* Escenario: grid encoge la columna del video al abrir paneles */}
      <div className="relative min-h-0 flex-1 overflow-hidden bg-black">
        <div
          className="absolute inset-0 grid transition-[grid-template-columns] duration-[280ms] ease-[cubic-bezier(0.22,1,0.36,1)]"
          style={{
            gridTemplateColumns: `${railOpen ? "min(280px,80vw)" : "0px"} minmax(0,1fr) ${
              sideOpen ? "min(360px,88vw)" : "0px"
            }`,
          }}
        >
          <div aria-hidden className="min-w-0 overflow-hidden" />
          <div
            className={`relative flex min-h-0 min-w-0 select-none items-center justify-center p-2 sm:p-3 ${
              frame ? "no-select touch-none" : ""
            }`}
            onSelectStart={frame ? (e) => e.preventDefault() : undefined}
            onDragStart={frame ? (e) => e.preventDefault() : undefined}
            onPointerDown={frame ? onStagePointerDown : undefined}
            onPointerMove={frame ? onStagePointerMove : undefined}
            onPointerUp={frame ? onStagePointerUp : undefined}
            onPointerCancel={() => {
              pointerRef.current = null;
              setScrubPreview(null);
            }}
          >
            <div className="relative h-full w-full max-w-full">
              {frame ? (
                <MediaStage src={frameSrc}>
                  {pistaOn && hasMasks && frame.trackMask && (
                    <>
                      <MaskLayer src={frame.trackMask} color="#22d3ee" opacity={0.42} />
                      <MaskLayer src={frame.sandMask} color="#f5a524" opacity={0.5} />
                    </>
                  )}

                  <AnnotationHUD frame={frame} visible={!playing || scrubPreview != null} />

                  <BrushLayer ref={brushRef} active={activeTool === "brush" && !correcting} onCountChange={setBrushCount} />
                  <PistaDraw
                    ref={pistaRef}
                    active={activeTool === "pista"}
                    onCountChange={setPistaCount}
                    onAccept={submitPista}
                    toast={toast}
                  />
                </MediaStage>
              ) : rawVideoSrc ? (
                <div className="relative h-full w-full overflow-hidden rounded-lg bg-black ring-1 ring-white/10">
                  <video
                    ref={rawVideoRef}
                    key={rawVideoSrc}
                    controls
                    playsInline
                    preload="metadata"
                    src={rawVideoSrc}
                    className="absolute inset-0 h-full w-full object-contain"
                    onLoadedMetadata={(e) => {
                      const d = e.currentTarget.duration;
                      if (Number.isFinite(d) && d > 0) setPreviewDurationSec(d);
                      setPreviewTimeSec(e.currentTarget.currentTime || 0);
                      setRawVideoError(false);
                    }}
                    onTimeUpdate={(e) => {
                      setPreviewTimeSec(e.currentTarget.currentTime || 0);
                    }}
                    onError={() => setRawVideoError(true)}
                  />
                  <div className="pointer-events-none absolute inset-x-0 top-0 z-10 bg-black/70 px-3 py-2.5 backdrop-blur-[2px]">
                    <div className="pointer-events-auto mx-auto flex max-w-xl flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
                      <div className="min-w-0 text-left">
                        <p className="text-sm font-medium text-text">Sin análisis todavía</p>
                        <p className="mt-0.5 text-[11px] leading-snug text-muted">
                          Reproduce el video para elegir el rango de análisis (inicio/fin) en el panel. Luego
                          configura y pulsa Analizar.
                        </p>
                        <p className="mt-1 text-[11px] leading-snug text-soft">
                          La primera reproducción por el túnel puede tardar cerca de 1 minuto en archivos de
                          varios MB.
                        </p>
                        {rawVideoError && (
                          <p className="mt-1 text-[11px] text-warn">
                            No se pudo cargar el video desde el motor. Revisa la URL del túnel.
                          </p>
                        )}
                      </div>
                      <div className="flex shrink-0 flex-wrap items-center gap-2">
                        <button
                          type="button"
                          data-chrome
                          onClick={() => {
                            setSideTab("config");
                            setSideOpen(true);
                            setAutoStart(true);
                          }}
                          className="rounded-md bg-accent px-3 py-1.5 text-xs font-semibold text-white"
                        >
                          Analizar ahora
                        </button>
                        <button
                          type="button"
                          data-chrome
                          onClick={() => openPanel("config")}
                          className="rounded-md bg-elevated px-3 py-1.5 text-xs font-semibold text-text ring-1 ring-border"
                        >
                          Abrir configuración
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex h-full w-full flex-col items-center justify-center gap-3 rounded-lg bg-black/80 px-6 text-center ring-1 ring-white/10">
                  <p className="text-sm font-medium text-text">Sin análisis todavía</p>
                  <p className="max-w-sm text-xs text-muted">
                    Configura el rango y pulsa Analizar, o abre Configuración para ajustar stride / rango.
                  </p>
                  <div className="flex flex-wrap items-center justify-center gap-2">
                    <button
                      type="button"
                      data-chrome
                      onClick={() => {
                        setSideTab("config");
                        setSideOpen(true);
                        setAutoStart(true);
                      }}
                      className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white"
                    >
                      Analizar ahora
                    </button>
                    <button
                      type="button"
                      data-chrome
                      onClick={() => openPanel("config")}
                      className="rounded-md bg-elevated px-4 py-2 text-sm font-semibold text-text ring-1 ring-border"
                    >
                      Abrir configuración
                    </button>
                  </div>
                </div>
              )}
            </div>

            {frame && !toolActive && (
              <>
                <button
                  type="button"
                  data-chrome
                  aria-label="Frame anterior"
                  onClick={(e) => {
                    e.stopPropagation();
                    step(-1);
                  }}
                  className="absolute left-0 top-0 z-20 flex h-full w-12 items-center justify-center bg-gradient-to-r from-black/45 to-transparent text-text/70 opacity-0 transition hover:opacity-100 focus:opacity-100"
                >
                  <SkipBack className="h-6 w-6" />
                </button>
                <button
                  type="button"
                  data-chrome
                  aria-label="Frame siguiente"
                  onClick={(e) => {
                    e.stopPropagation();
                    step(1);
                  }}
                  className="absolute right-0 top-0 z-20 flex h-full w-12 items-center justify-center bg-gradient-to-l from-black/45 to-transparent text-text/70 opacity-0 transition hover:opacity-100 focus:opacity-100"
                >
                  <SkipForward className="h-6 w-6" />
                </button>
              </>
            )}

            {toolActive && (
              <div className="pointer-events-none absolute bottom-3 left-1/2 z-30 -translate-x-1/2 rounded-full bg-black/70 px-3 py-1.5 text-[11px] text-muted ring-1 ring-white/10">
                {activeTool === "brush"
                  ? "Pinta sobre el atleta · Aceptar / Limpiar arriba"
                  : "Tap = punto · mantén y arrastra = trazar · Aceptar arriba"}
              </div>
            )}

            {scrubPreview != null && frames[scrubPreview] && (
              <div className="pointer-events-none absolute left-1/2 top-1/2 z-30 w-44 -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-md shadow-2xl ring-1 ring-border">
                <img src={frames[scrubPreview].raw} alt="" className="aspect-video w-full object-contain bg-black" />
                <div className="bg-surface px-2 py-1 text-center text-[11px] tabular-nums text-muted">
                  frame {frames[scrubPreview].frameId}
                </div>
              </div>
            )}
          </div>
          <div aria-hidden className="min-w-0 overflow-hidden" />
        </div>

        {/* Rieles fuera del stage con scrub — no roban play/pause */}
        <AthleteRail
          athlete={athlete}
          currentSessionId={session.id}
          open={railOpen}
          onClose={() => setRailOpen(false)}
          onSelect={(s) => {
            setRailOpen(false);
            onSelectSession(s);
            toast(`Abriendo “${s.title}”`);
          }}
        />

        <SidePanel
          open={sideOpen}
          tab={sideTab}
          setTab={setSideTab}
          config={config}
          setConfig={setConfig}
          metrics={metrics}
          overlayNote={data?.overlayNote}
          hasMasks={hasMasks}
          hasAnalysis={hasAnalysis}
          successPct={successPct}
          apiEnabled={isApi}
          videoName={data?.videoName || session.videoName}
          videoPath={data?.videoPath || session.videoPath}
          autoStart={autoStart}
          onAutoStartConsumed={() => setAutoStart(false)}
          onAnalyze={isApi ? handleAnalyze : null}
          onApplyScale={isApi ? handleApplyScale : null}
          currentTimeSec={currentTimeSec}
          durationSec={durationSec}
          onSeekHop={(i) => {
            seek(i);
            setPlaying(false);
          }}
          onClose={() => setSideOpen(false)}
          toast={toast}
        />
      </div>

      {/* Timeline solo con frames; sin análisis aún, SidePanel basta para Analizar */}
      {frameCount > 0 ? (
        <Timeline
          frames={frames}
          frameCount={frameCount}
          frameIndex={displayIndex}
          playing={playing}
          onTogglePlay={togglePlay}
          contacts={contacts}
          onSeek={(i) => {
            seek(i);
            setPlaying(false);
          }}
          onAddPhase={addPhase}
          onMoveContact={moveContact}
        />
      ) : (
        <div className="shrink-0 border-t border-border bg-surface/95 px-4 py-3 text-center text-xs text-muted">
          Vista previa del video · el timeline de fases aparece tras el análisis
        </div>
      )}
    </div>
  );
}
