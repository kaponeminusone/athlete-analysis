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
} from "lucide-react";

const angleColors = {
  LATERAL: "#f47c30",
  SEMI_BACK: "#d86b25",
  SEMI_FRONT: "#2fc66d",
  FRONTAL: "#dfc83f",
  UNKNOWN: "#64717f",
};

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
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toFixed(digits);
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
  const [propagationEndFrame, setPropagationEndFrame] = useState(null);
  const [isReanalyzing, setIsReanalyzing] = useState(false);
  const [reanalysisJob, setReanalysisJob] = useState(null);
  const [refinedOutputDir, setRefinedOutputDir] = useState(null);
  const [detections, setDetections] = useState([]);
  const [correctionVersion, setCorrectionVersion] = useState(0);
  const videoRef = useRef(null);

  const frames = project?.analysis?.frames || [];
  const currentFrame = frames[frameIndex] || null;
  const summary = project?.analysis?.data?.summary || {};
  const analysisExists = Boolean(project?.analysis?.exists);
  const frameImage = getFrameAsset(project, currentFrame, correctionVersion);

  useEffect(() => {
    loadVideos();
  }, []);

  useEffect(() => {
    setDetections([]);
  }, [frameIndex]);

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

  async function openVideo(path = videoPath) {
    if (!path.trim()) {
      setNotice("Indica la ruta del video.");
      setNoticeStrong(true);
      return;
    }

    setLoadingVideoPath(path.trim());
    setNotice("Buscando analysis.json...");
    setNoticeStrong(false);
    try {
      const nextProject = await fetchJson(apiUrl("/api/project", { video_path: path.trim() }));
      setVideoPath(path.trim());
      setProject(nextProject);
      setFrameIndex(0);
      setMode(nextProject.video.exists ? "video" : "annotation");
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
      openVideo(match.path);
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
      const result = await fetchJson("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: path, stride }),
      });
      setAnalysisLog(`Analisis iniciado. Job: ${result.job_id}\n`);
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
      const result = await fetchJson("/api/reanalyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_path: path,
          stride,
          seed_start_frame: seedStart,
          seed_end_frame:   seedEnd,
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

  async function refreshProject(path = project?.video?.path || videoPath) {
    if (!path) return;
    const nextProject = await fetchJson(apiUrl("/api/project", { video_path: path }));
    setProject(nextProject);
    setCorrectionVersion((version) => version + 1);
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
        }),
      });
      await refreshProject(project.video.path);
      setDetections([]);
      setPropagationEndFrame(null);
      setNotice(`Correccion aplicada: ${result.total_affected} frames afectados.`);
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
                onOpenVideo={openVideo}
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
                correctionMode={correctionMode}
                detections={detections}
                correctionVersion={correctionVersion}
                isCorrecting={isCorrecting}
                isLoadingDetections={isLoadingDetections}
                onSubmitCorrection={submitCorrection}
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
                correctionMode={correctionMode}
                setCorrectionMode={setCorrectionMode}
                sotBackend={sotBackend}
                setSotBackend={setSotBackend}
                stride={stride}
                setStride={setStride}
                propagationEndFrame={propagationEndFrame}
                setPropagationEndFrame={setPropagationEndFrame}
                currentFrameIdx={currentFrame?.frame_idx ?? null}
                onLoadDetections={loadDetections}
                isCorrecting={isCorrecting}
                isLoadingDetections={isLoadingDetections}
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
                const isActive = project?.video?.path === video.path;
                const isLoading = loadingVideoPath === video.path;
                return (
                  <button
                    className={`relative grid grid-cols-[96px_1fr] gap-2 border bg-editor-850 p-1 text-left transition ${
                      isActive ? "border-accent" : "border-editor-700 hover:border-editor-500"
                    } ${isLoading ? "opacity-80" : ""}`}
                    key={video.path}
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
  correctionMode,
  detections,
  isCorrecting,
  isLoadingDetections,
  onSubmitCorrection,
}) {
  const [dragStart, setDragStart] = useState(null);
  const [dragBox, setDragBox] = useState(null);
  const [maskPoints, setMaskPoints] = useState([]);
  const [isPainting, setIsPainting] = useState(false);
  const stageRef = useRef(null);

  useEffect(() => {
    setDragStart(null);
    setDragBox(null);
    setMaskPoints([]);
    setIsPainting(false);
  }, [currentFrame?.frame_idx, correctionMode]);

  function mediaMetrics() {
    const stage = stageRef.current;
    if (!stage) return null;
    const rect = stage.getBoundingClientRect();

    // Read actual CSS padding so we don't assume a hardcoded value
    const cs = window.getComputedStyle(stage);
    const padL = parseFloat(cs.paddingLeft)  || 0;
    const padT = parseFloat(cs.paddingTop)   || 0;
    const padR = parseFloat(cs.paddingRight) || 0;
    const padB = parseFloat(cs.paddingBottom)|| 0;

    // Content area available to the img / video element
    const contentW = rect.width  - padL - padR;
    const contentH = rect.height - padT - padB;

    // Prefer live video dimensions; fall back to analysis metadata
    const naturalWidth  = (videoRef.current?.videoWidth  > 0 ? videoRef.current.videoWidth  : 0)
                       || project?.analysis?.data?.video_info?.width  || 1;
    const naturalHeight = (videoRef.current?.videoHeight > 0 ? videoRef.current.videoHeight : 0)
                       || project?.analysis?.data?.video_info?.height || 1;

    // object-contain scale within the content area
    const scale = Math.min(contentW / naturalWidth, contentH / naturalHeight);
    const displayWidth  = naturalWidth  * scale;
    const displayHeight = naturalHeight * scale;

    // Offset from the stage rect top-left to the actual image top-left
    const offsetX = padL + (contentW - displayWidth)  / 2;
    const offsetY = padT + (contentH - displayHeight) / 2;

    return { rect, naturalWidth, naturalHeight, scale, displayWidth, displayHeight, offsetX, offsetY };
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

  function handlePointerDown(event) {
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
    if (correctionMode === "bbox_correction" && dragStart) {
      setDragBox({ start: dragStart, end: point });
    }
    if (correctionMode === "mask_correction" && isPainting) {
      setMaskPoints((points) => [...points, point]);
    }
  }

  function handlePointerUp(event) {
    if (correctionMode === "bbox_correction" && dragBox) {
      const point = eventToFramePoint(event) || dragBox.end;
      const x1 = Math.min(dragBox.start.x, point.x);
      const y1 = Math.min(dragBox.start.y, point.y);
      const x2 = Math.max(dragBox.start.x, point.x);
      const y2 = Math.max(dragBox.start.y, point.y);
      setDragStart(null);
      setDragBox(null);
      if (Math.abs(x2 - x1) > 8 && Math.abs(y2 - y1) > 8) {
        onSubmitCorrection("bbox_correction", { x1, y1, x2, y2 });
      }
    }
    if (correctionMode === "mask_correction") {
      setIsPainting(false);
    }
  }

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
            className="h-full w-full select-none object-contain"
            src={project.video.url}
            controls
            draggable={false}
            onTimeUpdate={onTimeUpdate}
            onDragStart={(event) => event.preventDefault()}
          />
        ) : frameImage ? (
          <img
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
        {correctionMode !== "inspect" ? (
          <div
            className="absolute inset-0 z-30 cursor-crosshair"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={() => setIsPainting(false)}
          />
        ) : null}
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

function Timeline({ frames, currentIndex, onSelect, duration, propagationEndFrame, onShiftSelect }) {
  if (!frames.length) {
    return <div className="m-2 border border-editor-700 bg-editor-850 px-2 py-2 text-[10px] uppercase text-slate-500">Timeline</div>;
  }

  const total = duration || frames.at(-1)?.timestamp_s || frames.length;
  const currentFrame = frames[currentIndex];

  // range for highlight: from current frame_idx to propagationEndFrame
  const rangeStart = currentFrame?.frame_idx ?? null;
  const rangeEnd   = propagationEndFrame;
  const hasRange   = rangeStart !== null && rangeEnd !== null && rangeEnd > rangeStart;

  function handleWheel(e) {
    e.preventDefault();
    onSelect(currentIndex + (e.deltaY > 0 ? 1 : -1));
  }

  function handleClick(e, index) {
    if (e.shiftKey) { onShiftSelect?.(index); }
    else            { onSelect(index); }
  }

  return (
    <div className="relative m-2 overflow-hidden border border-editor-600 bg-editor-850" onWheel={handleWheel}>
      {/* Header con leyenda */}
      <div className="absolute left-2 top-1 z-10 flex items-center gap-3 text-[10px] text-slate-500">
        <span className="uppercase">Timeline</span>
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
      </div>

      <div className="relative h-full min-h-14 pt-5">
        {frames.map((frame, index) => {
          const start = ((frame.timestamp_s || 0) / total) * 100;
          const nextTime = frames[index + 1]?.timestamp_s ?? total;
          const width = Math.max(((nextTime - (frame.timestamp_s || 0)) / total) * 100, 0.25);
          const detected = frame.person_detected;
          const angleColor = detected
            ? (angleColors[frame.camera_angle] || angleColors.UNKNOWN)
            : "#1e2433";
          const isActive = index === currentIndex;
          const inRange  = hasRange && frame.frame_idx > rangeStart && frame.frame_idx <= rangeEnd;

          return (
            <button
              key={`${frame.frame_idx}-${index}`}
              type="button"
              className={`absolute bottom-0 top-5 overflow-hidden border-r border-black/30 p-0 ${
                isActive ? "ring-2 ring-white ring-inset z-10" : ""
              } ${inRange ? "brightness-125" : ""}`}
              style={{
                left: `${start}%`,
                width: `${width}%`,
                outline: inRange ? "1px solid rgba(250,204,21,0.6)" : undefined,
              }}
              title={`Frame ${frame.frame_idx} · ${frame.timestamp_s?.toFixed(2)}s\n${
                detected ? `Deteccion: SI · ${frame.camera_angle}` : "Deteccion: NO"
              }${frame.quality_score != null ? ` · Q=${frame.quality_score}` : ""}${
                inRange ? "\n[en rango de propagacion]" : ""
              }`}
              onClick={(e) => handleClick(e, index)}
            >
              {/* Franja superior: estado de deteccion (40% del alto) */}
              <div
                className="absolute left-0 right-0 top-0"
                style={{
                  height: "40%",
                  background: detected ? "#22c55e" : "#1e2433",
                  opacity: detected ? 0.85 : 1,
                }}
              />
              {/* Franja inferior: angulo de camara (60% del alto) */}
              <div
                className="absolute bottom-0 left-0 right-0"
                style={{
                  height: "60%",
                  background: angleColor,
                  opacity: detected ? 1 : 0.25,
                }}
              />
            </button>
          );
        })}

        {/* Playhead */}
        <div
          className="absolute bottom-0 top-0 z-20 w-0.5 bg-red-400 pointer-events-none"
          style={{ left: `${(((currentFrame?.timestamp_s || 0) / total) * 100).toFixed(3)}%` }}
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
  correctionMode,
  setCorrectionMode,
  sotBackend,
  setSotBackend,
  stride,
  setStride,
  propagationEndFrame,
  setPropagationEndFrame,
  currentFrameIdx,
  onLoadDetections,
  isCorrecting,
  isLoadingDetections,
}) {
  return (
    <aside className="h-full overflow-auto bg-editor-800">
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
          </>
        ) : null}
      </div>

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

      <div className="mt-2 px-2">
        <div className="mb-1 text-[11px] font-bold uppercase text-slate-300">Analisis</div>
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
          {isReanalyzing ? <AnalysisProgress job={reanalysisJob} /> : null}
        </div>
      ) : null}
    </aside>
  );
}

function PanelTitle({ children }) {
  return <div className="px-2 py-2 text-[11px] font-bold uppercase text-slate-300">{children}</div>;
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

function BottomPanel({ project, summary, frames, currentIndex, onSelectFrame, propagationEndFrame, onShiftSelect }) {
  const distribution = Object.entries(summary.camera_angle_distribution || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
  const duration = project?.analysis?.data?.video_info?.duration_s;

  return (
    <div className="grid h-full grid-rows-[72px_32px] bg-editor-850">
      <div className="border-b border-black bg-editor-900">
        <Timeline
          frames={frames}
          currentIndex={currentIndex}
          onSelect={onSelectFrame}
          duration={duration}
          propagationEndFrame={propagationEndFrame}
          onShiftSelect={onShiftSelect}
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
