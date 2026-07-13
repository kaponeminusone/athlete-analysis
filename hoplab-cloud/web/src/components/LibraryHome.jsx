import { useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  FileVideo,
  Loader2,
  Play,
  Plus,
  Search,
  Upload,
  X,
} from "lucide-react";
import { uploadVideo } from "../api/client";
import { ANALYSIS_LABEL, successPct, successTone } from "../mock/data";

const ACCEPT_VIDEO = "video/mp4,video/quicktime,.mp4,.mov,.avi,.mkv,.m4v";

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function Avatar({ athlete, size = "h-8 w-8", text = "text-xs" }) {
  return (
    <span
      className={`flex ${size} shrink-0 items-center justify-center rounded-full font-display font-bold text-white ring-2 ring-black/30`}
      style={{ backgroundColor: athlete.accent }}
      title={athlete.name}
    >
      <span className={text}>{athlete.initials}</span>
    </span>
  );
}

function Thumb({ src, alt, className }) {
  const [loaded, setLoaded] = useState(false);
  if (!src) {
    return <div className={`relative overflow-hidden bg-elevated ${className}`} aria-hidden />;
  }
  return (
    <div className={`relative overflow-hidden bg-elevated ${className}`}>
      {!loaded && <div className="absolute inset-0 skeleton" />}
      <img
        src={src}
        alt={alt}
        className={`h-full w-full object-cover transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
        onLoad={() => setLoaded(true)}
        draggable={false}
      />
    </div>
  );
}

function StateChip({ analysis }) {
  const map = {
    full: { icon: CheckCircle2, cls: "bg-ok/15 text-ok ring-ok/40" },
    partial: { icon: CircleDashed, cls: "bg-warn/15 text-warn ring-warn/40" },
    none: { icon: CircleDashed, cls: "bg-accent/15 text-accent ring-accent/40" },
  };
  const { icon: Icon, cls } = map[analysis] || map.none;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold ring-1 backdrop-blur-sm ${cls}`}>
      <Icon className="h-3 w-3" />
      {ANALYSIS_LABEL[analysis]}
    </span>
  );
}

function SessionCard({ athlete, session, onClick, onAnalyze, toast }) {
  const pct = successPct(session);
  const tone = successTone(pct);
  return (
    <div className="group w-[236px] shrink-0">
      <button type="button" onClick={onClick} className="w-full text-left focus:outline-none">
        <div className="relative aspect-video overflow-hidden rounded-lg bg-elevated shadow-[0_8px_24px_rgba(0,0,0,0.45)] ring-1 ring-border/60 transition duration-200 group-hover:scale-[1.02] group-hover:ring-text/30">
          <Thumb src={session.thumb} alt={session.title} className="absolute inset-0" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/10 to-black/40" />

          <div className="absolute left-2 top-2">
            <StateChip analysis={session.analysis} />
          </div>

          {pct != null && (
            <div className={`absolute right-2 top-2 rounded-md px-1.5 py-0.5 text-[11px] font-bold tabular-nums ring-1 backdrop-blur-sm ${tone.bg} ${tone.text} ${tone.ring}`}>
              {pct}%
            </div>
          )}

          <div className="absolute inset-0 flex items-center justify-center opacity-0 transition group-hover:opacity-100">
            <span className="flex h-11 w-11 items-center justify-center rounded-full bg-accent/95 text-white shadow-lg">
              <Play className="ml-0.5 h-5 w-5 fill-current" />
            </span>
          </div>

          <span className="absolute bottom-2 right-2 rounded bg-black/75 px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-text">
            {session.durationLabel}
          </span>
          <div className="absolute bottom-2 left-2 flex items-center gap-1.5">
            <Avatar athlete={athlete} size="h-6 w-6" text="text-[10px]" />
            <span className="text-[11px] font-semibold text-text drop-shadow">{athlete.short}</span>
          </div>
        </div>
      </button>

      <div className="mt-2 px-0.5">
        <p className="truncate text-sm font-semibold text-text">{session.title}</p>
        <p className="mt-0.5 flex items-center gap-1.5 text-xs text-muted">
          <span className="tabular-nums">{session.date || session.videoName || "—"}</span>
        </p>
        {session.note && <p className="mt-0.5 line-clamp-1 text-xs text-soft">{session.note}</p>}
        {session.analysis === "none" && (
          <button
            type="button"
            onClick={() => {
              const isApi = Boolean(session.videoPath || session.source === "api");
              if (isApi) {
                onAnalyze(athlete, session);
              } else {
                toast(`Sesión demo · sin motor para analizar ${athlete.short}`);
              }
            }}
            className="mt-1.5 inline-flex items-center gap-1 rounded-md bg-accent/15 px-2 py-1 text-[11px] font-semibold text-accent ring-1 ring-accent/30 transition hover:bg-accent/25"
          >
            <Play className="h-3 w-3 fill-current" /> Analizar
          </button>
        )}
      </div>
    </div>
  );
}

function Row({ athlete, sessions, onOpenSession, onAnalyze, toast }) {
  const best = useMemo(() => {
    const pcts = sessions.map((s) => successPct(s)).filter((p) => p != null);
    return pcts.length ? Math.max(...pcts) : null;
  }, [sessions]);
  const tone = successTone(best);
  return (
    <section className="mb-5">
      <div className="mb-2 flex items-center gap-2.5 px-6">
        <Avatar athlete={athlete} />
        <div className="min-w-0">
          <h2 className="font-display text-base font-semibold tracking-wide text-text">{athlete.name}</h2>
          <p className="truncate text-[11px] text-soft">{athlete.note}</p>
        </div>
        {best != null && (
          <span className={`ml-auto rounded-full px-2 py-0.5 text-[11px] font-bold tabular-nums ring-1 ${tone.bg} ${tone.text} ${tone.ring}`}>
            Mejor {best}%
          </span>
        )}
      </div>
      <div className="rail-scroll flex gap-3 overflow-x-auto px-6 pb-1">
        {sessions.map((s) => (
          <SessionCard
            key={s.id}
            athlete={athlete}
            session={s}
            onClick={() => onOpenSession(athlete, s)}
            onAnalyze={onAnalyze}
            toast={toast}
          />
        ))}
      </div>
    </section>
  );
}

function IngestModal({ connected, onClose, onUploaded, onConnectMotor, toast }) {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  function pickFile(f) {
    if (!f) return;
    setError(null);
    setFile(f);
  }

  async function handleUpload() {
    if (!file || uploading) return;
    setUploading(true);
    setError(null);
    setProgress(0);
    try {
      const entry = await uploadVideo(file, {
        onProgress: (pct) => setProgress(pct),
      });
      toast(`Video subido: ${entry.name || file.name}`);
      onUploaded?.();
      onClose();
    } catch (err) {
      setError(err.message || "Error al subir el video");
      setUploading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center p-4">
      <button
        type="button"
        aria-label="Cerrar"
        className="absolute inset-0 bg-black/60"
        onClick={() => !uploading && onClose()}
      />
      <div
        className="relative w-[min(460px,94%)] rounded-xl border border-border bg-surface p-5 shadow-2xl"
        style={{ animation: "slide-up 0.24s cubic-bezier(0.22,1,0.36,1) both" }}
      >
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Upload className="h-4 w-4 text-accent" />
            <p className="font-display text-sm font-semibold uppercase tracking-wide">Ingresar video</p>
          </div>
          <button
            type="button"
            onClick={() => !uploading && onClose()}
            disabled={uploading}
            className="rounded-md p-1.5 text-muted hover:bg-elevated hover:text-text disabled:opacity-40"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {!connected ? (
          <div className="flex flex-col items-center gap-3 rounded-lg border border-warn/30 bg-warn/5 px-4 py-6 text-center">
            <AlertTriangle className="h-6 w-6 text-warn" />
            <p className="text-sm text-muted">
              Conectá el motor Colab antes de subir un video.
            </p>
            <button
              type="button"
              onClick={() => {
                onClose();
                onConnectMotor?.();
              }}
              className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110"
            >
              Conectar motor
            </button>
          </div>
        ) : (
          <>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT_VIDEO}
              className="hidden"
              onChange={(e) => pickFile(e.target.files?.[0])}
            />

            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                pickFile(e.dataTransfer.files?.[0]);
              }}
              disabled={uploading}
              className={`mb-3 flex w-full items-center gap-3 rounded-lg border border-dashed px-3 py-4 text-left text-sm transition ${
                dragOver ? "border-accent bg-accent/10 text-text" : "border-border bg-elevated/40 text-muted hover:border-muted"
              } disabled:opacity-60`}
            >
              {file ? <FileVideo className="h-5 w-5 shrink-0 text-accent" /> : <Upload className="h-4 w-4 shrink-0" />}
              {file ? (
                <span className="min-w-0">
                  <span className="block truncate font-medium text-text">{file.name}</span>
                  <span className="text-xs text-soft">{formatBytes(file.size)}</span>
                </span>
              ) : (
                <span>Elegí o arrastrá un video (.mp4, .mov, .avi, .mkv, .m4v)</span>
              )}
            </button>

            {uploading && (
              <div className="mb-3">
                <div className="mb-1 flex justify-between text-xs text-muted">
                  <span>Subiendo…</span>
                  <span className="tabular-nums">{progress}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full rounded-full bg-accent transition-[width] duration-200"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {error && (
              <p className="mb-3 rounded-md bg-red-400/10 px-3 py-2 text-xs text-red-400">{error}</p>
            )}

            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={onClose}
                disabled={uploading}
                className="rounded-md px-4 py-2 text-sm text-muted hover:text-text disabled:opacity-40"
              >
                Cancelar
              </button>
              <button
                type="button"
                onClick={handleUpload}
                disabled={!file || uploading}
                className="inline-flex items-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-50"
              >
                {uploading && <Loader2 className="h-4 w-4 animate-spin" />}
                {uploading ? "Subiendo…" : error ? "Reintentar" : "Subir"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function LibraryHome({
  athletes,
  onOpenSession,
  onAnalyzeSession,
  onReloadLibrary,
  onConnectMotor,
  toast,
  loading = false,
  librarySource = "mock",
}) {
  const [ingest, setIngest] = useState(false);
  const [query, setQuery] = useState("");
  const connected = librarySource === "api";

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return athletes;
    return athletes
      .map((a) => {
        const athleteMatch = a.name.toLowerCase().includes(q);
        const sessions = athleteMatch
          ? a.sessions
          : a.sessions.filter(
              (s) =>
                s.title.toLowerCase().includes(q) ||
                (s.note || "").toLowerCase().includes(q) ||
                (s.date || "").includes(q) ||
                (s.videoName || "").toLowerCase().includes(q),
            );
        return { ...a, sessions };
      })
      .filter((a) => a.sessions.length > 0);
  }, [athletes, query]);

  return (
    <div className="flex h-full flex-col bg-bg">
      {/* Barra superior compacta: búsqueda + ingesta (sin hero ni nav) */}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border/80 px-6">
        <span className="flex h-7 w-7 items-center justify-center rounded-md bg-accent font-display text-sm font-bold text-white">
          H
        </span>
        <div className="relative w-full max-w-sm">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-soft" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Buscar atleta, sesión o fecha (ej. 07-05)…"
            className="h-9 w-full rounded-full bg-elevated pl-9 pr-9 text-sm text-text outline-none ring-1 ring-border transition placeholder:text-soft focus:ring-muted"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1 text-soft hover:text-text"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        {librarySource === "api" && (
          <span className="hidden text-[10px] font-medium uppercase tracking-wider text-ok sm:inline">API</span>
        )}
        {librarySource === "mock" && !loading && (
          <span className="hidden text-[10px] font-medium uppercase tracking-wider text-warn sm:inline">Demo</span>
        )}
        <button
          type="button"
          onClick={() => setIngest(true)}
          className="ml-auto inline-flex items-center gap-1.5 rounded-md bg-accent px-3.5 py-2 text-sm font-semibold text-white shadow transition hover:brightness-110"
        >
          <Plus className="h-4 w-4" />
          Ingresar video
        </button>
      </header>

      <main className="rail-scroll min-h-0 flex-1 overflow-y-auto py-5">
        {loading ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-muted">
            <div className="h-8 w-8 animate-pulse rounded-full bg-elevated ring-1 ring-border" />
            <p className="text-sm">Cargando biblioteca…</p>
          </div>
        ) : filtered.length === 0 ? (
          query ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted">
              <Search className="h-6 w-6 text-soft" />
              <p className="text-sm">Sin resultados para “{query}”.</p>
            </div>
          ) : connected ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-muted">
              <FileVideo className="h-8 w-8 text-soft" />
              <p className="text-sm">No hay videos. Subí uno con “Ingresar video”.</p>
              <button
                type="button"
                onClick={() => setIngest(true)}
                className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3.5 py-2 text-sm font-semibold text-white shadow transition hover:brightness-110"
              >
                <Plus className="h-4 w-4" />
                Ingresar video
              </button>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-muted">
              <Search className="h-6 w-6 text-soft" />
              <p className="text-sm">Biblioteca vacía.</p>
            </div>
          )
        ) : (
          filtered.map((athlete) => (
            <Row
              key={athlete.id}
              athlete={athlete}
              sessions={athlete.sessions}
              onOpenSession={onOpenSession}
              onAnalyze={onAnalyzeSession || onOpenSession}
              toast={toast}
            />
          ))
        )}
      </main>

      {ingest && (
        <IngestModal
          connected={connected}
          onClose={() => setIngest(false)}
          onUploaded={onReloadLibrary}
          onConnectMotor={onConnectMotor}
          toast={toast}
        />
      )}
    </div>
  );
}
