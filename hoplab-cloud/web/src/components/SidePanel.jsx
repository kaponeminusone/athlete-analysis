import { useRef, useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import {
  deltaChip,
  formatTime,
  POSE_LABEL_TEXT,
  successTone,
} from "../mock/data";

function Toggle({ checked, onChange, label, hint, disabled, disabledHint }) {
  return (
    <label
      className={`flex items-start justify-between gap-4 rounded-lg bg-elevated/60 px-3 py-3 ring-1 ring-border ${
        disabled ? "opacity-55" : "cursor-pointer"
      }`}
      title={disabled ? disabledHint : undefined}
    >
      <div>
        <p className="text-sm font-medium text-text">{label}</p>
        {(disabled ? disabledHint : hint) && (
          <p className="mt-0.5 text-xs text-muted">{disabled ? disabledHint : hint}</p>
        )}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!disabled) onChange(!checked);
        }}
        className={`relative mt-0.5 h-7 w-12 shrink-0 rounded-full ring-1 transition-colors duration-200 ${
          checked && !disabled
            ? "bg-accent ring-accent/80"
            : "bg-[#3a3a3a] ring-white/35"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 h-6 w-6 rounded-full bg-white shadow-md transition-transform duration-200 ${
            checked && !disabled ? "translate-x-5" : "translate-x-0"
          }`}
        />
      </button>
    </label>
  );
}

function poseBarClass(label) {
  if (label === "buena") return "bg-ok";
  if (label === "regular") return "bg-warn";
  return "bg-accent";
}

function poseBadgeClass(label) {
  if (label === "buena") return "border-ok/50 text-ok";
  if (label === "regular") return "border-warn/50 text-warn";
  return "border-accent/50 text-accent";
}

function stopPanelPointer(e) {
  e.stopPropagation();
}

/**
 * Cajón lateral derecho: Configuración / Estadísticas.
 * Fuera del stage de scrub; stopPropagation en backdrop + aside.
 */
export default function SidePanel({
  open,
  tab,
  setTab,
  config,
  setConfig,
  metrics,
  overlayNote,
  hasMasks,
  hasAnalysis,
  successPct,
  apiEnabled = false,
  onAnalyze = null,
  onApplyScale = null,
  currentTimeSec = 0,
  durationSec = null,
  onSeekHop,
  onClose,
  toast,
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [progress, setProgress] = useState(null);
  const [progressMsg, setProgressMsg] = useState(null);
  const [lightbox, setLightbox] = useState(null);
  const [busyScale, setBusyScale] = useState(false);
  const timer = useRef(null);
  const tone = successTone(successPct);

  async function analyze() {
    if (progress != null) return;

    if (onAnalyze) {
      setProgress(0);
      setProgressMsg(null);
      try {
        await onAnalyze(config, (pct, msg) => {
          setProgress(Math.min(99, Math.max(0, pct)));
          if (msg) setProgressMsg(msg);
        });
        setProgress(100);
        setProgressMsg("Listo");
        toast("Análisis completado");
        window.setTimeout(() => {
          setProgress(null);
          setProgressMsg(null);
        }, 600);
      } catch (err) {
        setProgress(null);
        setProgressMsg(null);
        toast(err.message || "Error al analizar");
      }
      return;
    }

    // Fallback mock si no hay API
    if (timer.current) return;
    setProgress(0);
    timer.current = window.setInterval(() => {
      setProgress((p) => {
        if (p >= 100) {
          window.clearInterval(timer.current);
          timer.current = null;
          window.setTimeout(() => setProgress(null), 500);
          toast("Análisis completado (mock)");
          return 100;
        }
        return Math.min(100, p + 12);
      });
    }, 160);
  }

  async function applyScale(meters) {
    const m = Number(meters);
    if (!Number.isFinite(m) || m <= 0) {
      toast("Longitud inválida");
      return;
    }
    if (!onApplyScale) {
      toast(`Escala ${m} m (solo local — sin API)`);
      return;
    }
    setBusyScale(true);
    try {
      await onApplyScale(m);
    } catch (err) {
      toast(err.message || "Error al escalar");
    } finally {
      setBusyScale(false);
    }
  }

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Cerrar panel"
          className="absolute inset-0 z-40 bg-black/30"
          onClick={onClose}
          onPointerDown={stopPanelPointer}
          style={{ animation: "fade-in 0.2s ease both" }}
        />
      )}
      <aside
        className={`absolute inset-y-0 right-0 z-50 flex w-[min(360px,88vw)] flex-col border-l border-border bg-surface p-4 shadow-2xl transition-transform duration-[240ms] ease-[cubic-bezier(0.22,1,0.36,1)] ${
          open ? "translate-x-0" : "translate-x-full pointer-events-none"
        }`}
        onPointerDown={stopPanelPointer}
        onClick={stopPanelPointer}
      >
        <div className="mb-4 flex items-center justify-between">
          <div className="flex gap-1 rounded-lg bg-elevated p-1 ring-1 ring-border">
            {[
              { id: "config", label: "Configuración" },
              { id: "stats", label: "Estadísticas" },
            ].map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                  tab === t.id ? "bg-surface text-text shadow" : "text-muted hover:text-text"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-muted transition hover:bg-elevated hover:text-text"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="rail-scroll flex-1 overflow-y-auto pr-1">
          {tab === "config" && (
            <div className="space-y-3">
              {!hasAnalysis && (
                <div className="rounded-lg border border-accent/30 bg-accent/5 p-3 text-xs leading-relaxed text-muted">
                  Video nuevo · aún sin analizar. Un clic hace{" "}
                  <span className="text-text">análisis</span>,{" "}
                  <span className="text-text">mapa de pista</span>,{" "}
                  <span className="text-text">refine</span> y{" "}
                  <span className="text-text">fases automáticas</span>.
                </div>
              )}

              <div className="rounded-lg bg-elevated/60 px-3 py-3 ring-1 ring-border">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-text">Muestreo (stride)</p>
                  <span className="font-display text-sm font-semibold tabular-nums text-accent">1 / {config.stride}</span>
                </div>
                <p className="mt-0.5 text-xs text-muted">Cuántos cuadros saltear. Menor = más detalle, más lento.</p>
                <input
                  type="range"
                  min={1}
                  max={4}
                  step={1}
                  value={config.stride}
                  onChange={(e) => setConfig((c) => ({ ...c, stride: Number(e.target.value) }))}
                  className="mt-2 w-full accent-[var(--color-accent)]"
                />
              </div>

              <div className="rounded-lg bg-elevated/60 px-3 py-3 ring-1 ring-border">
                <p className="mb-1 text-sm font-medium text-text">Rango de análisis</p>
                <p className="mb-2 text-xs text-muted">
                  Desde / hasta (segundos). Vacío al final = hasta el cierre del video.
                  {durationSec != null && Number.isFinite(durationSec) ? (
                    <span className="text-soft"> · video ≈ {durationSec.toFixed(1)} s</span>
                  ) : null}
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <label className="block text-[11px] text-soft">
                    Inicio (s)
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      value={config.startSec ?? 0}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        setConfig((c) => ({
                          ...c,
                          startSec: Number.isFinite(v) ? Math.max(0, v) : 0,
                          range: "custom",
                        }));
                      }}
                      className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm tabular-nums text-text outline-none focus:ring-1 focus:ring-accent"
                    />
                  </label>
                  <label className="block text-[11px] text-soft">
                    Fin (s)
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      placeholder="fin"
                      value={config.endSec ?? ""}
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === "") {
                          setConfig((c) => ({ ...c, endSec: null, range: "custom" }));
                          return;
                        }
                        const v = Number(raw);
                        setConfig((c) => ({
                          ...c,
                          endSec: Number.isFinite(v) ? Math.max(0, v) : null,
                          range: "custom",
                        }));
                      }}
                      className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm tabular-nums text-text outline-none focus:ring-1 focus:ring-accent"
                    />
                  </label>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    onClick={() =>
                      setConfig((c) => ({
                        ...c,
                        startSec: Math.max(0, Number(currentTimeSec) || 0),
                        range: "custom",
                      }))
                    }
                    className="rounded-md bg-surface px-2 py-1 text-[11px] font-semibold text-muted ring-1 ring-border transition hover:text-text"
                  >
                    Inicio = playhead
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setConfig((c) => ({
                        ...c,
                        endSec: Math.max(0, Number(currentTimeSec) || 0),
                        range: "custom",
                      }))
                    }
                    className="rounded-md bg-surface px-2 py-1 text-[11px] font-semibold text-muted ring-1 ring-border transition hover:text-text"
                  >
                    Fin = playhead
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setConfig((c) => ({
                        ...c,
                        startSec: 0,
                        endSec: null,
                        range: "all",
                      }))
                    }
                    className="rounded-md bg-surface px-2 py-1 text-[11px] font-semibold text-muted ring-1 ring-border transition hover:text-text"
                  >
                    Video completo
                  </button>
                </div>
              </div>

              <Toggle
                checked={config.refine}
                onChange={(v) => setConfig((c) => ({ ...c, refine: v }))}
                label="Mejorar seguimiento"
                hint="Refinado + fases automáticas (recomendado)"
              />
              <Toggle
                checked={Boolean(config.useVenueMap)}
                onChange={(v) => setConfig((c) => ({ ...c, useVenueMap: v }))}
                label="Usar mapa de pista y arena"
                hint={
                  hasMasks
                    ? "Mapear con CNN / perfil del estadio al analizar"
                    : "Si hay CNN o perfil, se aplica al analizar"
                }
              />

              <button
                type="button"
                onClick={analyze}
                disabled={progress != null}
                className="mt-1 w-full overflow-hidden rounded-md bg-accent py-2.5 text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-80"
              >
                {progress == null
                  ? "Analizar"
                  : progressMsg
                    ? `${progressMsg} ${progress}%`
                    : `Analizando… ${progress}%`}
              </button>
              {progress != null && (
                <div className="h-1 overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full rounded-full bg-accent transition-[width] duration-200"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}

              <div className="pt-3">
                <button
                  type="button"
                  onClick={() => setAdvancedOpen((v) => !v)}
                  className="text-xs font-medium text-soft underline-offset-2 transition hover:text-muted hover:underline"
                >
                  {advancedOpen ? "Ocultar avanzado" : "Avanzado ▸"}
                </button>
                {advancedOpen && (
                  <div className="mt-3 space-y-3 rounded-lg border border-warn/30 bg-warn/5 p-3">
                    <div className="flex gap-2 text-warn">
                      <AlertTriangle className="h-4 w-4 shrink-0" />
                      <p className="text-xs leading-relaxed">
                        Acciones peligrosas: pueden dañar el modelo si la sesión está mal etiquetada. No es de uso diario.
                      </p>
                    </div>

                    <div className="rounded-md bg-elevated/80 px-3 py-2.5 ring-1 ring-border">
                      <label className="block text-sm font-medium text-text" htmlFor="hops-corridor-m">
                        Longitud pista de hops (m)
                      </label>
                      <p className="mt-0.5 text-[11px] text-muted">
                        Del 1er hop al aterrizaje. Por defecto 10 m (misma pista para todos).
                      </p>
                      <div className="mt-2 flex items-center gap-2">
                        <input
                          id="hops-corridor-m"
                          type="number"
                          min={5}
                          max={20}
                          step={0.5}
                          value={config.hopsCorridorM ?? 10}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            if (!Number.isFinite(v)) return;
                            setConfig((c) => ({ ...c, hopsCorridorM: Math.min(20, Math.max(5, v)) }));
                          }}
                          className="w-24 rounded-md border border-border bg-surface px-2 py-1.5 text-sm tabular-nums text-text outline-none focus:ring-1 focus:ring-accent"
                        />
                        <span className="text-xs text-soft">metros</span>
                        <button
                          type="button"
                          disabled={busyScale}
                          onClick={() => {
                            setConfig((c) => ({ ...c, hopsCorridorM: 10 }));
                            applyScale(10);
                          }}
                          className="ml-auto text-[11px] font-medium text-muted underline-offset-2 hover:text-text hover:underline disabled:opacity-50"
                        >
                          Usar 10 m
                        </button>
                      </div>
                      {apiEnabled && (
                        <button
                          type="button"
                          disabled={busyScale}
                          onClick={() => applyScale(config.hopsCorridorM ?? 10)}
                          className="mt-2 w-full rounded-md bg-surface py-1.5 text-xs font-semibold text-text ring-1 ring-border transition hover:ring-accent/40 disabled:opacity-60"
                        >
                          {busyScale ? "Aplicando…" : "Aplicar escala"}
                        </button>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={() => toast("Aprender de este video (mock) — no ejecutado")}
                      className="w-full rounded-md bg-elevated py-2 text-sm text-text ring-1 ring-border transition hover:ring-warn/50"
                    >
                      Aprender de este video
                    </button>
                    <button
                      type="button"
                      onClick={() => toast("Entrenar mapa (mock) — no ejecutado")}
                      className="w-full rounded-md bg-elevated py-2 text-sm text-text ring-1 ring-border transition hover:ring-warn/50"
                    >
                      Entrenar mapa
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {tab === "stats" &&
            (!hasAnalysis ? (
              <div className="flex flex-col items-center gap-3 rounded-lg border border-border bg-elevated/40 p-6 text-center">
                <p className="text-sm text-muted">Este video aún no tiene análisis.</p>
                <button
                  type="button"
                  onClick={() => setTab("config")}
                  className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110"
                >
                  Ir a Configuración
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className={`rounded-lg p-3 ring-1 ${tone.bg} ${tone.ring}`}>
                  <p className="text-[10px] uppercase tracking-wider text-soft">Éxito general de la práctica</p>
                  <div className="mt-1 flex items-baseline gap-2">
                    <span className={`font-display text-3xl font-bold tabular-nums ${tone.text}`}>{successPct}%</span>
                    <span className="text-xs text-muted">promedio de pose por hop + salto final</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  {[
                    { k: "Tiempo", v: formatTime(metrics?.general?.totalTimeMs || 0) },
                    { k: "Vel. media", v: `${metrics?.general?.avgSpeed ?? 0} m/s` },
                  ].map((c) => (
                    <div key={c.k} className="rounded-lg bg-elevated px-3 py-2.5 ring-1 ring-border">
                      <p className="text-[10px] uppercase tracking-wider text-soft">{c.k}</p>
                      <p className="mt-1 font-display text-lg font-semibold tabular-nums">{c.v}</p>
                    </div>
                  ))}
                </div>

                {overlayNote && (
                  <p className="rounded-md bg-warn/10 px-2.5 py-1.5 text-[10px] leading-relaxed text-warn ring-1 ring-warn/30">
                    {overlayNote} · ámbar = general · cian = esta toma
                  </p>
                )}

                {metrics?.finalFlightNote && (
                  <p className="text-[11px] leading-relaxed text-muted">{metrics.finalFlightNote}</p>
                )}

                <div>
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-soft">
                    Hops · tocá para ir al frame
                  </p>
                  <div className="space-y-2.5">
                    {metrics.hops?.map((h) => {
                      const chip = deltaChip(h.deltaVsGeneral);
                      const pct = Math.round((h.poseQuality || 0) * 100);
                      const labelKey = h.poseLabel || "débil";
                      return (
                        <div
                          key={h.id}
                          className="overflow-hidden rounded-lg bg-elevated ring-1 ring-border"
                        >
                          <button
                            type="button"
                            onClick={() => onSeekHop(h.index)}
                            className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition hover:bg-surface/60"
                          >
                            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-surface font-display text-sm font-bold text-accent">
                              {h.label}
                            </span>
                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-1.5">
                                <p className="text-sm font-medium tabular-nums">{h.speed} m/s</p>
                                <span
                                  className={`rounded px-1.5 py-0.5 text-[10px] font-bold tabular-nums ring-1 ${chip.cls}`}
                                  title="vs general"
                                >
                                  {chip.glyph} {chip.text} vs gen.
                                </span>
                              </div>
                              <p className="text-xs text-soft">
                                {formatTime(h.timeMs)}
                                {h.isFinalFlight ? " · vuelo final" : ""}
                              </p>
                            </div>
                          </button>

                          <div className="space-y-2 border-t border-border px-3 py-2.5">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-[10px] uppercase tracking-wider text-soft">Calidad de pose</span>
                              <span
                                className={`rounded border px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide ${poseBadgeClass(labelKey)}`}
                              >
                                {POSE_LABEL_TEXT[labelKey] || "—"} {pct}%
                              </span>
                            </div>
                            <div className="h-1.5 overflow-hidden rounded-sm bg-surface">
                              <div
                                className={`h-full transition-[width] ${poseBarClass(labelKey)}`}
                                style={{ width: `${pct}%` }}
                              />
                            </div>

                            {h.poseOverlayUrl && (
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setLightbox({
                                    src: h.poseOverlayUrl,
                                    title: `${h.label} — General (ámbar) vs esta toma (cian)`,
                                  });
                                }}
                                className="group relative block w-full overflow-hidden rounded-md ring-1 ring-border transition hover:ring-muted"
                                title="Ampliar overlay de pose"
                              >
                                <img
                                  src={h.poseOverlayUrl}
                                  alt={`Overlay pose ${h.label}`}
                                  className="aspect-[2/1] w-full bg-black object-contain"
                                  loading="lazy"
                                  onError={(ev) => {
                                    ev.currentTarget.style.display = "none";
                                  }}
                                />
                                <span className="pointer-events-none absolute bottom-1 left-1 rounded bg-black/70 px-1.5 py-0.5 text-[9px] text-muted opacity-0 transition group-hover:opacity-100">
                                  Ampliar
                                </span>
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ))}
        </div>
      </aside>

      {lightbox && (
        <div
          className="fixed inset-0 z-[80] flex items-center justify-center bg-black/80 p-4"
          onPointerDown={stopPanelPointer}
          onClick={() => setLightbox(null)}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="relative max-h-[90vh] w-full max-w-3xl overflow-hidden rounded-lg bg-surface shadow-2xl ring-1 ring-border"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-border px-3 py-2">
              <p className="text-xs font-medium text-muted">{lightbox.title}</p>
              <button
                type="button"
                onClick={() => setLightbox(null)}
                className="rounded-md p-1.5 text-muted transition hover:bg-elevated hover:text-text"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <img src={lightbox.src} alt="" className="max-h-[80vh] w-full bg-black object-contain" />
            <p className="px-3 py-2 text-[10px] text-soft">Ámbar = general · cian = esta toma</p>
          </div>
        </div>
      )}
    </>
  );
}
