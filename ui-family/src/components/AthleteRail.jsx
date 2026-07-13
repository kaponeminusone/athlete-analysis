import { useState } from "react";
import { CheckCircle2, CircleDashed, X } from "lucide-react";
import { ANALYSIS_LABEL, successPct, successTone } from "../mock/data";

function Thumb({ src }) {
  const [loaded, setLoaded] = useState(false);
  if (!src) {
    return <div className="relative aspect-video w-24 shrink-0 overflow-hidden rounded-md bg-elevated" />;
  }
  return (
    <div className="relative aspect-video w-24 shrink-0 overflow-hidden rounded-md bg-elevated">
      {!loaded && <div className="absolute inset-0 skeleton" />}
      <img
        src={src}
        alt=""
        className={`h-full w-full object-cover transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
        onLoad={() => setLoaded(true)}
        draggable={false}
      />
    </div>
  );
}

/**
 * Barra lateral izquierda colapsable con las sesiones del atleta actual.
 * Se superpone al borde izquierdo del player (no encoge el video de forma dura).
 * Cambio rápido entre videos del mismo atleta.
 */
export default function AthleteRail({ athlete, currentSessionId, open, onClose, onSelect }) {
  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Cerrar sesiones"
          className="absolute inset-0 z-40 bg-black/30"
          onClick={onClose}
          onPointerDown={(e) => e.stopPropagation()}
          style={{ animation: "fade-in 0.2s ease both" }}
        />
      )}
      <aside
        className={`no-select absolute inset-y-0 left-0 z-50 flex w-[min(280px,80vw)] flex-col border-r border-border bg-surface/97 shadow-2xl backdrop-blur-md transition-transform duration-[240ms] ease-[cubic-bezier(0.22,1,0.36,1)] ${
          open ? "translate-x-0" : "-translate-x-full pointer-events-none"
        }`}
        onSelectStart={(e) => e.preventDefault()}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 border-b border-border px-3 py-3">
          <span
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full font-display text-xs font-bold text-white ring-2 ring-black/30"
            style={{ backgroundColor: athlete.accent }}
          >
            {athlete.initials}
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate font-display text-sm font-semibold text-text">{athlete.name}</p>
            <p className="text-[11px] text-soft">{athlete.sessions.length} sesiones</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-muted transition hover:bg-elevated hover:text-text"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="rail-scroll flex-1 space-y-1.5 overflow-y-auto p-2">
          {athlete.sessions.map((s) => {
            const pct = successPct(s);
            const tone = successTone(pct);
            const activeCard = s.id === currentSessionId;
            const StateIcon = s.analysis === "full" ? CheckCircle2 : CircleDashed;
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => {
                  if (!activeCard) onSelect(s);
                }}
                className={`flex w-full items-center gap-2.5 rounded-lg p-2 text-left ring-1 transition ${
                  activeCard
                    ? "bg-elevated ring-accent/60"
                    : "bg-elevated/40 ring-border hover:bg-elevated hover:ring-muted"
                }`}
              >
                <Thumb src={s.thumb} />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-semibold text-text">{s.title}</p>
                  <p className="mt-0.5 text-[11px] tabular-nums text-muted">{s.date || s.videoName || "—"}</p>
                  <div className="mt-1 flex items-center gap-1.5">
                    <span className={`inline-flex items-center gap-1 text-[10px] font-medium ${
                      s.analysis === "none" ? "text-accent" : s.analysis === "partial" ? "text-warn" : "text-ok"
                    }`}>
                      <StateIcon className="h-3 w-3" />
                      {ANALYSIS_LABEL[s.analysis]}
                    </span>
                    {pct != null && (
                      <span className={`rounded px-1 py-px text-[10px] font-bold tabular-nums ${tone.text}`}>
                        {pct}%
                      </span>
                    )}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </aside>
    </>
  );
}
