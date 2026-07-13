import { useMemo, useRef, useState } from "react";
import { Pause, Play, Plus } from "lucide-react";
import { phaseColor } from "../mock/data";

/**
 * Timeline PERMANENTE (barra inferior fija del watch). Sin cajón, siempre visible.
 *   - Scrub con preview (thumb JPEG real).
 *   - Marcadores de hop: tap → seek · long-press (~450 ms) + drag → reposicionar.
 *   - Menú Fase: mover hop existente aquí / crear hop / aterrizaje / carrera.
 */
export default function Timeline({
  frames,
  frameCount,
  frameIndex,
  playing,
  onTogglePlay,
  contacts,
  onSeek,
  onPreviewChange,
  onAddPhase,
  onMoveContact,
}) {
  const trackRef = useRef(null);
  const longPress = useRef(null);
  const dragHop = useRef(null);
  const [preview, setPreview] = useState(null);
  const [menu, setMenu] = useState(null);
  const [ghost, setGhost] = useState(null); // { id, label, color, index }

  const shown = preview ?? frameIndex;
  const pct = (i) => `${(i / Math.max(1, frameCount - 1)) * 100}%`;

  const hopOrder = useMemo(() => {
    const map = {};
    let n = 0;
    contacts.forEach((c) => {
      if (c.type === "hop") map[c.id] = n++;
    });
    return map;
  }, [contacts]);

  const existingHops = useMemo(
    () => contacts.filter((c) => c.type === "hop" || c.type === "landing"),
    [contacts],
  );

  const ticks = useMemo(() => {
    if (!frameCount || !frames?.length) return [];
    const count = 6;
    return Array.from({ length: count }, (_, k) => {
      const i = Math.round((k / (count - 1)) * (frameCount - 1));
      return { i, frameId: frames[i]?.frameId ?? i };
    });
  }, [frames, frameCount]);

  function indexFromClientX(clientX) {
    const el = trackRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return Math.round(t * (frameCount - 1));
  }

  function clearLongPress() {
    if (longPress.current) {
      window.clearTimeout(longPress.current);
      longPress.current = null;
    }
  }

  function onTrackDown(e) {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    if (e.target.closest?.("[data-hop-marker]")) return;
    const i = indexFromClientX(e.clientX);
    setPreview(i);
    onPreviewChange?.(i);
    e.currentTarget.setPointerCapture?.(e.pointerId);
    longPress.current = window.setTimeout(() => {
      setMenu({ index: i, x: e.clientX, y: e.clientY });
      longPress.current = null;
    }, 480);
  }
  function onTrackMove(e) {
    if (preview == null) return;
    clearLongPress();
    const i = indexFromClientX(e.clientX);
    setPreview(i);
    onPreviewChange?.(i);
  }
  function onTrackUp() {
    clearLongPress();
    if (menu) return;
    if (preview != null) onSeek(preview);
    setPreview(null);
    onPreviewChange?.(null);
  }

  function onMarkerPointerDown(e, c, color) {
    e.stopPropagation();
    if (e.button !== 0 && e.pointerType === "mouse") return;
    const startX = e.clientX;
    const startIndex = c.index;
    dragHop.current = { id: c.id, label: c.label, color, startX, startIndex, dragging: false, moved: false };
    e.currentTarget.setPointerCapture?.(e.pointerId);
    clearLongPress();
    longPress.current = window.setTimeout(() => {
      if (!dragHop.current || dragHop.current.id !== c.id) return;
      dragHop.current.dragging = true;
      setGhost({ id: c.id, label: c.label, color, index: c.index });
      longPress.current = null;
    }, 450);
  }

  function onMarkerPointerMove(e) {
    const d = dragHop.current;
    if (!d) return;
    const dx = Math.abs(e.clientX - d.startX);
    if (dx > 6) d.moved = true;
    if (d.dragging) {
      e.stopPropagation();
      const i = indexFromClientX(e.clientX);
      setGhost((g) => (g ? { ...g, index: i } : g));
      return;
    }
    if (d.moved) clearLongPress();
  }

  function onMarkerPointerUp(e) {
    e.stopPropagation();
    clearLongPress();
    const d = dragHop.current;
    dragHop.current = null;
    if (!d) return;
    if (d.dragging) {
      const i = ghost?.index ?? indexFromClientX(e.clientX);
      setGhost(null);
      if (i !== d.startIndex) onMoveContact?.(d.id, i);
      else onSeek(d.startIndex);
      return;
    }
    setGhost(null);
    if (!d.moved) onSeek(d.startIndex);
  }

  return (
    <div
      data-chrome
      className="no-select relative z-30 shrink-0 touch-none border-t border-border bg-surface/95 px-4 pb-2.5 pt-2 backdrop-blur-md"
      onSelectStart={(e) => e.preventDefault()}
      onDragStart={(e) => e.preventDefault()}
    >
      <div className="mb-1.5 flex items-center gap-3">
        <button
          type="button"
          onClick={onTogglePlay}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-text text-bg transition hover:scale-105"
          title={playing ? "Pausar" : "Reproducir"}
        >
          {playing ? <Pause className="h-4 w-4 fill-current" /> : <Play className="ml-0.5 h-4 w-4 fill-current" />}
        </button>

        <div className="flex items-baseline gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-soft">Frame</span>
          <span className="font-display text-lg font-bold tabular-nums leading-none text-text">
            {frames[shown].frameId.toString().padStart(6, "0")}
          </span>
          <span className="text-xs tabular-nums text-muted">
            {shown + 1} / {frameCount}
          </span>
        </div>

        <button
          type="button"
          onClick={() => setMenu({ index: shown, center: true })}
          className="ml-auto inline-flex items-center gap-1 rounded-md bg-elevated px-2.5 py-1.5 text-[11px] font-semibold text-muted ring-1 ring-border transition hover:text-text"
        >
          <Plus className="h-3 w-3" /> Fase
        </button>
      </div>

      {/* Marcadores de hop (encima de la pista, pointer-events propios) */}
      <div className="relative z-20 mb-1 h-6 pointer-events-none">
        {contacts.map((c) => {
          const color = phaseColor(c, hopOrder[c.id] ?? 0);
          const hiding = ghost?.id === c.id;
          return (
            <button
              key={c.id}
              type="button"
              data-hop-marker
              onPointerDown={(e) => onMarkerPointerDown(e, c, color)}
              onPointerMove={onMarkerPointerMove}
              onPointerUp={onMarkerPointerUp}
              onPointerCancel={() => {
                clearLongPress();
                dragHop.current = null;
                setGhost(null);
              }}
              className={`pointer-events-auto absolute top-0 z-30 -translate-x-1/2 touch-none rounded-md px-2 py-0.5 text-[11px] font-extrabold text-black shadow-md transition hover:scale-110 ${
                hiding ? "opacity-30" : ""
              }`}
              style={{ left: pct(c.index), backgroundColor: color, boxShadow: `0 2px 8px ${color}66` }}
              title={`${c.label} · frame ${c.frameId} · mantén para mover`}
            >
              {c.label}
            </button>
          );
        })}
        {ghost && (
          <div
            className="pointer-events-none absolute top-0 z-40 -translate-x-1/2 rounded-md px-2 py-0.5 text-[11px] font-extrabold text-black opacity-80 ring-2 ring-white/80"
            style={{ left: pct(ghost.index), backgroundColor: ghost.color }}
          >
            {ghost.label}
          </div>
        )}
      </div>

      <div className="relative">
        {preview != null && !ghost && (
          <div
            className="pointer-events-none absolute bottom-full z-20 mb-2 w-36 -translate-x-1/2 overflow-hidden rounded-md shadow-xl ring-1 ring-border"
            style={{ left: pct(shown) }}
          >
            <img src={frames[shown].raw} alt="" className="aspect-video w-full object-cover" />
            <div className="bg-elevated px-2 py-1 text-center text-[10px] tabular-nums text-muted">
              {frames[shown].frameId.toString().padStart(6, "0")}
            </div>
          </div>
        )}

        <div
          ref={trackRef}
          className="relative z-10 h-12 cursor-ew-resize touch-none overflow-hidden rounded-md bg-elevated ring-1 ring-border"
          onPointerDown={onTrackDown}
          onPointerMove={onTrackMove}
          onPointerUp={onTrackUp}
          onPointerCancel={() => {
            setPreview(null);
            onPreviewChange?.(null);
            clearLongPress();
          }}
        >
          <div className="absolute inset-0 flex">
            {frames
              // Cada 5.º frame + lazy: evita stampede de N×7s al abrir (túnel Colab)
              .filter((_, i) => i % 5 === 0)
              .map((f) => (
                <img
                  key={f.index}
                  src={f.raw}
                  alt=""
                  loading="lazy"
                  decoding="async"
                  className="h-full w-[64px] shrink-0 object-cover opacity-70"
                  draggable={false}
                />
              ))}
          </div>
          <div className="absolute inset-0 bg-gradient-to-t from-black/45 to-transparent" />

          {contacts.map((c) => {
            const color = phaseColor(c, hopOrder[c.id] ?? 0);
            return (
              <div
                key={c.id}
                className="pointer-events-none absolute inset-y-0 z-10 w-0.5 -translate-x-1/2"
                style={{ left: pct(c.index), backgroundColor: color, opacity: 0.85 }}
              />
            );
          })}
          {ghost && (
            <div
              className="pointer-events-none absolute inset-y-0 z-15 w-0.5 -translate-x-1/2 bg-white/80"
              style={{ left: pct(ghost.index) }}
            />
          )}

          <div
            className="pointer-events-none absolute inset-y-0 z-20 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_8px_rgba(255,255,255,0.7)]"
            style={{ left: pct(shown) }}
          />
        </div>

        <div className="relative mt-1 h-3">
          {ticks.map((t) => (
            <span
              key={t.i}
              className="absolute -translate-x-1/2 text-[9px] tabular-nums text-soft"
              style={{ left: pct(t.i) }}
            >
              {t.frameId}
            </span>
          ))}
        </div>
      </div>

      {menu && (
        <div
          className="fixed z-[60] max-h-[70vh] w-48 overflow-y-auto rounded-lg border border-border bg-elevated p-1 shadow-xl"
          style={
            menu.center
              ? { left: "50%", bottom: 120, transform: "translateX(-50%)" }
              : { left: Math.min(menu.x, window.innerWidth - 200), top: Math.max(12, menu.y - 180) }
          }
          onPointerDown={(e) => e.stopPropagation()}
        >
          <p className="px-2 py-1 text-[10px] uppercase tracking-wider text-soft">
            Frame {frames[menu.index].frameId.toString().padStart(6, "0")}
          </p>
          {existingHops.length > 0 && (
            <>
              <p className="px-2 pt-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Mover aquí</p>
              {existingHops.map((c) => (
                <button
                  key={`move-${c.id}`}
                  type="button"
                  className="block w-full rounded px-3 py-2 text-left text-sm transition hover:bg-surface"
                  onClick={() => {
                    onMoveContact?.(c.id, menu.index);
                    setMenu(null);
                    setPreview(null);
                  }}
                >
                  Mover {c.label} aquí
                </button>
              ))}
              <div className="my-1 h-px bg-border" />
            </>
          )}
          {[
            { label: "Crear hop", type: "hop" },
            { label: "Crear aterrizaje", type: "landing" },
            { label: "Etiquetar carrera", type: "approach" },
          ].map((opt) => (
            <button
              key={opt.type}
              type="button"
              className="block w-full rounded px-3 py-2 text-left text-sm transition hover:bg-surface"
              onClick={() => {
                onAddPhase(menu.index, opt.type);
                setMenu(null);
                setPreview(null);
              }}
            >
              {opt.label}
            </button>
          ))}
          <button
            type="button"
            className="block w-full rounded px-3 py-2 text-left text-sm text-muted transition hover:bg-surface"
            onClick={() => {
              setMenu(null);
              setPreview(null);
            }}
          >
            Cancelar
          </button>
        </div>
      )}
    </div>
  );
}
