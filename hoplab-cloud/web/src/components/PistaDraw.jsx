import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

/**
 * Capa "Editar pista" — dibuja el polígono de la pista DIRECTO sobre el frame
 * (sin sheet ni oscurecido). Controles (Aceptar / Deshacer / Limpiar) en el
 * mini-header, comandados por ref.
 *   - tap = coloca un punto (vértice).
 *   - long-press + arrastrar = traza la polilínea en vivo (animada); soltar termina.
 */
const PistaDraw = forwardRef(function PistaDraw({ active, onCountChange, onAccept, toast }, ref) {
  const svgRef = useRef(null);
  const longPress = useRef(null);
  const mode = useRef("idle"); // idle | pending | dragging
  const start = useRef(null);
  const [points, setPoints] = useState([]);
  const [live, setLive] = useState(null);

  useEffect(() => {
    onCountChange?.(points.length);
  }, [points, onCountChange]);

  useImperativeHandle(ref, () => ({
    undo() {
      setPoints((p) => p.slice(0, -1));
    },
    clear() {
      setPoints([]);
      setLive(null);
      mode.current = "idle";
    },
    accept() {
      if (points.length < 3) {
        toast("Colocá al menos 3 puntos para definir la pista");
        return false;
      }
      // Envía el polígono al motor (calibración real); onAccept maneja sus toasts.
      onAccept?.(points.map((p) => ({ x: p.x, y: p.y })));
      setPoints([]);
      setLive(null);
      mode.current = "idle";
      return true;
    },
  }));

  function toLocal(e) {
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * 100,
      y: ((e.clientY - rect.top) / rect.height) * 100,
    };
  }

  function onDown(e) {
    if (!active) return;
    e.stopPropagation();
    const pt = toLocal(e);
    start.current = pt;
    mode.current = "pending";
    e.currentTarget.setPointerCapture?.(e.pointerId);
    longPress.current = window.setTimeout(() => {
      mode.current = "dragging";
      longPress.current = null;
      setLive([start.current]);
    }, 300);
  }

  function onMove(e) {
    if (!active) return;
    if (mode.current === "pending") {
      const pt = toLocal(e);
      if (Math.hypot(pt.x - start.current.x, pt.y - start.current.y) > 2) {
        if (longPress.current) {
          window.clearTimeout(longPress.current);
          longPress.current = null;
        }
        mode.current = "idle";
      }
      return;
    }
    if (mode.current !== "dragging") return;
    e.stopPropagation();
    const pt = toLocal(e);
    setLive((prev) => {
      if (!prev) return [pt];
      const last = prev[prev.length - 1];
      if (Math.hypot(pt.x - last.x, pt.y - last.y) < 1.2) return prev;
      return [...prev, pt];
    });
  }

  function onUp(e) {
    if (!active) return;
    e.stopPropagation();
    if (longPress.current) {
      window.clearTimeout(longPress.current);
      longPress.current = null;
    }
    if (mode.current === "pending") {
      setPoints((p) => [...p, start.current]);
    } else if (mode.current === "dragging" && live) {
      setPoints((p) => [...p, ...live]);
    }
    mode.current = "idle";
    setLive(null);
  }

  const all = live ? [...points, ...live] : points;
  const polyPath = all.length
    ? all.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ")
    : "";

  if (!active && points.length === 0) return null;

  return (
    <svg
      ref={svgRef}
      data-chrome
      className={`absolute inset-0 z-30 h-full w-full touch-none select-none ${active ? "pointer-events-auto cursor-crosshair" : "pointer-events-none"}`}
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      onSelectStart={(e) => e.preventDefault()}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={() => {
        mode.current = "idle";
        setLive(null);
        if (longPress.current) window.clearTimeout(longPress.current);
      }}
    >
      {all.length > 1 && (
        <path
          d={polyPath}
          fill={all.length > 2 ? "rgba(34,211,238,0.14)" : "none"}
          stroke="#22d3ee"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
          className={live ? "pista-live" : ""}
        />
      )}
      {all.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r={i === all.length - 1 ? 1.4 : 1}
          fill={i === all.length - 1 ? "#fff" : "#22d3ee"}
          className={i === all.length - 1 && live ? "pista-pulse" : ""}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
});

export default PistaDraw;
