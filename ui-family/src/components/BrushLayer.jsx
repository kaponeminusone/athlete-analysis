import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

const MASK_COLS = 160;
const MASK_ROWS = 90;
/** Expert usaba ±3; triple → radio 9 celdas. */
const STAMP_RADIUS = 9;
/** Expert visual 2.6; triple. */
const STROKE_WIDTH = 7.8;

/**
 * Rasteriza polilíneas en coords percentuales (0–100) a grilla binaria 160×90.
 * Muestrea a lo largo de cada segmento para trazos continuos, no solo vértices.
 */
export function strokesToMask(strokes, cols = MASK_COLS, rows = MASK_ROWS, radius = STAMP_RADIUS) {
  const mask = Array.from({ length: rows }, () => Array(cols).fill(0));

  function stamp(cx, cy) {
    const x0 = Math.max(0, cx - radius);
    const x1 = Math.min(cols - 1, cx + radius);
    const y0 = Math.max(0, cy - radius);
    const y1 = Math.min(rows - 1, cy + radius);
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) {
        mask[y][x] = 1;
      }
    }
  }

  function cellOf(pt) {
    const nx = pt.x / 100;
    const ny = pt.y / 100;
    const cx = Math.max(0, Math.min(cols - 1, Math.floor(nx * cols)));
    const cy = Math.max(0, Math.min(rows - 1, Math.floor(ny * rows)));
    return { cx, cy };
  }

  for (const stroke of strokes || []) {
    if (!stroke?.length) continue;
    for (let i = 0; i < stroke.length; i += 1) {
      const a = cellOf(stroke[i]);
      stamp(a.cx, a.cy);
      if (i === 0) continue;
      const b = cellOf(stroke[i - 1]);
      const dx = a.cx - b.cx;
      const dy = a.cy - b.cy;
      const steps = Math.max(Math.abs(dx), Math.abs(dy), 1);
      for (let s = 1; s < steps; s += 1) {
        const t = s / steps;
        stamp(Math.round(b.cx + dx * t), Math.round(b.cy + dy * t));
      }
    }
  }

  return mask;
}

/**
 * Capa "Corregir atleta" — pincel que se pinta DIRECTO sobre el frame, sin
 * oscurecerlo ni cubrirlo con una sheet. Los controles (Aceptar / Limpiar)
 * viven en el mini-header; se comandan por ref.
 */
const BrushLayer = forwardRef(function BrushLayer({ active, onCountChange }, ref) {
  const svgRef = useRef(null);
  const drawing = useRef(null);
  const strokesRef = useRef([]);
  const [strokes, setStrokes] = useState([]);
  const [localStroke, setLocalStroke] = useState(null);

  useEffect(() => {
    strokesRef.current = strokes;
  }, [strokes]);

  useEffect(() => {
    onCountChange?.(strokes.length);
  }, [strokes, onCountChange]);

  useImperativeHandle(ref, () => ({
    clear() {
      setStrokes([]);
      setLocalStroke(null);
      drawing.current = null;
    },
    /** @returns {{ mask: number[][] } | null} */
    accept() {
      const current = strokesRef.current;
      if (!current.length) return null;
      const mask = strokesToMask(current);
      return { mask };
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
    drawing.current = [pt];
    setLocalStroke([pt]);
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
  function onMove(e) {
    if (!active || !drawing.current) return;
    e.stopPropagation();
    drawing.current.push(toLocal(e));
    setLocalStroke([...drawing.current]);
  }
  function onUp(e) {
    if (!active || !drawing.current) return;
    e.stopPropagation();
    if (drawing.current.length > 1) setStrokes((s) => [...s, drawing.current]);
    drawing.current = null;
    setLocalStroke(null);
  }

  const pathOf = (pts) =>
    !pts?.length ? "" : pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");

  if (!active && strokes.length === 0) return null;

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
        drawing.current = null;
        setLocalStroke(null);
      }}
    >
      {[...strokes, localStroke].filter(Boolean).map((s, i) => (
        <path
          key={i}
          d={pathOf(s)}
          fill="none"
          stroke="#22d3ee"
          strokeWidth={STROKE_WIDTH}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
          opacity="0.92"
        />
      ))}
    </svg>
  );
});

export default BrushLayer;
