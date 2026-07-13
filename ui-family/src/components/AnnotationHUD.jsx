/**
 * HUD de anotación relativo al viewport: MISMO tamaño en CSS sin importar la
 * resolución del frame (usa unidades de viewport, no del <img>).
 * Muestra calidad del frame, número de frame y badge SKIP.
 */
export default function AnnotationHUD({ frame, visible, simplified = false }) {
  if (!visible || !frame) return null;

  const q = frame.quality;
  const qColor = q >= 80 ? "text-ok" : q >= 70 ? "text-warn" : "text-accent";

  return (
    <div
      className="pointer-events-none absolute left-[0.9vh] top-[0.9vh] z-20 flex flex-col gap-[0.6vh] font-ui leading-none"
      style={{ fontSize: "clamp(10px, 1.15vh, 13px)" }}
    >
      <div className="inline-flex items-center gap-2 rounded-md bg-black/65 px-2 py-1.5 ring-1 ring-white/10 backdrop-blur-sm">
        <span className="text-soft">Calidad</span>
        <span className={`font-semibold tabular-nums ${qColor}`}>{q}%</span>
      </div>
      <div className="inline-flex items-center gap-2 rounded-md bg-black/65 px-2 py-1.5 ring-1 ring-white/10 backdrop-blur-sm">
        <span className="text-soft">Frame</span>
        <span className="font-semibold tabular-nums text-text">{frame.frameId.toString().padStart(6, "0")}</span>
        {!simplified && <span className="text-soft">#{frame.index}</span>}
      </div>
      {frame.skip ? (
        <div className="inline-flex w-fit items-center gap-1.5 rounded-md bg-accent/90 px-2 py-1.5 font-semibold text-white">
          SKIP
        </div>
      ) : (
        !simplified && (
          <div className="inline-flex w-fit items-center gap-1.5 rounded-md bg-black/55 px-2 py-1.5 text-soft ring-1 ring-white/10">
            no-skip
          </div>
        )
      )}
    </div>
  );
}
