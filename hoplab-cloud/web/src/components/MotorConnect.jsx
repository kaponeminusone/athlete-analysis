/**
 * Pantalla/modal "Conectar motor Colab".
 * Aparece cuando la API no responde o el usuario lo abre manualmente.
 */
import { useState } from "react";
import { checkApi, saveMotorUrl, getApiBase } from "../api/client";

export default function MotorConnect({ onConnected, onClose }) {
  const [url, setUrl] = useState(getApiBase);
  const [status, setStatus] = useState(null); // null | "checking" | "ok" | "error"
  const [error, setError] = useState("");

  async function handleTest() {
    const clean = url.trim().replace(/\/$/, "");
    if (!clean.startsWith("http")) {
      setStatus("error");
      setError("La URL debe empezar por https://");
      return;
    }
    setStatus("checking");
    setError("");
    saveMotorUrl(clean);
    // Forzar recarga del módulo en runtime sería complejo; usamos window.location
    const ok = await checkApi();
    if (ok) {
      setStatus("ok");
      onConnected?.(clean);
    } else {
      setStatus("error");
      setError("El motor no respondió. Asegúrate de que el notebook Colab está corriendo.");
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") handleTest();
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-bg/90 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-2xl border border-border bg-elevated shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-6 py-5 border-b border-border flex items-start justify-between gap-2">
          <div>
          <h1 className="text-lg font-semibold text-text">Conectar motor</h1>
          <p className="text-sm text-muted mt-1">
            El análisis se ejecuta en Google Colab. Necesitas arrancar el notebook y pegar
            la URL del túnel aquí.
          </p>
          </div>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="text-muted hover:text-text text-xl leading-none mt-0.5 flex-shrink-0"
              aria-label="Cerrar"
            >
              ✕
            </button>
          )}
        </div>

        {/* Steps */}
        <div className="px-6 py-4 bg-surface/50 border-b border-border">
          <ol className="text-sm text-muted space-y-1 list-none">
            <li>
              <span className="text-accent font-semibold mr-2">1.</span>
              Abre el notebook Colab:{" "}
              <a
                href="https://colab.research.google.com/github/TU-USUARIO/TU-REPO/blob/main/hoplab-cloud/colab/HopLab_Server.ipynb"
                target="_blank"
                rel="noreferrer"
                className="text-accent underline"
              >
                Open in Colab ↗
              </a>
            </li>
            <li>
              <span className="text-accent font-semibold mr-2">2.</span>
              <strong className="text-text">Runtime → Run all</strong> (espera ~2 min)
            </li>
            <li>
              <span className="text-accent font-semibold mr-2">3.</span>
              Copia la URL <code className="text-soft">https://…trycloudflare.com</code> que aparece
            </li>
            <li>
              <span className="text-accent font-semibold mr-2">4.</span>
              Pégala abajo y pulsa <strong className="text-text">Conectar</strong>
            </li>
          </ol>
        </div>

        {/* Input */}
        <div className="px-6 py-5 space-y-3">
          <label className="block">
            <span className="text-xs font-medium text-soft uppercase tracking-wide">URL del motor</span>
            <input
              type="url"
              value={url}
              onChange={(e) => { setUrl(e.target.value); setStatus(null); }}
              onKeyDown={handleKeyDown}
              placeholder="https://abc123.trycloudflare.com"
              className="mt-1.5 w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-muted outline-none focus:ring-2 focus:ring-accent"
            />
          </label>

          {status === "error" && (
            <p className="text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">{error}</p>
          )}
          {status === "ok" && (
            <p className="text-sm text-green-400 bg-green-400/10 rounded-lg px-3 py-2">
              ✅ Motor conectado correctamente
            </p>
          )}

          <button
            type="button"
            onClick={handleTest}
            disabled={status === "checking"}
            className="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-bg transition hover:opacity-90 disabled:opacity-50"
          >
            {status === "checking" ? "Comprobando…" : "Probar y conectar"}
          </button>
        </div>

        {/* Footer note */}
        <div className="px-6 pb-4 text-xs text-muted text-center">
          La URL cambia cada sesión de Colab. Los datos se guardan en tu Google Drive.
        </div>
      </div>
    </div>
  );
}
