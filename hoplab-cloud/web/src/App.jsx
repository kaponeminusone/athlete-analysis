import { useCallback, useEffect, useRef, useState } from "react";
import LibraryHome from "./components/LibraryHome";
import WatchPage from "./components/WatchPage";
import MotorConnect from "./components/MotorConnect";
import { loadLibraryFromApi } from "./api/mapSession";
import { checkApi, getApiBase, saveMotorUrl } from "./api/client";
import { ATHLETES } from "./mock/data";

export default function App() {
  const [view, setView] = useState("library");
  const [current, setCurrent] = useState(null);
  const [toastMsg, setToastMsg] = useState(null);
  const [athletes, setAthletes] = useState(ATHLETES);
  const [librarySource, setLibrarySource] = useState("mock");
  const [libraryLoading, setLibraryLoading] = useState(true);

  // Estado del motor Colab
  const [motorOk, setMotorOk] = useState(false);
  const [showMotorConnect, setShowMotorConnect] = useState(false);
  const initDone = useRef(false);

  const toast = useCallback((msg) => {
    setToastMsg({ msg, id: Date.now() });
    window.setTimeout(() => setToastMsg(null), 2800);
  }, []);

  /** Carga la biblioteca desde la API y actualiza el estado. */
  const loadLibrary = useCallback(async () => {
    setLibraryLoading(true);
    try {
      const { athletes: apiAthletes, empty } = await loadLibraryFromApi();
      if (empty || !apiAthletes.length) {
        setAthletes(ATHLETES);
        setLibrarySource("mock");
      } else {
        setAthletes(apiAthletes);
        setLibrarySource("api");
      }
    } catch {
      setAthletes(ATHLETES);
      setLibrarySource("mock");
    } finally {
      setLibraryLoading(false);
    }
  }, []);

  /** Comprueba si hay motor disponible; si no, muestra pantalla de conexión. */
  const checkMotor = useCallback(async (showConnectIfFail = true) => {
    const ok = await checkApi();
    setMotorOk(ok);
    if (!ok && showConnectIfFail) {
      setShowMotorConnect(true);
    }
    return ok;
  }, []);

  // Al arrancar: verificar motor → cargar biblioteca
  useEffect(() => {
    if (initDone.current) return;
    initDone.current = true;
    (async () => {
      const hasBase = Boolean(getApiBase());
      if (!hasBase) {
        // Sin URL configurada: mostrar pantalla de conexión directamente
        setLibraryLoading(false);
        setShowMotorConnect(true);
        return;
      }
      const ok = await checkMotor(true);
      if (ok) {
        await loadLibrary();
      } else {
        setLibraryLoading(false);
      }
    })();
  }, [checkMotor, loadLibrary]);

  /** Callback cuando el usuario conecta el motor con éxito. */
  const handleMotorConnected = useCallback(
    async (url) => {
      saveMotorUrl(url);
      setMotorOk(true);
      setShowMotorConnect(false);
      toast("Motor conectado — cargando biblioteca...");
      await loadLibrary();
    },
    [loadLibrary, toast],
  );

  const openSession = useCallback((athlete, sess) => {
    setCurrent({ athlete, session: sess });
    setView("watch");
  }, []);

  const selectSession = useCallback((sess) => {
    setCurrent((c) => (c ? { ...c, session: sess } : c));
  }, []);

  const patchSession = useCallback((sessionId, patch) => {
    setAthletes((prev) =>
      prev.map((a) => ({
        ...a,
        sessions: a.sessions.map((s) => (s.id === sessionId ? { ...s, ...patch } : s)),
      })),
    );
    setCurrent((c) => {
      if (!c || c.session.id !== sessionId) return c;
      return { ...c, session: { ...c.session, ...patch } };
    });
  }, []);

  return (
    <div className="h-full w-full overflow-hidden bg-bg text-text">
      {/* Botón flotante "Conectar motor" en biblioteca */}
      {view === "library" && (
        <button
          type="button"
          onClick={() => setShowMotorConnect(true)}
          title="Configurar URL del motor Colab"
          className={`fixed bottom-4 right-4 z-50 rounded-full px-3 py-2 text-xs font-semibold shadow-lg transition ${
            motorOk
              ? "bg-green-500/20 text-green-400 border border-green-500/30"
              : "bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse"
          }`}
        >
          {motorOk ? "⚡ Motor activo" : "⚠ Sin motor"}
        </button>
      )}

      {view === "library" && (
        <LibraryHome
          athletes={athletes}
          onOpenSession={openSession}
          toast={toast}
          loading={libraryLoading}
          librarySource={librarySource}
        />
      )}

      {view === "watch" && current && (
        <WatchPage
          athlete={current.athlete}
          session={current.session}
          onBack={() => setView("library")}
          onSelectSession={selectSession}
          onSessionPatched={patchSession}
          toast={toast}
        />
      )}

      {/* Modal de conexión al motor */}
      {showMotorConnect && (
        <MotorConnect
          onConnected={handleMotorConnected}
          onClose={() => setShowMotorConnect(false)}
        />
      )}

      {toastMsg && (
        <div
          key={toastMsg.id}
          className="toast-pop fixed bottom-6 left-1/2 z-[90] -translate-x-1/2 rounded-lg border border-border bg-elevated px-4 py-2.5 text-sm font-medium text-text shadow-lg"
        >
          {toastMsg.msg}
        </div>
      )}
    </div>
  );
}
