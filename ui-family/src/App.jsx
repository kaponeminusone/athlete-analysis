import { useCallback, useEffect, useState } from "react";
import LibraryHome from "./components/LibraryHome";
import WatchPage from "./components/WatchPage";
import { loadLibraryFromApi } from "./api/mapSession";
import { ATHLETES } from "./mock/data";

export default function App() {
  const [view, setView] = useState("library");
  // { athlete, session } — mantenemos el atleta para la barra lateral de sesiones.
  const [current, setCurrent] = useState(null);
  const [toastMsg, setToastMsg] = useState(null);
  const [athletes, setAthletes] = useState(ATHLETES);
  const [librarySource, setLibrarySource] = useState("mock"); // mock | api
  const [libraryLoading, setLibraryLoading] = useState(true);

  const toast = useCallback((msg) => {
    setToastMsg({ msg, id: Date.now() });
    window.setTimeout(() => setToastMsg(null), 2800);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLibraryLoading(true);
      try {
        const { athletes: apiAthletes, empty } = await loadLibraryFromApi();
        if (cancelled) return;
        if (empty || !apiAthletes.length) {
          setAthletes(ATHLETES);
          setLibrarySource("mock");
          toast("API sin videos — usando biblioteca de demostración");
        } else {
          setAthletes(apiAthletes);
          setLibrarySource("api");
        }
      } catch {
        if (cancelled) return;
        setAthletes(ATHLETES);
        setLibrarySource("mock");
        toast("API no disponible — usando datos de demostración");
      } finally {
        if (!cancelled) setLibraryLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [toast]);

  const openSession = useCallback((athlete, sess) => {
    setCurrent({ athlete, session: sess });
    setView("watch");
  }, []);

  // Cambio rápido de sesión desde la barra lateral (mismo atleta).
  const selectSession = useCallback((sess) => {
    setCurrent((c) => (c ? { ...c, session: sess } : c));
  }, []);

  // Tras reanálisis, refrescar la sesión en biblioteca + current.
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
