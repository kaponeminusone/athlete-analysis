/**
 * Caché de frames en memoria (blob URLs) para scrub fluido sobre túnel Colab.
 * Precarga en background con concurrencia limitada; prioriza el frame actual y vecinos.
 */

const MAX_CONCURRENT = 4;

/** @type {Map<string, string>} cacheKey -> blob URL */
const blobByKey = new Map();
/** @type {Map<string, Promise<string>>} */
const inflight = new Map();

/** @type {{ url: string, version: number }[]} */
let queue = [];
/** @type {Set<string>} */
const queued = new Set();
let active = 0;
let generation = 0;

export function cacheKey(url, version = 0) {
  if (!url) return "";
  const v = version ?? 0;
  return `${url}${url.includes("?") ? "&" : "?"}v=${v}`;
}

export function clearFrameCache() {
  generation += 1;
  for (const blob of blobByKey.values()) {
    try {
      URL.revokeObjectURL(blob);
    } catch {
      /* ignore */
    }
  }
  blobByKey.clear();
  inflight.clear();
  queue = [];
  queued.clear();
  active = 0;
}

export function isFrameCached(url, version = 0) {
  const key = cacheKey(url, version);
  return key ? blobByKey.has(key) : false;
}

export function ensureBlobUrl(url, version = 0) {
  const key = cacheKey(url, version);
  if (!key) return Promise.reject(new Error("empty url"));
  const hit = blobByKey.get(key);
  if (hit) return Promise.resolve(hit);

  const pending = inflight.get(key);
  if (pending) return pending;

  const p = fetch(key)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.blob();
    })
    .then((blob) => {
      const objectUrl = URL.createObjectURL(blob);
      blobByKey.set(key, objectUrl);
      inflight.delete(key);
      return objectUrl;
    })
    .catch((err) => {
      inflight.delete(key);
      throw err;
    });

  inflight.set(key, p);
  return p;
}

function pump(gen) {
  while (active < MAX_CONCURRENT && queue.length > 0 && gen === generation) {
    const item = queue.shift();
    const key = cacheKey(item.url, item.version);
    queued.delete(key);
    active += 1;
    ensureBlobUrl(item.url, item.version)
      .catch(() => {})
      .finally(() => {
        active -= 1;
        if (gen === generation) pump(gen);
      });
  }
}

function enqueueItems(items, { front = false } = {}) {
  const gen = generation;
  const toAdd = [];
  for (const item of items) {
    const key = cacheKey(item.url, item.version ?? 0);
    if (!key || blobByKey.has(key) || inflight.has(key) || queued.has(key)) continue;
    toAdd.push({ url: item.url, version: item.version ?? 0 });
    queued.add(key);
  }
  if (!toAdd.length) return;
  if (front) queue = [...toAdd, ...queue];
  else queue.push(...toAdd);
  pump(gen);
}

/** Precarga todos los frames. onProgress({ done, total, label }). */
export function warmupFrameUrls(urls, { version = 0, onProgress, label = "frames" } = {}) {
  const gen = generation;
  const unique = [...new Set(urls.filter(Boolean))];
  const total = unique.length;
  let done = 0;

  if (total === 0) {
    onProgress?.({ done: 0, total: 0, label });
    return () => {};
  }

  onProgress?.({ done: 0, total, label });

  const tick = () => {
    if (gen !== generation) return;
    let cached = 0;
    for (const url of unique) {
      if (isFrameCached(url, version)) cached += 1;
    }
    if (cached !== done) {
      done = cached;
      onProgress?.({ done, total, label });
    }
  };

  const interval = window.setInterval(tick, 400);
  enqueueItems(unique.map((url) => ({ url, version })));

  return () => window.clearInterval(interval);
}

/** Prioriza frame actual ± vecinos (al frente de la cola). */
export function prioritizeFrameUrls(urls, { version = 0 } = {}) {
  enqueueItems(
    urls.filter(Boolean).map((url) => ({ url, version })),
    { front: true },
  );
}

export function frameUrlsFromList(frames, { annotated = true } = {}) {
  if (!frames?.length) return [];
  return frames.map((f) => (annotated ? f.annotated : f.raw)).filter(Boolean);
}
