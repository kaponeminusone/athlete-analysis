"""Quick latency bench against local api_server."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
PATH = urllib.parse.quote("C:/Users/RALSEI/Documents/Cursor/prototype/VOD2.MOV")


def t(label: str, url: str) -> None:
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
        status = r.status
    ms = (time.perf_counter() - t0) * 1000
    print(f"{label:14} {ms:7.0f} ms  {status}  bytes={len(data)}")


def main() -> None:
    t("status", f"{BASE}/status")
    t("videos", f"{BASE}/api/videos")
    t("project", f"{BASE}/api/project?video_path={PATH}")
    t("frame0", f"{BASE}/frame/VOD2/0?annotated=0")
    t("frame0_again", f"{BASE}/frame/VOD2/0?annotated=0")
    t("frame1", f"{BASE}/frame/VOD2/1?annotated=0")
    t("frame0a", f"{BASE}/frame/VOD2/0?annotated=1")
    t("frame0a_again", f"{BASE}/frame/VOD2/0?annotated=1")
    t("frame50", f"{BASE}/frame/VOD2/50?annotated=0")
    t("media", f"{BASE}/media?path={PATH}")
    t("venue_m", f"{BASE}/api/venue/model")
    with urllib.request.urlopen(f"{BASE}/api/venue/model", timeout=30) as r:
        print("venue:", r.read().decode()[:300])
    with urllib.request.urlopen(f"{BASE}/api/videos", timeout=30) as r:
        payload = json.loads(r.read().decode())
    videos = payload.get("videos", payload if isinstance(payload, list) else [])
    print(f"video_count={len(videos)}")
    for v in videos[:8]:
        print(
            f"  - {v.get('name')} analyzed={v.get('has_analysis')} "
            f"refined={v.get('has_refined')}"
        )


if __name__ == "__main__":
    main()
