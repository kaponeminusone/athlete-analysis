"""
Local review server for the YOLO prototype UI.

It serves the UI, video/frame media, discovers output/<video_name>/analysis.json,
and can launch run.py when the analysis does not exist yet.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
UI_ROOT = ROOT / "ui"
UI_DIST_ROOT = UI_ROOT / "dist"
JOBS: dict[str, dict] = {}


def frontend_root() -> Path:
    return UI_DIST_ROOT if UI_DIST_ROOT.exists() else UI_ROOT


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def media_url(path: Path | None) -> str:
    if not path:
        return ""
    return f"/media?path={path.resolve().as_posix()}"


def video_stem(video_path: str) -> str:
    return Path(video_path).stem or Path(video_path).name


def find_output_dir(video_path: str) -> Path:
    stem = video_stem(video_path)
    candidates = [
        OUTPUT_ROOT / stem,
        OUTPUT_ROOT / stem.upper(),
        OUTPUT_ROOT / stem.lower(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return OUTPUT_ROOT / stem


def indexed_images(folder: Path, prefix: str) -> dict[str, str]:
    images: dict[str, str] = {}
    if not folder.exists():
        return images

    for path in folder.glob("*.jpg"):
        stem = path.stem.replace(prefix, "")
        key = stem.lstrip("0") or "0"
        padded = stem if stem.isdigit() else key
        images[key] = media_url(path)
        images[padded] = media_url(path)
    return images


def project_payload(video_path: str) -> dict:
    output_dir = find_output_dir(video_path)
    analysis_path = output_dir / "analysis.json"
    chart_path = output_dir / "charts" / "camera_angle_timeline.png"
    video_file = Path(video_path)

    analysis_data = None
    if analysis_path.exists():
        with analysis_path.open("r", encoding="utf-8") as f:
            analysis_data = json.load(f)

    return {
        "video": {
            "path": str(video_file),
            "name": video_file.name or f"{output_dir.name}.mp4",
            "exists": video_file.exists(),
            "url": media_url(video_file) if video_file.exists() else "",
        },
        "output": {
            "path": str(output_dir),
            "exists": output_dir.exists(),
        },
        "analysis": {
            "path": str(analysis_path),
            "exists": analysis_path.exists(),
            "data": analysis_data,
            "frames": analysis_data.get("frames", []) if analysis_data else [],
        },
        "assets": {
            "annotated": indexed_images(output_dir / "annotated", "annotated_"),
            "frames": indexed_images(output_dir / "frames", "frame_"),
            "chart": media_url(chart_path) if chart_path.exists() else "",
        },
    }


def run_analysis_job(job_id: str, video_path: str) -> None:
    output_dir = find_output_dir(video_path)
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        video_path,
        "--output",
        str(output_dir),
        "--annotate-every",
        "1",
    ]
    JOBS[job_id].update({"status": "running", "command": command, "log": ""})
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            JOBS[job_id]["log"] += line
        return_code = process.wait()
        JOBS[job_id]["status"] = "done" if return_code == 0 else "failed"
        JOBS[job_id]["return_code"] = return_code
    except Exception as exc:  # noqa: BLE001 - surfaced in local UI log
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["log"] += f"\n[server] {exc}\n"


class ReviewHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean_path = unquote(parsed.path)
        root = frontend_root()
        if clean_path == "/":
            return str(root / "index.html")
        if clean_path.startswith("/ui/"):
            relative = clean_path.removeprefix("/ui/") or "index.html"
            return str(root / relative)
        return str(root / clean_path.lstrip("/"))

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        if parsed.path == "/api/project":
            video_path = parse_qs(parsed.query).get("video_path", [""])[0]
            if not video_path:
                json_response(self, {"error": "Missing video_path"}, 400)
                return
            json_response(self, project_payload(video_path))
            return

        if parsed.path == "/api/demo":
            demo_analysis = next(OUTPUT_ROOT.glob("*/analysis.json"), None)
            if not demo_analysis:
                json_response(self, {"error": "No demo analysis found in output/"}, 404)
                return
            demo_video = demo_analysis.parent.name + ".mp4"
            json_response(self, project_payload(demo_video))
            return

        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            if not job:
                json_response(self, {"error": "Job not found"}, 404)
                return
            json_response(self, job)
            return

        if parsed.path == "/media":
            path = parse_qs(parsed.query).get("path", [""])[0]
            self.serve_media(Path(path))
            return

        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            json_response(self, {"error": "Not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        video_path = payload.get("video_path", "")
        if not video_path:
            json_response(self, {"error": "Missing video_path"}, 400)
            return
        if not Path(video_path).exists():
            json_response(self, {"error": f"Video not found: {video_path}"}, 404)
            return

        job_id = str(int(time.time() * 1000))
        JOBS[job_id] = {"job_id": job_id, "video_path": video_path, "status": "queued", "log": ""}
        thread = threading.Thread(target=run_analysis_job, args=(job_id, video_path), daemon=True)
        thread.start()
        json_response(self, {"job_id": job_id})

    def serve_media(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, "Media not found")
            return

        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1

        if range_header and range_header.startswith("bytes="):
            raw_start, _, raw_end = range_header.replace("bytes=", "").partition("-")
            start = int(raw_start or 0)
            end = int(raw_end or end)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        else:
            self.send_response(200)

        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()

        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1024 * 512, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local YOLO review UI server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.chdir(ROOT)
    server = ThreadingHTTPServer(("localhost", args.port), ReviewHandler)
    print(f"Review UI: http://localhost:{args.port}/ui/")
    server.serve_forever()


if __name__ == "__main__":
    main()
