"""
Global in-memory job registry.
Thread-safe: all mutations go through Job.update() which holds a lock.
"""

import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class Job:
    job_id:            str
    status:            str   = "pending"   # pending | running | done | failed
    stage:             str   = ""
    message:           str   = ""
    current_frame:     int   = 0
    total_frames:      int   = 0
    analyzed_frames:   int   = 0
    percent:           float = 0.0
    last_log:          str   = ""
    started_at:        Optional[float] = None
    finished_at:       Optional[float] = None
    result_video_name: Optional[str]   = None
    error:             Optional[str]   = None
    _lock:             threading.Lock  = field(default_factory=threading.Lock, repr=False)

    def update(self, event: dict) -> None:
        with self._lock:
            for key, value in event.items():
                if hasattr(self, key) and not key.startswith("_"):
                    setattr(self, key, value)

    def to_dict(self) -> dict:
        return {
            "job_id":            self.job_id,
            "status":            self.status,
            "stage":             self.stage,
            "message":           self.message,
            "current_frame":     self.current_frame,
            "total_frames":      self.total_frames,
            "analyzed_frames":   self.analyzed_frames,
            "percent":           round(self.percent, 1),
            "last_log":          self.last_log,
            "started_at":        self.started_at,
            "finished_at":       self.finished_at,
            "result_video_name": self.result_video_name,
            "error":             self.error,
        }

    def start(self) -> None:
        self.update({"status": "running", "started_at": time.time()})

    def finish(self, video_name: str = "") -> None:
        self.update({
            "status":            "done",
            "stage":             "done",
            "percent":           100.0,
            "finished_at":       time.time(),
            "result_video_name": video_name,
        })

    def fail(self, error: str) -> None:
        self.update({
            "status":      "failed",
            "error":       error,
            "finished_at": time.time(),
        })


# Global registry
_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def create_job() -> Job:
    job_id = uuid.uuid4().hex[:10]
    job = Job(job_id=job_id)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def list_jobs() -> list[dict]:
    with _JOBS_LOCK:
        return [j.to_dict() for j in _JOBS.values()]


# Type alias for progress callbacks used throughout the pipeline
ProgressCallback = Callable[[dict], None]


def noop_progress(event: dict) -> None:
    """No-op callback for CLI usage."""
    pass


def print_progress(event: dict) -> None:
    """Simple print callback for CLI usage."""
    msg = event.get("message", "")
    if msg:
        print(f"  [{event.get('stage', '')}] {msg}")
