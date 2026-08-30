"""
Background jobs with honest progress.

WHY A QUEUE AT ALL: ingesting a large PDF takes minutes (hnsw.pdf is 107s to
parse alone). A synchronous upload endpoint would hold an HTTP connection open
for that long, and the design's document list needs QUEUED / PREPARING states
that a blocking call cannot express.

THREE DECISIONS, each a tradeoff worth stating rather than a default:

1. ONE WORKER THREAD, FIFO. Not a process pool, not Celery. Ingest rebuilds the
   whole index, so two concurrent ingests would race on the same files -- and the
   fix for a race is not more workers, it is serialising the work that cannot be
   parallel. FIFO also makes QUEUED a truthful state: a queued job really is
   waiting for a specific job ahead of it.

2. RECORDS ON DISK, not just in memory. A job whose status vanishes when the
   server restarts is indistinguishable from a job that never ran -- the same
   ambiguous-silence failure as `> /dev/null` losing 120 API calls. On-disk
   records also mean a crashed ingest leaves evidence.

3. PROGRESS IS PER-DOCUMENT, NOT PER-PAGE. The design shows
   "Reading pages 12 of 48". We cannot produce that: pymupdf4llm parses a
   document in one opaque call and emits nothing until it finishes. So progress
   reports the document and the stage ("parsing", "chunking", "embedding") and
   the UI must not draw a page counter. Inventing a smooth fake progress bar for
   an opaque operation is the same class of lie as filling in a truncated table
   with plausible numbers.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from . import config

JobState = Literal["queued", "running", "done", "failed", "cancelled"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Progress:
    stage: str = ""            # parsing | chunking | embedding | indexing
    current: int = 0           # documents completed
    total: int = 0             # documents in this job
    detail: str = ""           # the document currently being worked on
    # Deliberately no `percent`. A percentage implies uniform per-document cost,
    # and measured cost ranges from 0.4s (manual.docx) to 107s (hnsw.pdf) -- a
    # 250x spread. A progress bar built on that would be wrong most of the time.

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    progress: Progress = field(default_factory=Progress)
    result: dict[str, Any] | None = None
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["progress"] = self.progress.to_json()
        return d


class JobStore:
    """FIFO single-worker job runner with on-disk records."""

    def __init__(self, dirpath: Path | None = None) -> None:
        self.dir = dirpath or (config.DATA_INDEX / "jobs")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue[tuple[str, Callable[[Job], dict[str, Any]]]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._load()

    # -- persistence ---------------------------------------------------------

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _save(self, job: Job) -> None:
        job.updated_at = _now()
        tmp = self._path(job.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(job.to_json(), indent=1), encoding="utf-8")
        tmp.replace(self._path(job.id))

    def _load(self) -> None:
        for p in sorted(self.dir.glob("*.json")):
            try:
                d = json.loads(p.read_text("utf-8"))
                prog = Progress(**d.pop("progress", {}))
                job = Job(progress=prog, **d)
                # A job recorded as running when the process starts cannot be
                # running -- the worker died with the process. Marked failed
                # rather than left as a permanent ghost.
                if job.state in ("running", "queued"):
                    job.state = "failed"
                    job.error = "server restarted while this job was in flight"
                    job.finished_at = _now()
                    self._save(job)
                self._jobs[job.id] = job
            except Exception:  # noqa: BLE001
                continue

    # -- api -----------------------------------------------------------------

    def submit(
        self, kind: str, fn: Callable[[Job], dict[str, Any]], **params: Any
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
            self._save(job)
        self._q.put((job.id, fn))
        self._ensure_worker()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, *, limit: int = 25) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def active(self) -> Job | None:
        return next((j for j in self._jobs.values() if j.state == "running"), None)

    def queued(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.state == "queued"]

    def update(self, job: Job, **progress: Any) -> None:
        for k, v in progress.items():
            setattr(job.progress, k, v)
        self._save(job)

    # -- worker --------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, name="ragkit-jobs", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                job_id, fn = self._q.get(timeout=30)
            except queue.Empty:
                return
            job = self._jobs.get(job_id)
            if job is None or job.state == "cancelled":
                continue
            job.state = "running"
            job.started_at = _now()
            self._save(job)
            # ONE LINE ON START, ONE ON FINISH, TO STDOUT.
            #
            # Job state lived only in memory and in data/index/jobs/, both of
            # which die with the container -- so a background crash left NOTHING
            # in the logs. That is exactly the situation where the logs are the
            # only evidence there is: a visitor lost a conversation, the
            # container had silently restarted at 92% CPU with no request in
            # flight, and whatever burned that CPU was invisible because nothing
            # it did was ever written where a log collector could see it.
            #
            # uvicorn's access lines already go to stdout, so these land in the
            # same stream and interleave in the right order.
            _t0 = time.time()
            print(f"JOB {job.id} start kind={job.kind} "
                  f"params={json.dumps(job.params, default=str)[:200]}", flush=True)
            try:
                job.result = fn(job)
                job.state = "done"
            except Exception as exc:  # noqa: BLE001
                job.state = "failed"
                # Full traceback, not just str(exc). A background failure with no
                # stack is a bug report with the evidence removed.
                job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            finally:
                job.finished_at = _now()
                self._save(job)
                # The result SUMMARY, not the result. A log line that is
                # sometimes enormous gets truncated by something downstream,
                # and it truncates the part you needed.
                _sum = ""
                if job.state == "done" and isinstance(job.result, dict):
                    _sum = " " + json.dumps(
                        {k: v for k, v in job.result.items()
                         if isinstance(v, (int, float, str, bool))}, default=str)[:200]
                elif job.state == "failed":
                    _sum = " " + (job.error or "").splitlines()[0][:200]
                print(f"JOB {job.id} {job.state} kind={job.kind} "
                      f"in={time.time() - _t0:.1f}s{_sum}", flush=True)


STORE = JobStore()
