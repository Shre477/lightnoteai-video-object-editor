import threading
from typing import Dict, Optional

from .models.job import Job


class JobStore:
    """A minimal thread-safe in-memory store.

    For this assignment's scope an in-memory dict is a deliberate,
    reasonable choice (see README "Architecture & Engineering Decisions").
    Swapping this for Redis/Postgres later only touches this one file.
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)


job_store = JobStore()
