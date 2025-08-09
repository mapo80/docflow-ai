from __future__ import annotations
import threading, queue, time, uuid, json
from typing import Any, Dict, Callable
from logger import get_logger
from config import *
log = get_logger(__name__)

class JobQueue:
    def __init__(self, maxsize: int = 100):
        self.q = queue.PriorityQueue(maxsize=maxsize)
        self.events = {}  # job_id -> list of SSE events (dict)
        self.results = {} # job_id -> result
        self.lock = threading.Lock()
        self.workers = []
        self.stop = False

    def start(self, n_workers: int, handler: Callable[[dict], dict]):
        for i in range(n_workers):
            t = threading.Thread(target=self._worker, args=(handler,), daemon=True)
            t.start(); self.workers.append(t)

    def _put_event(self, job_id: str, typ: str, **data):
        ev = {"event": typ, "data": data, "ts": int(time.time()*1000)}
        with self.lock:
            self.events.setdefault(job_id, []).append(ev)
        log.debug("SSE %s %s", job_id, typ)

    def submit(self, payload: dict, priority: int = 5) -> str:
        job_id = str(uuid.uuid4())
        payload = dict(payload); payload["job_id"] = job_id
        try:
            self.q.put_nowait((priority, time.time(), payload))
        except queue.Full:
            raise RuntimeError("QueueFull")
        self._put_event(job_id, "queued", priority=priority)
        return job_id

    def _worker(self, handler: Callable[[dict], dict]):
        while not self.stop:
            try:
                prio, ts, payload = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            job_id = payload["job_id"]
            self._put_event(job_id, "started")
            try:
                res = handler(payload)
                with self.lock:
                    self.results[job_id] = res
                self._put_event(job_id, "done")
            except Exception as e:
                with self.lock:
                    self.results[job_id] = {"error": str(e)}
                self._put_event(job_id, "error", error=str(e))
            finally:
                self.q.task_done()

    def get_events(self, job_id: str):
        with self.lock:
            return list(self.events.get(job_id, []))

    def get_result(self, job_id: str):
        with self.lock:
            return self.results.get(job_id)

global_q = JobQueue(maxsize=int(os.getenv("JOB_QUEUE_MAXSIZE","100")))
