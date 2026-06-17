# backend/ratelimit.py
"""A small in-process sliding-window rate limiter.

No external dependency, so the service deploys as-is. Note the limits are
per-process: with multiple Cloud Run instances each holds its own counters, so
the effective global limit scales with instance count. That's fine as a basic
abuse/cost guard; use Redis or an API gateway if you need a hard global cap.
"""
import time
import threading
from collections import deque


class SlidingWindowLimiter:
    def __init__(self, max_events: int, window_sec: float):
        self.max_events = max_events
        self.window_sec = window_sec
        self._hits = {}  # key -> deque[timestamps]
        self._lock = threading.Lock()
        self._last_prune = 0.0

    def _maybe_prune(self, now: float):
        """Drop fully-expired keys periodically so memory doesn't grow unbounded."""
        if now - self._last_prune < self.window_sec:
            return
        self._last_prune = now
        cutoff = now - self.window_sec
        for key in list(self._hits.keys()):
            dq = self._hits[key]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if not dq:
                del self._hits[key]

    def allow(self, key: str) -> bool:
        """Record a hit for `key`; return False if it exceeds the window limit."""
        now = time.time()
        cutoff = now - self.window_sec
        with self._lock:
            self._maybe_prune(now)
            dq = self._hits.get(key)
            if dq is None:
                dq = deque()
                self._hits[key] = dq
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_events:
                return False
            dq.append(now)
            return True
