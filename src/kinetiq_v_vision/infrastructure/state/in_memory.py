import threading
from collections import deque
from collections.abc import Callable

from kinetiq_v_vision.application.ports.state import AnalysisRepositoryPort
from kinetiq_v_vision.domain.entities import Analysis, Observation
from kinetiq_v_vision.domain.exceptions import (
    CursorExpiredError,
    IdempotencyConflictError,
)


class InMemoryAnalysisRepository(AnalysisRepositoryPort):
    """Thread-safe in-memory store for analyses and bounded observation buffers."""

    def __init__(self, buffer_capacity: int = 300) -> None:
        self._lock = threading.Lock()
        self._buffer_capacity = buffer_capacity
        self._analyses: dict[str, Analysis] = {}
        self._buffers: dict[str, deque[Observation]] = {}
        self._oldest_seen: dict[str, str] = {}
        # idempotency_key -> (analysis_id, request_fingerprint)
        self._idempotency_keys: dict[str, tuple[str, str]] = {}

    def save(self, analysis: Analysis) -> None:
        with self._lock:
            self._save_locked(analysis)

    def _save_locked(self, analysis: Analysis) -> None:
        self._analyses[analysis.analysis_id] = analysis
        if analysis.analysis_id not in self._buffers:
            self._buffers[analysis.analysis_id] = deque(maxlen=self._buffer_capacity)

    def create_or_get_by_idempotency_key(
        self,
        *,
        idempotency_key: str | None,
        request_fingerprint: str,
        factory: Callable[[], Analysis],
    ) -> Analysis:
        with self._lock:
            if idempotency_key:
                existing = self._idempotency_keys.get(idempotency_key)
                if existing is not None:
                    existing_analysis_id, existing_fingerprint = existing
                    if existing_fingerprint != request_fingerprint:
                        raise IdempotencyConflictError(idempotency_key)
                    resolved = self._analyses.get(existing_analysis_id)
                    if resolved is not None:
                        return resolved
                    # Receipt exists but the analysis itself was deleted
                    # (e.g. stopped and reaped) -- fall through and
                    # recreate, re-recording the same key/fingerprint.

            analysis = factory()
            self._save_locked(analysis)
            if idempotency_key:
                self._idempotency_keys[idempotency_key] = (
                    analysis.analysis_id,
                    request_fingerprint,
                )
            return analysis

    def get_by_id(self, analysis_id: str) -> Analysis | None:
        with self._lock:
            return self._analyses.get(analysis_id)

    def delete(self, analysis_id: str) -> None:
        with self._lock:
            self._analyses.pop(analysis_id, None)
            self._buffers.pop(analysis_id, None)
            self._oldest_seen.pop(analysis_id, None)

    def append_observation(self, analysis_id: str, observation: Observation) -> None:
        with self._lock:
            if analysis_id not in self._buffers:
                self._buffers[analysis_id] = deque(maxlen=self._buffer_capacity)
            buf = self._buffers[analysis_id]
            if len(buf) == self._buffer_capacity and buf:
                # The oldest item is about to be evicted
                oldest = buf[0]
                self._oldest_seen[analysis_id] = f"{oldest.epoch}:{oldest.sequence}"
            buf.append(observation)

    def get_observations_after(
        self,
        analysis_id: str,
        after_cursor: str | None,
        limit: int = 50,
    ) -> tuple[list[Observation], str | None, bool]:
        with self._lock:
            buf = list(self._buffers.get(analysis_id, deque()))

        if not buf:
            return [], None, False

        start_index = 0
        if after_cursor:
            parts = after_cursor.split(":")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(
                    f"Malformed cursor format: '{after_cursor}'. Expected 'epoch:sequence'"
                )
            req_epoch = int(parts[0])
            req_seq = int(parts[1])

            oldest_in_buf = buf[0]
            if (req_epoch, req_seq) < (oldest_in_buf.epoch, oldest_in_buf.sequence):
                oldest_cursor = f"{oldest_in_buf.epoch}:{oldest_in_buf.sequence}"
                raise CursorExpiredError(
                    requested_cursor=after_cursor, oldest_cursor=oldest_cursor
                )

            found = False
            for idx, obs in enumerate(buf):
                if obs.epoch == req_epoch and obs.sequence == req_seq:
                    start_index = idx + 1
                    found = True
                    break
                if (obs.epoch, obs.sequence) > (req_epoch, req_seq):
                    start_index = idx
                    found = True
                    break

            if not found:
                # Requested position is beyond what we currently have
                return [], after_cursor, False

        items = buf[start_index : start_index + limit]
        has_more = (start_index + limit) < len(buf)
        next_cursor = (
            f"{items[-1].epoch}:{items[-1].sequence}" if items else after_cursor
        )

        return items, next_cursor, has_more
