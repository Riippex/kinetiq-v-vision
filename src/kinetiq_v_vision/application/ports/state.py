from abc import ABC, abstractmethod
from collections.abc import Callable

from kinetiq_v_vision.domain.entities import Analysis, Observation


class AnalysisRepositoryPort(ABC):
    """Port for persisting and retrieving analysis contexts and transient observation buffers."""

    @abstractmethod
    def save(self, analysis: Analysis) -> None:
        """Persist or update an analysis entity."""

    @abstractmethod
    def create_or_get_by_idempotency_key(
        self,
        *,
        idempotency_key: str | None,
        request_fingerprint: str,
        factory: Callable[[], Analysis],
    ) -> Analysis:
        """Atomically resolve an idempotent create.

        - If `idempotency_key` is falsy, always calls `factory()`, saves and
          returns a new `Analysis` (no idempotency tracking).
        - If `idempotency_key` was already recorded with the same
          `request_fingerprint`, returns the existing `Analysis` without
          calling `factory()`.
        - If `idempotency_key` was already recorded with a *different*
          `request_fingerprint`, raises `IdempotencyConflictError`.
        - Otherwise calls `factory()`, saves the result, records the key,
          and returns it.

        Implementations must perform the check-and-create atomically (e.g.
        under a single lock acquisition) so concurrent callers with the same
        key cannot both observe "not yet recorded" and create two analyses.
        """

    @abstractmethod
    def get_by_id(self, analysis_id: str) -> Analysis | None:
        """Retrieve an analysis context by ID."""

    @abstractmethod
    def delete(self, analysis_id: str) -> None:
        """Release and remove an analysis context."""

    @abstractmethod
    def append_observation(self, analysis_id: str, observation: Observation) -> None:
        """Append an observation to the analysis sliding window buffer."""

    @abstractmethod
    def get_observations_after(
        self,
        analysis_id: str,
        after_cursor: str | None,
        limit: int = 50,
    ) -> tuple[list[Observation], str | None, bool]:
        """Fetch observations strictly after the cursor.

        Returns:
            tuple of (observations, next_cursor, has_more)
        """
