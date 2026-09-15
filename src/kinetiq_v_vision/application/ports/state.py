from abc import ABC, abstractmethod

from kinetiq_v_vision.domain.entities import Analysis, Observation


class AnalysisRepositoryPort(ABC):
    """Port for persisting and retrieving analysis contexts and transient observation buffers."""

    @abstractmethod
    def save(self, analysis: Analysis) -> None:
        """Persist or update an analysis entity."""

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
