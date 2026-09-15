from dataclasses import dataclass

from kinetiq_v_vision.application.ports.state import AnalysisRepositoryPort
from kinetiq_v_vision.domain.entities import Observation
from kinetiq_v_vision.domain.exceptions import AnalysisNotFoundError


@dataclass(frozen=True)
class PollObservationsQuery:
    analysis_id: str
    after_cursor: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class PollObservationsResult:
    observations: list[Observation]
    next_cursor: str | None
    has_more: bool


class PollObservationsUseCase:
    def __init__(self, repository: AnalysisRepositoryPort) -> None:
        self._repository = repository

    def execute(self, query: PollObservationsQuery) -> PollObservationsResult:
        analysis = self._repository.get_by_id(query.analysis_id)
        if analysis is None:
            raise AnalysisNotFoundError(query.analysis_id)

        bounded_limit = max(1, min(query.limit, 100))
        observations, next_cursor, has_more = self._repository.get_observations_after(
            analysis_id=query.analysis_id,
            after_cursor=query.after_cursor,
            limit=bounded_limit,
        )

        return PollObservationsResult(
            observations=observations,
            next_cursor=next_cursor,
            has_more=has_more,
        )
