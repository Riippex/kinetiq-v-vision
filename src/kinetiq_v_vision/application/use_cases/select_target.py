from dataclasses import dataclass

from kinetiq_v_vision.application.ports.state import AnalysisRepositoryPort
from kinetiq_v_vision.application.ports.telemetry import TelemetryPort
from kinetiq_v_vision.domain.entities import Analysis
from kinetiq_v_vision.domain.exceptions import AnalysisNotFoundError


@dataclass(frozen=True)
class SelectTargetCommand:
    analysis_id: str
    candidate_id: str
    expected_epoch: int
    idempotency_key: str | None = None


class SelectTargetUseCase:
    def __init__(
        self,
        repository: AnalysisRepositoryPort,
        telemetry: TelemetryPort,
    ) -> None:
        self._repository = repository
        self._telemetry = telemetry

    def execute(self, command: SelectTargetCommand) -> Analysis:
        analysis = self._repository.get_by_id(command.analysis_id)
        if analysis is None:
            raise AnalysisNotFoundError(command.analysis_id)

        analysis.select_target(
            candidate_id=command.candidate_id,
            expected_epoch=command.expected_epoch,
        )
        self._repository.save(analysis)
        self._telemetry.record_event(
            "target_selected",
            analysis_id=analysis.analysis_id,
            target_person_id=command.candidate_id,
            epoch=analysis.epoch,
        )
        return analysis
