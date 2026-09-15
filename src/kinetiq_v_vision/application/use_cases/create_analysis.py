from dataclasses import dataclass

from kinetiq_v_vision.application.ports.state import AnalysisRepositoryPort
from kinetiq_v_vision.application.ports.telemetry import TelemetryPort
from kinetiq_v_vision.domain.entities import Analysis
from kinetiq_v_vision.domain.value_objects import ExerciseKey


@dataclass(frozen=True)
class CreateAnalysisCommand:
    session_id: str
    source_id: str
    exercise_key: str
    exercise_version: int = 1
    idempotency_key: str | None = None


class CreateAnalysisUseCase:
    def __init__(
        self,
        repository: AnalysisRepositoryPort,
        telemetry: TelemetryPort,
    ) -> None:
        self._repository = repository
        self._telemetry = telemetry

    def execute(self, command: CreateAnalysisCommand) -> Analysis:
        analysis = Analysis(
            session_id=command.session_id,
            source_id=command.source_id,
            exercise_key=ExerciseKey(command.exercise_key),
            exercise_version=command.exercise_version,
        )
        self._repository.save(analysis)
        self._telemetry.record_event(
            "analysis_created",
            analysis_id=analysis.analysis_id,
            session_id=analysis.session_id,
            exercise_key=analysis.exercise_key.value,
        )
        return analysis
