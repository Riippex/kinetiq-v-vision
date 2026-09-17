import hashlib
import json
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

    def fingerprint(self) -> str:
        """Identifies the logical request an idempotency_key was issued
        for, so a retry with the same key but different parameters is
        rejected rather than silently returning a mismatched analysis."""
        payload = {
            "session_id": self.session_id,
            "source_id": self.source_id,
            "exercise_key": self.exercise_key,
            "exercise_version": self.exercise_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class CreateAnalysisUseCase:
    def __init__(
        self,
        repository: AnalysisRepositoryPort,
        telemetry: TelemetryPort,
    ) -> None:
        self._repository = repository
        self._telemetry = telemetry

    def execute(self, command: CreateAnalysisCommand) -> Analysis:
        def factory() -> Analysis:
            return Analysis(
                session_id=command.session_id,
                source_id=command.source_id,
                exercise_key=ExerciseKey(command.exercise_key),
                exercise_version=command.exercise_version,
            )

        analysis = self._repository.create_or_get_by_idempotency_key(
            idempotency_key=command.idempotency_key,
            request_fingerprint=command.fingerprint(),
            factory=factory,
        )
        self._telemetry.record_event(
            "analysis_created",
            analysis_id=analysis.analysis_id,
            session_id=analysis.session_id,
            exercise_key=analysis.exercise_key.value,
        )
        return analysis
