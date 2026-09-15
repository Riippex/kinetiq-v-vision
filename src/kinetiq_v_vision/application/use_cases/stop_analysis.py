from kinetiq_v_vision.application.ports.state import AnalysisRepositoryPort
from kinetiq_v_vision.application.ports.telemetry import TelemetryPort
from kinetiq_v_vision.domain.exceptions import AnalysisNotFoundError


class StopAnalysisUseCase:
    def __init__(
        self,
        repository: AnalysisRepositoryPort,
        telemetry: TelemetryPort,
    ) -> None:
        self._repository = repository
        self._telemetry = telemetry

    def execute(self, analysis_id: str) -> None:
        analysis = self._repository.get_by_id(analysis_id)
        if analysis is None:
            raise AnalysisNotFoundError(analysis_id)

        analysis.stop()
        self._repository.save(analysis)
        self._telemetry.record_event("analysis_stopped", analysis_id=analysis_id)
