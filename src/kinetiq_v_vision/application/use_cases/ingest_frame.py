from dataclasses import dataclass
from typing import Any

from kinetiq_v_vision.application.ports.inference import PersonDetectorPort
from kinetiq_v_vision.application.ports.state import AnalysisRepositoryPort
from kinetiq_v_vision.application.ports.telemetry import TelemetryPort
from kinetiq_v_vision.domain.entities import CandidatePerson, MediaFrame
from kinetiq_v_vision.domain.exceptions import (
    AnalysisNotFoundError,
    AnalysisStoppedError,
)
from kinetiq_v_vision.domain.value_objects import AnalysisState


@dataclass(frozen=True)
class IngestFrameCommand:
    analysis_id: str
    frame_index: int = 0
    timestamp_ms: float = 0.0
    width: int = 0
    height: int = 0
    frame_data: Any = None


class IngestFrameUseCase:
    """Runs person detection on a submitted frame and merges the results
    into the analysis's candidate set -- the "frames/inference populate
    candidates" step of the target-enrollment lifecycle, decoupled from
    both analysis creation and target confirmation: a client (or, in a
    live deployment, a capture/worker process) submits frames here between
    `POST /v1/analyses` and `POST /v1/analyses/{id}/target`, and
    `GET /v1/analyses/{id}/candidates` reflects whatever has been detected
    so far.
    """

    def __init__(
        self,
        repository: AnalysisRepositoryPort,
        detector: PersonDetectorPort,
        telemetry: TelemetryPort,
    ) -> None:
        self._repository = repository
        self._detector = detector
        self._telemetry = telemetry

    def execute(self, command: IngestFrameCommand) -> list[CandidatePerson]:
        analysis = self._repository.get_by_id(command.analysis_id)
        if analysis is None:
            raise AnalysisNotFoundError(command.analysis_id)

        if analysis.state == AnalysisState.STOPPED:
            raise AnalysisStoppedError(command.analysis_id)

        frame = MediaFrame(
            frame_index=command.frame_index,
            timestamp_ms=command.timestamp_ms,
            width=command.width,
            height=command.height,
            data=command.frame_data,
        )
        detected = self._detector.detect_candidates(frame)
        for candidate in detected:
            analysis.add_candidate(candidate)
        self._repository.save(analysis)

        self._telemetry.record_event(
            "frame_ingested",
            analysis_id=analysis.analysis_id,
            frame_index=command.frame_index,
            candidates_detected=len(detected),
            candidates_total=len(analysis.candidates),
        )

        return list(analysis.candidates.values())
