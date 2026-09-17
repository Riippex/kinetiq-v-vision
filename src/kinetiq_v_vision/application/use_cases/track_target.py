from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from kinetiq_v_vision.application.ports.state import AnalysisRepositoryPort
from kinetiq_v_vision.application.ports.telemetry import TelemetryPort
from kinetiq_v_vision.domain.entities import CandidatePerson, Observation
from kinetiq_v_vision.domain.exceptions import (
    AnalysisNotFoundError,
    AnalysisNotTrackingError,
    InvalidTargetError,
)
from kinetiq_v_vision.domain.target_tracker import TargetTracker, TargetTrackerConfig
from kinetiq_v_vision.domain.value_objects import (
    AnalysisState,
)


@dataclass(frozen=True)
class TrackTargetCommand:
    analysis_id: str
    candidates: Sequence[CandidatePerson] = field(default_factory=list)
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(UTC))


class TrackTargetUseCase:
    """Application use case for processing video frame candidates and updating target tracking."""

    def __init__(
        self,
        repository: AnalysisRepositoryPort,
        telemetry: TelemetryPort,
        tracker_config: TargetTrackerConfig | None = None,
    ) -> None:
        self._repository = repository
        self._telemetry = telemetry
        self._tracker_config = tracker_config or TargetTrackerConfig()
        self._trackers: dict[tuple[str, int], TargetTracker] = {}

    def _get_or_create_tracker(
        self, analysis_id: str, epoch: int, target_person_id: str
    ) -> TargetTracker:
        # Keyed by (analysis_id, epoch): a new epoch (stream restart,
        # reconnection, re-targeting) must never reuse a prior epoch's
        # tracker state (last known bbox, stable-frame counters, etc.).
        key = (analysis_id, epoch)
        tracker = self._trackers.get(key)
        if tracker is None or tracker.target_person_id != target_person_id:
            tracker = TargetTracker(
                target_person_id=target_person_id,
                config=self._tracker_config,
            )
            self._trackers[key] = tracker
        return tracker

    def execute(self, command: TrackTargetCommand) -> Observation:
        analysis = self._repository.get_by_id(command.analysis_id)
        if analysis is None:
            raise AnalysisNotFoundError(command.analysis_id)

        if analysis.target_person_id is None:
            raise InvalidTargetError("No target person has been selected for this analysis")

        if analysis.state != AnalysisState.TRACKING:
            raise AnalysisNotTrackingError(command.analysis_id, analysis.state.value)

        tracker = self._get_or_create_tracker(
            command.analysis_id, analysis.epoch, analysis.target_person_id
        )
        result = tracker.process_frame(command.candidates, command.timestamp_utc)

        sequence = analysis.next_sequence()

        observation = Observation(
            session_id=analysis.session_id,
            epoch=analysis.epoch,
            sequence=sequence,
            timestamp_utc=command.timestamp_utc,
            target_person_id=analysis.target_person_id,
            exercise_key=analysis.exercise_key.value,
            exercise_version=analysis.exercise_version,
            tracking_state=result.tracking_state,
            visibility_state=result.visibility_state,
            reason_code=result.reason_code,
            associated_candidate_id=(
                result.target_candidate.candidate_id if result.target_candidate else None
            ),
        )

        self._repository.append_observation(analysis.analysis_id, observation)
        self._repository.save(analysis)


        self._telemetry.record_event(
            "target_tracked",
            analysis_id=analysis.analysis_id,
            epoch=analysis.epoch,
            sequence=sequence,
            tracking_state=result.tracking_state.value,
            reason_code=result.reason_code.value,
        )

        return observation
