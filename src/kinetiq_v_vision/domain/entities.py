import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kinetiq_v_vision.domain.exceptions import (
    InvalidEpochError,
    InvalidTargetError,
    StaleEpochError,
)
from kinetiq_v_vision.domain.value_objects import (
    AnalysisState,
    BoundingBox,
    ExerciseKey,
    ReasonCode,
    TrackingState,
    VisibilityState,
)


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    confidence: float | None = None
    z: float | None = None
    visibility: float | None = None


@dataclass(frozen=True)
class MediaFrame:
    frame_index: int
    timestamp_ms: float
    width: int
    height: int
    data: Any


@dataclass(frozen=True)
class RepetitionEvent:
    repetition_index: int
    start_timestamp: datetime
    end_timestamp: datetime
    confidence: float
    quality_score: float
    form_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HoldEvent:
    elapsed_seconds: float
    is_holding: bool
    confidence: float
    stability_score: float
    form_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidatePerson:
    candidate_id: str
    bbox: BoundingBox
    confidence: float
    detected_at: datetime
    # Normalized [0.0, 1.0] full-frame auxiliary keypoints decoded from the
    # detector's raw output, in upstream mp_persondet.py order: mid-hip,
    # full-body point, shoulder-center, upper-body point. Used by pose
    # preprocessing to size the person ROI (see PoseEstimationPreprocessor);
    # None when the detector path does not expose them (e.g. stub, direct
    # detections array).
    keypoints: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class Observation:
    session_id: str
    epoch: int
    sequence: int
    timestamp_utc: datetime
    target_person_id: str
    exercise_key: str
    exercise_version: int
    tracking_state: TrackingState
    visibility_state: VisibilityState
    reason_code: ReasonCode
    repetitions: list[RepetitionEvent] | None = None
    hold: HoldEvent | None = None

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise InvalidEpochError(self.epoch)
        if self.sequence < 1:
            raise ValueError(f"Sequence must be an integer >= 1, got {self.sequence}")


@dataclass
class Analysis:
    session_id: str
    source_id: str
    exercise_key: ExerciseKey
    exercise_version: int = 1
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    epoch: int = 1
    sequence_counter: int = 0
    state: AnalysisState = AnalysisState.AWAITING_SELECTION
    target_person_id: str | None = None
    last_valid_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    candidates: dict[str, CandidatePerson] = field(default_factory=dict)

    def select_target(self, candidate_id: str, expected_epoch: int) -> None:
        """Explicitly confirm the candidate target person for this analysis context."""
        if expected_epoch != self.epoch:
            raise StaleEpochError(
                expected_epoch=expected_epoch, current_epoch=self.epoch
            )

        if (
            candidate_id not in self.candidates
            and candidate_id != self.target_person_id
        ):
            raise InvalidTargetError(candidate_id)

        # If re-selecting a target after initial confirmation, advance the epoch
        if self.target_person_id is not None and self.target_person_id != candidate_id:
            self.epoch += 1
            self.sequence_counter = 0

        self.target_person_id = candidate_id
        self.state = AnalysisState.TRACKING
        self.last_valid_at = datetime.now(UTC)

    def increment_epoch(self) -> int:
        """Advance epoch on stream restart, reconnection, or re-targeting."""
        self.epoch += 1
        self.sequence_counter = 0
        self.state = AnalysisState.AWAITING_SELECTION
        return self.epoch

    def next_sequence(self) -> int:
        """Increment and return the strictly monotonic sequence for the current epoch."""
        self.sequence_counter += 1
        return self.sequence_counter

    def add_candidate(self, candidate: CandidatePerson) -> None:
        self.candidates[candidate.candidate_id] = candidate

    def stop(self) -> None:
        self.state = AnalysisState.STOPPED
        self.candidates.clear()
