from datetime import UTC, datetime

import pytest

from kinetiq_v_vision.domain.entities import (
    Analysis,
    CandidatePerson,
    Observation,
    RepetitionEvent,
)
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


def test_analysis_creation_defaults() -> None:
    analysis = Analysis(
        session_id="session-123",
        source_id="camera-front",
        exercise_key=ExerciseKey.BODYWEIGHT_SQUAT,
    )
    assert analysis.session_id == "session-123"
    assert analysis.source_id == "camera-front"
    assert analysis.exercise_key == ExerciseKey.BODYWEIGHT_SQUAT
    assert analysis.epoch == 1
    assert analysis.sequence_counter == 0
    assert analysis.state == AnalysisState.AWAITING_SELECTION
    assert analysis.target_person_id is None


def test_target_selection_and_stale_epoch_rejection() -> None:
    analysis = Analysis(
        session_id="session-123",
        source_id="camera-front",
        exercise_key=ExerciseKey.PUSH_UP,
    )
    candidate = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.1, 0.1, 0.5, 0.8),
        confidence=0.95,
        detected_at=datetime.now(UTC),
    )
    analysis.add_candidate(candidate)

    # Reject stale epoch
    with pytest.raises(StaleEpochError) as excinfo:
        analysis.select_target(candidate_id="person_01", expected_epoch=2)
    assert excinfo.value.expected_epoch == 2
    assert excinfo.value.current_epoch == 1

    # Successful selection
    analysis.select_target(candidate_id="person_01", expected_epoch=1)
    assert analysis.target_person_id == "person_01"
    assert analysis.state == AnalysisState.TRACKING
    assert analysis.epoch == 1

    # Select unknown target
    with pytest.raises(InvalidTargetError):
        analysis.select_target(candidate_id="unknown_target", expected_epoch=1)


def test_epoch_increment_and_sequence_monotonicity() -> None:
    analysis = Analysis(
        session_id="session-123",
        source_id="camera-front",
        exercise_key=ExerciseKey.PLANK,
    )
    seq1 = analysis.next_sequence()
    seq2 = analysis.next_sequence()
    assert seq1 == 1
    assert seq2 == 2

    # Stream interruption / epoch advance
    new_epoch = analysis.increment_epoch()
    assert new_epoch == 2
    assert analysis.sequence_counter == 0
    assert analysis.state == AnalysisState.AWAITING_SELECTION

    new_seq1 = analysis.next_sequence()
    assert new_seq1 == 1


def test_observation_validation() -> None:
    now = datetime.now(UTC)
    obs = Observation(
        session_id="session-123",
        epoch=1,
        sequence=1,
        timestamp_utc=now,
        target_person_id="person_01",
        exercise_key="bodyweight_squat",
        exercise_version=1,
        tracking_state=TrackingState.CONFIRMED,
        visibility_state=VisibilityState.FULL,
        reason_code=ReasonCode.OK,
        repetitions=[
            RepetitionEvent(
                repetition_index=1,
                start_timestamp=now,
                end_timestamp=now,
                confidence=0.95,
                quality_score=0.90,
            )
        ],
    )
    assert obs.sequence == 1
    assert obs.reason_code == ReasonCode.OK

    with pytest.raises(InvalidEpochError):
        Observation(
            session_id="session-123",
            epoch=0,
            sequence=1,
            timestamp_utc=now,
            target_person_id="person_01",
            exercise_key="bodyweight_squat",
            exercise_version=1,
            tracking_state=TrackingState.CONFIRMED,
            visibility_state=VisibilityState.FULL,
            reason_code=ReasonCode.OK,
        )

    with pytest.raises(ValueError):
        Observation(
            session_id="session-123",
            epoch=1,
            sequence=0,
            timestamp_utc=now,
            target_person_id="person_01",
            exercise_key="bodyweight_squat",
            exercise_version=1,
            tracking_state=TrackingState.CONFIRMED,
            visibility_state=VisibilityState.FULL,
            reason_code=ReasonCode.OK,
        )
