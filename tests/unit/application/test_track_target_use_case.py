from datetime import UTC, datetime

import pytest

from kinetiq_v_vision.application.use_cases.create_analysis import (
    CreateAnalysisCommand,
    CreateAnalysisUseCase,
)
from kinetiq_v_vision.application.use_cases.select_target import (
    SelectTargetCommand,
    SelectTargetUseCase,
)
from kinetiq_v_vision.application.use_cases.track_target import (
    TrackTargetCommand,
    TrackTargetUseCase,
)
from kinetiq_v_vision.domain.entities import CandidatePerson
from kinetiq_v_vision.domain.exceptions import (
    AnalysisNotFoundError,
    AnalysisNotTrackingError,
    InvalidTargetError,
)
from kinetiq_v_vision.domain.target_tracker import TargetTrackerConfig
from kinetiq_v_vision.domain.value_objects import (
    AnalysisState,
    BoundingBox,
    ReasonCode,
    TrackingState,
)
from kinetiq_v_vision.infrastructure.state.in_memory import InMemoryAnalysisRepository
from kinetiq_v_vision.infrastructure.telemetry.logger import LoggingTelemetryAdapter


def test_track_target_use_case_lifecycle() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()

    create_uc = CreateAnalysisUseCase(repo, telemetry)
    select_uc = SelectTargetUseCase(repo, telemetry)
    track_uc = TrackTargetUseCase(
        repo,
        telemetry,
        tracker_config=TargetTrackerConfig(reacquisition_required_frames=1),
    )

    # 1. Create analysis
    analysis = create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-100",
            source_id="cam-1",
            exercise_key="bodyweight_squat",
        )
    )

    # Attempting track before target selection raises InvalidTargetError
    with pytest.raises(InvalidTargetError):
        track_uc.execute(
            TrackTargetCommand(
                analysis_id=analysis.analysis_id,
                candidates=[],
            )
        )

    # 2. Add candidate & select target
    cand = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.2, 0.2, 0.4, 0.7),
        confidence=0.95,
        detected_at=datetime.now(UTC),
    )
    analysis.add_candidate(cand)
    repo.save(analysis)

    select_uc.execute(
        SelectTargetCommand(
            analysis_id=analysis.analysis_id,
            candidate_id="person_01",
            expected_epoch=1,
        )
    )

    # 3. Track target frame 1 -> sequence 1
    t1 = datetime.now(UTC)
    obs1 = track_uc.execute(
        TrackTargetCommand(
            analysis_id=analysis.analysis_id,
            candidates=[cand],
            timestamp_utc=t1,
        )
    )
    assert obs1.epoch == 1
    assert obs1.sequence == 1
    assert obs1.target_person_id == "person_01"
    assert obs1.tracking_state == TrackingState.CONFIRMED
    assert obs1.reason_code == ReasonCode.OK
    assert obs1.associated_candidate_id == "person_01"

    # 4. Track target frame 2 -> sequence 2 (sequence monotonicity)
    obs2 = track_uc.execute(
        TrackTargetCommand(
            analysis_id=analysis.analysis_id,
            candidates=[cand],
            timestamp_utc=t1,
        )
    )
    assert obs2.epoch == 1
    assert obs2.sequence == 2

    # Check observations persisted in repository
    observations, next_cursor, _has_more = repo.get_observations_after(
        analysis.analysis_id, after_cursor=None
    )
    assert len(observations) == 2
    assert observations[0].sequence == 1
    assert observations[1].sequence == 2
    assert next_cursor == "1:2"



def test_track_target_nonexistent_analysis() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    track_uc = TrackTargetUseCase(repo, telemetry)

    with pytest.raises(AnalysisNotFoundError):
        track_uc.execute(
            TrackTargetCommand(
                analysis_id="nonexistent-id",
                candidates=[],
            )
        )


def _create_and_select(
    repo: InMemoryAnalysisRepository, telemetry: LoggingTelemetryAdapter
) -> str:
    create_uc = CreateAnalysisUseCase(repo, telemetry)
    select_uc = SelectTargetUseCase(repo, telemetry)

    analysis = create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-200",
            source_id="cam-1",
            exercise_key="bodyweight_squat",
        )
    )
    cand = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.2, 0.2, 0.4, 0.7),
        confidence=0.95,
        detected_at=datetime.now(UTC),
    )
    analysis.add_candidate(cand)
    repo.save(analysis)
    select_uc.execute(
        SelectTargetCommand(
            analysis_id=analysis.analysis_id,
            candidate_id="person_01",
            expected_epoch=1,
        )
    )
    return analysis.analysis_id


def test_track_target_rejects_when_analysis_not_tracking() -> None:
    """A stopped analysis still has target_person_id set (stop() does not
    clear it), so this exercises the TRACKING-state gate specifically,
    distinct from the target-not-selected check."""
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    track_uc = TrackTargetUseCase(repo, telemetry)

    analysis_id = _create_and_select(repo, telemetry)
    analysis = repo.get_by_id(analysis_id)
    assert analysis is not None
    analysis.stop()
    repo.save(analysis)

    with pytest.raises(AnalysisNotTrackingError) as excinfo:
        track_uc.execute(TrackTargetCommand(analysis_id=analysis_id, candidates=[]))
    assert excinfo.value.current_state == AnalysisState.STOPPED.value


def test_track_target_requires_reconfirmation_after_epoch_increment() -> None:
    """After a stream restart (increment_epoch), tracking must be rejected
    until the target is explicitly reconfirmed for the new epoch."""
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    track_uc = TrackTargetUseCase(repo, telemetry)

    analysis_id = _create_and_select(repo, telemetry)
    analysis = repo.get_by_id(analysis_id)
    assert analysis is not None
    assert analysis.state == AnalysisState.TRACKING

    analysis.increment_epoch()
    repo.save(analysis)

    assert analysis.target_person_id is None
    assert analysis.state == AnalysisState.AWAITING_SELECTION

    with pytest.raises(InvalidTargetError):
        track_uc.execute(TrackTargetCommand(analysis_id=analysis_id, candidates=[]))


def test_track_target_uses_fresh_tracker_state_per_epoch() -> None:
    """A tracker's stable-frame counters must not survive a stream restart:
    the reacquisition requirement should re-apply in the new epoch rather
    than inheriting the prior epoch's confirmed streak."""
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    # Default reacquisition_required_frames=3: a fresh tracker needs 3
    # consecutive matching frames before CONFIRMED.
    track_uc = TrackTargetUseCase(repo, telemetry)

    analysis_id = _create_and_select(repo, telemetry)
    cand = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.2, 0.2, 0.4, 0.7),
        confidence=0.95,
        detected_at=datetime.now(UTC),
    )

    # Reach CONFIRMED in epoch 1 by supplying 3 consecutive matching frames.
    obs1 = None
    for _ in range(3):
        obs1 = track_uc.execute(
            TrackTargetCommand(analysis_id=analysis_id, candidates=[cand])
        )
    assert obs1 is not None
    assert obs1.tracking_state == TrackingState.CONFIRMED

    # Restart the stream and reconfirm the same target for the new epoch.
    analysis = repo.get_by_id(analysis_id)
    assert analysis is not None
    analysis.increment_epoch()
    analysis.add_candidate(cand)
    repo.save(analysis)

    select_uc = SelectTargetUseCase(repo, telemetry)
    select_uc.execute(
        SelectTargetCommand(
            analysis_id=analysis_id,
            candidate_id="person_01",
            expected_epoch=2,
        )
    )

    obs2 = track_uc.execute(
        TrackTargetCommand(analysis_id=analysis_id, candidates=[cand])
    )
    assert obs2.epoch == 2
    # A fresh tracker starts its reacquisition streak at zero, not carrying
    # over epoch 1's confirmed streak -- proving the (analysis_id, epoch)
    # cache key actually isolates tracker state across epochs rather than
    # reusing a stale instance. A single matching frame is not yet enough to
    # reach CONFIRMED again.
    assert obs2.tracking_state == TrackingState.SEARCHING
