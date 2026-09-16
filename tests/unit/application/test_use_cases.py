from datetime import UTC, datetime

import pytest

from kinetiq_v_vision.application.use_cases.create_analysis import (
    CreateAnalysisCommand,
    CreateAnalysisUseCase,
)
from kinetiq_v_vision.application.use_cases.poll_observations import (
    PollObservationsQuery,
    PollObservationsUseCase,
)
from kinetiq_v_vision.application.use_cases.select_target import (
    SelectTargetCommand,
    SelectTargetUseCase,
)
from kinetiq_v_vision.application.use_cases.stop_analysis import StopAnalysisUseCase
from kinetiq_v_vision.domain.entities import CandidatePerson, Observation
from kinetiq_v_vision.domain.exceptions import (
    CursorExpiredError,
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
from kinetiq_v_vision.infrastructure.state.in_memory import InMemoryAnalysisRepository
from kinetiq_v_vision.infrastructure.telemetry.logger import LoggingTelemetryAdapter


def test_create_and_select_target_flow() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()

    create_uc = CreateAnalysisUseCase(repo, telemetry)
    select_uc = SelectTargetUseCase(repo, telemetry)

    # 1. Create analysis
    cmd = CreateAnalysisCommand(
        session_id="session-456",
        source_id="clip-test-1",
        exercise_key="bodyweight_squat",
    )
    analysis = create_uc.execute(cmd)
    assert analysis.session_id == "session-456"
    assert analysis.exercise_key == ExerciseKey.BODYWEIGHT_SQUAT
    assert analysis.epoch == 1
    assert analysis.state == AnalysisState.AWAITING_SELECTION

    # 2. Add detected candidate
    candidate = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.1, 0.1, 0.5, 0.8),
        confidence=0.98,
        detected_at=datetime.now(UTC),
    )
    analysis.add_candidate(candidate)
    repo.save(analysis)

    # 3. Select target with expected epoch
    sel_cmd = SelectTargetCommand(
        analysis_id=analysis.analysis_id,
        candidate_id="person_01",
        expected_epoch=1,
    )
    updated = select_uc.execute(sel_cmd)
    assert updated.target_person_id == "person_01"
    assert updated.state == AnalysisState.TRACKING


def test_select_target_stale_epoch_fails() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)
    select_uc = SelectTargetUseCase(repo, telemetry)

    analysis = create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-456",
            source_id="clip-test-1",
            exercise_key="push_up",
        )
    )
    candidate = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.1, 0.1, 0.5, 0.8),
        confidence=0.98,
        detected_at=datetime.now(UTC),
    )
    analysis.add_candidate(candidate)
    repo.save(analysis)

    with pytest.raises(StaleEpochError):
        select_uc.execute(
            SelectTargetCommand(
                analysis_id=analysis.analysis_id,
                candidate_id="person_01",
                expected_epoch=2,
            )
        )


def test_poll_observations_cursor_and_expiry() -> None:
    repo = InMemoryAnalysisRepository(buffer_capacity=3)
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)
    poll_uc = PollObservationsUseCase(repo)

    analysis = create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-789",
            source_id="clip-test-1",
            exercise_key="plank",
        )
    )

    now = datetime.now(UTC)
    for seq in range(1, 6):
        obs = Observation(
            session_id=analysis.session_id,
            epoch=1,
            sequence=seq,
            timestamp_utc=now,
            target_person_id="target_01",
            exercise_key="plank",
            exercise_version=1,
            tracking_state=TrackingState.CONFIRMED,
            visibility_state=VisibilityState.FULL,
            reason_code=ReasonCode.OK,
        )
        repo.append_observation(analysis.analysis_id, obs)

    # Buffer capacity is 3, so buffer holds sequences [3, 4, 5]
    # Requesting sequence 1 must raise CursorExpiredError (expired from buffer)
    with pytest.raises(CursorExpiredError) as excinfo:
        poll_uc.execute(
            PollObservationsQuery(
                analysis_id=analysis.analysis_id,
                after_cursor="1:1",
            )
        )
    assert excinfo.value.requested_cursor == "1:1"
    assert excinfo.value.oldest_cursor == "1:3"

    # Requesting after 1:3 should return 4 and 5
    result = poll_uc.execute(
        PollObservationsQuery(
            analysis_id=analysis.analysis_id,
            after_cursor="1:3",
        )
    )
    assert len(result.observations) == 2
    assert result.observations[0].sequence == 4
    assert result.observations[1].sequence == 5
    assert result.next_cursor == "1:5"
    assert not result.has_more


def test_stop_analysis_releases_resources() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)
    stop_uc = StopAnalysisUseCase(repo, telemetry)

    analysis = create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-999",
            source_id="clip-test-1",
            exercise_key="glute_bridge",
        )
    )
    assert analysis.state == AnalysisState.AWAITING_SELECTION

    stop_uc.execute(analysis.analysis_id)
    retrieved = repo.get_by_id(analysis.analysis_id)
    assert retrieved is not None
    assert retrieved.state == AnalysisState.STOPPED
