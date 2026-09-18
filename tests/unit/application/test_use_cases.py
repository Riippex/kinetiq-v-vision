from datetime import UTC, datetime

import pytest

from kinetiq_v_vision.application.use_cases.create_analysis import (
    CreateAnalysisCommand,
    CreateAnalysisUseCase,
)
from kinetiq_v_vision.application.use_cases.ingest_frame import (
    IngestFrameCommand,
    IngestFrameUseCase,
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
    AnalysisNotFoundError,
    AnalysisStoppedError,
    CursorExpiredError,
    IdempotencyConflictError,
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
from kinetiq_v_vision.infrastructure.inference.stub import StubPersonDetectorAdapter
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


def test_create_analysis_is_idempotent_on_key_with_matching_request() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)

    cmd = CreateAnalysisCommand(
        session_id="session-idem-1",
        source_id="camera-front",
        exercise_key="push_up",
        exercise_version=1,
        idempotency_key="idem-key-1",
    )

    first = create_uc.execute(cmd)
    second = create_uc.execute(cmd)

    assert first.analysis_id == second.analysis_id


def test_create_analysis_rejects_key_reuse_with_different_request() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)

    create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-idem-2",
            source_id="camera-front",
            exercise_key="push_up",
            idempotency_key="idem-key-2",
        )
    )

    # Same key, different exercise_key -- a genuinely different request.
    with pytest.raises(IdempotencyConflictError, match="idem-key-2"):
        create_uc.execute(
            CreateAnalysisCommand(
                session_id="session-idem-2",
                source_id="camera-front",
                exercise_key="plank",
                idempotency_key="idem-key-2",
            )
        )


def test_create_analysis_without_key_always_creates_new() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)

    cmd = CreateAnalysisCommand(
        session_id="session-idem-3",
        source_id="camera-front",
        exercise_key="glute_bridge",
    )

    first = create_uc.execute(cmd)
    second = create_uc.execute(cmd)

    assert first.analysis_id != second.analysis_id


def test_create_analysis_idempotent_replay_resolves_after_resave() -> None:
    """If the recorded analysis was deleted (e.g. stopped and reaped) but
    the idempotency receipt survives, a replay recreates it under the same
    key/fingerprint rather than raising or resolving to a missing record."""
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)

    cmd = CreateAnalysisCommand(
        session_id="session-idem-4",
        source_id="camera-front",
        exercise_key="bodyweight_squat",
        idempotency_key="idem-key-4",
    )
    first = create_uc.execute(cmd)
    repo.delete(first.analysis_id)

    second = create_uc.execute(cmd)
    assert repo.get_by_id(second.analysis_id) is not None


def test_ingest_frame_populates_candidates_from_detector() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)
    ingest_uc = IngestFrameUseCase(
        repository=repo,
        detector=StubPersonDetectorAdapter(),
        telemetry=telemetry,
    )

    analysis = create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-frame-1",
            source_id="camera-front",
            exercise_key="push_up",
        )
    )
    assert analysis.candidates == {}

    candidates = ingest_uc.execute(
        IngestFrameCommand(analysis_id=analysis.analysis_id, frame_index=0)
    )

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "candidate_01"

    persisted = repo.get_by_id(analysis.analysis_id)
    assert persisted is not None
    assert "candidate_01" in persisted.candidates


def test_ingest_frame_merges_repeated_detections_without_duplicating() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)
    ingest_uc = IngestFrameUseCase(
        repository=repo,
        detector=StubPersonDetectorAdapter(),
        telemetry=telemetry,
    )

    analysis = create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-frame-2",
            source_id="camera-front",
            exercise_key="plank",
        )
    )

    ingest_uc.execute(IngestFrameCommand(analysis_id=analysis.analysis_id, frame_index=0))
    candidates = ingest_uc.execute(
        IngestFrameCommand(analysis_id=analysis.analysis_id, frame_index=1)
    )

    assert len(candidates) == 1


def test_ingest_frame_raises_for_unknown_analysis() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    ingest_uc = IngestFrameUseCase(
        repository=repo,
        detector=StubPersonDetectorAdapter(),
        telemetry=telemetry,
    )

    with pytest.raises(AnalysisNotFoundError):
        ingest_uc.execute(IngestFrameCommand(analysis_id="missing-analysis"))


def test_ingest_frame_rejects_stopped_analysis() -> None:
    repo = InMemoryAnalysisRepository()
    telemetry = LoggingTelemetryAdapter()
    create_uc = CreateAnalysisUseCase(repo, telemetry)
    stop_uc = StopAnalysisUseCase(repo, telemetry)
    ingest_uc = IngestFrameUseCase(
        repository=repo,
        detector=StubPersonDetectorAdapter(),
        telemetry=telemetry,
    )

    analysis = create_uc.execute(
        CreateAnalysisCommand(
            session_id="session-frame-3",
            source_id="camera-front",
            exercise_key="glute_bridge",
        )
    )
    stop_uc.execute(analysis.analysis_id)

    with pytest.raises(AnalysisStoppedError):
        ingest_uc.execute(IngestFrameCommand(analysis_id=analysis.analysis_id))
