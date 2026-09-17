from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
from kinetiq_v_vision.domain.entities import CandidatePerson, Observation
from kinetiq_v_vision.domain.target_tracker import TargetTrackerConfig
from kinetiq_v_vision.domain.value_objects import (
    BoundingBox,
    TrackingState,
)
from kinetiq_v_vision.infrastructure.state.in_memory import InMemoryAnalysisRepository
from kinetiq_v_vision.infrastructure.telemetry.logger import LoggingTelemetryAdapter


@dataclass(frozen=True)
class TrackingScenarioFrame:
    frame_index: int
    timestamp_utc: datetime
    ground_truth_target_id: str | None
    candidates: list[CandidatePerson]


@dataclass(frozen=True)
class TrackingScenario:
    name: str
    description: str
    target_person_id: str
    initial_bbox: BoundingBox
    frames: list[TrackingScenarioFrame]


@dataclass
class TrackingEvaluationMetrics:
    total_frames: int = 0
    target_tracked_frames: int = 0
    attribution_accuracy_pct: float = 0.0
    target_switch_count: int = 0
    false_pause_count: int = 0
    correct_pause_count: int = 0
    reacquisition_count: int = 0
    mean_reacquisition_latency_ms: float = 0.0
    is_synthetic: bool = True

    @property
    def is_release_eligible(self) -> bool:
        """Any silent target switch blocks release.

        There used to be a second condition here, `bystander_attributed_reps
        == 0`. It has been removed rather than "fixed": repetition detection
        does not exist yet in this codebase (VV-501/VV-502 are still
        Blocked/Ready), `TrackTargetUseCase` never populates
        `Observation.repetitions`, and this evaluation harness never emitted
        any repetition events for that field to count -- the metric was
        permanently and vacuously zero. Re-add it, computed from real
        emitted repetitions, once repetition evaluation actually exists;
        until then a hardcoded-zero field claiming release eligibility on a
        capability that does not exist is exactly the kind of fabricated
        evidence this correction is meant to eliminate.
        """
        return self.target_switch_count == 0


def create_synthetic_tracking_scenarios() -> list[TrackingScenario]:
    """Generate standard synthetic tracking scenarios: clean tracking, distractor crossing, exit/re-entry."""
    base_time = datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC)
    target_id = "user_primary"
    target_bbox = BoundingBox(0.2, 0.2, 0.3, 0.6)

    # 1. Clean Tracking Scenario
    clean_frames: list[TrackingScenarioFrame] = []
    for i in range(10):
        t = base_time + timedelta(milliseconds=i * 100)
        cand = CandidatePerson(
            candidate_id=target_id,
            bbox=BoundingBox(0.2 + i * 0.005, 0.2, 0.3, 0.6),
            confidence=0.95,
            detected_at=t,
        )
        clean_frames.append(
            TrackingScenarioFrame(
                frame_index=i,
                timestamp_utc=t,
                ground_truth_target_id=target_id,
                candidates=[cand],
            )
        )
    s1 = TrackingScenario(
        name="Clean Tracking",
        description="Single target moving across frame with no distractors",
        target_person_id=target_id,
        initial_bbox=target_bbox,
        frames=clean_frames,
    )

    # 2. Distractor Crossing Scenario
    distractor_frames: list[TrackingScenarioFrame] = []
    for i in range(15):
        t = base_time + timedelta(milliseconds=i * 100)
        target_cand = CandidatePerson(
            candidate_id=target_id,
            bbox=BoundingBox(0.25, 0.2, 0.3, 0.6),
            confidence=0.92,
            detected_at=t,
        )
        candidates = [target_cand]

        # Distractor walks right in front of target during frames 5..9
        if 5 <= i <= 9:
            bystander_cand = CandidatePerson(
                candidate_id="bystander_pet",
                bbox=BoundingBox(0.26, 0.21, 0.3, 0.6),
                confidence=0.90,
                detected_at=t,
            )
            candidates.append(bystander_cand)

        distractor_frames.append(
            TrackingScenarioFrame(
                frame_index=i,
                timestamp_utc=t,
                ground_truth_target_id=target_id if not (5 <= i <= 9) else None,
                candidates=candidates,
            )
        )
    s2 = TrackingScenario(
        name="Distractor Crossing",
        description="Bystander crosses directly over target causing temporal ambiguity",
        target_person_id=target_id,
        initial_bbox=target_bbox,
        frames=distractor_frames,
    )

    # 3. Exit and Re-entry Scenario
    exit_frames: list[TrackingScenarioFrame] = []
    for i in range(20):
        t = base_time + timedelta(milliseconds=i * 100)
        # Target present frames 0..4, absent 5..14, re-enters 15..19
        if i <= 4 or i >= 15:
            target_cand = CandidatePerson(
                candidate_id=target_id,
                bbox=BoundingBox(0.25, 0.2, 0.3, 0.6),
                confidence=0.93,
                detected_at=t,
            )
            cands = [target_cand]
            gt_id = target_id
        else:
            cands = []
            gt_id = None

        exit_frames.append(
            TrackingScenarioFrame(
                frame_index=i,
                timestamp_utc=t,
                ground_truth_target_id=gt_id,
                candidates=cands,
            )
        )
    s3 = TrackingScenario(
        name="Exit and Re-entry",
        description="Target leaves camera view for 1 second and re-enters",
        target_person_id=target_id,
        initial_bbox=target_bbox,
        frames=exit_frames,
    )

    return [s1, s2, s3, create_bystander_only_reassociation_scenario()]


def create_bystander_only_reassociation_scenario() -> TrackingScenario:
    """Adversarial scenario: the true target is present only long enough to
    seed a spatial reference (`last_known_bbox`), then permanently leaves,
    replaced by a bystander at nearly the same position for every remaining
    frame -- the true target never reappears.

    This is not a hypothetical: `TargetTracker.process_frame`'s spatial
    fallback (used whenever there is no direct candidate_id match) scores
    candidates purely by IoU/centroid distance against `last_known_bbox`,
    with no check that the best-scoring candidate shares any identity with
    the enrolled target. A same-position bystander therefore reaches
    CONFIRMED under a *different* candidate_id -- a real, currently
    unfixed silent target switch, reproduced deterministically here rather
    than asserted about in the abstract.
    """
    base_time = datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC)
    target_id = "user_primary"
    shared_bbox = BoundingBox(0.25, 0.2, 0.3, 0.6)

    frames: list[TrackingScenarioFrame] = []
    for i in range(10):
        t = base_time + timedelta(milliseconds=i * 100)
        if i < 2:
            # True target briefly present, at the exact bbox the bystander
            # will later occupy, to seed the tracker's spatial reference.
            cand = CandidatePerson(
                candidate_id=target_id,
                bbox=shared_bbox,
                confidence=0.95,
                detected_at=t,
            )
            gt_id = target_id
        else:
            # Target permanently gone; only a same-position bystander remains.
            cand = CandidatePerson(
                candidate_id="bystander_only",
                bbox=shared_bbox,
                confidence=0.93,
                detected_at=t,
            )
            gt_id = None

        frames.append(
            TrackingScenarioFrame(
                frame_index=i,
                timestamp_utc=t,
                ground_truth_target_id=gt_id,
                candidates=[cand],
            )
        )

    return TrackingScenario(
        name="Bystander-Only Reassociation",
        description=(
            "True target present only long enough to seed a spatial reference, "
            "then permanently absent while a same-position bystander remains -- "
            "adversarial probe for silent spatial reassociation onto a "
            "non-target identity"
        ),
        target_person_id=target_id,
        initial_bbox=shared_bbox,
        frames=frames,
    )


def run_tracking_evaluation(
    scenarios: Sequence[TrackingScenario] | None = None,
    tracker_config: TargetTrackerConfig | None = None,
) -> TrackingEvaluationMetrics:
    """Run target tracking evaluation across scenarios and compute aggregate quality metrics."""
    if scenarios is None:
        scenarios = create_synthetic_tracking_scenarios()

    config = tracker_config or TargetTrackerConfig(reacquisition_required_frames=2)
    metrics = TrackingEvaluationMetrics(is_synthetic=True)

    reacq_latencies: list[float] = []

    for scenario in scenarios:
        repo = InMemoryAnalysisRepository()
        telemetry = LoggingTelemetryAdapter()

        create_uc = CreateAnalysisUseCase(repo, telemetry)
        select_uc = SelectTargetUseCase(repo, telemetry)
        track_uc = TrackTargetUseCase(repo, telemetry, tracker_config=config)

        analysis = create_uc.execute(
            CreateAnalysisCommand(
                session_id="eval-session-01",
                source_id="eval-source-01",
                exercise_key="bodyweight_squat",
            )
        )

        # Pre-populate candidate for selection
        initial_cand = CandidatePerson(
            candidate_id=scenario.target_person_id,
            bbox=scenario.initial_bbox,
            confidence=0.95,
            detected_at=datetime.now(UTC),
        )
        analysis.add_candidate(initial_cand)
        repo.save(analysis)

        select_uc.execute(
            SelectTargetCommand(
                analysis_id=analysis.analysis_id,
                candidate_id=scenario.target_person_id,
                expected_epoch=1,
            )
        )

        lost_start_time: datetime | None = None

        for frame in scenario.frames:
            metrics.total_frames += 1
            obs: Observation = track_uc.execute(
                TrackTargetCommand(
                    analysis_id=analysis.analysis_id,
                    candidates=frame.candidates,
                    timestamp_utc=frame.timestamp_utc,
                )
            )

            # Silent target switch: the tracker reports CONFIRMED (i.e. not
            # AMBIGUOUS/SEARCHING/LOST) while the candidate it actually
            # spatially associated this frame with (`associated_candidate_id`)
            # is not the enrolled target. `obs.target_person_id` is always
            # the enrolled ID by construction (TrackTargetUseCase echoes
            # `analysis.target_person_id`), so comparing it against
            # `scenario.target_person_id` can never differ -- that
            # comparison was checking a value against itself. The real
            # attribution signal is `associated_candidate_id`.
            if (
                obs.tracking_state == TrackingState.CONFIRMED
                and obs.associated_candidate_id != scenario.target_person_id
            ):
                metrics.target_switch_count += 1

            if obs.tracking_state == TrackingState.CONFIRMED:
                metrics.target_tracked_frames += 1
                if frame.ground_truth_target_id == scenario.target_person_id:
                    # Correct attribution
                    pass
                else:
                    # Attributed target when ground truth was absent/bystander
                    metrics.false_pause_count += 1
            elif obs.tracking_state in (TrackingState.SEARCHING, TrackingState.LOST):
                if frame.ground_truth_target_id is None:
                    metrics.correct_pause_count += 1
                else:
                    metrics.false_pause_count += 1
                if lost_start_time is None:
                    lost_start_time = frame.timestamp_utc

            # Check reacquisition
            if (
                lost_start_time is not None
                and obs.tracking_state == TrackingState.CONFIRMED
            ):
                reacq_dt = (
                    frame.timestamp_utc - lost_start_time
                ).total_seconds() * 1000.0
                reacq_latencies.append(reacq_dt)
                metrics.reacquisition_count += 1
                lost_start_time = None

    if metrics.total_frames > 0:
        metrics.attribution_accuracy_pct = round(
            (metrics.target_tracked_frames / metrics.total_frames) * 100.0, 2
        )

    if reacq_latencies:
        metrics.mean_reacquisition_latency_ms = round(
            sum(reacq_latencies) / len(reacq_latencies), 2
        )

    return metrics
