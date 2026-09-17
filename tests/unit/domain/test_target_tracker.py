from datetime import UTC, datetime, timedelta

import pytest

from kinetiq_v_vision.domain.entities import CandidatePerson
from kinetiq_v_vision.domain.target_tracker import (
    TargetTracker,
    TargetTrackerConfig,
    calculate_centroid_distance,
    calculate_iou,
)
from kinetiq_v_vision.domain.value_objects import (
    BoundingBox,
    ReasonCode,
    TrackingState,
    VisibilityState,
)


def test_calculate_iou_and_centroid_distance() -> None:
    box_a = BoundingBox(0.1, 0.1, 0.4, 0.4)
    box_b = BoundingBox(0.1, 0.1, 0.4, 0.4)
    assert pytest.approx(calculate_iou(box_a, box_b)) == 1.0
    assert pytest.approx(calculate_centroid_distance(box_a, box_b)) == 0.0

    # No overlap
    box_c = BoundingBox(0.6, 0.6, 0.2, 0.2)
    assert calculate_iou(box_a, box_c) == 0.0

    # Half overlap
    box_d = BoundingBox(0.1, 0.1, 0.2, 0.4)
    assert pytest.approx(calculate_iou(box_a, box_d)) == 0.5


def test_target_tracker_direct_id_match() -> None:
    now = datetime.now(UTC)
    tracker = TargetTracker(
        target_person_id="person_01",
        config=TargetTrackerConfig(reacquisition_required_frames=1),
    )

    candidate = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.2, 0.2, 0.3, 0.6),
        confidence=0.9,
        detected_at=now,
    )

    res = tracker.process_frame([candidate], now)
    assert res.tracking_state == TrackingState.CONFIRMED
    assert res.visibility_state == VisibilityState.FULL
    assert res.reason_code == ReasonCode.OK
    assert res.is_attributable is True
    assert res.target_candidate == candidate


def test_target_tracker_spatial_only_match_never_confirms() -> None:
    """Conservative identity policy (VV-401 correction): candidate IDs are
    assigned from per-frame detection order, not a persistent identity, so
    spatial proximity to a non-ID-matching candidate is never sufficient to
    reach CONFIRMED on its own -- it must land AMBIGUOUS and require an
    explicit reconfirmation (select_target) or the original ID reappearing."""
    now = datetime.now(UTC)
    initial_box = BoundingBox(0.2, 0.2, 0.3, 0.6)
    tracker = TargetTracker(
        target_person_id="person_01",
        initial_bbox=initial_box,
        config=TargetTrackerConfig(reacquisition_required_frames=1),
    )

    # Candidate with different candidate_id, but close spatial overlap
    candidate = CandidatePerson(
        candidate_id="det_105",
        bbox=BoundingBox(0.21, 0.21, 0.3, 0.6),
        confidence=0.85,
        detected_at=now,
    )

    res = tracker.process_frame([candidate], now)
    assert res.tracking_state == TrackingState.AMBIGUOUS
    assert res.is_attributable is False
    assert res.target_candidate.candidate_id == "det_105"

    # Even many consecutive frames of the same spatial match must never
    # promote to CONFIRMED without an ID match or explicit reconfirmation.
    for _ in range(5):
        res = tracker.process_frame([candidate], now)
        assert res.tracking_state == TrackingState.AMBIGUOUS
        assert res.is_attributable is False


def test_target_tracker_id_match_with_implausible_teleport_stays_ambiguous() -> None:
    """Validates spatial continuity even when the candidate_id matches: an
    ID coincidentally reused for a different, unrelated person at an
    implausible location must not be silently confirmed."""
    now = datetime.now(UTC)
    initial_box = BoundingBox(0.2, 0.2, 0.1, 0.1)
    tracker = TargetTracker(
        target_person_id="person_01",
        initial_bbox=initial_box,
        config=TargetTrackerConfig(reacquisition_required_frames=1),
    )

    # Same candidate_id as the enrolled target, but at a completely
    # different location -- no plausible continuous motion could explain it.
    teleported = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.85, 0.85, 0.1, 0.1),
        confidence=0.9,
        detected_at=now,
    )

    res = tracker.process_frame([teleported], now)
    assert res.tracking_state == TrackingState.AMBIGUOUS
    assert res.is_attributable is False


def test_target_tracker_ambiguity_detection() -> None:
    now = datetime.now(UTC)
    initial_box = BoundingBox(0.2, 0.2, 0.3, 0.6)
    tracker = TargetTracker(
        target_person_id="person_01",
        initial_bbox=initial_box,
        config=TargetTrackerConfig(ambiguity_iou_threshold=0.3),
    )

    # Primary candidate
    target_cand = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.2, 0.2, 0.3, 0.6),
        confidence=0.9,
        detected_at=now,
    )

    # Distractor candidate overlapping heavily
    distractor_cand = CandidatePerson(
        candidate_id="bystander_02",
        bbox=BoundingBox(0.22, 0.22, 0.3, 0.6),
        confidence=0.88,
        detected_at=now,
    )

    res = tracker.process_frame([target_cand, distractor_cand], now)
    assert res.tracking_state == TrackingState.AMBIGUOUS
    assert res.reason_code == ReasonCode.TARGET_AMBIGUOUS
    assert res.is_attributable is False


def test_target_tracker_loss_and_expiry_timeouts() -> None:
    t0 = datetime.now(UTC)
    tracker = TargetTracker(
        target_person_id="person_01",
        initial_bbox=BoundingBox(0.1, 0.1, 0.3, 0.5),
        config=TargetTrackerConfig(
            loss_timeout_seconds=2.0,
            expiry_timeout_seconds=10.0,
            reacquisition_required_frames=1,
        ),
    )

    target_cand = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.1, 0.1, 0.3, 0.5),
        confidence=0.9,
        detected_at=t0,
    )
    # Confirm target initially
    tracker.process_frame([target_cand], t0)
    assert tracker.tracking_state == TrackingState.CONFIRMED

    # 1 second later: missing target -> SEARCHING
    t1 = t0 + timedelta(seconds=1.0)
    res1 = tracker.process_frame([], t1)
    assert res1.tracking_state == TrackingState.SEARCHING

    # 3 seconds later: missing target > loss timeout -> LOST
    t2 = t0 + timedelta(seconds=3.0)
    res2 = tracker.process_frame([], t2)
    assert res2.tracking_state == TrackingState.LOST

    # 12 seconds later: > expiry timeout -> LOST & ABSENT
    t3 = t0 + timedelta(seconds=12.0)
    res3 = tracker.process_frame([], t3)
    assert res3.tracking_state == TrackingState.LOST
    assert res3.visibility_state == VisibilityState.ABSENT


def test_target_tracker_reacquisition_buffer() -> None:
    t0 = datetime.now(UTC)
    tracker = TargetTracker(
        target_person_id="person_01",
        initial_bbox=BoundingBox(0.1, 0.1, 0.3, 0.5),
        config=TargetTrackerConfig(reacquisition_required_frames=3),
    )

    candidate = CandidatePerson(
        candidate_id="person_01",
        bbox=BoundingBox(0.1, 0.1, 0.3, 0.5),
        confidence=0.9,
        detected_at=t0,
    )

    # Frame 1: SEARCHING (stable count 1 < 3)
    res1 = tracker.process_frame([candidate], t0)
    assert res1.tracking_state == TrackingState.SEARCHING
    assert res1.is_attributable is False

    # Frame 2: SEARCHING (stable count 2 < 3)
    res2 = tracker.process_frame([candidate], t0 + timedelta(milliseconds=100))
    assert res2.tracking_state == TrackingState.SEARCHING
    assert res2.is_attributable is False

    # Frame 3: CONFIRMED (stable count 3 >= 3)
    res3 = tracker.process_frame([candidate], t0 + timedelta(milliseconds=200))
    assert res3.tracking_state == TrackingState.CONFIRMED
    assert res3.is_attributable is True
