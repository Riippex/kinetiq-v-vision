import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from kinetiq_v_vision.domain.entities import CandidatePerson
from kinetiq_v_vision.domain.value_objects import (
    BoundingBox,
    ReasonCode,
    TrackingState,
    VisibilityState,
)


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x_left = max(box_a.x, box_b.x)
    y_top = max(box_a.y, box_b.y)
    x_right = min(box_a.x + box_a.width, box_b.x + box_b.width)
    y_bottom = min(box_a.y + box_a.height, box_b.y + box_b.height)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = box_a.width * box_a.height
    area_b = box_b.width * box_b.height
    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def calculate_centroid_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calculate normalized Euclidean distance between centroids of two bounding boxes."""
    center_a_x = box_a.x + box_a.width / 2.0
    center_a_y = box_a.y + box_a.height / 2.0
    center_b_x = box_b.x + box_b.width / 2.0
    center_b_y = box_b.y + box_b.height / 2.0

    dx = center_a_x - center_b_x
    dy = center_a_y - center_b_y
    return math.sqrt(dx * dx + dy * dy)


@dataclass(frozen=True)
class TargetTrackerConfig:
    min_association_score: float = 0.2
    ambiguity_delta: float = 0.15
    ambiguity_iou_threshold: float = 0.4
    loss_timeout_seconds: float = 2.0
    expiry_timeout_seconds: float = 15.0
    reacquisition_required_frames: int = 3
    iou_weight: float = 0.7
    distance_weight: float = 0.3


@dataclass(frozen=True)
class TargetTrackerResult:
    target_candidate: CandidatePerson | None
    tracking_state: TrackingState
    visibility_state: VisibilityState
    reason_code: ReasonCode
    association_score: float = 0.0
    is_attributable: bool = False


@dataclass
class TargetTracker:
    target_person_id: str
    initial_bbox: BoundingBox | None = None
    config: TargetTrackerConfig = field(default_factory=TargetTrackerConfig)
    last_known_bbox: BoundingBox | None = None
    last_valid_at: datetime | None = None
    consecutive_missing_count: int = 0
    consecutive_stable_frames: int = 0
    tracking_state: TrackingState = TrackingState.SEARCHING
    visibility_state: VisibilityState = VisibilityState.ABSENT
    reason_code: ReasonCode = ReasonCode.OUT_OF_FRAME

    def __post_init__(self) -> None:
        if self.initial_bbox is not None and self.last_known_bbox is None:
            self.last_known_bbox = self.initial_bbox

    def compute_association_score(
        self, candidate: CandidatePerson, reference_bbox: BoundingBox
    ) -> float:
        """Compute spatial association score combining IoU and normalized centroid distance."""
        iou = calculate_iou(candidate.bbox, reference_bbox)
        dist = calculate_centroid_distance(candidate.bbox, reference_bbox)
        # Distance penalty: 1.0 when identical, decays as distance grows
        dist_score = max(0.0, 1.0 - dist)
        return self.config.iou_weight * iou + self.config.distance_weight * dist_score

    def process_frame(
        self,
        candidates: Sequence[CandidatePerson],
        timestamp_utc: datetime,
    ) -> TargetTrackerResult:
        """Process a frame's candidate detections and update target tracking state."""
        # 1. Check for expiration
        if (
            self.last_valid_at is not None
            and (timestamp_utc - self.last_valid_at).total_seconds()
            > self.config.expiry_timeout_seconds
        ):
            self.tracking_state = TrackingState.LOST
            self.visibility_state = VisibilityState.ABSENT
            self.reason_code = ReasonCode.OUT_OF_FRAME
            self.consecutive_stable_frames = 0
            return TargetTrackerResult(
                target_candidate=None,
                tracking_state=TrackingState.LOST,
                visibility_state=VisibilityState.ABSENT,
                reason_code=ReasonCode.OUT_OF_FRAME,
                association_score=0.0,
                is_attributable=False,
            )

        if not candidates:
            return self._handle_missing_target(
                timestamp_utc, reason=ReasonCode.OUT_OF_FRAME
            )

        # 2. If we have a selected target_person_id, check for direct ID match first
        direct_match = next(
            (c for c in candidates if c.candidate_id == self.target_person_id), None
        )
        if direct_match is not None:
            reference_bbox = self.last_known_bbox or direct_match.bbox
            score = self.compute_association_score(direct_match, reference_bbox)
            # Check for distractors interfering with the direct match
            distractor_conflict = any(
                c.candidate_id != self.target_person_id
                and calculate_iou(c.bbox, direct_match.bbox)
                > self.config.ambiguity_iou_threshold
                for c in candidates
            )
            if distractor_conflict:
                return self._handle_ambiguous_target(
                    direct_match, score, timestamp_utc
                )

            return self._handle_confirmed_target(direct_match, score, timestamp_utc)

        # 3. Spatial tracking using reference bounding box
        if self.last_known_bbox is None:
            # No reference bbox and target_person_id not in candidates
            return self._handle_missing_target(
                timestamp_utc, reason=ReasonCode.TARGET_AMBIGUOUS
            )

        ref_bbox = self.last_known_bbox
        scored_candidates = [
            (c, self.compute_association_score(c, ref_bbox)) for c in candidates
        ]
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        best_candidate, best_score = scored_candidates[0]

        # 4. Check minimum association score
        if best_score < self.config.min_association_score:
            return self._handle_missing_target(
                timestamp_utc, reason=ReasonCode.LOW_CONFIDENCE
            )

        # 5. Check ambiguity (second candidate has very close score or high IoU overlap)
        if len(scored_candidates) > 1:
            second_candidate, second_score = scored_candidates[1]
            score_diff = best_score - second_score
            iou_overlap = calculate_iou(best_candidate.bbox, second_candidate.bbox)

            if (
                score_diff < self.config.ambiguity_delta
                or iou_overlap > self.config.ambiguity_iou_threshold
            ):
                return self._handle_ambiguous_target(
                    best_candidate, best_score, timestamp_utc
                )

        # 6. Unambiguous match found! Check reacquisition requirement if recovering from loss/search
        return self._handle_confirmed_target(
            best_candidate, best_score, timestamp_utc
        )

    def _handle_confirmed_target(
        self, candidate: CandidatePerson, score: float, timestamp_utc: datetime
    ) -> TargetTrackerResult:
        self.last_known_bbox = candidate.bbox
        self.last_valid_at = timestamp_utc
        self.consecutive_missing_count = 0

        # Require consecutive stable frames if coming from SEARCHING/LOST
        if self.tracking_state in (TrackingState.SEARCHING, TrackingState.LOST):
            self.consecutive_stable_frames += 1
            if (
                self.consecutive_stable_frames
                < self.config.reacquisition_required_frames
            ):
                return TargetTrackerResult(
                    target_candidate=candidate,
                    tracking_state=TrackingState.SEARCHING,
                    visibility_state=VisibilityState.PARTIAL,
                    reason_code=ReasonCode.LOW_CONFIDENCE,
                    association_score=score,
                    is_attributable=False,
                )

        self.consecutive_stable_frames = max(
            self.consecutive_stable_frames + 1, self.config.reacquisition_required_frames
        )
        self.tracking_state = TrackingState.CONFIRMED

        vis_state = (
            VisibilityState.FULL
            if candidate.confidence >= 0.7
            else VisibilityState.PARTIAL
        )
        reason = (
            ReasonCode.OK if candidate.confidence >= 0.5 else ReasonCode.LOW_CONFIDENCE
        )
        self.visibility_state = vis_state
        self.reason_code = reason

        is_attr = vis_state in (VisibilityState.FULL, VisibilityState.PARTIAL) and (
            reason == ReasonCode.OK
        )

        return TargetTrackerResult(
            target_candidate=candidate,
            tracking_state=TrackingState.CONFIRMED,
            visibility_state=vis_state,
            reason_code=reason,
            association_score=score,
            is_attributable=is_attr,
        )

    def _handle_ambiguous_target(
        self, candidate: CandidatePerson, score: float, timestamp_utc: datetime
    ) -> TargetTrackerResult:
        self.consecutive_stable_frames = 0
        self.tracking_state = TrackingState.AMBIGUOUS
        self.visibility_state = VisibilityState.PARTIAL
        self.reason_code = ReasonCode.TARGET_AMBIGUOUS

        return TargetTrackerResult(
            target_candidate=candidate,
            tracking_state=TrackingState.AMBIGUOUS,
            visibility_state=VisibilityState.PARTIAL,
            reason_code=ReasonCode.TARGET_AMBIGUOUS,
            association_score=score,
            is_attributable=False,
        )

    def _handle_missing_target(
        self, timestamp_utc: datetime, reason: ReasonCode
    ) -> TargetTrackerResult:
        self.consecutive_missing_count += 1
        self.consecutive_stable_frames = 0

        # Check loss timeout
        if (
            self.last_valid_at is not None
            and (timestamp_utc - self.last_valid_at).total_seconds()
            > self.config.loss_timeout_seconds
        ):
            self.tracking_state = TrackingState.LOST
            self.visibility_state = VisibilityState.ABSENT
            self.reason_code = reason
        else:
            self.tracking_state = TrackingState.SEARCHING
            self.visibility_state = VisibilityState.OCCLUDED
            self.reason_code = reason

        return TargetTrackerResult(
            target_candidate=None,
            tracking_state=self.tracking_state,
            visibility_state=self.visibility_state,
            reason_code=self.reason_code,
            association_score=0.0,
            is_attributable=False,
        )
