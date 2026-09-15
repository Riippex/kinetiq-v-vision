from datetime import datetime, timezone
from typing import Any

from kinetiq_v_vision.application.ports.inference import (
    PersonDetectorPort,
    PoseInferencePort,
)
from kinetiq_v_vision.domain.entities import CandidatePerson, Landmark
from kinetiq_v_vision.domain.value_objects import BoundingBox


class StubPersonDetectorAdapter(PersonDetectorPort):
    """Deterministic stub person detector returning a single mock candidate."""

    def __init__(self, default_candidate_id: str = "candidate_01") -> None:
        self._default_candidate_id = default_candidate_id

    def detect_candidates(self, frame: Any) -> list[CandidatePerson]:
        return [
            CandidatePerson(
                candidate_id=self._default_candidate_id,
                bbox=BoundingBox(x=0.2, y=0.1, width=0.6, height=0.8),
                confidence=0.95,
                detected_at=datetime.now(timezone.utc),
            )
        ]


class StubPoseInferenceAdapter(PoseInferencePort):
    """Deterministic stub pose inference returning key landmarks."""

    def infer_pose(self, frame: Any, candidate_id: str) -> list[Landmark]:
        return [
            Landmark(name="nose", x=0.5, y=0.15, confidence=0.98),
            Landmark(name="left_shoulder", x=0.45, y=0.25, confidence=0.96),
            Landmark(name="right_shoulder", x=0.55, y=0.25, confidence=0.96),
            Landmark(name="left_hip", x=0.46, y=0.55, confidence=0.94),
            Landmark(name="right_hip", x=0.54, y=0.55, confidence=0.94),
            Landmark(name="left_knee", x=0.46, y=0.75, confidence=0.92),
            Landmark(name="right_knee", x=0.54, y=0.75, confidence=0.92),
            Landmark(name="left_ankle", x=0.46, y=0.95, confidence=0.90),
            Landmark(name="right_ankle", x=0.54, y=0.95, confidence=0.90),
        ]
