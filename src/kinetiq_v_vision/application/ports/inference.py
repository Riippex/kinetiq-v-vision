from abc import ABC, abstractmethod
from typing import Any

from kinetiq_v_vision.domain.entities import CandidatePerson, Landmark, MediaFrame
from kinetiq_v_vision.domain.value_objects import BoundingBox


class PersonDetectorPort(ABC):
    """Port for running person detection on raw video frames."""

    @abstractmethod
    def detect_candidates(self, frame: MediaFrame | Any) -> list[CandidatePerson]:
        """Detect potential human candidates in a video frame."""


class PoseInferencePort(ABC):
    """Port for running landmark pose inference for a bounded person target."""

    @abstractmethod
    def infer_pose(
        self,
        frame: MediaFrame | Any,
        candidate_id: str,
        candidate_bbox: BoundingBox | None = None,
    ) -> list[Landmark]:
        """Infer body landmarks for a confirmed person target."""
