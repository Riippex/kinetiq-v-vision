from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from kinetiq_v_vision.domain.entities import CandidatePerson
from kinetiq_v_vision.domain.target_tracker import TargetTrackerResult


class TargetTrackerPort(ABC):
    """Port for tracking a target person across video frames."""

    @abstractmethod
    def track_target(
        self,
        session_id: str,
        target_person_id: str,
        candidates: Sequence[CandidatePerson],
        timestamp_utc: datetime,
    ) -> TargetTrackerResult:
        """Process candidate detections and update target tracking state."""
