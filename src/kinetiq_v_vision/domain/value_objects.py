from dataclasses import dataclass
from enum import Enum


class TrackingState(str, Enum):
    CONFIRMED = "CONFIRMED"
    SEARCHING = "SEARCHING"
    AMBIGUOUS = "AMBIGUOUS"
    LOST = "LOST"


class VisibilityState(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    OCCLUDED = "OCCLUDED"
    ABSENT = "ABSENT"


class ReasonCode(str, Enum):
    OK = "OK"
    TARGET_AMBIGUOUS = "TARGET_AMBIGUOUS"
    OCCLUSION = "OCCLUSION"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class AnalysisState(str, Enum):
    AWAITING_SELECTION = "AWAITING_SELECTION"
    CALIBRATING = "CALIBRATING"
    TRACKING = "TRACKING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class ExerciseKey(str, Enum):
    BODYWEIGHT_SQUAT = "bodyweight_squat"
    PUSH_UP = "push_up"
    PLANK = "plank"
    GLUTE_BRIDGE = "glute_bridge"


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("Bounding box dimensions cannot be negative")
