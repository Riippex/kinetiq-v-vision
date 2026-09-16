"""Domain layer for Kinetiq V Vision.

Contains core entities, value objects, invariants, and domain exceptions.
Must remain free of framework dependencies (no FastAPI, OpenCV, boto3, or MLflow).
"""

from kinetiq_v_vision.domain.entities import (
    Analysis,
    CandidatePerson,
    HoldEvent,
    Landmark,
    MediaFrame,
    Observation,
    RepetitionEvent,
)
from kinetiq_v_vision.domain.exceptions import (
    AnalysisNotFoundError,
    CursorExpiredError,
    DomainError,
    InvalidEpochError,
    InvalidTargetError,
    StaleEpochError,
    TargetAmbiguousError,
)
from kinetiq_v_vision.domain.value_objects import (
    AnalysisState,
    BoundingBox,
    ExerciseKey,
    ReasonCode,
    TrackingState,
    VisibilityState,
)

__all__ = [
    "Analysis",
    "AnalysisNotFoundError",
    "AnalysisState",
    "BoundingBox",
    "CandidatePerson",
    "CursorExpiredError",
    "DomainError",
    "ExerciseKey",
    "HoldEvent",
    "InvalidEpochError",
    "InvalidTargetError",
    "Landmark",
    "MediaFrame",
    "Observation",
    "ReasonCode",
    "RepetitionEvent",
    "StaleEpochError",
    "TargetAmbiguousError",
    "TrackingState",
    "VisibilityState",
]
