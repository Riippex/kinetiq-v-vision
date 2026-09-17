"""Abstract ports for outbound adapters."""

from kinetiq_v_vision.application.ports.inference import (
    PersonDetectorPort,
    PoseInferencePort,
)
from kinetiq_v_vision.application.ports.media import FrameSourcePort
from kinetiq_v_vision.application.ports.state import AnalysisRepositoryPort
from kinetiq_v_vision.application.ports.target_tracker_port import TargetTrackerPort
from kinetiq_v_vision.application.ports.telemetry import TelemetryPort

__all__ = [
    "AnalysisRepositoryPort",
    "FrameSourcePort",
    "PersonDetectorPort",
    "PoseInferencePort",
    "TargetTrackerPort",
    "TelemetryPort",
]
