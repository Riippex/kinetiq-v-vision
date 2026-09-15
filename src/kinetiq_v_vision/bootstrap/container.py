from fastapi import FastAPI

from kinetiq_v_vision.application.use_cases.create_analysis import CreateAnalysisUseCase
from kinetiq_v_vision.application.use_cases.poll_observations import (
    PollObservationsUseCase,
)
from kinetiq_v_vision.application.use_cases.select_target import SelectTargetUseCase
from kinetiq_v_vision.application.use_cases.stop_analysis import StopAnalysisUseCase
from kinetiq_v_vision.bootstrap.settings import VisionSettings
from kinetiq_v_vision.infrastructure.inference.stub import (
    StubPersonDetectorAdapter,
    StubPoseInferenceAdapter,
)
from kinetiq_v_vision.infrastructure.state.in_memory import InMemoryAnalysisRepository
from kinetiq_v_vision.infrastructure.telemetry.logger import LoggingTelemetryAdapter
from kinetiq_v_vision.interfaces.rest.app import create_app
from kinetiq_v_vision.interfaces.rest.routes import (
    get_create_use_case,
    get_poll_use_case,
    get_repository,
    get_select_use_case,
    get_stop_use_case,
)


class Container:
    """Dependency injection container and composition root."""

    def __init__(self, settings: VisionSettings | None = None) -> None:
        self.settings = settings or VisionSettings.from_env()

        # Infrastructure adapters
        self.repository = InMemoryAnalysisRepository(
            buffer_capacity=self.settings.buffer_capacity
        )
        self.telemetry = LoggingTelemetryAdapter()
        self.detector = StubPersonDetectorAdapter()
        self.pose_inference = StubPoseInferenceAdapter()

        # Application use cases
        self.create_analysis_use_case = CreateAnalysisUseCase(
            repository=self.repository,
            telemetry=self.telemetry,
        )
        self.select_target_use_case = SelectTargetUseCase(
            repository=self.repository,
            telemetry=self.telemetry,
        )
        self.poll_observations_use_case = PollObservationsUseCase(
            repository=self.repository
        )
        self.stop_analysis_use_case = StopAnalysisUseCase(
            repository=self.repository,
            telemetry=self.telemetry,
        )

    def create_configured_app(self) -> FastAPI:
        """Instantiate and configure the FastAPI application with dependency overrides."""
        app = create_app()

        app.dependency_overrides[get_create_use_case] = lambda: self.create_analysis_use_case
        app.dependency_overrides[get_select_use_case] = lambda: self.select_target_use_case
        app.dependency_overrides[get_poll_use_case] = lambda: self.poll_observations_use_case
        app.dependency_overrides[get_stop_use_case] = lambda: self.stop_analysis_use_case
        app.dependency_overrides[get_repository] = lambda: self.repository

        return app
