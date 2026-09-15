from abc import ABC, abstractmethod


class TelemetryPort(ABC):
    """Port for logging, timing, and metrics instrumentation."""

    @abstractmethod
    def record_latency(self, operation: str, duration_ms: float) -> None:
        """Record the elapsed time of a pipeline stage in milliseconds."""

    @abstractmethod
    def record_event(self, event_name: str, **attributes: object) -> None:
        """Record an operational lifecycle or tracking event."""
