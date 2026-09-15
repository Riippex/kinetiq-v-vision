import logging

from kinetiq_v_vision.application.ports.telemetry import TelemetryPort

logger = logging.getLogger("kinetiq_v_vision")


class LoggingTelemetryAdapter(TelemetryPort):
    """Standard logger telemetry adapter."""

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._logger = logger
        self._logger.setLevel(log_level)

    def record_latency(self, operation: str, duration_ms: float) -> None:
        self._logger.debug("Latency [%s]: %.2f ms", operation, duration_ms)

    def record_event(self, event_name: str, **attributes: object) -> None:
        self._logger.info("Event [%s]: %s", event_name, attributes)
