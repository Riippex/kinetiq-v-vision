import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionSettings:
    """Environment configuration settings for Kinetiq V Vision."""

    host: str = "0.0.0.0"
    port: int = 8000
    buffer_capacity: int = 300
    log_level: str = "INFO"
    environment: str = "development"

    @classmethod
    def from_env(cls) -> "VisionSettings":
        return cls(
            host=os.getenv("VISION_HOST", "0.0.0.0"),
            port=int(os.getenv("VISION_PORT", "8000")),
            buffer_capacity=int(os.getenv("VISION_BUFFER_CAPACITY", "300")),
            log_level=os.getenv("VISION_LOG_LEVEL", "INFO"),
            environment=os.getenv("VISION_ENV", "development"),
        )
