from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class FrameSourcePort(ABC):
    """Port for resolving and streaming video frames from an authorized source ID."""

    @abstractmethod
    def open_stream(self, source_id: str) -> Iterator[Any]:
        """Open a video frame generator for a verified source."""

    @abstractmethod
    def close_stream(self, source_id: str) -> None:
        """Release underlying stream or clip resources."""
