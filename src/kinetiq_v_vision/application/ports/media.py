from abc import ABC, abstractmethod
from collections.abc import Iterator

from kinetiq_v_vision.domain.entities import MediaFrame


class FrameSourcePort(ABC):
    """Port for resolving and streaming video frames from an authorized source ID."""

    @abstractmethod
    def open_stream(self, source_id: str) -> Iterator[MediaFrame]:
        """Open a video frame generator for a verified source."""

    @abstractmethod
    def close_stream(self, source_id: str) -> None:
        """Release underlying stream or clip resources."""
