"""Controlled media source adapter for resolving and streaming authorized video frames."""

import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2

from kinetiq_v_vision.application.ports.media import FrameSourcePort
from kinetiq_v_vision.domain.entities import MediaFrame


class UnauthorizedSourceError(ValueError):
    """Raised when an untrusted or malformed media source identifier is supplied."""


class SourceNotFoundError(FileNotFoundError):
    """Raised when an authorized media source cannot be located on disk."""


SAFE_SOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)*$")
URL_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-\.]*://")


class ControlledMediaSourceAdapter(FrameSourcePort):
    """Resolves authorized media sources and streams frames via OpenCV VideoCapture.

    Enforces strict access control:
    - Rejects path traversal (`..`, absolute paths, drive letters).
    - Rejects arbitrary user-supplied URL schemes (`http://`, `file://`, `ftp://`).
    - Confines file access strictly within an authorized media root directory.
    - Supports in-memory registered synthetic streams for deterministic CI/unit evaluation.
    """

    def __init__(
        self,
        media_root: Path | str | None = None,
        allowed_extensions: set[str] | None = None,
    ) -> None:
        self._media_root = Path(media_root).resolve() if media_root else None
        self._allowed_extensions = allowed_extensions or {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
        }
        self._active_captures: dict[str, Any] = {}
        self._synthetic_streams: dict[str, list[MediaFrame]] = {}

    def register_synthetic_stream(
        self, source_id: str, frames: list[MediaFrame]
    ) -> None:
        """Register an in-memory frame sequence for testing and synthetic pipelines."""
        self._synthetic_streams[source_id] = frames

    def resolve_source_path(self, source_id: str) -> Path:
        """Validate and resolve a controlled source ID to an authorized local Path."""
        if not source_id or not isinstance(source_id, str):
            raise UnauthorizedSourceError("Source ID must be a non-empty string.")

        if URL_SCHEME_PATTERN.match(source_id) or "://" in source_id:
            raise UnauthorizedSourceError(
                f"Arbitrary URL schemes are prohibited: {source_id}"
            )

        if ":" in source_id:
            raise UnauthorizedSourceError(
                f"Source ID contains prohibited colon or drive specifier: {source_id}"
            )

        if not SAFE_SOURCE_ID_PATTERN.match(source_id):
            raise UnauthorizedSourceError(
                f"Source ID contains invalid characters: {source_id}"
            )

        parts = source_id.split("/")
        if any(part == ".." or part == "." or not part for part in parts):
            raise UnauthorizedSourceError(
                f"Path traversal is prohibited: {source_id}"
            )

        if self._media_root is None:
            raise UnauthorizedSourceError(
                "No media root directory configured on controlled media adapter."
            )

        resolved_path = (self._media_root / source_id).resolve()

        if not resolved_path.is_relative_to(self._media_root):
            raise UnauthorizedSourceError(
                f"Source resolved outside authorized media directory: {source_id}"
            )

        if resolved_path.suffix.lower() not in self._allowed_extensions:
            raise UnauthorizedSourceError(
                f"File extension '{resolved_path.suffix}' is not permitted."
            )

        if not resolved_path.is_file():
            raise SourceNotFoundError(
                f"Authorized media source not found: {resolved_path}"
            )

        return resolved_path

    def open_stream(self, source_id: str) -> Iterator[MediaFrame]:
        """Open a video frame generator for an authorized source."""
        if source_id in self._synthetic_streams:
            yield from self._synthetic_streams[source_id]
            return

        resolved_path = self.resolve_source_path(source_id)

        cap = cv2.VideoCapture(str(resolved_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV failed to open video source: {resolved_path}"
            )

        self._active_captures[source_id] = cap

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        frame_interval_ms = 1000.0 / fps
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                height, width = frame.shape[:2]
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_ms = (
                    pos_msec if pos_msec > 0 else frame_idx * frame_interval_ms
                )

                yield MediaFrame(
                    frame_index=frame_idx,
                    timestamp_ms=float(timestamp_ms),
                    width=int(width),
                    height=int(height),
                    data=frame,
                )
                frame_idx += 1
        finally:
            self.close_stream(source_id)

    def close_stream(self, source_id: str) -> None:
        """Release active capture resources for a source ID."""
        cap = self._active_captures.pop(source_id, None)
        if cap is not None:
            cap.release()
