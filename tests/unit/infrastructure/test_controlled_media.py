"""Unit tests for ControlledMediaSourceAdapter."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from kinetiq_v_vision.domain.entities import MediaFrame
from kinetiq_v_vision.infrastructure.media.controlled_media import (
    ControlledMediaSourceAdapter,
    SourceNotFoundError,
    UnauthorizedSourceError,
)


def test_rejects_unauthorized_url_schemes() -> None:
    adapter = ControlledMediaSourceAdapter(media_root=Path("/fake/media"))
    with pytest.raises(UnauthorizedSourceError, match="Arbitrary URL schemes are prohibited"):
        adapter.resolve_source_path("http://malicious.site/stream.mp4")

    with pytest.raises(UnauthorizedSourceError, match="Arbitrary URL schemes are prohibited"):
        adapter.resolve_source_path("file:///etc/passwd")

    with pytest.raises(UnauthorizedSourceError, match="Arbitrary URL schemes are prohibited"):
        adapter.resolve_source_path("s3://bucket/clip.mp4")


def test_rejects_prohibited_colon_and_drive_specifiers() -> None:
    adapter = ControlledMediaSourceAdapter(media_root=Path("/fake/media"))
    with pytest.raises(UnauthorizedSourceError, match="prohibited colon"):
        adapter.resolve_source_path("C:/Windows/System32/clip.mp4")


def test_rejects_path_traversal(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    adapter = ControlledMediaSourceAdapter(media_root=media_dir)

    with pytest.raises(UnauthorizedSourceError, match="Path traversal is prohibited"):
        adapter.resolve_source_path("../outside.mp4")

    with pytest.raises(UnauthorizedSourceError, match="Path traversal is prohibited"):
        adapter.resolve_source_path("sub/../../outside.mp4")


def test_rejects_unallowed_extension(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    secret_file = media_dir / "secret.txt"
    secret_file.write_text("classified")

    adapter = ControlledMediaSourceAdapter(media_root=media_dir)
    with pytest.raises(UnauthorizedSourceError, match="not permitted"):
        adapter.resolve_source_path("secret.txt")


def test_rejects_missing_file(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    adapter = ControlledMediaSourceAdapter(media_root=media_dir)

    with pytest.raises(SourceNotFoundError, match="Authorized media source not found"):
        adapter.resolve_source_path("missing.mp4")


def test_resolves_valid_clip_path(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    clip_file = media_dir / "workout_01.mp4"
    clip_file.write_bytes(b"dummy video bytes")

    adapter = ControlledMediaSourceAdapter(media_root=media_dir)
    resolved = adapter.resolve_source_path("workout_01.mp4")
    assert resolved == clip_file.resolve()


def test_synthetic_stream_iteration() -> None:
    adapter = ControlledMediaSourceAdapter()
    dummy_frames = [
        MediaFrame(frame_index=0, timestamp_ms=0.0, width=640, height=480, data=np.zeros((480, 640, 3), dtype=np.uint8)),
        MediaFrame(frame_index=1, timestamp_ms=33.3, width=640, height=480, data=np.zeros((480, 640, 3), dtype=np.uint8)),
    ]
    adapter.register_synthetic_stream("synth_source_01", dummy_frames)

    stream = list(adapter.open_stream("synth_source_01"))
    assert len(stream) == 2
    assert stream[0].frame_index == 0
    assert stream[1].frame_index == 1
    assert stream[0].width == 640


def test_opencv_video_stream_reading(tmp_path: Path) -> None:
    """Create a real 3-frame video clip via OpenCV VideoWriter and stream it through the adapter."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    video_path = media_dir / "test_clip.mp4"

    # Write a small 64x64 video with 3 frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
    assert out.isOpened()
    for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        out.write(frame)
    out.release()

    adapter = ControlledMediaSourceAdapter(media_root=media_dir)
    frames = list(adapter.open_stream("test_clip.mp4"))

    assert len(frames) == 3
    for i, frame in enumerate(frames):
        assert frame.frame_index == i
        assert frame.width == 64
        assert frame.height == 64
        assert frame.data.shape == (64, 64, 3)
        assert frame.timestamp_ms >= 0.0

    # Ensure resources released
    adapter.close_stream("test_clip.mp4")
