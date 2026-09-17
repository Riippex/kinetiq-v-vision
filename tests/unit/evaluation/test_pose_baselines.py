"""Unit tests for pose baseline evaluation module and CLI command.

The `mediapipe_opencv` candidate now downloads and loads real pinned ONNX
weights (see `model_downloader.py`) instead of using golden/fixed tensors, so
tests that exercise `create_candidate_pipelines()` or the full benchmark
require network access on first run (the verified download is cached under
`weights/` for subsequent runs). Synthetic fixtures contain no real person,
so the real model's person/pose coverage on them is honestly 0% — these
tests assert on genuine mechanics (frames processed, real positive latency)
rather than forcing a coverage expectation the real model cannot honestly
satisfy against non-person imagery.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from kinetiq_v_vision.evaluation.baselines import (
    AuthorizedMediaUnavailableError,
    BenchmarkEnvironment,
    CandidateBenchmarkReport,
    CandidateMetrics,
    MediaIntegrityError,
    create_candidate_pipelines,
    run_pose_baseline_benchmark,
)
from kinetiq_v_vision.interfaces.cli.main import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DATA_DIR = REPO_ROOT / "fixtures" / "data"
MANIFEST_PATH = FIXTURES_DATA_DIR / "synthetic_manifest.json"

_STUB_PIPELINES: dict[str, tuple[str, Any]] = {
    "noop": ("No-op Pipeline", lambda frame: (True, [])),
}


def _write_tiny_clip(path: Path) -> None:
    """Write a minimal real 2-frame .mp4 so ControlledMediaSourceAdapter can
    resolve and open it like an authorized clip."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    assert out.isOpened()
    for color in [(10, 20, 30), (40, 50, 60)]:
        out.write(np.full((64, 64, 3), color, dtype=np.uint8))
    out.release()


def _base_clip(clip_id: str, source_uri: str, sha256: str) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "source_uri": source_uri,
        "sha256": sha256,
        "participant_id": "synthetic-user-01",
        "session_id": "session-integrity-01",
        "split": "development",
        "exercise": "bodyweight_squat",
        "perspective": "FRONTAL",
        "resolution": {"width": 1920, "height": 1080},
        "fps": 30.0,
        "duration_seconds": 1.0,
        "consent_scope": "authorized-internal",
        "provenance": {
            "source": "unit-test",
            "license": "Apache-2.0",
            "collected_at": "2026-09-17T00:00:00Z",
            "collector": "unit-test",
        },
        "conditions": ["clean_framing"],
        "annotation_ref": "annotations/does_not_matter.json",
        "annotation_version": "v1.0",
    }


def _write_manifest(tmp_path: Path, clips: list[dict[str, Any]]) -> Path:
    manifest = {
        "manifest_version": "1.0.0",
        "dataset_name": "media-integrity-unit-test",
        "updated_at": "2026-09-17T00:00:00Z",
        "clips": clips,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_benchmark_environment_capture() -> None:
    env = BenchmarkEnvironment()
    assert env.platform_system != ""
    assert env.opencv_version != ""
    assert env.python_version != ""
    assert env.timestamp_utc.endswith("+00:00") or "Z" in env.timestamp_utc


def test_candidate_metrics_finalize() -> None:
    m = CandidateMetrics(
        candidate_id="test_cand",
        candidate_name="Test Candidate",
        total_frames=10,
        detected_person_frames=8,
        valid_pose_frames=7,
        latencies_ms=[10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        confidence_values=[0.9, 0.95, 0.85],
        visibility_values=[0.8, 0.85, 0.9],
    )
    m.finalize()

    assert m.person_coverage_pct == 80.0
    assert m.pose_coverage_pct == 70.0
    assert m.p50_latency_ms == pytest.approx(27.5, abs=1.0)
    assert m.p95_latency_ms > m.p50_latency_ms
    assert m.min_latency_ms == 10.0
    assert m.max_latency_ms == 50.0
    assert m.throughput_fps > 0.0
    assert m.mean_landmark_confidence == pytest.approx(0.9, abs=0.01)
    assert m.mean_landmark_visibility == pytest.approx(0.85, abs=0.01)


def test_candidate_benchmark_report_serialization(tmp_path: Path) -> None:
    env = BenchmarkEnvironment()
    metrics = CandidateMetrics(
        candidate_id="c1",
        candidate_name="Candidate 1",
        total_frames=5,
        detected_person_frames=5,
        valid_pose_frames=5,
        latencies_ms=[10.0, 12.0, 11.0, 13.0, 12.0],
    )
    metrics.finalize()

    report = CandidateBenchmarkReport(
        environment=env,
        manifest_path=str(MANIFEST_PATH),
        split="development",
        total_clips_evaluated=2,
        candidate_metrics={"c1": metrics},
    )

    md = report.summary_markdown()
    assert "# Pose Baseline Benchmark Report: DEVELOPMENT Split" in md
    assert "Candidate 1" in md
    assert "p50 Latency (ms)" in md

    as_dict = report.to_dict()
    assert as_dict["split"] == "development"
    assert "c1" in as_dict["candidates"]

    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["split"] == "development"


def test_create_candidate_pipelines() -> None:
    """Requires network access on first run to download+verify the pinned
    ONNX weights (cached under weights/ afterward)."""
    pipelines = create_candidate_pipelines()
    assert "mediapipe_opencv" in pipelines
    assert "reference_stub" in pipelines

    for name, fn in pipelines.values():
        assert isinstance(name, str)
        assert callable(fn)


def test_run_pose_baseline_benchmark_on_development_split() -> None:
    """Synthetic mode must be explicit. The real `mediapipe_opencv` candidate
    genuinely runs (real weights, real net.forward()), but the synthetic
    fixture frames contain no real person, so person/pose coverage is
    honestly 0% — this test asserts real mechanics (frames processed,
    strictly positive latency), not a forced coverage expectation.
    """
    report = run_pose_baseline_benchmark(
        manifest_path=MANIFEST_PATH,
        split="development",
        max_frames_per_clip=3,
        warmup_frames=1,
        allow_synthetic=True,
    )

    assert report.environment.is_synthetic is True
    assert report.total_clips_evaluated == 4
    assert len(report.candidate_metrics) == 2
    assert "mediapipe_opencv" in report.candidate_metrics
    assert "reference_stub" in report.candidate_metrics

    for m in report.candidate_metrics.values():
        assert m.total_frames == 12  # 4 clips * 3 frames
        assert 0.0 <= m.person_coverage_pct <= 100.0
        assert 0.0 <= m.pose_coverage_pct <= 100.0
        assert m.p50_latency_ms > 0.0
        assert m.p95_latency_ms >= m.p50_latency_ms
        assert m.throughput_fps > 0.0
        assert 0.0 <= m.mean_landmark_confidence <= 1.0

    # The deterministic stub is designed to always "detect" on any frame;
    # the real model is proven to have actually run via its non-zero latency.
    assert report.candidate_metrics["reference_stub"].person_coverage_pct == 100.0
    assert report.candidate_metrics["mediapipe_opencv"].p50_latency_ms > 0.0


def test_run_pose_baseline_benchmark_filters_candidates() -> None:
    report = run_pose_baseline_benchmark(
        manifest_path=MANIFEST_PATH,
        split="development",
        selected_candidate_ids=["reference_stub"],
        max_frames_per_clip=2,
        allow_synthetic=True,
    )

    assert len(report.candidate_metrics) == 1
    assert "reference_stub" in report.candidate_metrics
    assert "mediapipe_opencv" not in report.candidate_metrics


def test_run_pose_baseline_benchmark_requires_real_media_by_default() -> None:
    """Regression test: the normal (non-synthetic) benchmark must resolve
    and process real authorized clips, failing clearly rather than silently
    substituting synthetic frames when the media is unavailable."""
    with pytest.raises(AuthorizedMediaUnavailableError, match="fixture_squat_01"):
        run_pose_baseline_benchmark(
            manifest_path=MANIFEST_PATH,
            split="development",
            max_frames_per_clip=2,
        )


def test_run_pose_baseline_benchmark_rejects_synthetic_mode_for_real_consent_scope(
    tmp_path: Path,
) -> None:
    """Regression test: synthetic mode must refuse to run against a clip that
    is not explicitly declared synthetic-no-person, so it can never be used
    to quietly fabricate results for what should be a real evaluation."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["clips"][0]["consent_scope"] = "authorized-internal"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="synthetic-no-person"):
        run_pose_baseline_benchmark(
            manifest_path=manifest_path,
            split="development",
            max_frames_per_clip=2,
            allow_synthetic=True,
        )


def test_run_pose_baseline_benchmark_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_pose_baseline_benchmark(tmp_path / "nonexistent.json")


def test_run_pose_baseline_benchmark_rejects_empty_split() -> None:
    with pytest.raises(ValueError, match="No clips found for split 'unknown_split'"):
        run_pose_baseline_benchmark(MANIFEST_PATH, split="unknown_split")


def test_cli_baseline_command(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out_json = tmp_path / "baseline_report.json"
    exit_code = cli_main(
        [
            "baseline",
            "--manifest",
            str(MANIFEST_PATH),
            "--split",
            "development",
            "--max-frames",
            "2",
            "--output",
            str(out_json),
            "--synthetic",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Pose Baseline Benchmark Report: DEVELOPMENT Split" in captured.out
    assert "SYNTHETIC MODE" in captured.out
    assert "MediaPipe Pose (OpenCV 5 Runtime)" in captured.out
    assert out_json.is_file()

    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)
        assert data["split"] == "development"
        assert "mediapipe_opencv" in data["candidates"]


def test_cli_baseline_command_fails_clearly_without_synthetic_or_media(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression test: omitting --synthetic against fixtures with no
    authorized local media must fail clearly, not silently synthesize
    frames or crash with an unrelated traceback."""
    exit_code = cli_main(
        [
            "baseline",
            "--manifest",
            str(MANIFEST_PATH),
            "--split",
            "development",
            "--max-frames",
            "2",
        ]
    )

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Error executing baseline benchmark" in captured.err
    assert "authorized" in captured.err.lower() or "unavailable" in captured.err.lower()


def test_cli_baseline_command_fails_on_missing_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    exit_code = cli_main(
        [
            "baseline",
            "--manifest",
            str(tmp_path / "does_not_exist.json"),
        ]
    )
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Error executing baseline benchmark" in captured.err


def test_02_pose_baselines_notebook_file_is_valid() -> None:
    nb_path = REPO_ROOT / "notebooks" / "02_pose_baselines.ipynb"
    assert nb_path.is_file()

    with open(nb_path, encoding="utf-8") as f:
        nb_data = json.load(f)

    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 3
    # Check that it imports run_pose_baseline_benchmark
    sources = "".join("".join(c["source"]) for c in nb_data["cells"])
    assert "run_pose_baseline_benchmark" in sources
    assert "summary_markdown" in sources


# --- Media integrity: sha256 verification and basename-collision guard -----


def test_run_pose_baseline_benchmark_accepts_clip_with_correct_sha256(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    clip_path = media_dir / "squat_01.mp4"
    _write_tiny_clip(clip_path)
    correct_sha256 = hashlib.sha256(clip_path.read_bytes()).hexdigest()

    manifest_path = _write_manifest(
        tmp_path,
        [_base_clip("clip_ok", "s3://bucket/squat_01.mp4", correct_sha256)],
    )

    report = run_pose_baseline_benchmark(
        manifest_path=manifest_path,
        split="development",
        max_frames_per_clip=1,
        media_root=media_dir,
        custom_pipelines=_STUB_PIPELINES,
    )

    assert report.total_clips_evaluated == 1
    assert report.candidate_metrics["noop"].total_frames == 1


def test_run_pose_baseline_benchmark_rejects_clip_with_incorrect_sha256(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    clip_path = media_dir / "squat_01.mp4"
    _write_tiny_clip(clip_path)
    wrong_sha256 = "f" * 64

    manifest_path = _write_manifest(
        tmp_path,
        [_base_clip("clip_bad_hash", "s3://bucket/squat_01.mp4", wrong_sha256)],
    )

    with pytest.raises(MediaIntegrityError, match="does not match"):
        run_pose_baseline_benchmark(
            manifest_path=manifest_path,
            split="development",
            max_frames_per_clip=1,
            media_root=media_dir,
            custom_pipelines=_STUB_PIPELINES,
        )


def test_run_pose_baseline_benchmark_rejects_missing_clip_file_clearly(
    tmp_path: Path,
) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    # No file written under media_dir for this clip.
    manifest_path = _write_manifest(
        tmp_path,
        [_base_clip("clip_missing", "s3://bucket/never_uploaded.mp4", "a" * 64)],
    )

    with pytest.raises(AuthorizedMediaUnavailableError, match="clip_missing"):
        run_pose_baseline_benchmark(
            manifest_path=manifest_path,
            split="development",
            max_frames_per_clip=1,
            media_root=media_dir,
            custom_pipelines=_STUB_PIPELINES,
        )


def test_run_pose_baseline_benchmark_basename_collision_does_not_silently_swap_clips(
    tmp_path: Path,
) -> None:
    """Two different clips whose source_uri basenames collide must not be
    treated as interchangeable: only the local file matching the clip's own
    pinned sha256 may be accepted, so a naming collision surfaces as an
    integrity failure rather than silently benchmarking the wrong clip."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    clip_path = media_dir / "clip.mp4"
    _write_tiny_clip(clip_path)
    actual_sha256 = hashlib.sha256(clip_path.read_bytes()).hexdigest()

    # clip_a's source_uri basename collides with clip_b's, but only one file
    # is present on disk and its hash matches clip_a, not clip_b.
    clip_a = _base_clip("clip_a", "s3://bucket-one/videos/clip.mp4", actual_sha256)
    clip_b = _base_clip(
        "clip_b", "s3://bucket-two/other/session/clip.mp4", "b" * 64
    )
    manifest_path = _write_manifest(tmp_path, [clip_a, clip_b])

    with pytest.raises(MediaIntegrityError, match="clip_b"):
        run_pose_baseline_benchmark(
            manifest_path=manifest_path,
            split="development",
            max_frames_per_clip=1,
            media_root=media_dir,
            custom_pipelines=_STUB_PIPELINES,
        )


def test_synthetic_mode_never_checks_sha256_of_fabricated_clips() -> None:
    """Synthetic clips have placeholder/fabricated sha256 values in the
    manifest (they describe media that was never recorded); synthetic mode
    must generate in-memory frames and must never attempt to verify a real
    file's hash against them."""
    report = run_pose_baseline_benchmark(
        manifest_path=MANIFEST_PATH,
        split="development",
        max_frames_per_clip=1,
        allow_synthetic=True,
        custom_pipelines=_STUB_PIPELINES,
    )
    assert report.environment.is_synthetic is True
    assert report.total_clips_evaluated == 4
