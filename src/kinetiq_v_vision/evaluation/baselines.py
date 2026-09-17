"""Reusable evaluation functions and CLI runner for pose model baselines.

Implements benchmark protocol for stage 02:
- Measures real hardware latency (p50, p90, p95, mean, min, max, FPS).
- Measures detection and pose landmark coverage.
- Evaluates landmark confidence and visibility statistics.
- Captures reproducible hardware and environment metadata.
- Compares candidates on the fixed development protocol without fabricated metrics.
"""

import itertools
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from kinetiq_v_vision.domain.entities import MediaFrame
from kinetiq_v_vision.evaluation.audit import validate_manifest_schema
from kinetiq_v_vision.infrastructure.inference.manifest import (
    calculate_file_sha256,
    load_model_manifest,
)
from kinetiq_v_vision.infrastructure.inference.model_downloader import (
    DEFAULT_WEIGHTS_DIR,
    resolve_model_artifact,
)
from kinetiq_v_vision.infrastructure.inference.opencv_adapters import (
    OpenCVPersonDetectorAdapter,
    OpenCVPoseInferenceAdapter,
)
from kinetiq_v_vision.infrastructure.inference.stub import (
    StubPersonDetectorAdapter,
    StubPoseInferenceAdapter,
)
from kinetiq_v_vision.infrastructure.media import (
    ControlledMediaSourceAdapter,
    SourceNotFoundError,
    UnauthorizedSourceError,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_FIXTURES_PATH = REPO_ROOT / "fixtures" / "golden" / "golden_fixtures.npz"
MODEL_MANIFESTS_DIR = REPO_ROOT / "model-manifests"


class AuthorizedMediaUnavailableError(RuntimeError):
    """Raised when the normal (non-synthetic) benchmark cannot resolve the
    authorized media a manifest clip references. Never silently substitutes
    synthetic frames for a real evaluation."""


class MediaIntegrityError(AuthorizedMediaUnavailableError):
    """Raised when a resolved local media file's content does not match the
    manifest clip's pinned `sha256`. This is also the guard against basename
    collision: `_derive_local_source_id` resolves a clip's local file by the
    basename of its `source_uri`, so two different clips whose source URIs
    happen to share a basename could otherwise silently resolve to the same
    (wrong) file on disk. Since the on-disk file can match at most one
    clip's pinned hash, a mismatch here always fails loudly instead of
    silently benchmarking the wrong clip's content."""


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """Captured hardware, runtime, and platform metadata."""

    platform_system: str = field(default_factory=platform.system)
    platform_release: str = field(default_factory=platform.release)
    platform_machine: str = field(default_factory=platform.machine)
    processor: str = field(default_factory=platform.processor)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    opencv_version: str = field(default_factory=lambda: cv2.__version__)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    is_synthetic: bool = False


@dataclass
class CandidateMetrics:
    """Recorded runtime performance, coverage, and confidence for a candidate pipeline."""

    candidate_id: str
    candidate_name: str
    total_frames: int = 0
    detected_person_frames: int = 0
    valid_pose_frames: int = 0
    person_coverage_pct: float = 0.0
    pose_coverage_pct: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    mean_landmark_confidence: float = 0.0
    mean_landmark_visibility: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    confidence_values: list[float] = field(default_factory=list)
    visibility_values: list[float] = field(default_factory=list)

    def finalize(self) -> None:
        """Compute summary percentiles, throughput, and coverage percentages."""
        if self.total_frames > 0:
            self.person_coverage_pct = round(
                (self.detected_person_frames / self.total_frames) * 100.0, 2
            )
            self.pose_coverage_pct = round(
                (self.valid_pose_frames / self.total_frames) * 100.0, 2
            )

        if self.latencies_ms:
            lats = np.array(self.latencies_ms, dtype=np.float64)
            self.mean_latency_ms = round(float(np.mean(lats)), 3)
            self.p50_latency_ms = round(float(np.percentile(lats, 50)), 3)
            self.p90_latency_ms = round(float(np.percentile(lats, 90)), 3)
            self.p95_latency_ms = round(float(np.percentile(lats, 95)), 3)
            self.min_latency_ms = round(float(np.min(lats)), 3)
            self.max_latency_ms = round(float(np.max(lats)), 3)
            self.throughput_fps = (
                round(1000.0 / self.mean_latency_ms, 2)
                if self.mean_latency_ms > 0
                else 0.0
            )

        if self.confidence_values:
            self.mean_landmark_confidence = round(
                float(np.mean(self.confidence_values)), 4
            )

        if self.visibility_values:
            self.mean_landmark_visibility = round(
                float(np.mean(self.visibility_values)), 4
            )


@dataclass
class CandidateBenchmarkReport:
    """Benchmark report collecting multi-candidate comparisons on a fixed split."""

    environment: BenchmarkEnvironment
    manifest_path: str
    split: str
    total_clips_evaluated: int
    candidate_metrics: dict[str, CandidateMetrics]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "environment": asdict(self.environment),
            "manifest_path": self.manifest_path,
            "split": self.split,
            "total_clips_evaluated": self.total_clips_evaluated,
            "candidates": {
                cid: {
                    k: v
                    for k, v in asdict(metrics).items()
                    if k not in ("latencies_ms", "confidence_values", "visibility_values")
                }
                for cid, metrics in self.candidate_metrics.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_markdown(self) -> str:
        """Render a GitHub-compatible markdown table summarizing benchmark metrics."""
        lines = [
            f"# Pose Baseline Benchmark Report: {self.split.upper()} Split",
            "",
        ]
        if self.environment.is_synthetic:
            synthetic_warning = (
                "> **SYNTHETIC MODE** — clip frames are synthetic (mechanics/latency "
                "validation only). Coverage and confidence numbers below do NOT represent "
                "real evaluation on authorized recorded movement and must never be reported "
                "as such."
            )
            lines.extend(["> [!WARNING]", synthetic_warning, ""])
        lines.extend([
            f"- **Manifest**: `{self.manifest_path}`",
            f"- **Clips Evaluated**: {self.total_clips_evaluated}",
            f"- **Platform**: `{self.environment.platform_system} {self.environment.platform_release} ({self.environment.platform_machine})`",
            f"- **Python**: `{self.environment.python_version}`",
            f"- **OpenCV**: `{self.environment.opencv_version}`",
            f"- **Benchmark Timestamp**: `{self.environment.timestamp_utc}`",
            "",
            "## Candidate Comparison Table",
            "",
            "| Candidate | Frames | Person Cov (%) | Pose Cov (%) | p50 Latency (ms) | p95 Latency (ms) | Throughput (FPS) | Mean Conf |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])

        for cid, m in self.candidate_metrics.items():
            lines.append(
                f"| **{m.candidate_name}** (`{cid}`) | {m.total_frames} | "
                f"{m.person_coverage_pct}% | {m.pose_coverage_pct}% | "
                f"{m.p50_latency_ms} ms | {m.p95_latency_ms} ms | "
                f"{m.throughput_fps} FPS | {m.mean_landmark_confidence} |"
            )

        lines.extend([
            "",
            "> [!NOTE]",
            "> All metrics represent wall-clock execution time on the local hardware protocol. No metrics are fabricated.",
        ])

        return "\n".join(lines)


def _load_golden_tensors() -> dict[str, np.ndarray]:
    if not GOLDEN_FIXTURES_PATH.is_file():
        raise FileNotFoundError(f"Golden fixtures archive not found at {GOLDEN_FIXTURES_PATH}")
    with np.load(GOLDEN_FIXTURES_PATH) as data:
        return {key: data[key] for key in data.files}


def create_candidate_pipelines(
    weights_dir: Path | str = DEFAULT_WEIGHTS_DIR,
) -> dict[str, tuple[str, Callable[[MediaFrame], tuple[bool, list[Any]]]]]:
    """Instantiate candidate detection + pose evaluation pipelines.

    Returns a mapping of candidate_id -> (candidate_name, pipeline_fn).

    The 'mediapipe_opencv' candidate downloads (if needed), verifies against
    the pinned manifest's size and SHA-256, and loads the real ONNX weights
    via `cv2.dnn.readNetFromONNX`, running genuine `net.forward()` inference
    per frame. Golden/fixed tensors are never used here — they remain
    reserved for the decode/preprocessing unit tests in
    tests/unit/infrastructure/test_opencv_adapters.py, which exercise the
    tensor-decoding logic in isolation from network/model availability.

    Raises whatever `resolve_model_artifact` raises (see
    `model_downloader.ModelArtifactUnavailableError`) if a pinned artifact
    cannot be obtained and verified — never silently falls back to a mock.
    """
    person_manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "person_detection_mediapipe_v1.json"
    )
    pose_manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "pose_estimation_mediapipe_v1.json"
    )

    person_model_path = resolve_model_artifact(person_manifest, weights_dir=weights_dir)
    pose_model_path = resolve_model_artifact(pose_manifest, weights_dir=weights_dir)

    opencv_detector = OpenCVPersonDetectorAdapter(
        model_path=person_model_path,
        confidence_threshold=0.5,
    )
    opencv_pose = OpenCVPoseInferenceAdapter(
        model_path=pose_model_path,
        confidence_threshold=0.5,
    )

    def run_opencv_pipeline(frame: MediaFrame) -> tuple[bool, list[Any]]:
        candidates = opencv_detector.detect_candidates(frame)
        if not candidates:
            return False, []
        target = candidates[0]
        landmarks = opencv_pose.infer_pose(
            frame,
            candidate_id=target.candidate_id,
            candidate_bbox=target.bbox,
            candidate_keypoints=target.keypoints,
        )
        return True, landmarks

    # Candidate 2: Deterministic Reference / Stub pipeline
    stub_detector = StubPersonDetectorAdapter()
    stub_pose = StubPoseInferenceAdapter()

    def run_stub_pipeline(frame: MediaFrame) -> tuple[bool, list[Any]]:
        candidates = stub_detector.detect_candidates(frame)
        if not candidates:
            return False, []
        target = candidates[0]
        landmarks = stub_pose.infer_pose(
            frame, candidate_id=target.candidate_id, candidate_bbox=target.bbox
        )
        return True, landmarks

    return {
        "mediapipe_opencv": ("MediaPipe Pose (OpenCV 5 Runtime)", run_opencv_pipeline),
        "reference_stub": ("Reference Baseline (Deterministic Stub)", run_stub_pipeline),
    }


def _create_synthetic_frames_for_clip(clip: dict[str, Any], count: int) -> list[MediaFrame]:
    """Generate synthetic MediaFrame instances matching clip specification for reproducible benchmarking."""
    res = clip.get("resolution", {})
    width = int(res.get("width", 1920))
    height = int(res.get("height", 1080))
    fps = float(clip.get("fps", 30.0))
    interval_ms = 1000.0 / fps

    golden_tensors = _load_golden_tensors()
    base_frame = golden_tensors["test_frame"]
    if base_frame.shape[0] != height or base_frame.shape[1] != width:
        base_frame = cv2.resize(base_frame, (width, height))

    frames = []
    for i in range(count):
        # Deterministic variation in pixel values per frame
        frame_img = base_frame.copy()
        if i > 0:
            frame_img[10:30, 10:30] = (i * 7) % 255
        frames.append(
            MediaFrame(
                frame_index=i,
                timestamp_ms=i * interval_ms,
                width=width,
                height=height,
                data=frame_img,
            )
        )
    return frames


def _derive_local_source_id(source_uri: str) -> str:
    """Map a manifest clip's `source_uri` (e.g. an s3:// URI) to the local,
    media-root-relative filename an authorized copy is expected under."""
    return Path(urlparse(source_uri).path).name


def run_pose_baseline_benchmark(
    manifest_path: str | Path,
    split: str = "development",
    selected_candidate_ids: list[str] | None = None,
    max_frames_per_clip: int | None = 10,
    warmup_frames: int = 2,
    custom_pipelines: dict[str, tuple[str, Callable[[MediaFrame], tuple[bool, list[Any]]]]] | None = None,
    media_root: str | Path | None = None,
    allow_synthetic: bool = False,
) -> CandidateBenchmarkReport:
    """Execute the pose baseline benchmark protocol on the specified split.

    Args:
        manifest_path: Path to dataset manifest JSON.
        split: Split name to evaluate ('development' or 'heldout').
        selected_candidate_ids: List of candidate IDs to benchmark, or None for all.
        max_frames_per_clip: Maximum number of frames per clip to evaluate.
        warmup_frames: Number of unmeasured warmup frames prior to recording latency.
        custom_pipelines: Optional dictionary of custom pipelines for extension/testing.
        media_root: Directory authorized recorded clips are read from. Each
            clip's `source_uri` basename must exist under this directory.
            Required (and used) only when `allow_synthetic` is False.
        allow_synthetic: When False (the default), every clip's authorized
            media must resolve and be processed for real; a missing or
            unauthorized source raises `AuthorizedMediaUnavailableError`
            rather than silently falling back to synthetic frames. When
            True, every evaluated clip must declare
            `consent_scope: "synthetic-no-person"` and the benchmark
            generates reproducible synthetic frames instead, explicitly
            labeling the returned report as synthetic
            (`report.environment.is_synthetic`). Synthetic mode validates
            mechanics and model load/forward latency only — it never
            substitutes for authorized held-out evaluation.

    Returns:
        CandidateBenchmarkReport containing hardware metadata and per-candidate metrics.
    """
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found at {manifest_file}")

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    # Validate schema
    errors = validate_manifest_schema(manifest)
    if errors:
        raise ValueError(f"Manifest validation failed: {errors}")

    # Filter clips strictly by split
    clips = [c for c in manifest.get("clips", []) if c.get("split") == split]
    if not clips:
        raise ValueError(f"No clips found for split '{split}' in manifest.")

    pipelines = custom_pipelines or create_candidate_pipelines()
    if selected_candidate_ids:
        pipelines = {cid: p for cid, p in pipelines.items() if cid in selected_candidate_ids}
        if not pipelines:
            raise ValueError(f"None of the selected candidates {selected_candidate_ids} are available.")

    candidate_metrics: dict[str, CandidateMetrics] = {
        cid: CandidateMetrics(candidate_id=cid, candidate_name=name)
        for cid, (name, _) in pipelines.items()
    }

    env = BenchmarkEnvironment(is_synthetic=allow_synthetic)
    media_adapter = ControlledMediaSourceAdapter(media_root=media_root)

    # Resolve exactly one source per clip: either a verified authorized
    # local file (normal mode) or an explicitly-registered synthetic stream
    # (opt-in mode only, and only for clips declared synthetic-no-person).
    frames_per_clip = max_frames_per_clip or 15
    source_id_by_clip: dict[str, str] = {}

    for clip in clips:
        clip_id = clip["clip_id"]
        if allow_synthetic:
            consent_scope = clip.get("consent_scope")
            if consent_scope != "synthetic-no-person":
                raise ValueError(
                    f"Refusing synthetic mode for clip '{clip_id}': consent_scope "
                    f"'{consent_scope}' is not 'synthetic-no-person'. Synthetic mode "
                    "may only run against clips explicitly declared synthetic."
                )
            frames = _create_synthetic_frames_for_clip(clip, count=frames_per_clip)
            media_adapter.register_synthetic_stream(clip_id, frames)
            source_id_by_clip[clip_id] = clip_id
        else:
            source_id = _derive_local_source_id(clip["source_uri"])
            try:
                resolved_path = media_adapter.resolve_source_path(source_id)
            except (UnauthorizedSourceError, SourceNotFoundError) as exc:
                raise AuthorizedMediaUnavailableError(
                    f"Clip '{clip_id}' references source_uri '{clip['source_uri']}' "
                    f"(expected authorized local file '{source_id}' under "
                    f"media_root={media_root!r}), but it is unavailable: {exc}. "
                    "Pass allow_synthetic=True to run mechanics-only synthetic "
                    "benchmarking instead of a real evaluation."
                ) from exc

            expected_sha256 = clip["sha256"].lower()
            actual_sha256 = calculate_file_sha256(resolved_path)
            if actual_sha256 != expected_sha256:
                raise MediaIntegrityError(
                    f"Clip '{clip_id}' resolved to local file '{resolved_path}' "
                    f"(source_uri '{clip['source_uri']}'), but its content does not "
                    f"match the manifest's pinned sha256 (expected {expected_sha256}, "
                    f"got {actual_sha256}). This is refused rather than benchmarked: "
                    "the file may be corrupt, stale, or -- since resolution is by "
                    "source_uri basename -- an unrelated clip's file that happens to "
                    "share the same basename."
                )
            source_id_by_clip[clip_id] = source_id

    # Run benchmarks per candidate
    for cid, (name, pipeline_fn) in pipelines.items():
        metrics = candidate_metrics[cid]

        for clip in clips:
            clip_id = clip["clip_id"]
            source_id = source_id_by_clip[clip_id]
            frame_stream = media_adapter.open_stream(source_id)
            frames = list(
                frame_stream
                if allow_synthetic
                else itertools.islice(frame_stream, max_frames_per_clip)
            )

            # Warmup
            for w_idx in range(min(warmup_frames, len(frames))):
                pipeline_fn(frames[w_idx])

            # Measured iterations
            for frame in frames:
                metrics.total_frames += 1

                t_start = time.perf_counter()
                detected_person, landmarks = pipeline_fn(frame)
                t_end = time.perf_counter()

                latency_ms = (t_end - t_start) * 1000.0
                metrics.latencies_ms.append(latency_ms)

                if detected_person:
                    metrics.detected_person_frames += 1

                if landmarks:
                    metrics.valid_pose_frames += 1
                    for lm in landmarks:
                        if hasattr(lm, "confidence") and lm.confidence is not None:
                            metrics.confidence_values.append(float(lm.confidence))
                        if hasattr(lm, "visibility") and lm.visibility is not None:
                            metrics.visibility_values.append(float(lm.visibility))

        metrics.finalize()

    return CandidateBenchmarkReport(
        environment=env,
        manifest_path=str(manifest_file),
        split=split,
        total_clips_evaluated=len(clips),
        candidate_metrics=candidate_metrics,
    )
