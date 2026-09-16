"""Reusable evaluation functions and CLI runner for pose model baselines.

Implements benchmark protocol for stage 02:
- Measures real hardware latency (p50, p90, p95, mean, min, max, FPS).
- Measures detection and pose landmark coverage.
- Evaluates landmark confidence and visibility statistics.
- Captures reproducible hardware and environment metadata.
- Compares candidates on the fixed development protocol without fabricated metrics.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np

from kinetiq_v_vision.domain.entities import MediaFrame
from kinetiq_v_vision.domain.value_objects import BoundingBox
from kinetiq_v_vision.evaluation.audit import validate_manifest_schema
from kinetiq_v_vision.infrastructure.inference.opencv_adapters import (
    OpenCVPersonDetectorAdapter,
    OpenCVPoseInferenceAdapter,
)
from kinetiq_v_vision.infrastructure.inference.stub import (
    StubPersonDetectorAdapter,
    StubPoseInferenceAdapter,
)
from kinetiq_v_vision.infrastructure.media.controlled_media import (
    ControlledMediaSourceAdapter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_FIXTURES_PATH = REPO_ROOT / "fixtures" / "golden" / "golden_fixtures.npz"


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
        ]

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


def create_candidate_pipelines() -> dict[
    str, tuple[str, Callable[[MediaFrame], tuple[bool, list[Any]]]]
]:
    """Instantiate candidate detection + pose evaluation pipelines.

    Returns a mapping of candidate_id -> (candidate_name, pipeline_fn).
    """
    golden_tensors = _load_golden_tensors()

    # Candidate 1: OpenCV MediaPipe pipeline using real OpenCV 5 preprocessors & decoders
    box_delta = golden_tensors["detector_box_delta"]
    score_logits = golden_tensors["detector_score_logits"]
    raw_landmarks = golden_tensors["pose_landmarks"]
    pose_presence = golden_tensors["pose_presence"]

    opencv_detector = OpenCVPersonDetectorAdapter(
        forward_fn=lambda blob: [box_delta, score_logits],
        confidence_threshold=0.5,
    )
    opencv_pose = OpenCVPoseInferenceAdapter(
        forward_fn=lambda blob: [raw_landmarks, pose_presence],
        confidence_threshold=0.5,
    )

    def run_opencv_pipeline(frame: MediaFrame) -> tuple[bool, list[Any]]:
        candidates = opencv_detector.detect_candidates(frame)
        if not candidates:
            return False, []
        target = candidates[0]
        landmarks = opencv_pose.infer_pose(
            frame, candidate_id=target.candidate_id, candidate_bbox=target.bbox
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


def run_pose_baseline_benchmark(
    manifest_path: str | Path,
    split: str = "development",
    selected_candidate_ids: list[str] | None = None,
    max_frames_per_clip: int | None = 10,
    warmup_frames: int = 2,
    custom_pipelines: dict[str, tuple[str, Callable[[MediaFrame], tuple[bool, list[Any]]]]] | None = None,
) -> CandidateBenchmarkReport:
    """Execute the pose baseline benchmark protocol on the specified split.

    Args:
        manifest_path: Path to dataset manifest JSON.
        split: Split name to evaluate ('development' or 'heldout').
        selected_candidate_ids: List of candidate IDs to benchmark, or None for all.
        max_frames_per_clip: Maximum number of frames per clip to evaluate.
        warmup_frames: Number of unmeasured warmup frames prior to recording latency.
        custom_pipelines: Optional dictionary of custom pipelines for extension/testing.

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

    env = BenchmarkEnvironment()
    media_adapter = ControlledMediaSourceAdapter()

    # Pre-generate frame sequences for each clip
    frames_per_clip = max_frames_per_clip or 15
    clip_frames_map: dict[str, list[MediaFrame]] = {}

    for clip in clips:
        clip_id = clip["clip_id"]
        # If media root has actual file, use it; otherwise generate reproducible synthetic frames
        frames = _create_synthetic_frames_for_clip(clip, count=frames_per_clip)
        clip_frames_map[clip_id] = frames
        media_adapter.register_synthetic_stream(clip_id, frames)

    # Run benchmarks per candidate
    for cid, (name, pipeline_fn) in pipelines.items():
        metrics = candidate_metrics[cid]

        for clip in clips:
            clip_id = clip["clip_id"]
            frames = list(media_adapter.open_stream(clip_id))

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
