"""Unit tests for pose baseline evaluation module and CLI command."""

import json
from pathlib import Path
import pytest

from kinetiq_v_vision.evaluation.baselines import (
    BenchmarkEnvironment,
    CandidateBenchmarkReport,
    CandidateMetrics,
    create_candidate_pipelines,
    run_pose_baseline_benchmark,
)
from kinetiq_v_vision.interfaces.cli.main import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DATA_DIR = REPO_ROOT / "fixtures" / "data"
MANIFEST_PATH = FIXTURES_DATA_DIR / "synthetic_manifest.json"


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
    pipelines = create_candidate_pipelines()
    assert "mediapipe_opencv" in pipelines
    assert "reference_stub" in pipelines

    for cid, (name, fn) in pipelines.items():
        assert isinstance(name, str)
        assert callable(fn)


def test_run_pose_baseline_benchmark_on_development_split() -> None:
    report = run_pose_baseline_benchmark(
        manifest_path=MANIFEST_PATH,
        split="development",
        max_frames_per_clip=3,
        warmup_frames=1,
    )

    assert report.total_clips_evaluated == 4
    assert len(report.candidate_metrics) == 2
    assert "mediapipe_opencv" in report.candidate_metrics
    assert "reference_stub" in report.candidate_metrics

    for cid, m in report.candidate_metrics.items():
        assert m.total_frames == 12  # 4 clips * 3 frames
        assert m.person_coverage_pct > 0.0
        assert m.pose_coverage_pct > 0.0
        assert m.p50_latency_ms > 0.0
        assert m.p95_latency_ms >= m.p50_latency_ms
        assert m.throughput_fps > 0.0
        assert 0.0 <= m.mean_landmark_confidence <= 1.0


def test_run_pose_baseline_benchmark_filters_candidates() -> None:
    report = run_pose_baseline_benchmark(
        manifest_path=MANIFEST_PATH,
        split="development",
        selected_candidate_ids=["reference_stub"],
        max_frames_per_clip=2,
    )

    assert len(report.candidate_metrics) == 1
    assert "reference_stub" in report.candidate_metrics
    assert "mediapipe_opencv" not in report.candidate_metrics


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
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Pose Baseline Benchmark Report: DEVELOPMENT Split" in captured.out
    assert "MediaPipe Pose (OpenCV 5 Runtime)" in captured.out
    assert out_json.is_file()

    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)
        assert data["split"] == "development"
        assert "mediapipe_opencv" in data["candidates"]


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
