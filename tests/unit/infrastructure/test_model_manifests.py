import copy
import hashlib
import sys
from pathlib import Path

import pytest

from kinetiq_v_vision.infrastructure.inference.manifest import (
    calculate_file_sha256,
    check_runtime_compatibility,
    get_model_manifest_schema_path,
    get_runtime_manifest_schema_path,
    load_model_manifest,
    load_runtime_manifest,
    run_opencv_smoke_check,
    validate_model_manifest,
    validate_runtime_manifest,
    verify_artifact_sha256,
)
from kinetiq_v_vision.interfaces.cli.main import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = REPO_ROOT / "model-manifests"


def test_schema_paths_exist() -> None:
    assert get_model_manifest_schema_path().exists()
    assert get_runtime_manifest_schema_path().exists()


def test_load_and_validate_person_detection_manifest() -> None:
    manifest_path = MANIFESTS_DIR / "person_detection_mediapipe_v1.json"
    assert manifest_path.exists()
    manifest = load_model_manifest(manifest_path)

    assert manifest["model_id"] == "person_detection_mediapipe_v1"
    assert manifest["task"] == "person_detection"
    assert manifest["artifact"]["filename"] == "person_detection_mediapipe_2023mar.onnx"
    assert (
        manifest["artifact"]["sha256"]
        == "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f"
    )
    assert manifest["artifact"]["size_bytes"] == 11990159
    assert len(manifest["artifact"]["download_urls"]) >= 1
    assert manifest["upstream"]["license"] == "Apache-2.0"
    assert manifest["preprocessing"]["input_shape"] == [1, 3, 224, 224]


def test_load_and_validate_pose_estimation_manifest() -> None:
    manifest_path = MANIFESTS_DIR / "pose_estimation_mediapipe_v1.json"
    assert manifest_path.exists()
    manifest = load_model_manifest(manifest_path)

    assert manifest["model_id"] == "pose_estimation_mediapipe_v1"
    assert manifest["task"] == "pose_estimation"
    assert manifest["artifact"]["filename"] == "pose_estimation_mediapipe_2023mar.onnx"
    assert (
        manifest["artifact"]["sha256"]
        == "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f"
    )
    assert manifest["artifact"]["size_bytes"] == 5557238
    assert len(manifest["artifact"]["download_urls"]) >= 1
    assert manifest["upstream"]["license"] == "Apache-2.0"
    # NHWC: the pose ONNX graph is channel-last, verified against upstream
    # mp_pose.py `_preprocess` (blob[np.newaxis, :, :, :] with no transpose).
    assert manifest["preprocessing"]["input_shape"] == [1, 256, 256, 3]
    assert manifest["postprocessing"]["landmark_count"] == 33


def test_load_and_validate_runtime_environment_manifest() -> None:
    manifest_path = MANIFESTS_DIR / "runtime-environment.json"
    assert manifest_path.exists()
    manifest = load_runtime_manifest(manifest_path)

    assert manifest["environment_name"] == "kinetiq-v-vision-runtime"
    assert manifest["python_spec"]["min_version"] == "3.12.0"
    assert manifest["python_spec"]["max_exclusive_version"] == "3.14.0"
    assert manifest["opencv_spec"]["min_version"] == "5.0.0"
    assert manifest["opencv_spec"]["required_major_version"] == 5
    assert "windows-x86_64" in manifest["supported_platforms"]
    assert "linux-x86_64" in manifest["supported_platforms"]


def test_rejects_invalid_model_manifest() -> None:
    manifest_path = MANIFESTS_DIR / "pose_estimation_mediapipe_v1.json"
    valid_data = load_model_manifest(manifest_path)

    # 1. Missing required field
    missing_key = copy.deepcopy(valid_data)
    del missing_key["artifact"]
    with pytest.raises(ValueError, match="Model manifest validation failed"):
        validate_model_manifest(missing_key)

    # 2. Invalid SHA-256 (not 64 hex characters)
    bad_sha = copy.deepcopy(valid_data)
    bad_sha["artifact"]["sha256"] = "invalid_hash"
    with pytest.raises(ValueError, match="Model manifest validation failed"):
        validate_model_manifest(bad_sha)

    # 3. Invalid task
    bad_task = copy.deepcopy(valid_data)
    bad_task["task"] = "semantic_segmentation"
    with pytest.raises(ValueError, match="Model manifest validation failed"):
        validate_model_manifest(bad_task)

    # 4. Extra property in upstream
    extra_prop = copy.deepcopy(valid_data)
    extra_prop["upstream"]["unexpected"] = "value"
    with pytest.raises(ValueError, match="Model manifest validation failed"):
        validate_model_manifest(extra_prop)


def test_rejects_invalid_runtime_manifest() -> None:
    manifest_path = MANIFESTS_DIR / "runtime-environment.json"
    valid_data = load_runtime_manifest(manifest_path)

    # Missing python_spec
    missing_spec = copy.deepcopy(valid_data)
    del missing_spec["python_spec"]
    with pytest.raises(ValueError, match="Runtime manifest validation failed"):
        validate_runtime_manifest(missing_spec)

    # Invalid platform
    bad_platform = copy.deepcopy(valid_data)
    bad_platform["supported_platforms"].append("solaris-sparc")
    with pytest.raises(ValueError, match="Runtime manifest validation failed"):
        validate_runtime_manifest(bad_platform)


def test_sha256_calculation_and_verification(tmp_path: Path) -> None:
    test_file = tmp_path / "dummy_artifact.bin"
    content = b"Kinetiq V Vision Test Model Weight Bytes"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert calculate_file_sha256(test_file) == expected_hash
    assert verify_artifact_sha256(test_file, expected_hash) is True
    assert verify_artifact_sha256(test_file, "0" * 64) is False


def test_current_runtime_compatibility() -> None:
    manifest_path = MANIFESTS_DIR / "runtime-environment.json"
    manifest = load_runtime_manifest(manifest_path)

    errors = check_runtime_compatibility(manifest)
    assert errors == [], (
        f"Current environment should be compatible with manifest: {errors}"
    )


def test_runtime_compatibility_detects_violations() -> None:
    manifest_path = MANIFESTS_DIR / "runtime-environment.json"
    manifest = load_runtime_manifest(manifest_path)

    # Incompatible python min version
    strict_py = copy.deepcopy(manifest)
    strict_py["python_spec"]["min_version"] = "3.20.0"
    errors = check_runtime_compatibility(strict_py)
    assert any("below minimum required" in e for e in errors)

    # Incompatible platform
    strict_platform = copy.deepcopy(manifest)
    strict_platform["supported_platforms"] = ["solaris-sparc64"]
    errors = check_runtime_compatibility(strict_platform)
    assert any("not in supported platforms" in e for e in errors)


def test_no_weight_binaries_committed() -> None:
    """Verify strictly that no raw weight binaries (.onnx, .pt, .pth, .tflite) exist in the repo."""
    forbidden_extensions = {".onnx", ".pt", ".pth", ".tflite"}
    for search_dir in [
        REPO_ROOT / "model-manifests",
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
    ]:
        if search_dir.exists():
            for file_path in search_dir.rglob("*"):
                if file_path.is_file():
                    assert file_path.suffix.lower() not in forbidden_extensions, (
                        f"Forbidden weight binary found committed: {file_path}"
                    )


def test_cli_check_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["check"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Runtime environment: compatible." in captured.out
    assert "Model manifest person_detection_mediapipe_v1.json: valid." in captured.out
    assert "Model manifest pose_estimation_mediapipe_v1.json: valid." in captured.out
    assert "Kinetiq V Vision engine: ready." in captured.out


def test_run_opencv_smoke_check_passes_with_installed_runtime() -> None:
    """OpenCV 5 is pinned and installed; the smoke check must execute real image
    processing rather than merely confirming that ``cv2`` imports."""
    assert run_opencv_smoke_check() is None


def test_run_opencv_smoke_check_reports_missing_cv2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    result = run_opencv_smoke_check()
    assert result is not None
    assert "OpenCV (cv2) is not installed" in result


def test_check_runtime_compatibility_detects_missing_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the readiness bug where `check` reported success without OpenCV.

    Simulates cv2 being absent even though it is installed in this test
    environment, proving the compatibility check no longer silently skips it.
    """
    manifest_path = MANIFESTS_DIR / "runtime-environment.json"
    manifest = load_runtime_manifest(manifest_path)

    monkeypatch.setitem(sys.modules, "cv2", None)
    errors = check_runtime_compatibility(manifest)
    assert any("OpenCV (cv2) is not installed" in e for e in errors)


def test_check_runtime_compatibility_detects_wrong_opencv_major_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    manifest_path = MANIFESTS_DIR / "runtime-environment.json"
    manifest = load_runtime_manifest(manifest_path)

    monkeypatch.setattr(cv2, "__version__", "4.10.0", raising=False)
    errors = check_runtime_compatibility(manifest)
    assert any(
        "does not match required major version 5" in e for e in errors
    )


def test_cli_check_command_fails_without_opencv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `kinetiq-vision check` CLI must return a nonzero exit code when the
    required OpenCV 5 runtime is unavailable, instead of reporting readiness."""
    monkeypatch.setitem(sys.modules, "cv2", None)

    exit_code = cli_main(["check"])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Runtime environment: compatible." not in captured.out
    assert "Kinetiq V Vision engine: ready." not in captured.out
