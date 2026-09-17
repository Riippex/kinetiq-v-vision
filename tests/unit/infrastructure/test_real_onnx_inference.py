"""Real ONNX model load and forward-pass evidence.

Proves the pinned MediaPipe person-detector and pose-estimation weights can
actually be downloaded, hash-verified, loaded via `cv2.dnn.readNetFromONNX`,
and executed with `net.forward()` — independent of whether any authorized
evaluation clip exists. This is the "real inference works" evidence called
for even when VV-302 stays blocked on authorized held-out media: it proves
the runner and the model path are genuine, not that evaluation coverage is
meaningful on non-person synthetic imagery (see test_pose_baselines.py).

Requires network access on first run to populate the git-ignored `weights/`
cache; subsequent runs reuse the verified local copy.
"""

from pathlib import Path

import cv2
import numpy as np

from kinetiq_v_vision.domain.entities import MediaFrame
from kinetiq_v_vision.domain.value_objects import BoundingBox
from kinetiq_v_vision.infrastructure.inference.manifest import load_model_manifest
from kinetiq_v_vision.infrastructure.inference.model_downloader import (
    resolve_model_artifact,
)
from kinetiq_v_vision.infrastructure.inference.opencv_adapters import (
    OpenCVPersonDetectorAdapter,
    OpenCVPoseInferenceAdapter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_MANIFESTS_DIR = REPO_ROOT / "model-manifests"


def _synthetic_rgb_frame(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_person_detector_artifact_downloads_and_verifies() -> None:
    manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "person_detection_mediapipe_v1.json"
    )
    path = resolve_model_artifact(manifest)

    assert path.is_file()
    assert path.stat().st_size == manifest["artifact"]["size_bytes"]
    assert path.suffix == ".onnx"


def test_pose_estimation_artifact_downloads_and_verifies() -> None:
    manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "pose_estimation_mediapipe_v1.json"
    )
    path = resolve_model_artifact(manifest)

    assert path.is_file()
    assert path.stat().st_size == manifest["artifact"]["size_bytes"]
    assert path.suffix == ".onnx"


def test_person_detector_real_forward_pass_executes() -> None:
    """Loads the real ONNX graph and runs a genuine net.forward() — not a
    golden/fixed tensor — proving actual OpenCV 5 DNN execution."""
    manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "person_detection_mediapipe_v1.json"
    )
    model_path = resolve_model_artifact(manifest)

    adapter = OpenCVPersonDetectorAdapter(model_path=model_path, confidence_threshold=0.5)
    assert adapter.net is not None, "Real ONNX net must be loaded, not a forward_fn stand-in"
    assert adapter.forward_fn is None

    frame = MediaFrame(
        frame_index=0,
        timestamp_ms=0.0,
        width=640,
        height=480,
        data=_synthetic_rgb_frame(640, 480),
    )

    # Must execute without raising, regardless of whether anything is detected.
    candidates = adapter.detect_candidates(frame)
    assert isinstance(candidates, list)


def test_pose_estimator_real_forward_pass_executes() -> None:
    manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "pose_estimation_mediapipe_v1.json"
    )
    model_path = resolve_model_artifact(manifest)

    adapter = OpenCVPoseInferenceAdapter(model_path=model_path, confidence_threshold=0.5)
    assert adapter.net is not None, "Real ONNX net must be loaded, not a forward_fn stand-in"
    assert adapter.forward_fn is None

    frame = MediaFrame(
        frame_index=0,
        timestamp_ms=0.0,
        width=640,
        height=480,
        data=_synthetic_rgb_frame(640, 480),
    )
    bbox = BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8)

    # Must execute without raising, regardless of whether landmarks decode.
    landmarks = adapter.infer_pose(frame, candidate_id="c1", candidate_bbox=bbox)
    assert isinstance(landmarks, list)


def test_readnetfromonnx_loads_pinned_weights_directly() -> None:
    """Bypasses the adapters entirely: proves cv2.dnn.readNetFromONNX loads
    the verified weight file and net.forward() produces real tensor output."""
    manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "person_detection_mediapipe_v1.json"
    )
    model_path = resolve_model_artifact(manifest)

    net = cv2.dnn.readNetFromONNX(str(model_path))
    blob = cv2.dnn.blobFromImage(
        _synthetic_rgb_frame(224, 224),
        scalefactor=1.0 / 127.5,
        size=(224, 224),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
    )
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    assert len(outputs) == 2
    for output in outputs:
        assert output is not None
        assert output.size > 0
