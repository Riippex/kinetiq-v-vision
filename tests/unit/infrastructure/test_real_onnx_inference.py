"""Real ONNX model load and forward-pass evidence.

Proves the pinned MediaPipe person-detector and pose-estimation weights can
actually be downloaded, hash-verified, loaded via `cv2.dnn.readNetFromONNX`,
and executed with `net.forward()` — independent of whether any authorized
evaluation clip exists. This is the "real inference works" evidence called
for even when VV-302 stays blocked on authorized held-out media: it proves
the runner and the model path are genuine, not that evaluation coverage is
meaningful on non-person synthetic imagery (see test_pose_baselines.py).

`test_pose_estimator_real_forward_pass_executes` is the VV-301 decode-gate
test: it runs the real detector and pose graphs on a real photograph of a
real person (fixtures/real/, see PROVENANCE.md) and requires exactly the 33
canonical landmarks with plausible normalized coordinates, confidence, and
visibility. It fails on an empty result. Golden/fixed tensors are never used
here; they remain reserved for the isolated decode/preprocessing unit tests
in test_opencv_adapters.py and test_opencv_preprocessing.py.

Requires network access on first run to populate the git-ignored `weights/`
cache; subsequent runs reuse the verified local copy.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from kinetiq_v_vision.domain.entities import MediaFrame
from kinetiq_v_vision.infrastructure.inference.manifest import load_model_manifest
from kinetiq_v_vision.infrastructure.inference.model_downloader import (
    resolve_model_artifact,
)
from kinetiq_v_vision.infrastructure.inference.opencv_adapters import (
    MEDIAPIPE_POSE_LANDMARKS,
    OpenCVPersonDetectorAdapter,
    OpenCVPoseInferenceAdapter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_MANIFESTS_DIR = REPO_ROOT / "model-manifests"
REAL_PERSON_IMAGE_PATH = REPO_ROOT / "fixtures" / "real" / "mediapipe_pose_reference.jpg"
# See fixtures/real/PROVENANCE.md: upstream MediaPipe pose-landmarker test
# asset (Apache-2.0), fetched from storage.googleapis.com/mediapipe-assets.
REAL_PERSON_IMAGE_SHA256 = (
    "c8a830ed683c0276d713dd5aeda28f415f10cd6291972084a40d0d8b934ed62b"
)


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
    """VV-301 real evidence: the pinned pose ONNX graph, run through genuine
    `net.forward()` (not golden/fixed tensors, not a mocked forward_fn) on a
    real photograph of a real person (see fixtures/real/PROVENANCE.md), must
    decode exactly the 33 canonical MediaPipe landmarks with plausible
    normalized coordinates, confidence, and visibility. An empty or
    short result fails this test outright -- VV-301 cannot be Verified on
    an empty landmark list, a person-less image, or synthetic/golden data.
    """
    assert REAL_PERSON_IMAGE_PATH.is_file(), (
        f"Real-person fixture missing at {REAL_PERSON_IMAGE_PATH}; "
        "see fixtures/real/PROVENANCE.md"
    )
    actual_sha256 = hashlib.sha256(REAL_PERSON_IMAGE_PATH.read_bytes()).hexdigest()
    assert actual_sha256 == REAL_PERSON_IMAGE_SHA256, (
        f"fixtures/real/mediapipe_pose_reference.jpg content changed unexpectedly "
        f"(expected sha256 {REAL_PERSON_IMAGE_SHA256}, got {actual_sha256})"
    )

    person_manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "person_detection_mediapipe_v1.json"
    )
    pose_manifest = load_model_manifest(
        MODEL_MANIFESTS_DIR / "pose_estimation_mediapipe_v1.json"
    )
    person_model_path = resolve_model_artifact(person_manifest)
    pose_model_path = resolve_model_artifact(pose_manifest)

    detector = OpenCVPersonDetectorAdapter(
        model_path=person_model_path, confidence_threshold=0.5
    )
    adapter = OpenCVPoseInferenceAdapter(model_path=pose_model_path, confidence_threshold=0.5)
    assert adapter.net is not None, "Real ONNX net must be loaded, not a forward_fn stand-in"
    assert adapter.forward_fn is None

    image = cv2.imread(str(REAL_PERSON_IMAGE_PATH))
    assert image is not None, f"Failed to decode {REAL_PERSON_IMAGE_PATH}"
    frame = MediaFrame(
        frame_index=0,
        timestamp_ms=0.0,
        width=image.shape[1],
        height=image.shape[0],
        data=image,
    )

    candidates = detector.detect_candidates(frame)
    assert len(candidates) >= 1, "Real person detector found no candidate on a real-person photo"
    target = candidates[0]

    landmarks = adapter.infer_pose(
        frame,
        candidate_id=target.candidate_id,
        candidate_bbox=target.bbox,
        candidate_keypoints=target.keypoints,
    )

    assert len(landmarks) == 33, (
        f"Expected exactly 33 canonical MediaPipe landmarks, got {len(landmarks)}"
    )
    assert [lm.name for lm in landmarks] == list(MEDIAPIPE_POSE_LANDMARKS)

    for lm in landmarks:
        assert 0.0 <= lm.x <= 1.0, f"{lm.name}: x={lm.x} out of normalized range"
        assert 0.0 <= lm.y <= 1.0, f"{lm.name}: y={lm.y} out of normalized range"
        assert lm.confidence is not None and 0.0 <= lm.confidence <= 1.0, (
            f"{lm.name}: confidence={lm.confidence} out of range"
        )
        assert lm.visibility is not None and 0.0 <= lm.visibility <= 1.0, (
            f"{lm.name}: visibility={lm.visibility} out of range"
        )

    # The model must be confidently locating landmarks on this unambiguous,
    # unoccluded full-body photo, not merely emitting in-range noise.
    mean_visibility = sum(lm.visibility for lm in landmarks) / len(landmarks)
    assert mean_visibility > 0.8, f"Mean visibility unexpectedly low: {mean_visibility}"

    # Sanity-check plausible body topology: shoulders above hips above ankles
    # (y grows downward in image coordinates).
    by_name = {lm.name: lm for lm in landmarks}
    shoulder_y = (by_name["left_shoulder"].y + by_name["right_shoulder"].y) / 2
    hip_y = (by_name["left_hip"].y + by_name["right_hip"].y) / 2
    ankle_y = (by_name["left_ankle"].y + by_name["right_ankle"].y) / 2
    assert shoulder_y < hip_y < ankle_y, (
        f"Implausible body topology: shoulder_y={shoulder_y}, hip_y={hip_y}, ankle_y={ankle_y}"
    )


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
