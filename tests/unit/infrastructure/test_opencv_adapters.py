"""Unit tests for OpenCV 5 person detector and pose inference adapters."""

from pathlib import Path

import numpy as np
import pytest

from kinetiq_v_vision.domain.entities import MediaFrame
from kinetiq_v_vision.domain.value_objects import BoundingBox
from kinetiq_v_vision.infrastructure.inference.opencv_adapters import (
    MEDIAPIPE_POSE_LANDMARKS,
    OpenCVPersonDetectorAdapter,
    OpenCVPoseInferenceAdapter,
    generate_mediapipe_person_anchors,
)
from kinetiq_v_vision.infrastructure.media.controlled_media import (
    ControlledMediaSourceAdapter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = REPO_ROOT / "fixtures" / "golden" / "golden_fixtures.npz"


@pytest.fixture(scope="module")
def golden_data() -> dict[str, np.ndarray]:
    with np.load(GOLDEN_PATH) as data:
        return {key: data[key] for key in data.files}


def test_mediapipe_person_anchors_generation() -> None:
    anchors = generate_mediapipe_person_anchors()

    assert anchors.shape == (2254, 2)
    assert anchors.dtype == np.float32
    assert 0.0 < anchors[:, 0].min()
    assert anchors[:, 0].max() < 1.0
    assert 0.0 < anchors[:, 1].min()
    assert anchors[:, 1].max() < 1.0


def test_person_detector_adapter_decodes_golden_tensors(
    golden_data: dict[str, np.ndarray],
) -> None:
    frame = golden_data["test_frame"]
    box_delta = golden_data["detector_box_delta"]
    score_logits = golden_data["detector_score_logits"]

    def mock_forward(blob: np.ndarray) -> list[np.ndarray]:
        return [box_delta, score_logits]

    adapter = OpenCVPersonDetectorAdapter(forward_fn=mock_forward, confidence_threshold=0.5)
    candidates = adapter.detect_candidates(frame)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.candidate_id == "candidate_01"
    assert c.confidence > 0.95
    assert 0.0 <= c.bbox.x <= 1.0
    assert 0.0 <= c.bbox.y <= 1.0
    assert 0.0 < c.bbox.width <= 1.0
    assert 0.0 < c.bbox.height <= 1.0
    assert c.bbox.x + c.bbox.width <= 1.0001
    assert c.bbox.y + c.bbox.height <= 1.0001


def test_person_detector_adapter_filters_low_confidence() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box_delta = np.zeros((1, 2254, 16), dtype=np.float32)
    low_scores = np.full((1, 2254, 1), -15.0, dtype=np.float32)  # sigmoid ~ 3e-7

    adapter = OpenCVPersonDetectorAdapter(
        forward_fn=lambda blob: [box_delta, low_scores],
        confidence_threshold=0.5,
    )
    candidates = adapter.detect_candidates(frame)
    assert len(candidates) == 0


def test_person_detector_adapter_direct_detections_array() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Direct detections format: [x1, y1, x2, y2, score] in letterbox 224x224 coordinates
    direct_dets = np.array(
        [
            [20.0, 50.0, 180.0, 200.0, 0.92],
            [10.0, 10.0, 30.0, 30.0, 0.25],  # Should be filtered by confidence threshold
        ],
        dtype=np.float32,
    )

    adapter = OpenCVPersonDetectorAdapter(
        forward_fn=lambda blob: [direct_dets],
        confidence_threshold=0.5,
    )
    candidates = adapter.detect_candidates(frame)

    assert len(candidates) == 1
    assert candidates[0].confidence == 0.92
    assert 0.0 <= candidates[0].bbox.x <= 1.0
    assert 0.0 <= candidates[0].bbox.y <= 1.0


def test_person_detector_adapter_raises_when_unconfigured() -> None:
    adapter = OpenCVPersonDetectorAdapter()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="no model loaded and no forward_fn"):
        adapter.detect_candidates(frame)


def test_pose_inference_adapter_decodes_golden_tensors(
    golden_data: dict[str, np.ndarray],
) -> None:
    frame = golden_data["test_frame"]
    raw_landmarks = golden_data["pose_landmarks"]
    pose_presence = golden_data["pose_presence"]

    def mock_forward(blob: np.ndarray) -> list[np.ndarray]:
        return [raw_landmarks, pose_presence]

    adapter = OpenCVPoseInferenceAdapter(forward_fn=mock_forward, confidence_threshold=0.5)
    bbox = BoundingBox(x=0.4, y=0.2, width=0.2, height=0.6)

    landmarks = adapter.infer_pose(frame, candidate_id="cand_01", candidate_bbox=bbox)

    assert len(landmarks) == 33
    assert len(MEDIAPIPE_POSE_LANDMARKS) == 33

    for i, lm in enumerate(landmarks):
        assert lm.name == MEDIAPIPE_POSE_LANDMARKS[i]
        assert 0.0 <= lm.x <= 1.0
        assert 0.0 <= lm.y <= 1.0
        assert 0.0 <= lm.confidence <= 1.0
        assert 0.0 <= lm.visibility <= 1.0
        assert lm.z is not None


def test_pose_inference_adapter_filters_low_overall_presence(
    golden_data: dict[str, np.ndarray],
) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_landmarks = golden_data["pose_landmarks"]
    low_presence = np.array([[0.2]], dtype=np.float32)  # Below 0.5 threshold

    adapter = OpenCVPoseInferenceAdapter(
        forward_fn=lambda blob: [raw_landmarks, low_presence],
        confidence_threshold=0.5,
    )
    landmarks = adapter.infer_pose(frame, candidate_id="cand_01")
    assert len(landmarks) == 0


def test_pose_inference_adapter_raises_when_unconfigured() -> None:
    adapter = OpenCVPoseInferenceAdapter()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="no model loaded and no forward_fn"):
        adapter.infer_pose(frame, candidate_id="cand_01")


def test_end_to_end_controlled_media_inference_pipeline(
    golden_data: dict[str, np.ndarray],
) -> None:
    """Validate full chain: ControlledMedia -> Preprocessing -> Detector -> Pose Inference."""
    frame_data = golden_data["test_frame"]
    test_frame = MediaFrame(
        frame_index=0,
        timestamp_ms=0.0,
        width=1920,
        height=1080,
        data=frame_data,
    )

    # 1. Controlled media source
    media_adapter = ControlledMediaSourceAdapter()
    media_adapter.register_synthetic_stream("authorized_clip_01", [test_frame])

    stream = list(media_adapter.open_stream("authorized_clip_01"))
    assert len(stream) == 1
    stream_frame = stream[0]

    # 2. Person detector
    box_delta = golden_data["detector_box_delta"]
    score_logits = golden_data["detector_score_logits"]
    detector = OpenCVPersonDetectorAdapter(
        forward_fn=lambda blob: [box_delta, score_logits],
        confidence_threshold=0.5,
    )
    candidates = detector.detect_candidates(stream_frame)
    assert len(candidates) >= 1
    selected_target = candidates[0]

    # 3. Pose inference for selected target
    raw_landmarks = golden_data["pose_landmarks"]
    pose_presence = golden_data["pose_presence"]
    pose_adapter = OpenCVPoseInferenceAdapter(
        forward_fn=lambda blob: [raw_landmarks, pose_presence],
        confidence_threshold=0.5,
    )

    landmarks = pose_adapter.infer_pose(
        stream_frame,
        candidate_id=selected_target.candidate_id,
        candidate_bbox=selected_target.bbox,
    )

    assert len(landmarks) == 33
    assert all(0.0 <= lm.x <= 1.0 for lm in landmarks)
    assert all(0.0 <= lm.y <= 1.0 for lm in landmarks)
    assert all(0.0 <= lm.confidence <= 1.0 for lm in landmarks)
    assert landmarks[0].name == "nose"
    assert landmarks[11].name == "left_shoulder"
    assert landmarks[12].name == "right_shoulder"
