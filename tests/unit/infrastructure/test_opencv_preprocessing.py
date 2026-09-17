"""Unit tests for OpenCV 5 preprocessing modules and golden tensor validation."""

from pathlib import Path

import numpy as np
import pytest

from kinetiq_v_vision.domain.value_objects import BoundingBox
from kinetiq_v_vision.infrastructure.inference.preprocessing import (
    LetterboxMetadata,
    PersonDetectionPreprocessor,
    PoseEstimationPreprocessor,
    RoiCropMetadata,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = REPO_ROOT / "fixtures" / "golden" / "golden_fixtures.npz"


@pytest.fixture(scope="module")
def golden_data() -> dict[str, np.ndarray]:
    with np.load(GOLDEN_PATH) as data:
        return {key: data[key] for key in data.files}


def test_person_detection_preprocessor_shape_and_range(
    golden_data: dict[str, np.ndarray],
) -> None:
    frame = golden_data["test_frame"]
    preprocessor = PersonDetectionPreprocessor(target_size=(224, 224))

    tensor, meta = preprocessor.preprocess(frame)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert -1.0 <= tensor.min() <= 1.0
    assert -1.0 <= tensor.max() <= 1.0

    assert meta.original_width == 1920
    assert meta.original_height == 1080
    assert meta.target_width == 224
    assert meta.target_height == 224
    assert meta.pad_left == 0
    assert meta.pad_top > 0  # 1920x1080 is wider than 1:1, so vertical padding is applied


def test_person_detection_preprocessor_matches_golden_tensor(
    golden_data: dict[str, np.ndarray],
) -> None:
    frame = golden_data["test_frame"]
    golden_tensor = golden_data["person_detection_input"]

    preprocessor = PersonDetectionPreprocessor()
    tensor, _ = preprocessor.preprocess(frame)

    # Assert byte-level numerical consistency within floating point tolerance
    diff = np.abs(tensor - golden_tensor).max()
    assert diff < 1e-5, f"Preprocessor diverged from golden tensor with max diff {diff}"


def test_letterbox_metadata_unprojects_coordinates() -> None:
    meta = LetterboxMetadata(
        original_width=1920,
        original_height=1080,
        target_width=224,
        target_height=224,
        scale_ratio=224.0 / 1920.0,
        pad_left=0,
        pad_top=round((224 - 1080 * (224 / 1920)) / 2),
    )

    # A box in the middle of the letterbox area
    bbox = meta.unproject_box(x1_pixel=50.0, y1_pixel=80.0, x2_pixel=150.0, y2_pixel=160.0)

    assert 0.0 <= bbox.x <= 1.0
    assert 0.0 <= bbox.y <= 1.0
    assert 0.0 < bbox.width <= 1.0
    assert 0.0 < bbox.height <= 1.0
    assert bbox.x + bbox.width <= 1.0001
    assert bbox.y + bbox.height <= 1.0001


def test_pose_estimation_preprocessor_shape_and_range(
    golden_data: dict[str, np.ndarray],
) -> None:
    frame = golden_data["test_frame"]
    bbox = BoundingBox(x=0.4, y=0.2, width=0.2, height=0.6)

    preprocessor = PoseEstimationPreprocessor(target_size=(256, 256), box_enlarge_factor=1.25)
    tensor, meta = preprocessor.preprocess(frame, bbox)

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= 1.0
    assert 0.0 <= tensor.max() <= 1.0

    assert meta.original_width == 1920
    assert meta.original_height == 1080
    assert meta.target_size == 256
    assert meta.roi_size > 0.0


def test_pose_estimation_preprocessor_matches_golden_tensor(
    golden_data: dict[str, np.ndarray],
) -> None:
    frame = golden_data["test_frame"]
    golden_tensor = golden_data["pose_estimation_input"]

    bbox = BoundingBox(x=800 / 1920.0, y=200 / 1080.0, width=320 / 1920.0, height=700 / 1080.0)
    preprocessor = PoseEstimationPreprocessor()
    tensor, _ = preprocessor.preprocess(frame, bbox)

    diff = np.abs(tensor - golden_tensor).max()
    assert diff < 1e-5, f"Pose preprocessor diverged from golden tensor with max diff {diff}"



def test_roi_crop_metadata_unprojects_points() -> None:
    meta = RoiCropMetadata(
        original_width=1920,
        original_height=1080,
        roi_x1=800.0,
        roi_y1=200.0,
        roi_size=875.0,
        target_size=256,
    )

    # Point at center of crop (0.5, 0.5)
    norm_x, norm_y = meta.unproject_point(0.5, 0.5)
    expected_x = round((800.0 + 0.5 * 875.0) / 1920.0, 6)
    expected_y = round((200.0 + 0.5 * 875.0) / 1080.0, 6)

    assert norm_x == expected_x
    assert norm_y == expected_y
    assert 0.0 <= norm_x <= 1.0
    assert 0.0 <= norm_y <= 1.0


def test_pose_preprocessor_handles_boundary_candidates() -> None:
    """Test candidate bounding boxes touching or exceeding frame boundaries."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    preprocessor = PoseEstimationPreprocessor()

    # Candidate near top-left corner
    corner_bbox = BoundingBox(x=0.01, y=0.01, width=0.1, height=0.2)
    tensor, meta = preprocessor.preprocess(frame, corner_bbox)
    assert tensor.shape == (1, 3, 256, 256)
    assert meta.roi_x1 < 0  # ROI extends past left border, padded with zeros

    # Candidate near bottom-right corner
    br_bbox = BoundingBox(x=0.9, y=0.85, width=0.1, height=0.15)
    tensor_br, _meta_br = preprocessor.preprocess(frame, br_bbox)
    assert tensor_br.shape == (1, 3, 256, 256)


def test_preprocessors_reject_invalid_inputs() -> None:
    det_prep = PersonDetectionPreprocessor()
    pose_prep = PoseEstimationPreprocessor()

    with pytest.raises(ValueError, match="Expected BGR image array"):
        det_prep.preprocess(np.zeros((100, 100)))  # 2D array

    with pytest.raises(ValueError, match="Expected BGR image array"):
        pose_prep.preprocess(np.zeros((100, 100)), BoundingBox(0, 0, 1, 1))

    with pytest.raises(ValueError, match="dimensions must be greater than zero"):
        det_prep.preprocess(np.zeros((0, 100, 3), dtype=np.uint8))
