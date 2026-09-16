"""OpenCV 5 preprocessing modules for person detection and pose estimation.

Implements preprocessing pipelines specified by:
- model-manifests/person_detection_mediapipe_v1.json (224x224 RGB, letterbox center pad, scale to [-1, 1])
- model-manifests/pose_estimation_mediapipe_v1.json (256x256 RGB, 1.25x ROI crop, scale to [0, 1])
"""

from dataclasses import dataclass
import cv2
import numpy as np

from kinetiq_v_vision.domain.value_objects import BoundingBox


@dataclass(frozen=True)
class LetterboxMetadata:
    """Metadata tracking letterbox padding and scaling for coordinate unprojection."""

    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale_ratio: float
    pad_left: int
    pad_top: int

    def unproject_box(
        self, x1_pixel: float, y1_pixel: float, x2_pixel: float, y2_pixel: float
    ) -> BoundingBox:
        """Unproject coordinates from letterbox frame back to normalized [0.0, 1.0] coordinates."""
        # Subtract letterbox padding
        unpad_x1 = (x1_pixel - self.pad_left) / self.scale_ratio
        unpad_y1 = (y1_pixel - self.pad_top) / self.scale_ratio
        unpad_x2 = (x2_pixel - self.pad_left) / self.scale_ratio
        unpad_y2 = (y2_pixel - self.pad_top) / self.scale_ratio

        # Normalize to original image dimensions [0.0, 1.0]
        norm_x1 = max(0.0, min(1.0, unpad_x1 / self.original_width))
        norm_y1 = max(0.0, min(1.0, unpad_y1 / self.original_height))
        norm_x2 = max(0.0, min(1.0, unpad_x2 / self.original_width))
        norm_y2 = max(0.0, min(1.0, unpad_y2 / self.original_height))

        norm_w = max(0.0, norm_x2 - norm_x1)
        norm_h = max(0.0, norm_y2 - norm_y1)

        return BoundingBox(
            x=round(float(norm_x1), 6),
            y=round(float(norm_y1), 6),
            width=round(float(norm_w), 6),
            height=round(float(norm_h), 6),
        )


@dataclass(frozen=True)
class RoiCropMetadata:
    """Metadata tracking candidate person ROI crop for landmark unprojection."""

    original_width: int
    original_height: int
    roi_x1: float
    roi_y1: float
    roi_size: float
    target_size: int = 256

    def unproject_point(
        self, norm_crop_x: float, norm_crop_y: float
    ) -> tuple[float, float]:
        """Unproject landmark point from normalized [0.0, 1.0] crop coordinates

        back to normalized [0.0, 1.0] full-frame coordinates.
        """
        pixel_x = self.roi_x1 + norm_crop_x * self.roi_size
        pixel_y = self.roi_y1 + norm_crop_y * self.roi_size

        norm_x = max(0.0, min(1.0, pixel_x / self.original_width))
        norm_y = max(0.0, min(1.0, pixel_y / self.original_height))

        return round(float(norm_x), 6), round(float(norm_y), 6)


class PersonDetectionPreprocessor:
    """OpenCV 5 preprocessor for MediaPipe person detection.

    Conforms to person_detection_mediapipe_v1.json:
    - Target shape: [1, 3, 224, 224] (NCHW)
    - Color format: RGB
    - Normalization: scale_to_minus_one_to_one ([-1.0, 1.0])
    - Aspect ratio: letterbox_center_pad
    """

    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        self.target_width, self.target_height = target_size

    def preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, LetterboxMetadata]:
        """Preprocess an input BGR image into an NCHW float32 tensor in [-1.0, 1.0]."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected BGR image array with shape (H, W, 3), got {type(image)}")

        orig_h, orig_w = image.shape[:2]
        if orig_h == 0 or orig_w == 0:
            raise ValueError("Input image dimensions must be greater than zero.")

        # 1. Color format BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Scale ratio preserving aspect ratio
        ratio = min(self.target_height / orig_h, self.target_width / orig_w)
        new_w = int(round(orig_w * ratio))
        new_h = int(round(orig_h * ratio))

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 3. Letterbox center pad
        pad_w = self.target_width - new_w
        pad_h = self.target_height - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # 4. Normalization: [0, 255] -> [-1.0, 1.0] via (x / 127.5) - 1.0
        normalized = (padded.astype(np.float32) / 127.5) - 1.0

        # 5. HWC -> CHW -> NCHW
        chw = np.transpose(normalized, (2, 0, 1))
        nchw = np.ascontiguousarray(chw[np.newaxis, :, :, :], dtype=np.float32)

        meta = LetterboxMetadata(
            original_width=orig_w,
            original_height=orig_h,
            target_width=self.target_width,
            target_height=self.target_height,
            scale_ratio=float(ratio),
            pad_left=pad_left,
            pad_top=pad_top,
        )

        return nchw, meta


class PoseEstimationPreprocessor:
    """OpenCV 5 preprocessor for MediaPipe pose estimation.

    Conforms to pose_estimation_mediapipe_v1.json:
    - Target shape: [1, 3, 256, 256] (NCHW)
    - Color format: RGB
    - Normalization: scale_to_zero_to_one ([0.0, 1.0])
    - Bounding box enlarge factor: 1.25
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        box_enlarge_factor: float = 1.25,
    ) -> None:
        self.target_width, self.target_height = target_size
        self.box_enlarge_factor = box_enlarge_factor

    def preprocess(
        self, image: np.ndarray, bbox: BoundingBox
    ) -> tuple[np.ndarray, RoiCropMetadata]:
        """Crop and preprocess candidate ROI into an NCHW float32 tensor in [0.0, 1.0]."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected BGR image array with shape (H, W, 3), got {type(image)}")

        orig_h, orig_w = image.shape[:2]
        if orig_h == 0 or orig_w == 0:
            raise ValueError("Input image dimensions must be greater than zero.")

        # Pixel bbox center and dimension
        pixel_cx = (bbox.x + bbox.width / 2.0) * orig_w
        pixel_cy = (bbox.y + bbox.height / 2.0) * orig_h
        pixel_bw = bbox.width * orig_w
        pixel_bh = bbox.height * orig_h

        # Square ROI enlarged by box_enlarge_factor (default 1.25)
        raw_size = max(pixel_bw, pixel_bh, 1.0)
        roi_size = raw_size * self.box_enlarge_factor

        roi_x1 = pixel_cx - roi_size / 2.0
        roi_y1 = pixel_cy - roi_size / 2.0
        roi_x2 = roi_x1 + roi_size
        roi_y2 = roi_y1 + roi_size

        # Crop with zero-padding when ROI extends beyond frame bounds
        src_x1 = int(max(0, round(roi_x1)))
        src_y1 = int(max(0, round(roi_y1)))
        src_x2 = int(min(orig_w, round(roi_x2)))
        src_y2 = int(min(orig_h, round(roi_y2)))

        pad_left = int(max(0, round(src_x1 - roi_x1)))
        pad_top = int(max(0, round(src_y1 - roi_y1)))
        pad_right = int(max(0, round(roi_x2 - src_x2)))
        pad_bottom = int(max(0, round(roi_y2 - src_y2)))

        cropped = image[src_y1:src_y2, src_x1:src_x2]
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        # Resize to 256x256
        resized = cv2.resize(
            cropped,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalization: [0, 255] -> [0.0, 1.0]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC -> CHW -> NCHW
        chw = np.transpose(normalized, (2, 0, 1))
        nchw = np.ascontiguousarray(chw[np.newaxis, :, :, :], dtype=np.float32)

        meta = RoiCropMetadata(
            original_width=orig_w,
            original_height=orig_h,
            roi_x1=float(roi_x1),
            roi_y1=float(roi_y1),
            roi_size=float(roi_size),
            target_size=self.target_width,
        )

        return nchw, meta
