"""OpenCV 5 adapters for person detection and pose estimation inference."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from kinetiq_v_vision.application.ports.inference import (
    PersonDetectorPort,
    PoseInferencePort,
)
from kinetiq_v_vision.domain.entities import CandidatePerson, Landmark, MediaFrame
from kinetiq_v_vision.domain.value_objects import BoundingBox
from kinetiq_v_vision.infrastructure.inference.preprocessing import (
    PersonDetectionPreprocessor,
    PoseEstimationPreprocessor,
)

MEDIAPIPE_POSE_LANDMARKS: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


def generate_mediapipe_person_anchors() -> np.ndarray:
    """Generate the 2254 anchor center points for 224x224 MediaPipe Person Detection.

    Grid specification:
    - 28x28 (stride 8): 2 anchors per cell -> 1568
    - 14x14 (stride 16): 2 anchors per cell -> 392
    - 7x7 (stride 32): 6 anchors per cell -> 294
    Total: 2254 anchors, matching OpenCV Zoo upstream exactly.
    """
    anchors: list[list[float]] = []

    # Grid 28x28
    for y in range(28):
        for x in range(28):
            for _ in range(2):
                anchors.append([(x + 0.5) / 28.0, (y + 0.5) / 28.0])

    # Grid 14x14
    for y in range(14):
        for x in range(14):
            for _ in range(2):
                anchors.append([(x + 0.5) / 14.0, (y + 0.5) / 14.0])

    # Grid 7x7
    for y in range(7):
        for x in range(7):
            for _ in range(6):
                anchors.append([(x + 0.5) / 7.0, (y + 0.5) / 7.0])

    return np.array(anchors, dtype=np.float32)


class OpenCVPersonDetectorAdapter(PersonDetectorPort):
    """Person detector adapter executing via OpenCV DNN or custom tensor evaluator."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        net: Any | None = None,
        forward_fn: Callable[[np.ndarray], list[np.ndarray]] | None = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        preprocessor: PersonDetectionPreprocessor | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.preprocessor = preprocessor or PersonDetectionPreprocessor()
        self.anchors = generate_mediapipe_person_anchors()
        self.forward_fn = forward_fn
        self.net = net

        if self.net is None and model_path is not None:
            path_str = str(model_path)
            if Path(path_str).is_file():
                self.net = cv2.dnn.readNetFromONNX(path_str)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_candidates(self, frame: MediaFrame | Any) -> list[CandidatePerson]:
        """Detect candidate people in the input frame and return bounding boxes in [0.0, 1.0]."""
        image = frame.data if isinstance(frame, MediaFrame) else frame
        blob, meta = self.preprocessor.preprocess(image)

        # Forward pass
        if self.forward_fn is not None:
            outputs = self.forward_fn(blob)
        elif self.net is not None:
            self.net.setInput(blob)
            out_names = self.net.getUnconnectedOutLayersNames()
            outputs = self.net.forward(out_names)
        else:
            raise RuntimeError(
                "OpenCVPersonDetectorAdapter has no model loaded and no forward_fn configured."
            )

        return self._decode_candidates(outputs, meta)

    def _decode_candidates(
        self, outputs: list[np.ndarray], meta: Any
    ) -> list[CandidatePerson]:
        """Decode raw detector tensors into normalized candidate persons."""
        if not outputs:
            return []

        # Case 1: Simplified/direct detections array [N, 5] (x1, y1, x2, y2, score)
        if len(outputs) == 1 and outputs[0].ndim == 2 and outputs[0].shape[1] >= 5:
            dets = outputs[0]
            candidates: list[CandidatePerson] = []
            for i, row in enumerate(dets):
                score = float(row[4])
                if score < self.confidence_threshold:
                    continue
                bbox = meta.unproject_box(row[0], row[1], row[2], row[3])
                candidates.append(
                    CandidatePerson(
                        candidate_id=f"candidate_{i+1:02d}",
                        bbox=bbox,
                        confidence=round(score, 4),
                        detected_at=datetime.now(UTC),
                    )
                )
            return candidates

        # Case 2: MediaPipe Person Detection SSD outputs [box_delta, score_logits]
        # output[0]: box delta (1, 2254, 16)
        # output[1]: classification scores (1, 2254, 1)
        raw_boxes = outputs[0]
        raw_scores = outputs[1]

        if raw_scores.ndim == 3:
            raw_scores = raw_scores[0, :, 0]
        elif raw_scores.ndim == 2:
            raw_scores = raw_scores[:, 0]

        if raw_boxes.ndim == 3:
            raw_boxes = raw_boxes[0]

        # Sigmoid scoring
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores.astype(np.float64), -100.0, 100.0)))

        # Filter by threshold prior to NMS
        valid_indices = np.where(scores >= self.confidence_threshold)[0]
        if len(valid_indices) == 0:
            return []

        cxy_delta = raw_boxes[valid_indices, :2] / 224.0
        wh_delta = raw_boxes[valid_indices, 2:4] / 224.0
        sel_anchors = self.anchors[valid_indices]

        # Coordinates in letterbox 224x224 space
        center_xy = (cxy_delta + sel_anchors) * 224.0
        half_wh = (wh_delta * 224.0) / 2.0
        xy1 = center_xy - half_wh
        xy2 = center_xy + half_wh

        pixel_boxes = np.column_stack([xy1, xy2])  # [M, 4]
        filtered_scores = scores[valid_indices]

        # OpenCV NMS
        # Format for cv2.dnn.NMSBoxes is [x, y, w, h] in integers/floats
        nms_boxes = [
            [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
            for b in pixel_boxes
        ]
        keep_indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            [float(s) for s in filtered_scores],
            score_threshold=float(self.confidence_threshold),
            nms_threshold=float(self.nms_threshold),
        )

        if len(keep_indices) == 0:
            return []

        keep = np.array(keep_indices).flatten()
        candidates = []

        for idx, k in enumerate(keep):
            box = pixel_boxes[k]
            score = float(filtered_scores[k])
            bbox = meta.unproject_box(box[0], box[1], box[2], box[3])
            candidates.append(
                CandidatePerson(
                    candidate_id=f"candidate_{idx+1:02d}",
                    bbox=bbox,
                    confidence=round(score, 4),
                    detected_at=datetime.now(UTC),
                )
            )

        return candidates


class OpenCVPoseInferenceAdapter(PoseInferencePort):
    """Pose inference adapter extracting 33 body landmarks via OpenCV DNN."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        net: Any | None = None,
        forward_fn: Callable[[np.ndarray], list[np.ndarray]] | None = None,
        confidence_threshold: float = 0.5,
        preprocessor: PoseEstimationPreprocessor | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.preprocessor = preprocessor or PoseEstimationPreprocessor()
        self.forward_fn = forward_fn
        self.net = net

        if self.net is None and model_path is not None:
            path_str = str(model_path)
            if Path(path_str).is_file():
                self.net = cv2.dnn.readNetFromONNX(path_str)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def infer_pose(
        self,
        frame: MediaFrame | Any,
        candidate_id: str,
        candidate_bbox: BoundingBox | None = None,
    ) -> list[Landmark]:
        """Infer 33 body landmarks for the specified candidate in [0.0, 1.0] normalized coordinates."""
        image = frame.data if isinstance(frame, MediaFrame) else frame
        bbox = candidate_bbox or BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)

        blob, meta = self.preprocessor.preprocess(image, bbox)

        if self.forward_fn is not None:
            outputs = self.forward_fn(blob)
        elif self.net is not None:
            self.net.setInput(blob)
            out_names = self.net.getUnconnectedOutLayersNames()
            outputs = self.net.forward(out_names)
        else:
            raise RuntimeError(
                "OpenCVPoseInferenceAdapter has no model loaded and no forward_fn configured."
            )

        return self._decode_pose(outputs, meta)

    def _decode_pose(
        self, outputs: list[np.ndarray], meta: Any
    ) -> list[Landmark]:
        """Decode raw pose tensors into 33 normalized body landmarks."""
        if not outputs:
            return []

        # Find landmarks tensor
        # MediaPipe Pose ONNX:
        # Output 0: landmarks [1, 195] (39 keypoints x 5 values: x, y, z, vis, pres)
        # Output 1: pose presence flag [1, 1]
        landmarks_raw = outputs[0]

        # Check pose confidence if present
        if len(outputs) > 1 and outputs[1].size == 1:
            overall_conf = float(outputs[1].ravel()[0])
            if overall_conf < self.confidence_threshold:
                return []

        # Reshape to (N, 5) or (N, >=3)
        if landmarks_raw.ndim == 3 and landmarks_raw.shape[0] == 1:
            landmarks_raw = landmarks_raw[0]
        elif landmarks_raw.ndim == 2 and landmarks_raw.shape[0] == 1:
            # Flattened [1, 195] -> [39, 5]
            if landmarks_raw.shape[1] == 195:
                landmarks_raw = landmarks_raw.reshape(39, 5)
            elif landmarks_raw.shape[1] == 33 * 5:
                landmarks_raw = landmarks_raw.reshape(33, 5)
            elif landmarks_raw.shape[1] == 33 * 3:
                landmarks_raw = landmarks_raw.reshape(33, 3)

        num_points = min(len(MEDIAPIPE_POSE_LANDMARKS), landmarks_raw.shape[0])
        landmarks: list[Landmark] = []

        for i in range(num_points):
            name = MEDIAPIPE_POSE_LANDMARKS[i]
            row = landmarks_raw[i]

            # In MediaPipe, x and y can be in [0, 256] or normalized [0, 1]
            raw_x = float(row[0])
            raw_y = float(row[1])

            # Normalize crop coordinates to [0.0, 1.0] if in [0, 256]
            crop_norm_x = raw_x / 256.0 if raw_x > 1.0 else raw_x
            crop_norm_y = raw_y / 256.0 if raw_y > 1.0 else raw_y

            # Unproject from crop to full frame [0.0, 1.0]
            norm_x, norm_y = meta.unproject_point(crop_norm_x, crop_norm_y)

            # z (relative depth)
            z_val = float(row[2]) if len(row) > 2 else 0.0

            # visibility / presence
            if len(row) >= 5:
                vis_logit = float(row[3])
                pres_logit = float(row[4])
                vis = 1.0 / (1.0 + np.exp(-np.clip(vis_logit, -50.0, 50.0)))
                pres = 1.0 / (1.0 + np.exp(-np.clip(pres_logit, -50.0, 50.0)))
                conf = float(vis * pres)
            elif len(row) >= 4:
                vis_logit = float(row[3])
                vis = 1.0 / (1.0 + np.exp(-np.clip(vis_logit, -50.0, 50.0)))
                conf = float(vis)
            else:
                vis = 1.0
                conf = 1.0

            landmarks.append(
                Landmark(
                    name=name,
                    x=norm_x,
                    y=norm_y,
                    confidence=round(float(conf), 4),
                    z=round(z_val, 4),
                    visibility=round(float(vis), 4),
                )
            )

        return landmarks
