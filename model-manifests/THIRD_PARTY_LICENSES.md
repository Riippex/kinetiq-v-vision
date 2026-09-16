# Third-Party Model Licenses & Attribution

This inventory records the provenance, licensing terms, and upstream citations for the third-party models pinned by Kinetiq V Vision.

---

## 1. MediaPipe Person Detection (`person_detection_mediapipe_2023mar.onnx`)

- **Origin / Maintainer:** OpenCV Zoo (`models/person_detection_mediapipe`) / Google MediaPipe authors / PINTO0309
- **Upstream Repository:** [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)
- **Pinned Commit:** `8e2b658d1007864a24cc2cf0e5b4288c74b32584`
- **Artifact:** `person_detection_mediapipe_2023mar.onnx`
- **SHA-256 Digest:** `47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f`
- **License:** Apache License 2.0
- **License URL:** [OpenCV Zoo Person Detection License](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/person_detection_mediapipe/LICENSE)
- **Notice & Attribution:**
  ```text
  Copyright (c) OpenCV Authors, Google LLC, and contributors.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ```

---

## 2. MediaPipe Pose Estimation (`pose_estimation_mediapipe_2023mar.onnx`)

- **Origin / Maintainer:** OpenCV Zoo (`models/pose_estimation_mediapipe`) / Google MediaPipe authors
- **Upstream Repository:** [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)
- **Pinned Commit:** `1f19f821d68288feff2ef5c53993b33da74b1509`
- **Artifact:** `pose_estimation_mediapipe_2023mar.onnx`
- **SHA-256 Digest:** `9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f`
- **License:** Apache License 2.0
- **License URL:** [OpenCV Zoo Pose Estimation License](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/pose_estimation_mediapipe/LICENSE)
- **Notice & Attribution:**
  ```text
  Copyright (c) OpenCV Authors, Google LLC, and contributors.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ```

---

## Repository Policy Note

Per `docs/model-selection.md`:
1. Kinetiq V Vision uses pretrained weights under their respective open-source licenses and attributes original authors.
2. Raw binary weights (`.onnx` files) are never committed to this Git repository. Weights are resolved through cryptographic hashes (`sha256`), download URLs, or private deployment artifacts in S3.
3. Kinetiq V Vision's own repository license (Apache 2.0) applies to original code contributions and does not re-license third-party weights, datasets, or user media.
