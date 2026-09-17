# Real-person test fixtures

These are genuine photographs (not synthetic/golden tensors) used exclusively
to provide real evidence that the pose model actually decodes 33 landmarks
from a real person, per VV-301. They must never be used as inputs to
"synthetic mode" benchmark paths.

## mediapipe_pose_reference.jpg

- Source: `https://storage.googleapis.com/mediapipe-assets/pose.jpg`
- Referenced by: `mediapipe/tasks/testdata/vision/BUILD` in
  `google-ai-edge/mediapipe` (commit tracked via `master` at time of fetch,
  2026-09-17), as the upstream test asset for MediaPipe's pose landmarker
  test suite -- i.e. it is the literal reference test image for the
  BlazePose/MediaPipe Pose model lineage our ONNX artifact
  (`pose_estimation_mediapipe_2023mar.onnx`) is converted from.
- License: Apache License 2.0 (`google-ai-edge/mediapipe` repository license;
  the referencing `BUILD` file carries the same Apache-2.0 header).
- SHA-256: `c8a830ed683c0276d713dd5aeda28f415f10cd6291972084a40d0d8b934ed62b`
- Size: 44100 bytes
- Content: a single adult person in a full-body standing yoga pose, outdoors,
  unobstructed -- suitable for exercising the 33-landmark pose contract.
