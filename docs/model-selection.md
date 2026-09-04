# Initial Pose Model and Licensing Recommendation

## Recommended Baseline

Evaluate the MediaPipe Pose ONNX model distributed in OpenCV Zoo, together with its companion MediaPipe person detector, through OpenCV DNN. Start with pose_estimation_mediapipe_2023mar.onnx before comparing quantized variants. The upstream directory documents 33 pose landmarks and segmentation output and explicitly states that its files are Apache-2.0 licensed. This is a practical initial candidate for the project's OpenCV-first integration; speed and accuracy on our workload remain unmeasured.

Sources: [pose model and license declaration](https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe), [pose license](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/pose_estimation_mediapipe/LICENSE), [detector license](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/person_detection_mediapipe/LICENSE).

The pipeline is person detection, pose inference, confidence/visibility filtering, temporal tracking and smoothing, exercise-specific movement analysis, and structured observations. Kinetiq V Vision is this engine; it must attribute the pretrained model rather than imply that the original weights were trained by this project. The product consumes observations to decide coaching and routine changes.

Do not infer clinical measurements, reliable physical depth, or exercise correctness from landmark count alone. Exercise-specific criteria, camera placement, occlusion handling, and evaluation determine supported claims. The initial single-participant workout scenario still needs explicit person selection and handling of other people entering the frame.

## Adoption Gate

Before adoption, pin upstream commits and exact detector/pose artifacts; record download URLs, SHA-256 hashes, license copies/notices, preprocessing, outputs, and conversion provenance. Verify OpenCV 5 execution on authorized clips, then the phone capture path. Record hardware, precision, input resolution, processing throughput, end-to-end latency, landmark stability, rep-count errors, visibility failures, and dropped frames. Include different camera angles, partial visibility, and interruptions. This proposal does not claim that the upstream example has already been tested with our selected OpenCV build.

If the baseline fails measured requirements, compare a specific lightweight RTMPose checkpoint and export. Check that checkpoint's weights, training-data terms, detector dependencies, and runtime compatibility separately; the repository's license alone does not establish every artifact's terms. [RTMPose paper and project](https://arxiv.org/abs/2303.07399).

## Repository License

Recommend retaining the existing Apache License 2.0 in both repositories. It permits reuse, modification, redistribution, and commercial use under its conditions, and includes an express contributor patent license. Preserve applicable attribution and notices, include the license, and identify modifications when distributing modified files. It does not require downstream improvements to be published, so it suits broad reuse and future commercial development. [Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0).

This license applies to our own contributions as designated; it does not relicense third-party weights, datasets, libraries, or user media. Keep dependency and model notices distinct. Progress photos and evaluation recordings are not made public or open-source by the repository license.