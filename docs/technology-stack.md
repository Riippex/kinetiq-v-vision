# Technology Stack

Status: architecture baseline approved; pipeline implementation, model benchmarks and infrastructure provisioning are pending.

| Area | Selected direction |
|---|---|
| Runtime | Python with uv and a locked environment |
| Vision | OpenCV 5, NumPy; model selected through evaluation |
| Service | FastAPI REST, Pydantic and Uvicorn |
| Model baseline | OpenCV Zoo MediaPipe Pose ONNX and companion detector, pending validation |
| Experiments | Jupyter notebooks and managed SageMaker MLflow |
| Artifacts | Versioned manifests, hashes and private S3 artifacts |
| Compute | CPU containers on ECS Fargate initially; benchmark before sizing |
| Infrastructure | Terraform; engine resources and state separate from product |
| Verification | Unit, contract, integration and held-out evaluation |

Production algorithms live in importable modules. Notebooks cover EDA, preprocessing, feature engineering, baselines, target tracking, temporal analysis, held-out evaluation and release review. They reuse production functions. Model licenses, data permissions and provenance are evaluated per artifact.

See [architecture and ML lifecycle](architecture.md), [model selection](model-selection.md), [target tracking](session-target-tracking.md), [evaluation protocol](../evaluation/protocol.md) and [notebooks](../notebooks/README.md).
