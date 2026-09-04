# Vision Architecture and ML Lifecycle v1

September 4, 2026. Requested architecture: independently deployed REST microservice, Clean Architecture, reproducible notebook research, and managed SageMaker MLflow. The design below is not an implementation claim.

## Layers

- Domain: target identity scoped to analysis, visibility states, repetition state machines, observation values, timestamps and confidence semantics. No FastAPI, OpenCV, boto3 or MLflow imports.
- Application: start/select/analyze/read/stop use cases; ports for pose inference, detection, frame source, clock, analysis state, artifact loading and telemetry. Bounded input policy and orchestration live here.
- Infrastructure: OpenCV DNN or selected inference runtime, NumPy frame preparation, media adapters, model loading, transient state adapters, AWS/MLflow experiment adapters.
- Interfaces: FastAPI REST, CLI evaluation entry points and notebook-facing functions.
- Composition root: settings, model/runtime selection, startup/readiness, ASGI app and process wiring.

```text
src/kinetiq_v_vision/
  domain/
  application/{use_cases,ports}/
  infrastructure/{inference,media,state,telemetry}/
  interfaces/{rest,cli}/
  bootstrap/
notebooks/
configs/
contracts/
evaluation/
model-manifests/
tests/{unit,integration,contract}/
infra/terraform/
```

Keep production algorithms in importable source modules. Notebooks call those modules for experiments and interpretation; the service never executes notebooks on a request. Avoid duplicate model preprocessing or repetition logic in notebooks. Worker processes handle inference so ASGI request handling is not blocked by CPU work.

Product service creates an analysis, explicitly selects its target, reads observations using a cursor, and stops it. Provisional endpoints are defined in contracts/rest-api.md. Authenticate service calls and validate the authorized product session and source; never accept an arbitrary user-supplied URL that the worker fetches. Use source IDs resolved through a controlled media adapter. A camera/stream change increments stream epoch and requires renewed selection.

Start with bounded HTTP observation polling. Media transport is separate: recorded authorized clips first, live phone/Ring transport after measured capture tests. Do not call the REST design evidence that live video transport has been solved. Use short calls, bounded pages and explicit overload/retry behavior. If measurement shows polling cannot meet the UX target, record a transport ADR before changing it.

## ML/Data Science Lifecycle

1. State the task, supported exercise/camera conditions, risk cases and evaluation criteria.
2. Assemble authorized data; version manifests and annotation provenance. Split by participant/session to reduce leakage. Keep real media private in S3.
3. Explore visibility, framing and annotation consistency in notebooks.
4. Run the pretrained baseline and compare detector/pose candidates using the same data and hardware protocol.
5. Tune tracking and temporal analysis on development data. Fine-tune weights only when evidence, rights and sufficient data justify it; training is not a required milestone by itself.
6. Freeze configuration and evaluate on held-out data. Inspect failures and regressions per condition, not only aggregate metrics.
7. Record runs in managed SageMaker MLflow, package a versioned pipeline release and promote only after evaluation review.
8. Deploy a pinned image/model/configuration combination, monitor operational metrics and maintain rollback. New data or failures return to a controlled evaluation cycle, not automatic online training on user videos.

MLflow records parameters, metrics, code revision, model hashes, dataset-manifest hash, environment and report artifacts. A release manifest includes detector and pose versions, preprocessing, target association, temporal rules, contract version, evaluation run, image digest, licenses and limits. Registry aliases aid workflow, but deploy immutable versions/digests. User photos are neither training consent nor evaluation data by default.

## Notebook Sequence

- 00_experiment_readiness.ipynb: task, inputs and manifest checks.
- 01_dataset_audit.ipynb: authorized dataset coverage, participant splits and annotation quality.
- 02_pose_baselines.ipynb: model/runtime comparisons and reproducible latency.
- 03_target_tracking.ipynb: distractors, occlusion, identity switches and recovery.
- 04_temporal_analysis.ipynb: repetitions, exercise phases and parameter sensitivity.
- 05_heldout_evaluation.ipynb: frozen pipeline evaluation and failure review.
- 06_release_review.ipynb: evidence, model lineage, limitations and promotion decision.

Initially only the readiness notebook is executable; later notebook stages depend on real data and production modules. The notebook index documents required inputs/outputs rather than presenting invented results. Execute notebooks from a fresh kernel in order, with explicit parameters, seeds and environment. Strip private outputs before Git; store authorized executed HTML/notebooks with the experiment artifacts. Run light synthetic checks in CI and full dataset evaluation in the controlled ML workflow. [Notebook execution](https://nbconvert.readthedocs.io/en/latest/index.html).

## EDA Boundary

Durable product events are handled by the monolith's outbox/EventBridge/SQS flow. Vision stays REST-facing and does not acquire a product database dependency. Model release and completed evaluation facts may later feed deployment automation from the release process; adoption requires a separate authenticated event contract. High-frequency frames and pose events do not belong on the business event bus.

## Initial Readiness Gates

Schema fixtures and state-machine tests; model license/provenance checks; clean-kernel notebook execution; actual OpenCV inference on authorized clips; target selection and distractor evaluation; load/capacity measurement; MLflow run write/read; release manifest and rollback rehearsal. No metrics or cloud provisioning are implied by these documents.
