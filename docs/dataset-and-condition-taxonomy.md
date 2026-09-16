# Dataset Governance, Split Policy, and Condition Taxonomy

This specification defines the data contracts, split isolation rules, and condition taxonomy for Kinetiq V Vision research, baselines, and evaluation.

## 1. Participant- and Session-Level Split Policy

To prevent data leakage across model baselines, tracking heuristics, and temporal parameter tuning:

1. **Strict Participant Isolation**: A `participant_id` must exclusively belong to either the `development` split or the `heldout` split. Under no circumstances may clips from the same participant appear in both splits.
2. **Strict Session Isolation**: A `session_id` must independently belong to exactly one split, checked separately from `participant_id`. A recording session can leak evaluation signal across splits even if its clips were (incorrectly) attributed to different participant identifiers, so session isolation is never inferred from participant isolation alone.
3. **Deterministic Assignment**: Participant and session splits are assigned prior to any tuning and frozen in the dataset manifest.
4. **Required Exercise Coverage Per Split**: Every canonical exercise (`bodyweight_squat`, `push_up`, `plank`, `glute_bridge`) must have at least one clip in `development` and at least one clip in `heldout`. An exercise present in only one required split fails the dataset audit even if its total clip count is nonzero.
5. **Held-Out Protocol**:
   - `development`: Used for exploratory data analysis (`01_dataset_audit.ipynb`), pose baseline comparisons (`02_pose_baselines.ipynb`), target tracking tuning (`03_target_tracking.ipynb`), and repetition/hold state machine design (`04_temporal_analysis.ipynb`).
   - `heldout`: Strictly reserved for frozen release evaluation (`05_heldout_evaluation.ipynb`) and release promotion review (`06_release_review.ipynb`). It is never used for parameter tuning.

## 1a. Annotation Reference Integrity

Each manifest clip's `annotation_ref` is the sole authoritative pointer to its ground-truth annotation:

1. The path is resolved against the dataset root and must stay within it; references escaping the permitted root are rejected.
2. The referenced file must exist. A missing file fails the audit for that clip — it is never treated as passable by falling back to a directory scan.
3. The resolved annotation's own `clip_id` must equal the referencing clip's `clip_id`. A mismatch fails the audit even if some other file in the annotations directory happens to declare the expected `clip_id`.
4. Duplicate `clip_id` values across manifest clips are rejected.

## 2. Condition Taxonomy

Evaluation clips are tagged with standardized condition labels to allow granular per-condition evaluation beyond aggregate numbers:

| Category | Condition Tag | Description | Expected Engine Behavior |
|---|---|---|---|
| **Framing** | `clean_framing` | Full body or required joints in view with clear contrast and static camera. | Standard tracking and scoring active. |
| **Distractors** | `bystander_crossing` | A secondary person walks across the frame between camera and target user. | Spatial continuity maintained; no switch to bystander; pause if occluded. |
| | `stationary_bystander` | Another person remains visible in the background or side during workout. | Zero observations or repetitions attributed to bystander. |
| | `pet_crossing` | A dog or cat enters the exercising area or crosses the frame. | Zero repetitions attributed; animal ignored or classified as non-target. |
| **Visibility** | `partial_occlusion` | Part of the target body is temporarily occluded by furniture, equipment, or mat. | Reason code `OCCLUSION` emitted; repetitions pause if critical joints obscured. |
| | `leaving_returning` | User steps out of frame and returns within session timeout. | Reason code `OUT_OF_FRAME`; tracking resumes upon re-entry; completed reps preserved. |
| **Stream Artifacts** | `camera_motion` | Phone or tripod vibrates, tilts, or shakes slightly during exercise. | Landmark coordinates smoothed; no false repetition counted. |
| | `dropped_frames` | Stream packets drop, creating timestamp jitter or sequence gaps. | Sequence jump detected; stale observations rejected. |
| | `low_light` | Dim room lighting or backlighting creating high contrast/shadows. | `LOW_CONFIDENCE` reason code emitted if joint certainty falls below threshold. |
| | `similar_clothing` | Target and bystander wear visually similar athletic clothing. | Pose geometry and spatial continuity prevent identity swap. |

## 3. Synthetic CI Fixture Boundary

- CI pipelines execute validations using purely synthetic manifests and annotations located in `fixtures/data/`.
- Synthetic entries use `"consent_scope": "synthetic-no-person"` and dummy SHA-256 digests.
- No private video, raw frames, or participant photos are committed to Git. Real media assets reside securely in authorized cloud storage (S3) referenced by private URIs.
