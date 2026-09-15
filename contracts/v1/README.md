# Vision Canonical Contract v1

This directory defines the canonical contract for Kinetiq V Vision (`kinetiq-v-vision`), which serves as the authoritative definition of API endpoints, capabilities, and real-time movement observation schemas for the Kinetiq ecosystem.

## Ownership Principles

1. **Canonical Source of Truth**: The Vision service owns observation and capability schemas. Consumer repositories (such as the product monolith `kinetiq-v`) consume versioned schemas and fixtures defined here.
2. **Explicit Capabilities**: Vision explicitly advertises supported exercises, perspectives, and metrics. Consumers must not assume an exercise is vision-supported unless present in `capabilities.v1.json`.
3. **Session, Epoch, and Sequence Invariants**:
   - Every observation belongs to an active `session_id`.
   - `epoch` starts at 1 and increments when stream continuity or target association changes (reconnection, camera change, re-selection). Consumers must drop or reject observations matching stale epochs.
   - `sequence` is a 1-indexed, strictly monotonic integer within each `epoch`. Missing sequence numbers indicate dropped frames or buffer gaps. Out-of-order sequence numbers must be rejected.
4. **Target Integrity & Ambiguity**: When target tracking becomes ambiguous (`TARGET_AMBIGUOUS`) or occluded/out of frame (`OCCLUSION`, `OUT_OF_FRAME`), repetitions/holds cease immediately. Confirmed repetitions are immutable and preserved.

## Directory Structure

- `schema/`:
  - `vision-capabilities.v1.schema.json`: JSON Schema (Draft 2020-12) for capability discovery.
  - `vision-observation.v1.schema.json`: JSON Schema (Draft 2020-12) for real-time observation events.
- `fixtures/`:
  - `capabilities.v1.json`: Canonical capabilities fixture (`bodyweight_squat`, `push_up`, `plank`, `glute_bridge`).
  - `observation_repetition.v1.json`: Valid repetition event fixture.
  - `observation_hold.v1.json`: Valid isometric hold fixture.
  - `observation_target_ambiguous.v1.json`: Target ambiguity reason code fixture.
  - `observation_visibility_lost.v1.json`: Target visibility loss fixture.
  - `negative/`: Suite of negative fixtures asserting validation failure for invalid formats, out-of-bound scores, missing required fields, or unauthorized properties.
- `rest-api.md`: Comprehensive REST endpoint, error envelope, and cursor pagination specification.
