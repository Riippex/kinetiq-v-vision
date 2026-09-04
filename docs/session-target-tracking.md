# Session Target Tracking and Visibility Recovery

September 4, 2026. Accepted product requirements: explicitly select the exercising user at session start, ignore other people and animals, retain short-lived tracking context, explain visibility-related pauses, and evaluate squat, biceps curl, and lateral raise first. Algorithms and timing defaults below are proposals to benchmark, not implemented guarantees.

## Session Enrollment

Show a live camera preview and ask the user to select their detected person region and confirm it. Show a clear selected-person outline before beginning calibration. If no credible human candidate is available, ask for a framing adjustment. Bind the selection to an ephemeral target ID scoped to the workout and camera stream; this is target selection, not verification of the account holder's identity.

Use short-lived spatial continuity and pose geometry to associate detections with the selected track. No persistent face recognition, face embedding, cross-session identity template, or saved enrollment photo is required. Do not promise perfect identity continuity when people cross or disappear. If evidence becomes ambiguous, ask for confirmation rather than choosing the nearest or largest person automatically. Camera changes and engine restarts require fresh confirmation in the initial design.

Other people and animals must never contribute repetitions or observations to the selected user. Achieving that is an evaluation requirement, not an assumption that a human detector never misclassifies animals. Incidental bystander detections may be needed transiently for association but should not be persisted or exposed as personal records. The selected model pipeline must be tested with multiple people and pets; add or change the detector/association component if the baseline cannot meet this requirement.

## Proposed State and Timing Policy

| State | Engine behavior | Product behavior |
|---|---|---|
| Awaiting selection / calibrating | No exercise scoring | Show selection and framing guidance |
| Tracking | Publish fresh, supported observations | Advance observed activity and show feedback |
| Temporarily missing or insufficient visibility | Retain target context; stop unsupported counts and assessments immediately | Show a brief reconnecting/visibility message; preserve progress |
| Ambiguous target | Stop attribution immediately | Pause the active exercise immediately and request target confirmation |
| Visibility pause | Continue looking for the confirmed target without crediting unseen activity | Pause guided active-exercise progression and explain the reason |
| Reacquiring | Require consistent fresh evidence of the selected target | Offer resume after stability; retain previous confirmed results |
| Selection expired | Discard target association context | Require a new selection; keep saved workout progress |

Initial tunable defaults: show degraded tracking immediately; pause guided exercise after 2 seconds of unusable evidence; require 0.5 seconds of stable fresh evidence to recover from a brief loss; expire target-association context after 15 seconds without usable observations. Evaluate these values against false pauses, incorrect reacquisition, and user experience. They are not measured latency claims.

Short unambiguous losses may recover automatically before the pause threshold. After an actual pause, ask the user to resume. On an identity ambiguity, require explicit selection regardless of elapsed time. Reset any partial repetition across a loss; do not reconstruct completed repetitions from unseen motion. Pause only the affected active exercise, not the entire stored workout or unrelated rest intervals. Wall-clock timestamps remain accurate while guided active time is paused.

Reasons must reflect evidence: selected person out of frame, required joints obscured, uncertain target, camera disconnected, or stale frames. Do not assert lighting or an animal caused the failure unless that cause is actually detected. Use concise guidance such as “Your knees are out of view. Step back to resume tracking.” only when the affected joints support that explanation.

## Cached State and Durability

Keep the hot tracking buffer in engine memory: target ID, camera/stream epoch, last valid timestamp, bounding region, recent landmarks/confidence, movement phase, and sequence counter. Redis may hold a minimal session-scoped status snapshot with expiration for display/recovery coordination; it is not mandatory to write every frame to Redis. Reset or expire transient state on camera changes, session completion, or prolonged loss. Never refresh a last-observed timestamp merely because a cached record was read.

Cached pose is stale evidence, not a fresh observation. An optional last-seen outline must be visibly marked as stale and fade; never use it to count repetitions, judge technique, or depict a live moving skeleton. Prediction may assist association but must be labeled and excluded from measured exercise results. Do not store video or face templates in this cache.

PostgreSQL remains authoritative for product sessions and confirmed results. Redis expiration or failure must not erase saved progress or silently change the tracked user. The engine owns tracking quality; the product owns the pause/resume transition and user message. Correlate commands/events by session, target, and stream epoch to reject old observations after reselection. Redis status does not authorize reconnecting a camera or reading another user's session.

## Proposed Output Contract

Include session ID, analysis ID, ephemeral target ID, stream epoch, sequence number, capture/processing timestamps, tracking state, reason code, last-valid timestamp and observation age. Return pose landmarks and their available confidence/visibility fields, movement phase, confirmed repetition events/counts, and exercise-specific observations with evidence and confidence. Unavailable values must be explicit, not fabricated. Separate observed from predicted/stale values and target-association confidence from pose confidence. The final schema and any score calibration remain to be validated.

Return only selected-user analysis to product clients. A missing required joint can invalidate a specific assessment without implying every visible joint is unavailable. Never claim overall technique correctness from pose availability alone.

## Evaluation and MLflow

Initial exercises: squat, biceps curl, lateral raise. Include authorized clips with another person crossing the user, a stationary bystander, a pet crossing, the user leaving/returning, similar clothing, partial occlusion, camera motion, dropped frames, and an engine restart. Hold out people and clips from parameter tuning where possible.

Measure target-switch errors, non-target repetitions attributed to the user, false pauses, correct reacquisition, time to pause/recover, rep-count error, and end-to-end latency. Treat any observed silent target switch or bystander-attributed repetition as a release-blocking failure requiring investigation; a clean test set is not proof of zero real-world risk.