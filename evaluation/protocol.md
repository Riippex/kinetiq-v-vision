# Evaluation Protocol v0.1

Evaluate squat, biceps curl and lateral raise with the selected user, including clean framing, partial visibility, another person crossing, a pet crossing, leaving/returning, similar clothing, camera motion and stream interruptions.

Dataset manifest fields: clip ID, private URI, SHA-256, participant pseudonym, consent scope, exercise, frame rate/resolution, camera angle, distractors, annotation version and split. Assign participants to development or held-out evaluation before tuning. If only one person is available, report that limitation. No clips or annotations exist yet.

Annotate target regions/visibility, repetition completion times, interruption and crossing events. Record uncertainty. Measure rep-count error, event precision/recall with a declared timing tolerance, target switches, bystander-attributed repetitions, false pauses, recovery delay, dropped frames and p50/p95 end-to-end latency. Report sample counts and results per exercise/condition. Silent target switches or bystander reps block release pending investigation. Set other performance gates after baseline measurement and before held-out comparison.

Managed SageMaker MLflow experiment name: kinetiq-v-vision-evaluation. Record code revision/dirty status, schema version, model and detector hashes, configuration, dataset-manifest hash, runtime/hardware, metrics and authorized failure examples. Releases pin model version, configuration digest and container digest; an alias alone does not reproduce a release. No managed server or experiment run exists yet.

contracts/observation.schema.json is the initial event shape. Coordinates are normalized to the original frame, potentially outside [0,1]. Confidence is a model score, not a calibrated probability. Null target is allowed only before selection/after expiry. Missing, ambiguous or expired states produce no new rep events. Do not emit stale landmarks as fresh. Deduplicate event IDs and reject old epochs/sequences. These semantic rules supplement JSON Schema.

Benchmark the baseline detector/pose combination before adopting weights. Pin upstream artifacts and preserve licenses. The 2-second pause and 15-second expiry are initial tuning values. Restarting the engine requires fresh target confirmation. Product state stays in PostgreSQL; MLflow is not a live-session store.
