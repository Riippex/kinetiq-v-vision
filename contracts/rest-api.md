# REST Analysis Contract v0.1

Design contract. Authentication is required on every route; private network placement alone is insufficient. Product-origin requests carry a scoped service credential, session authorization context and correlation ID. Request schema/identity implementation will be pinned during scaffolding.

| Method and route | Input | Result |
|---|---|---|
| POST /v1/analyses | session_id, source_id, exercise_id/version, idempotency_key | 201 analysis_id, epoch, state; identical retry returns existing resource |
| POST /v1/analyses/{id}/target | candidate_id, expected_epoch, idempotency_key | 200 selected ephemeral target and new epoch |
| GET /v1/analyses/{id} | Authorized resource ID | 200 status, last_valid_at, epoch, capabilities |
| GET /v1/analyses/{id}/candidates | Authorized resource ID | 200 transient candidate IDs and bounding regions for user confirmation |
| GET /v1/analyses/{id}/observations?after={cursor}&limit={n} | Cursor and bounded limit (max 100) | 200 events, next_cursor, has_more; empty page when nothing new |
| DELETE /v1/analyses/{id} | Authorized resource ID | 204 idempotent closure |

Observation entries conform to observation.schema.json. Candidate IDs are scoped to the current analysis/epoch and expire; reject stale selections. A cursor is scoped to analysis/epoch. Return 410 CURSOR_EXPIRED when the recent buffer no longer contains the requested position, with a latest-status recovery path and explicit coverage gap. Do not silently pretend that transient observations are durably replayable. Product must persist confirmed events as received; buffer loss can leave incomplete coverage, which is shown to users.

Use 401 for absent/invalid credentials, 403 for disallowed actions where resource disclosure is acceptable, 404 for missing/inaccessible analyses, 409 for epoch/idempotency conflicts, 422 for invalid input, and 429/503 with Retry-After for capacity/temporary unavailability. Error envelope: code, message, correlation_id, retryable. Set client timeouts and retry budgets. Never retry a non-idempotent operation blindly.

source_id resolves through an authorized source registry; it is not a URL fetch endpoint. Camera frames/media are not GraphQL or JSON observation payloads. Recorded-source ingestion and live media adapters have separate validation gates.
