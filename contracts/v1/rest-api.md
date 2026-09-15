# Vision REST API Contract v1.0.0

This specification establishes the canonical REST API interface for `kinetiq-v-vision`.
All endpoints require authentication using service credentials, session context, and a tracing correlation ID (`X-Correlation-ID`).

## Route Specifications

| Method | Path | Request Body / Query Params | Success Response | Error Codes | Description |
|---|---|---|---|---|---|
| `POST` | `/v1/analyses` | `session_id` (UUID), `source_id` (string), `exercise_key` (string), `exercise_version` (int), `idempotency_key` (string) | `201 Created`<br>`{analysis_id, session_id, epoch: 1, state: "AWAITING_SELECTION"}` | 400, 401, 403, 409, 422 | Initializes an analysis context for an authorized source. Idempotent on key. |
| `POST` | `/v1/analyses/{id}/target` | `candidate_id` (string), `expected_epoch` (int), `idempotency_key` (string) | `200 OK`<br>`{target_person_id, epoch, state: "CONFIRMED"}` | 400, 401, 403, 404, 409, 422 | Explicitly confirms the ephemeral target candidate. Increments epoch if re-targeting. Reject if expected_epoch != current_epoch with 409. |
| `GET` | `/v1/analyses/{id}` | None | `200 OK`<br>`{analysis_id, session_id, epoch, state, last_valid_at}` | 401, 403, 404 | Retrieves analysis lifecycle status and metadata. |
| `GET` | `/v1/analyses/{id}/candidates` | None | `200 OK`<br>`{candidates: [{candidate_id, bbox: [x,y,w,h], confidence, expires_at}]}` | 401, 403, 404 | Lists active candidate person targets detected in the active frame for selection. |
| `GET` | `/v1/analyses/{id}/observations` | Query: `after` (string, optional: `"{epoch}:{sequence}"`), `limit` (int, 1..100, default: 50) | `200 OK`<br>`{observations: [...], next_cursor: string \| null, has_more: bool}` | 401, 403, 404, 410 | Bounded polling cursor for observation events. Returns 410 CURSOR_EXPIRED if requested cursor has dropped out of buffer. |
| `DELETE` | `/v1/analyses/{id}` | None | `204 No Content` | 401, 403, 404 | Idempotently stops the analysis and releases inference resources. |

## Epoch and Sequence Semantics

1. **Epoch Rules**:
   - Every session starts at `epoch: 1`.
   - Any physical disruption (stream restart, camera angle change, target re-acquisition after complete loss) or manual target change increments the `epoch`.
   - Incoming target selection requests must supply `expected_epoch`. If the expected epoch is older than the current epoch, the engine returns `409 CONFLICT` with code `STALE_EPOCH`.
   - Consumers must reject or drop observations whose `epoch` is less than their highest received epoch.

2. **Sequence Rules**:
   - Within a given `epoch`, `sequence` is a 1-indexed strictly increasing integer (`1, 2, 3, ...`).
   - Every frame processed that emits an observation increments `sequence` by 1.
   - Consumers can verify continuity: if `sequence` jumps (e.g. from 10 to 14), frames were dropped due to latency or load shedding.
   - Out-of-order or duplicate sequences within the same epoch are invalid and must be rejected by consumers.

3. **Cursor and Buffer Retention**:
   - The cursor string format is `{epoch}:{sequence}`.
   - Vision retains a sliding window buffer of recent observations (e.g., last 300 frames / 10 seconds).
   - If a client requests `after={epoch}:{sequence}` where `sequence` has aged out of the buffer, Vision returns `410 Gone` with code `CURSOR_EXPIRED` and includes the oldest available cursor in the response body to allow recovery.

## Error Response Envelope

All error responses return a standardized JSON envelope:
```json
{
  "error": {
    "code": "STALE_EPOCH",
    "message": "Expected epoch 1 does not match active epoch 2",
    "correlation_id": "c1f7a080-3278-4394-bb9e-6441ef31ebaa",
    "retryable": false,
    "details": {}
  }
}
```
