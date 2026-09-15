from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import pytest

from kinetiq_v_vision.bootstrap.container import Container
from kinetiq_v_vision.domain.entities import CandidatePerson, Observation, RepetitionEvent
from kinetiq_v_vision.domain.value_objects import (
    BoundingBox,
    ReasonCode,
    TrackingState,
    VisibilityState,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "v1"
    / "schema"
    / "vision-observation.v1.schema.json"
)


@pytest.fixture(scope="module")
def observation_schema() -> dict[str, Any]:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def client() -> TestClient:
    container = Container()
    app = container.create_configured_app()
    return TestClient(app)


def test_health_check(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "kinetiq-v-vision"


def test_rest_analysis_lifecycle(
    client: TestClient, observation_schema: dict[str, Any]
) -> None:
    # 1. Create analysis
    create_payload = {
        "session_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
        "source_id": "phone-camera-01",
        "exercise_key": "bodyweight_squat",
        "exercise_version": 1,
    }
    resp = client.post("/v1/analyses", json=create_payload)
    assert resp.status_code == 201
    analysis_data = resp.json()
    analysis_id = analysis_data["analysis_id"]
    assert analysis_data["session_id"] == create_payload["session_id"]
    assert analysis_data["epoch"] == 1
    assert analysis_data["state"] == "AWAITING_SELECTION"

    # 2. Get status
    status_resp = client.get(f"/v1/analyses/{analysis_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["analysis_id"] == analysis_id

    # 3. Add a candidate and get candidates
    container: Container = client.app.dependency_overrides.get(None, None)  # type: ignore
    # Add candidate via repo directly to simulate detector discovery
    repo = client.app.dependency_overrides[
        list(client.app.dependency_overrides.keys())[4]
    ]()
    analysis = repo.get_by_id(analysis_id)
    analysis.add_candidate(
        CandidatePerson(
            candidate_id="person_01",
            bbox=BoundingBox(0.2, 0.1, 0.6, 0.8),
            confidence=0.96,
            detected_at=datetime.now(timezone.utc),
        )
    )
    repo.save(analysis)

    cand_resp = client.get(f"/v1/analyses/{analysis_id}/candidates")
    assert cand_resp.status_code == 200
    candidates = cand_resp.json()["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == "person_01"

    # 4. Reject stale epoch selection
    stale_resp = client.post(
        f"/v1/analyses/{analysis_id}/target",
        json={"candidate_id": "person_01", "expected_epoch": 99},
    )
    assert stale_resp.status_code == 409
    err = stale_resp.json()["error"]
    assert err["code"] == "STALE_EPOCH"
    assert not err["retryable"]

    # 5. Successfully select target
    sel_resp = client.post(
        f"/v1/analyses/{analysis_id}/target",
        json={"candidate_id": "person_01", "expected_epoch": 1},
    )
    assert sel_resp.status_code == 200
    assert sel_resp.json()["target_person_id"] == "person_01"
    assert sel_resp.json()["state"] == "TRACKING"

    # 6. Append observation and query observations
    now = datetime.now(timezone.utc)
    obs = Observation(
        session_id=analysis.session_id,
        epoch=1,
        sequence=1,
        timestamp_utc=now,
        target_person_id="person_01",
        exercise_key="bodyweight_squat",
        exercise_version=1,
        tracking_state=TrackingState.CONFIRMED,
        visibility_state=VisibilityState.FULL,
        reason_code=ReasonCode.OK,
        repetitions=[
            RepetitionEvent(
                repetition_index=1,
                start_timestamp=now,
                end_timestamp=now,
                confidence=0.95,
                quality_score=0.92,
                form_flags=[],
            )
        ],
    )
    repo.append_observation(analysis_id, obs)

    obs_resp = client.get(f"/v1/analyses/{analysis_id}/observations")
    assert obs_resp.status_code == 200
    page = obs_resp.json()
    assert len(page["observations"]) == 1
    obs_dict = page["observations"][0]

    # Validate output conforms to canonical JSON Schema
    validator = Draft202012Validator(observation_schema)
    errors = list(validator.iter_errors(obs_dict))
    assert not errors, f"Observation failed JSON Schema validation: {errors}"

    # 7. Stop analysis
    del_resp = client.delete(f"/v1/analyses/{analysis_id}")
    assert del_resp.status_code == 204
