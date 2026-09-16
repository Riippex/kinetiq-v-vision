import copy
import json
from pathlib import Path

import pytest

from kinetiq_v_vision.evaluation.audit import (
    audit_annotations,
    audit_condition_taxonomy,
    audit_exercise_coverage,
    audit_splits_and_participants,
    generate_dataset_audit_report,
    validate_manifest_schema,
)

FIXTURES_DATA_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "data"
MANIFEST_PATH = FIXTURES_DATA_DIR / "synthetic_manifest.json"


@pytest.fixture
def synthetic_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def synthetic_annotations(synthetic_manifest: dict) -> dict[str, dict]:
    annotations = {}
    for clip in synthetic_manifest["clips"]:
        ref = clip["annotation_ref"]
        ann_path = FIXTURES_DATA_DIR / ref
        assert ann_path.exists(), f"Missing annotation at {ann_path}"
        with open(ann_path, encoding="utf-8") as f:
            annotations[clip["clip_id"]] = json.load(f)
    return annotations


def test_manifest_schema_validation_passes(synthetic_manifest: dict) -> None:
    errors = validate_manifest_schema(synthetic_manifest)
    assert not errors, f"Manifest failed validation: {errors}"


def test_manifest_schema_validation_catches_invalid_hash(
    synthetic_manifest: dict,
) -> None:
    bad_manifest = copy.deepcopy(synthetic_manifest)
    bad_manifest["clips"][0]["sha256"] = "invalid_hash_not_64_chars"
    errors = validate_manifest_schema(bad_manifest)
    assert errors, "Expected validation error for invalid sha256"
    assert any("sha256" in e for e in errors)


def test_split_audit_detects_isolation(synthetic_manifest: dict) -> None:
    audit = audit_splits_and_participants(synthetic_manifest)
    assert audit["is_isolated"] is True
    assert len(audit["leakage_participants"]) == 0
    assert audit["total_participants"] == 3
    assert audit["clips_by_split"]["development"] == 4
    assert audit["clips_by_split"]["heldout"] == 2


def test_split_audit_catches_leakage(synthetic_manifest: dict) -> None:
    leaking_manifest = copy.deepcopy(synthetic_manifest)
    # Give participant synthetic-user-01 a clip in heldout as well
    leaking_manifest["clips"].append(
        {
            **leaking_manifest["clips"][0],
            "clip_id": "fixture_leaking_clip",
            "split": "heldout",
        }
    )
    audit = audit_splits_and_participants(leaking_manifest)
    assert audit["is_isolated"] is False
    assert "synthetic-user-01" in audit["leakage_participants"]
    assert audit["leakage_participants"]["synthetic-user-01"] == {
        "development",
        "heldout",
    }


def test_exercise_coverage_audit(synthetic_manifest: dict) -> None:
    coverage = audit_exercise_coverage(synthetic_manifest)
    assert coverage["has_full_coverage"] is True
    assert len(coverage["missing_canonical_exercises"]) == 0
    matrix = coverage["matrix"]
    assert matrix["bodyweight_squat"]["total"] >= 1
    assert matrix["push_up"]["total"] >= 1
    assert matrix["plank"]["total"] >= 1
    assert matrix["glute_bridge"]["total"] >= 1


def test_condition_taxonomy_audit(synthetic_manifest: dict) -> None:
    taxonomy = audit_condition_taxonomy(synthetic_manifest)
    assert "clean_framing" in taxonomy["condition_counts"]
    assert "bystander_crossing" in taxonomy["condition_counts"]
    assert "pet_crossing" in taxonomy["condition_counts"]
    assert taxonomy["total_tagged_conditions"] >= 5


def test_annotations_audit_passes(
    synthetic_manifest: dict, synthetic_annotations: dict
) -> None:
    res = audit_annotations(synthetic_manifest, synthetic_annotations)
    assert res["is_valid"] is True
    assert res["valid_count"] == len(synthetic_manifest["clips"])
    assert len(res["missing_annotations"]) == 0
    assert len(res["timing_violations"]) == 0


def test_annotations_audit_catches_timing_violations(
    synthetic_manifest: dict, synthetic_annotations: dict
) -> None:
    bad_annotations = copy.deepcopy(synthetic_annotations)
    # Invert repetition timestamps (start > inflection)
    bad_annotations["fixture_squat_01"]["repetitions"][0]["start_time"] = 3.0
    bad_annotations["fixture_squat_01"]["repetitions"][0]["inflection_time"] = 2.0

    res = audit_annotations(synthetic_manifest, bad_annotations)
    assert res["is_valid"] is False
    assert any("timestamp ordering" in v for v in res["timing_violations"])


def test_generate_dataset_audit_report() -> None:
    report = generate_dataset_audit_report(MANIFEST_PATH)
    assert report.is_valid is True
    summary = report.summary_markdown()
    assert "Dataset Audit Report" in summary
    assert "Split Isolation" in summary
    assert "Exercise Coverage" in summary
    assert "Condition Taxonomy Coverage" in summary
    assert "Annotation Integrity" in summary
