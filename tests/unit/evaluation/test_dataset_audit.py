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
    resolve_annotation_references,
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
    assert len(audit["leakage_sessions"]) == 0
    assert audit["total_participants"] == 4
    assert audit["total_sessions"] == 5
    assert audit["clips_by_split"]["development"] == 4
    assert audit["clips_by_split"]["heldout"] == 4


def test_split_audit_catches_participant_leakage(synthetic_manifest: dict) -> None:
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


def test_split_audit_catches_session_leakage_with_different_participant_ids(
    synthetic_manifest: dict,
) -> None:
    """A session must never appear in both splits, even when the clip's
    participant_id was (incorrectly) varied so participant-level isolation
    alone would miss the leak."""
    leaking_manifest = copy.deepcopy(synthetic_manifest)
    dev_clip = leaking_manifest["clips"][0]
    assert dev_clip["session_id"] == "session-synth-01"

    leaking_manifest["clips"].append(
        {
            **dev_clip,
            "clip_id": "fixture_session_leak_clip",
            "split": "heldout",
            # Different participant_id, same session_id as a development clip.
            "participant_id": "synthetic-user-99",
        }
    )

    audit = audit_splits_and_participants(leaking_manifest)
    assert audit["is_isolated"] is False
    assert "synthetic-user-99" not in audit["leakage_participants"]
    assert "synthetic-user-01" not in audit["leakage_participants"]
    assert audit["leakage_sessions"]["session-synth-01"] == {
        "development",
        "heldout",
    }


def test_exercise_coverage_audit(synthetic_manifest: dict) -> None:
    coverage = audit_exercise_coverage(synthetic_manifest)
    assert coverage["has_full_coverage"] is True
    assert len(coverage["missing_canonical_exercises"]) == 0
    assert coverage["missing_split_coverage"] == {}
    matrix = coverage["matrix"]
    for exercise in ("bodyweight_squat", "push_up", "plank", "glute_bridge"):
        assert matrix[exercise]["development"] >= 1, exercise
        assert matrix[exercise]["heldout"] >= 1, exercise


def test_exercise_coverage_audit_fails_when_a_split_lacks_an_exercise(
    synthetic_manifest: dict,
) -> None:
    """An exercise present only in development (never held-out, or vice
    versa) must fail coverage even though its total count is nonzero."""
    partial_manifest = copy.deepcopy(synthetic_manifest)
    partial_manifest["clips"] = [
        clip
        for clip in partial_manifest["clips"]
        if not (clip["exercise"] == "push_up" and clip["split"] == "heldout")
    ]

    coverage = audit_exercise_coverage(partial_manifest)
    assert coverage["has_full_coverage"] is False
    assert coverage["missing_split_coverage"]["push_up"] == ["heldout"]
    assert coverage["matrix"]["push_up"]["total"] >= 1


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
    assert report.reference_audit["is_valid"] is True
    assert report.reference_audit["reference_errors"] == []
    summary = report.summary_markdown()
    assert "Dataset Audit Report" in summary
    assert "Split Isolation" in summary
    assert "Exercise Coverage" in summary
    assert "Condition Taxonomy Coverage" in summary
    assert "Annotation Reference Integrity" in summary
    assert "Annotation Timing Integrity" in summary


def _minimal_clip(clip_id: str, annotation_ref: str, **overrides: object) -> dict:
    clip = {
        "clip_id": clip_id,
        "source_uri": f"s3://bucket/{clip_id}.mp4",
        "sha256": "0" * 64,
        "participant_id": "p1",
        "session_id": "s1",
        "split": "development",
        "exercise": "bodyweight_squat",
        "annotation_ref": annotation_ref,
        "annotation_version": "v1.0",
    }
    clip.update(overrides)
    return clip


def _write_annotation(path: Path, clip_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"clip_id": clip_id, "exercise": "bodyweight_squat"}),
        encoding="utf-8",
    )


def test_resolve_annotation_references_accepts_valid_reference(
    tmp_path: Path,
) -> None:
    _write_annotation(tmp_path / "annotations" / "clip_a.json", "clip_a")
    manifest = {
        "clips": [_minimal_clip("clip_a", "annotations/clip_a.json")]
    }

    result = resolve_annotation_references(manifest, tmp_path)

    assert result["is_valid"] is True
    assert result["reference_errors"] == []
    assert "clip_a" in result["resolved_annotations"]


def test_resolve_annotation_references_rejects_missing_reference(
    tmp_path: Path,
) -> None:
    """Regression test: a nonexistent annotation_ref must fail the audit."""
    manifest = {
        "clips": [_minimal_clip("clip_missing", "annotations/does_not_exist.json")]
    }

    result = resolve_annotation_references(manifest, tmp_path)

    assert result["is_valid"] is False
    assert "clip_missing" not in result["resolved_annotations"]
    assert any(
        "does not resolve to an existing annotation file" in e
        for e in result["reference_errors"]
    )


def test_resolve_annotation_references_rejects_mismatched_clip_id(
    tmp_path: Path,
) -> None:
    """Regression test: an annotation_ref that resolves to a file declaring a
    different clip_id than the referencing clip must fail the audit."""
    _write_annotation(tmp_path / "annotations" / "wrong.json", "some_other_clip")
    manifest = {
        "clips": [_minimal_clip("clip_incorrect_ref", "annotations/wrong.json")]
    }

    result = resolve_annotation_references(manifest, tmp_path)

    assert result["is_valid"] is False
    assert "clip_incorrect_ref" not in result["resolved_annotations"]
    assert any(
        "declaring clip_id 'some_other_clip'" in e for e in result["reference_errors"]
    )


def test_resolve_annotation_references_does_not_substitute_directory_scan_match(
    tmp_path: Path,
) -> None:
    """Reproduces the original bug: a manifest clip whose annotation_ref is
    missing or wrong must fail even when some other JSON file in the
    annotations directory happens to declare the expected clip_id."""
    # A decoy file that a naive directory-scan-by-clip_id approach would
    # incorrectly accept as clip_id's annotation.
    _write_annotation(tmp_path / "annotations" / "decoy.json", "clip_target")
    manifest = {
        "clips": [
            _minimal_clip("clip_target", "annotations/does_not_exist.json")
        ]
    }

    result = resolve_annotation_references(manifest, tmp_path)

    assert result["is_valid"] is False
    assert "clip_target" not in result["resolved_annotations"]


def test_resolve_annotation_references_rejects_path_outside_dataset_root(
    tmp_path: Path,
) -> None:
    outside_dir = tmp_path.parent / "outside_dataset_root"
    _write_annotation(outside_dir / "secret.json", "clip_escape")
    manifest = {
        "clips": [
            _minimal_clip(
                "clip_escape",
                f"../{outside_dir.name}/secret.json",
            )
        ]
    }

    result = resolve_annotation_references(manifest, tmp_path)

    assert result["is_valid"] is False
    assert "clip_escape" not in result["resolved_annotations"]
    assert any(
        "resolves outside the permitted dataset root" in e
        for e in result["reference_errors"]
    )


def test_resolve_annotation_references_rejects_duplicate_clip_ids(
    tmp_path: Path,
) -> None:
    _write_annotation(tmp_path / "annotations" / "clip_dup.json", "clip_dup")
    manifest = {
        "clips": [
            _minimal_clip("clip_dup", "annotations/clip_dup.json"),
            _minimal_clip("clip_dup", "annotations/clip_dup.json"),
        ]
    }

    result = resolve_annotation_references(manifest, tmp_path)

    assert result["is_valid"] is False
    assert result["duplicate_clip_ids"] == ["clip_dup"]
