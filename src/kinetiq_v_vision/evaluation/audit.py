from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

CANONICAL_EXERCISES = {
    "bodyweight_squat",
    "push_up",
    "plank",
    "glute_bridge",
}

DEFAULT_MANIFEST_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "data"
    / "schema"
    / "dataset-manifest.schema.json"
)
DEFAULT_ANNOTATION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "data"
    / "schema"
    / "annotation.schema.json"
)


def _load_schema(schema_path: Path) -> dict[str, Any]:
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def validate_manifest_schema(
    manifest_data: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> list[str]:
    """Validate manifest data against Draft 2020-12 schema."""
    if schema is None:
        schema = _load_schema(DEFAULT_MANIFEST_SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(manifest_data), key=lambda e: str(e.path))
    return [f"path {list(e.path)}: {e.message}" for e in errors]


def audit_splits_and_participants(manifest_data: dict[str, Any]) -> dict[str, Any]:
    """Audit participant distribution and strictly check for split leakage."""
    clips = manifest_data.get("clips", [])
    participant_splits: dict[str, set[str]] = {}
    clips_by_split: dict[str, int] = {"development": 0, "heldout": 0}
    participants_by_split: dict[str, set[str]] = {"development": set(), "heldout": set()}

    for clip in clips:
        p_id = clip.get("participant_id")
        split = clip.get("split")
        if p_id and split:
            participant_splits.setdefault(p_id, set()).add(split)
            if split in clips_by_split:
                clips_by_split[split] += 1
                participants_by_split[split].add(p_id)

    leakage_participants = {
        p: splits for p, splits in participant_splits.items() if len(splits) > 1
    }

    return {
        "is_isolated": len(leakage_participants) == 0,
        "total_participants": len(participant_splits),
        "total_clips": len(clips),
        "clips_by_split": clips_by_split,
        "participants_by_split": {
            k: sorted(list(v)) for k, v in participants_by_split.items()
        },
        "leakage_participants": leakage_participants,
    }


def audit_exercise_coverage(manifest_data: dict[str, Any]) -> dict[str, Any]:
    """Audit exercise representation across development and held-out splits."""
    clips = manifest_data.get("clips", [])
    coverage_matrix: dict[str, dict[str, int]] = {
        ex: {"development": 0, "heldout": 0, "total": 0}
        for ex in sorted(CANONICAL_EXERCISES)
    }

    unexpected_exercises: set[str] = set()

    for clip in clips:
        ex = clip.get("exercise")
        split = clip.get("split")
        if ex in coverage_matrix:
            if split in ("development", "heldout"):
                coverage_matrix[ex][split] += 1
                coverage_matrix[ex]["total"] += 1
        elif ex:
            unexpected_exercises.add(ex)

    missing_exercises = {
        ex for ex, counts in coverage_matrix.items() if counts["total"] == 0
    }

    return {
        "matrix": coverage_matrix,
        "missing_canonical_exercises": sorted(list(missing_exercises)),
        "unexpected_exercises": sorted(list(unexpected_exercises)),
        "has_full_coverage": len(missing_exercises) == 0,
    }


def audit_condition_taxonomy(manifest_data: dict[str, Any]) -> dict[str, Any]:
    """Audit representation of condition tags across the dataset."""
    clips = manifest_data.get("clips", [])
    condition_counts: dict[str, int] = {}
    condition_by_exercise: dict[str, dict[str, int]] = {}

    for clip in clips:
        ex = clip.get("exercise", "unknown")
        for cond in clip.get("conditions", []):
            condition_counts[cond] = condition_counts.get(cond, 0) + 1
            if cond not in condition_by_exercise:
                condition_by_exercise[cond] = {}
            condition_by_exercise[cond][ex] = condition_by_exercise[cond].get(ex, 0) + 1

    return {
        "condition_counts": condition_counts,
        "condition_by_exercise": condition_by_exercise,
        "total_tagged_conditions": sum(condition_counts.values()),
    }


def audit_annotations(
    manifest_data: dict[str, Any],
    annotations: dict[str, dict[str, Any]],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate annotation files against schema and semantic timing invariants."""
    if schema is None:
        schema = _load_schema(DEFAULT_ANNOTATION_SCHEMA_PATH)

    validator = Draft202012Validator(schema)
    clips = manifest_data.get("clips", [])

    missing_annotations: list[str] = []
    schema_errors: dict[str, list[str]] = {}
    timing_violations: list[str] = []
    valid_count = 0

    for clip in clips:
        clip_id = clip.get("clip_id")
        duration = clip.get("duration_seconds", 0.0)

        if clip_id not in annotations:
            missing_annotations.append(clip_id)
            continue

        ann = annotations[clip_id]
        errors = sorted(validator.iter_errors(ann), key=lambda e: str(e.path))
        if errors:
            schema_errors[clip_id] = [e.message for e in errors]
            continue

        # Check timing invariants
        clip_timing_valid = True

        # Repetition timing
        reps = ann.get("repetitions", [])
        last_end = 0.0
        for r in reps:
            start = r.get("start_time", 0.0)
            inflection = r.get("inflection_time", 0.0)
            end = r.get("end_time", 0.0)

            if not (start < inflection < end):
                timing_violations.append(
                    f"{clip_id}: Rep {r.get('repetition_index')} has invalid timestamp ordering "
                    f"({start} < {inflection} < {end} failed)"
                )
                clip_timing_valid = False

            if start < last_end:
                timing_violations.append(
                    f"{clip_id}: Rep {r.get('repetition_index')} overlaps with prior repetition"
                )
                clip_timing_valid = False
            last_end = end

            if end > duration:
                timing_violations.append(
                    f"{clip_id}: Rep {r.get('repetition_index')} end ({end}s) exceeds clip duration ({duration}s)"
                )
                clip_timing_valid = False

        # Hold timing
        hold = ann.get("hold")
        if hold:
            start = hold.get("start_time", 0.0)
            end = hold.get("end_time", 0.0)
            dur = hold.get("duration_seconds", 0.0)
            if start >= end or end > duration:
                timing_violations.append(
                    f"{clip_id}: Hold interval [{start}, {end}] is invalid for duration {duration}s"
                )
                clip_timing_valid = False
            if round(end - start, 2) < round(dur, 2):
                timing_violations.append(
                    f"{clip_id}: Hold duration {dur}s exceeds elapsed interval {end - start}s"
                )
                clip_timing_valid = False

        if clip_timing_valid:
            valid_count += 1

    is_valid = (
        len(missing_annotations) == 0
        and len(schema_errors) == 0
        and len(timing_violations) == 0
    )

    return {
        "is_valid": is_valid,
        "total_clips": len(clips),
        "valid_count": valid_count,
        "missing_annotations": missing_annotations,
        "schema_errors": schema_errors,
        "timing_violations": timing_violations,
    }


@dataclass
class DatasetAuditReport:
    manifest_errors: list[str] = field(default_factory=list)
    split_audit: dict[str, Any] = field(default_factory=dict)
    exercise_audit: dict[str, Any] = field(default_factory=dict)
    taxonomy_audit: dict[str, Any] = field(default_factory=dict)
    annotation_audit: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return (
            len(self.manifest_errors) == 0
            and self.split_audit.get("is_isolated", False)
            and self.exercise_audit.get("has_full_coverage", False)
            and self.annotation_audit.get("is_valid", False)
        )

    def summary_markdown(self) -> str:
        lines = [
            "# Dataset Audit Report",
            f"**Overall Audit Valid:** {self.is_valid}",
            "",
            "## 1. Split Isolation",
            f"- Isolated (No Leakage): {self.split_audit.get('is_isolated')}",
            f"- Total Participants: {self.split_audit.get('total_participants')}",
            f"- Clips: dev={self.split_audit.get('clips_by_split', {}).get('development', 0)}, "
            f"heldout={self.split_audit.get('clips_by_split', {}).get('heldout', 0)}",
        ]
        if self.split_audit.get("leakage_participants"):
            lines.append(f"- **LEAKAGE DETECTED:** {self.split_audit['leakage_participants']}")

        lines.extend([
            "",
            "## 2. Exercise Coverage",
            "| Exercise | Development | Heldout | Total |",
            "|---|---|---|---|",
        ])
        matrix = self.exercise_audit.get("matrix", {})
        for ex, counts in matrix.items():
            lines.append(f"| `{ex}` | {counts['development']} | {counts['heldout']} | {counts['total']} |")

        lines.extend([
            "",
            "## 3. Condition Taxonomy Coverage",
        ])
        for cond, count in sorted(self.taxonomy_audit.get("condition_counts", {}).items()):
            lines.append(f"- `{cond}`: {count} clips")

        lines.extend([
            "",
            "## 4. Annotation Integrity",
            f"- Valid Annotations: {self.annotation_audit.get('valid_count')} / {self.annotation_audit.get('total_clips')}",
            f"- Missing Annotations: {len(self.annotation_audit.get('missing_annotations', []))}",
            f"- Schema Errors: {len(self.annotation_audit.get('schema_errors', {}))}",
            f"- Timing Violations: {len(self.annotation_audit.get('timing_violations', []))}",
        ])

        return "\n".join(lines)


def generate_dataset_audit_report(
    manifest_path: str | Path,
    annotations_dir: str | Path | None = None,
) -> DatasetAuditReport:
    manifest_p = Path(manifest_path)
    with open(manifest_p, encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_errors = validate_manifest_schema(manifest_data)
    split_audit = audit_splits_and_participants(manifest_data)
    exercise_audit = audit_exercise_coverage(manifest_data)
    taxonomy_audit = audit_condition_taxonomy(manifest_data)

    annotations: dict[str, dict[str, Any]] = {}
    if annotations_dir is None:
        annotations_dir = manifest_p.parent / "annotations"

    ann_p = Path(annotations_dir)
    if ann_p.exists():
        for fpath in ann_p.glob("*.json"):
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
                clip_id = data.get("clip_id")
                if clip_id:
                    annotations[clip_id] = data

    annotation_audit = audit_annotations(manifest_data, annotations)

    return DatasetAuditReport(
        manifest_errors=manifest_errors,
        split_audit=split_audit,
        exercise_audit=exercise_audit,
        taxonomy_audit=taxonomy_audit,
        annotation_audit=annotation_audit,
    )
