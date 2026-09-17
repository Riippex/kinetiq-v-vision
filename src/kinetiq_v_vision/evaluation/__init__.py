"""Evaluation and audit module for Kinetiq V Vision."""

from kinetiq_v_vision.evaluation.audit import (
    DatasetAuditReport,
    audit_annotations,
    audit_condition_taxonomy,
    audit_exercise_coverage,
    audit_splits_and_participants,
    generate_dataset_audit_report,
    resolve_annotation_references,
    validate_manifest_schema,
)
from kinetiq_v_vision.evaluation.baselines import (
    BenchmarkEnvironment,
    CandidateBenchmarkReport,
    CandidateMetrics,
    create_candidate_pipelines,
    run_pose_baseline_benchmark,
)
from kinetiq_v_vision.evaluation.tracking import (
    TrackingEvaluationMetrics,
    TrackingScenario,
    create_synthetic_tracking_scenarios,
    run_tracking_evaluation,
)

__all__ = [
    "BenchmarkEnvironment",
    "CandidateBenchmarkReport",
    "CandidateMetrics",
    "DatasetAuditReport",
    "TrackingEvaluationMetrics",
    "TrackingScenario",
    "audit_annotations",
    "audit_condition_taxonomy",
    "audit_exercise_coverage",
    "audit_splits_and_participants",
    "create_candidate_pipelines",
    "create_synthetic_tracking_scenarios",
    "generate_dataset_audit_report",
    "resolve_annotation_references",
    "run_pose_baseline_benchmark",
    "run_tracking_evaluation",
    "validate_manifest_schema",
]
