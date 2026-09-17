from kinetiq_v_vision.evaluation.tracking import (
    create_bystander_only_reassociation_scenario,
    create_synthetic_tracking_scenarios,
    run_tracking_evaluation,
)


def test_run_tracking_evaluation_synthetic_scenarios() -> None:
    """The full default scenario set now includes the adversarial
    bystander-only scenario, so this is release-INeligible by construction
    -- see test_bystander_only_scenario_exposes_known_reassociation_defect
    for why that is the correct, honest result rather than a regression."""
    scenarios = create_synthetic_tracking_scenarios()
    assert len(scenarios) == 4

    metrics = run_tracking_evaluation(scenarios)
    assert metrics.total_frames == 55
    assert metrics.target_tracked_frames > 0
    assert metrics.is_synthetic is True
    assert metrics.target_switch_count == 8
    assert metrics.is_release_eligible is False


def test_clean_scenarios_alone_remain_release_eligible() -> None:
    """The three original scenarios (Clean Tracking, Distractor Crossing,
    Exit and Re-entry) are unaffected by the attribution-tracking fix and
    remain switch-free on their own."""
    scenarios = create_synthetic_tracking_scenarios()[:3]

    metrics = run_tracking_evaluation(scenarios)
    assert metrics.total_frames == 45
    assert metrics.target_tracked_frames > 0
    assert metrics.target_switch_count == 0
    assert metrics.is_release_eligible is True


def test_bystander_only_scenario_exposes_known_reassociation_defect() -> None:
    """Regression/disclosure test, not a bug in the test itself: once a
    tracker has a spatial reference (`last_known_bbox`) and the true target
    permanently leaves, a same-position bystander with a *different*
    candidate_id reaches CONFIRMED via TargetTracker's spatial fallback
    (domain/target_tracker.py), which scores candidates by IoU/centroid
    distance only and never checks identity. This evaluation harness must
    detect that as a target switch, not hide it -- VV-402 stays Blocked
    until the tracker itself rejects/degrades a spatially-plausible but
    identity-mismatched candidate instead of silently confirming it."""
    scenario = create_bystander_only_reassociation_scenario()

    metrics = run_tracking_evaluation([scenario])
    assert metrics.total_frames == 10
    assert metrics.target_switch_count == 8
    assert metrics.is_release_eligible is False
