from kinetiq_v_vision.evaluation.tracking import (
    create_bystander_only_reassociation_scenario,
    create_synthetic_tracking_scenarios,
    run_tracking_evaluation,
)


def test_run_tracking_evaluation_synthetic_scenarios() -> None:
    """The full default scenario set, including the adversarial
    bystander-only scenario, is switch-free and release eligible now that
    TargetTracker's spatial fallback never silently confirms a
    non-ID-matching candidate (see
    test_bystander_only_scenario_no_longer_causes_spatial_reassociation)."""
    scenarios = create_synthetic_tracking_scenarios()
    assert len(scenarios) == 4

    metrics = run_tracking_evaluation(scenarios)
    assert metrics.total_frames == 55
    assert metrics.target_tracked_frames > 0
    assert metrics.is_synthetic is True
    assert metrics.target_switch_count == 0
    assert metrics.is_release_eligible is True


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


def test_bystander_only_scenario_no_longer_causes_spatial_reassociation() -> None:
    """VV-401 correction: once a tracker has a spatial reference
    (`last_known_bbox`) and the true target permanently leaves, a
    same-position bystander under a *different* candidate_id used to reach
    CONFIRMED via TargetTracker's spatial fallback (a real, reproduced
    defect -- see git history). The fallback now never promotes a
    non-ID-matching candidate past AMBIGUOUS, so this adversarial scenario
    is switch-free and release eligible: the bystander's frames are
    correctly reported as identity-unresolved (AMBIGUOUS), not silently
    attributed to the enrolled target."""
    scenario = create_bystander_only_reassociation_scenario()

    metrics = run_tracking_evaluation([scenario])
    assert metrics.total_frames == 10
    assert metrics.target_switch_count == 0
    assert metrics.is_release_eligible is True
