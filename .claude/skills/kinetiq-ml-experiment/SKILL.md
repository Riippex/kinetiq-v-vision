---
name: kinetiq-ml-experiment
description: Develop Kinetiq Vision EDA, feature engineering, model experiments, evaluation notebooks or model-release evidence.
---

# kinetiq-ml-experiment

Read docs/architecture.md, docs/model-selection.md, docs/session-target-tracking.md and evaluation/protocol.md as relevant. Separate exploratory data analysis from event-driven architecture. Keep participant/session splits fixed before tuning; avoid held-out leakage. Track code, data manifests, weights, preprocessing and temporal configuration together in managed MLflow. Notebooks call importable production functions and run from a clean kernel. Mark synthetic fixtures and never fabricate model results. Evaluate target switches, distractors, occlusion and latency alongside rep error. Preserve media permissions and weight licenses. Promote immutable evaluated versions, not an untracked alias.
