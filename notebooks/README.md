# Notebook Workflow

The [architecture document](../docs/architecture.md) defines stages 00–06. 00_experiment_readiness.ipynb is the initial executable notebook. It checks required metadata and participant split isolation using explicitly synthetic examples; it contains no model results. Subsequent stages are created when their real dataset and source modules exist.

Run from a fresh kernel. Keep reusable implementation in src/kinetiq_v_vision; notebooks import it. Configure data paths and MLflow through environment/configuration, never embedded credentials. Do not publish private media or identifying outputs. Archive authorized executed notebooks/reports with their run ID; clear outputs in source control.

Expected artifacts by stage: manifest audit; dataset coverage report; baseline comparison; target-switch/occlusion analysis; temporal error report; frozen held-out evaluation; release decision with immutable artifact references. No stage may substitute mock metrics for missing measurements.
