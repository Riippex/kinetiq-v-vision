# Kinetiq V Vision

A vision engine for session-specific person tracking and exercise movement analysis with OpenCV 5 and AWS.

**Status:** architecture, initial REST/event contracts and an executable synthetic readiness notebook. No model benchmark or deployed inference service is claimed.

This repository owns pose inference, target tracking, temporal analysis and model evaluation. The [Kinetiq V product](https://github.com/Riippex/kinetiq-v) owns routines, persistent user/session state, clients and Alexa+ MCP.

- [Technology stack](docs/technology-stack.md)
- [Architecture and ML lifecycle](docs/architecture.md)
- [Model selection and licensing](docs/model-selection.md)
- [Target tracking](docs/session-target-tracking.md)
- [REST contract](contracts/rest-api.md)
- [Observation schema](contracts/observation.schema.json)
- [Evaluation protocol](evaluation/protocol.md)
- [Notebooks](notebooks/README.md)
- [Contributor workflow](docs/runbooks/pull-requests.md)
- [Agent skills and graph tools](docs/runbooks/agent-skills.md)

Public documentation lives in docs/. Local private planning belongs in ignored documents/. Raw evaluation data, weights and private notebook outputs are not published by default.

License: [Apache-2.0](LICENSE). Third-party models and datasets retain their own terms.
