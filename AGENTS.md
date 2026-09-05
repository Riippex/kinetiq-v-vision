# Agent Workflow

- Repository text, code comments and artifacts are English; conversation may be Spanish.
- Vision owns pose inference, target tracking, temporal analysis, the REST contract, evaluation and model lineage. Product profiles, routines, persistent sessions and Alexa+ belong to kinetiq-v.
- Read [architecture](docs/architecture.md), [stack](docs/technology-stack.md), and relevant executable contracts before boundary changes.
- Follow the [delivery workflow](docs/runbooks/pull-requests.md): direct validated commits to develop are allowed during active construction, while scoped branches and pull requests remain optional. Never update or promote main without explicit authorization.
- Public material belongs in docs/. Private notes belong in ignored documents/. Read AGENTS.local.md when present for local context; never publish it.
- Use [agent skill routing](docs/runbooks/agent-skills.md) when the task matches a skill. Local graph indexes are not authoritative or publishable.
- Preserve unrelated changes and distinguish planned, implemented and verified behavior. Do not spawn subagents unless requested.
