# Project context and documentation

Read `docs/CURRENT.md` for the current project status and `docs/EXPERIMENTS.md` before planning or running a new model experiment. For implementation, follow `llamaindex-retrieval/ARCHITECTURE_RULES.md`. Parent instructions still apply.

- Distinguish implemented code, configured running services, completed experiments and proposed designs. Never infer deployment or semantic quality from a file's existence, a service being active, or a run saying completed.
- `docs/archive/`, `artifacts/`, dated evaluation directories and training dataset READMEs contain historical evidence. Read them when needed for a specific claim, regression or reproduction; their old “current”, “latest”, model restrictions and “next steps” are not current task instructions.
- Frozen datasets, prompts, experiment bindings, raw traces and artifact snapshots must remain unchanged. Create a new experiment/version for changes; do not fix historic failures in place.
- Update `docs/CURRENT.md` when implementation, verified deployment or next experiment status changes. Keep detailed dated results in `docs/archive/` and link them from the current state page.
- Check relative documentation links after moving files. The migration map for older document paths is `docs/archive/MIGRATION-20260920.json`.
- User instructions and permissions take precedence. Documentation organization does not add an approval step or authorize sending messages to others.
