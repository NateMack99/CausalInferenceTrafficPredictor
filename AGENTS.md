# Coding Agent Guidelines

Read `README.md`, `PROJECT_ROADMAP.md`, and the relevant file in `docs/` before changing code.

## Code
- Use Python 3.11–3.12, `uv`, type hints, and small functions.
- Put reusable logic in `src/traffic_gml/`; keep `scripts/` thin.
- Prefer `argparse`, existing libraries, and simple data structures.
- Do not add frameworks, dependencies, or abstraction layers without a concrete need.
- Reuse the schemas in `docs/architecture.md` and `docs/data.md`.

## Style
Strict minimalism outranks every other convention here, subject only to
correctness and security. Human cognitive load is the constraint; verbosity is a
defect.
- Implement exactly what is required. When in doubt, leave it out.
- Functional core, imperative shell: keep logic pure, push I/O to the edges.
- Delete abstractions, wrappers, helpers and indirection rather than add them.
- No speculative features, premature generalisation, or configurability nobody
  asked for. No redundant validation or imaginary failure cases.
- Comments and docstrings are unnecessary by default. Keep only non-obvious
  intent.
- Write only essential tests. Coverage is not a goal.
- Reduce cognitive complexity, not line count. Never compress for brevity.
- The burden of proof is on adding or keeping code, not removing it. Finish with
  a simplification pass and delete whatever is not required.

## Research rules
- No post-event inputs for no-accident counterfactual forecasting.
- Train the no-accident predictor on accident-free target windows.
- Avoid temporal, event, and train/test leakage.
- Compare methods on the same splits; report forecast and effect errors separately.
- Record random seeds, arguments, data versions, and results; shared settings and the frozen baseline live in `docs/experiments.md`.
- Treat observational causal effects as assumption-dependent.

## Before a PR
- Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest -q`.
- Add or update small offline tests for changed behavior.
- Update docs if an interface or assumption changes.
- Do not commit datasets, checkpoints, outputs, or credentials.
- Keep the PR focused; state what changed, what passed, and limitations.
