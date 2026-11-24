# Coding Agent Guidelines

Read `README.md`, `PROJECT_ROADMAP.md`, and the relevant file in `docs/` before changing code.

## Code
- Use Python 3.11–3.12, `uv`, type hints, and small functions.
- Put reusable logic in `src/traffic_gml/`; keep `scripts/` thin.
- Prefer `argparse`, existing libraries, and simple data structures.
- Do not add frameworks, dependencies, or abstraction layers without a concrete need.
- Reuse the schemas in `docs/architecture.md` and `docs/data.md`.

## Style
Human readability is the primary constraint. Be aggressively minimalist.
- Write the minimum code the requirement needs. Prefer deletion over addition.
- Functional core, imperative shell: keep logic pure, push I/O to the edges.
- No speculative features, unrequested flexibility, or defensive checks for
  conditions that cannot occur. Trust internal contracts, validate boundaries.
- Comment only what the code cannot express.
- Write the fewest tests that give confidence. Avoid redundant coverage.
- Do not cut lines with clever syntax or cryptic names.
- Treat every added line as a maintenance liability. Code must stay small
  enough for a human to review comfortably without AI.

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
