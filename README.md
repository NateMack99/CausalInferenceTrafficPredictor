# Chicago Traffic Accident Spillovers

Estimate how crashes change traffic speeds on affected and neighboring roads over time. We compare non-graph models with spatiotemporal GNNs that forecast **no-accident** speeds, then evaluate estimated effects using matched controls and semi-synthetic ground truth.

## Quickstart

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
bash scripts/setup.sh
```

Setup creates `.venv` with uv if needed, syncs the locked Python environment, verifies your Chicago portal token,
and optionally downloads and prepares the data.

Modeling (PR #3) runs without real data, on mock tables written in the processed schema:

```bash
uv run python scripts/make_mock_data.py --out data/mock
uv run python scripts/run_experiment.py --data data/mock --model gcn --seed 42
```

Point `--data` at `data/processed` once PR #1 and PR #2 land. `--model` takes `persistence`, `historical_mean`, `mlp`, `gcn` or `sage`; see `--help` for the window, model and optimisation flags. Each run records its arguments and metrics under `outputs/`. The dataset and causal-evaluation CLI commands will be added as their roadmap PRs are implemented.

## Layout

- `src/traffic_gml/` — reusable data, graph, model, and evaluation code
- `scripts/` — thin command-line entry points (added with features)
- `tests/` — small, offline tests
- `docs/` — architecture, data contracts, and methodology
- `data/`, `outputs/` — local datasets and generated results; not committed
- `PROJECT_ROADMAP.md` — sprint and PR plan

## Documentation

- [Architecture and interfaces](docs/architecture.md)
- [Data sources and schemas](docs/data.md)
- [Research methodology](docs/methodology.md)
- [Guidelines for coding agents](AGENTS.md)

## Research and paper

Keep train/test event windows separate, and never use post-accident measurements as inputs to a no-accident forecast. Evaluate speed forecasts separately from estimated causal effects; real-world effect estimates require explicit assumptions and robustness checks.

**Overleaf is the source of truth for the paper.** Any LaTeX in Git is a convenience copy.
