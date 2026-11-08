# Chicago Traffic Accident Spillovers

Estimate how crashes change traffic speeds on affected and neighboring roads over time. We compare non-graph models with spatiotemporal GNNs that forecast **no-accident** speeds, then evaluate estimated effects using matched controls and semi-synthetic ground truth.

## Quickstart

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
bash scripts/setup.sh
```

Setup creates `.venv` with uv if needed, syncs the locked Python environment, verifies your Chicago portal token,
and optionally downloads and prepares the data.

Modeling runs without real data, on mock tables written in the processed schema:

```bash
uv run python -m traffic_gml.mock --out data/mock
uv run python -m traffic_gml.train --data data/mock --model gcn --seed 42
```

Point `--data` at `data/processed` once the collection and graph steps have run. `--model` takes `persistence`, `historical_mean`, `mlp`, `gcn` or `sage`; see `--help` for the window, model and optimisation flags. Each run records its arguments and metrics under `outputs/`. `scripts/midpoint.py` runs the whole pipeline: it joins the speed, crash and road-graph outputs into those tables, trains every model on identical splits, estimates crash effects against matched accident-free controls, and writes the figures and `results.csv` under `outputs/midpoint/`.

## Layout

- `src/traffic_gml/` — reusable data, graph, model, and evaluation code
- `scripts/` — command-line entry points for collection, the road graph, and the midpoint run
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

The research rules that constrain the code — no post-event inputs, accident-free training windows, forecast accuracy reported separately from causal effects — are in [AGENTS.md](AGENTS.md).

Overleaf is the source of truth for the paper; see [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md).
