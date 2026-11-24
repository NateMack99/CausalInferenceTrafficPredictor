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

On real data, build the road graph first, then run the pipeline:

```bash
uv run python scripts/build_graph.py --no-fetch \
    --crash-window 2025-03-01 2025-05-01 --speed-window 2025-03-01 2025-05-01
uv run python scripts/experiment.py --speeds data/parquet/chicago-speeds-history \
    --graph outputs/<stem>_segments.graphml --crashes outputs/<stem>_crash_segments.csv \
    --window 2025-03-01 2025-05-01 --static-features length_m degree betweenness
```

Point `--data` at `data/processed` once the collection and graph steps have run. `--model` takes `persistence`, `historical_mean`, `mlp`, `gcn`, `sage`, `tgru` or `stgcn`; see `--help` for the window, model and optimisation flags. Read the graph effect off `gcn` against `mlp`, and `stgcn` against `tgru`: each pair differs only in having edges, at matched parameter counts. Each run records its arguments and metrics under `outputs/`. `scripts/experiment.py` runs the whole pipeline: it joins the speed, crash and road-graph outputs into those tables, trains every model on identical splits, estimates crash effects against matched accident-free controls, and writes the figures and `results.csv` under `--out` (default `outputs/run`). `outputs/midpoint/` holds the frozen March-April pilot and is never written to again; `docs/hpc-run.md` covers the final cluster runs.

Two further entry points need no real data to be useful:

```bash
uv run python scripts/benchmark.py --data data/mock --out outputs/benchmark
```

`benchmark.py` injects disruptions of known size into held-out crash-free windows and
scores how well each forecaster recovers them, by hop ring and horizon step, reporting
forecast error separately. Recovering a known effect is the evidence the estimator works;
a low forecast MAE is not.

## Layout

- `src/traffic_gml/` — reusable data, graph, model, and evaluation code
- `scripts/` — command-line entry points for collection, the road graph, the midpoint run, and the semi-synthetic known-effect benchmark (`benchmark.py`)
- `tests/` — small, offline tests
- `docs/` — architecture, data contracts, and methodology
- `data/`, `outputs/` — local datasets and generated results; not committed
- `PROJECT_ROADMAP.md` — sprint and PR plan

## Documentation

- [Architecture and interfaces](docs/architecture.md)
- [Data sources and schemas](docs/data.md)
- [Research methodology](docs/methodology.md)
- [Experiment settings, exclusion rules, frozen baseline](docs/experiments.md)
- [HPC run handoff](docs/hpc-run.md)
- [Guidelines for coding agents](AGENTS.md)

## Research and paper

The research rules that constrain the code — no post-event inputs, accident-free training windows, forecast accuracy reported separately from causal effects — are in [AGENTS.md](AGENTS.md).

Overleaf is the source of truth for the paper; see [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md).
