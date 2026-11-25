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

## What to run

| Entry point | What it does | Real data |
| --- | --- | --- |
| `scripts/experiment.py` | The whole pipeline: joins the speed, crash and graph outputs into the contract tables, trains every model on identical splits, estimates effects against matched accident-free controls | yes |
| `scripts/event_study.py` | Matched difference-in-differences on observed speeds, with pre-trend placebo and covariate balance. Uses no forecaster, so it cannot leak | yes |
| `scripts/benchmark.py` | Injects disruptions of known size into held-out crash-free windows and scores how well each forecaster recovers them, by hop ring and horizon step | no, runs on mock |
| `scripts/aggregate.py` | Collects a finished run into the paper tables, the ablation figure and `result_index.csv` | no |
| `scripts/hpc/submit.sh` | Queues the whole Run A matrix on Slurm with dependencies ([runbook](docs/hpc-run.md)) | on the cluster |

```bash
uv run python scripts/benchmark.py --data data/mock --out outputs/benchmark
```

Recovering a known effect is the evidence the estimator works; a low forecast MAE
is not.

Point `--data` at `data/processed` once the collection and graph steps have run.
`--model` takes `persistence`, `historical_mean`, `mlp`, `gcn`, `sage`, `tgru` or
`stgcn`. Read the graph effect off `gcn` against `mlp`, and `stgcn` against
`tgru`: each pair differs only in having edges, at matched parameter counts.

The settings every reported run uses are in
[docs/experiments.md](docs/experiments.md). Argparse defaults match them, so a
bare local run is the paper's configuration.

## Where results live

Every entry point writes `summary.json` under `--out` with its arguments,
metrics, `code_version` and `data_version`. `aggregate.py` then writes
`result_index.csv`, mapping each manuscript label (`tab:forecast`,
`fig:eventcurves`) to the file holding it and whether it is ready.
`outputs/midpoint/` is the frozen March-April pilot and is never written to
again.

## Layout

- `src/traffic_gml/` — the library: data, graph, event windows, models, training, causal estimation, evaluation
- `scripts/` — the entry points above, plus `setup.sh` and `fetch_*.sh` for collection; `scripts/hpc/` is the cluster layer over the same ones
- `tests/` — offline tests, mostly scientific-validity checks
- `docs/` — data contracts, experiment settings, methodology, Rivanna runbook
- `Proposal/` — LaTeX working copy: the manuscript (`main.tex`), the midpoint report, the original proposal. Overleaf is the source of truth
- `data/`, `outputs/` — local datasets and generated results; not committed
- `PROJECT_ROADMAP.md` — workstreams and acceptance criteria

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
