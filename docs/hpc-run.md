# HPC run handoff

Everything needed to run the expanded-window experiments on Rivanna and send the
results back. Written for whoever operates the cluster; no knowledge of the
modelling code is assumed. Commands are copy-paste ready.

The frozen March–April pilot lives in `outputs/midpoint/` and `data/processed/`.
**Nothing here writes to either.** If a command ever asks to, stop.

## 1. Code

```bash
git checkout main
git pull
git rev-parse --short HEAD    # record this; it goes in the results you send back
```

Run from a commit on `main`. Do not run from a feature branch: `main` is the only
branch with the input-masking fix, the spatiotemporal models and the benchmark
all present together.

## 2. Environment and CUDA

```bash
uv venv --python 3.12 .venv
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The lockfile resolves CUDA wheels on Linux automatically; no extra index is
needed. If `cuda.is_available()` is `False`, stop and report it rather than
running on CPU: a full run is roughly nine hours of CPU time.

Then confirm the code itself works before queueing anything long:

```bash
uv run ruff check . && uv run pytest -q          # expect 68 passed
uv run python -m traffic_gml.mock --out data/mock
uv run python -m traffic_gml.train --data data/mock --model stgcn --epochs 2 --device auto
```

The last command is the GPU smoke test. Record the device line it prints.

## 3. Data to copy

These are gitignored, so they must be copied from Kyle's machine into the same
relative paths:

| Path | Size | Needed for |
| --- | ---: | --- |
| `data/parquet/chicago-speeds/` | 91 MB | everything (338 `date=` partitions, 2025-01-01 to 2025-12-06) |
| `data/parquet/chicago-crashes/` | included above | the graph build |
| `data/chicago_road_network.graphml` | 41 MB | the road graph |
| `data/chicago-traffic-segments.json` | 372 KB | segment geometry |
| `data/speed_roads.parquet` | 184 KB | all three are needed together, or `prepare_roads` |
| `data/speed_segments.parquet` | 32 KB | reruns and overwrites them (harmless, just slow) |
| `data/speed_road_network.graphml` | 4.5 MB | |
| `data/dense_80pct_segments.csv` | 8 KB | only for `--only-80-percent`, which we do not use |

Check after copying:

```bash
ls data/parquet/chicago-speeds | grep -c '^date='     # expect 338
```

## 4. Build the graph for the expanded window

```bash
uv run python scripts/build_graph.py --no-fetch \
    --crash-window 2025-01-01 2025-12-01 \
    --speed-window 2025-01-01 2025-12-01
```

The window stops at December 1 because the feed only reaches December 6, and a
partial month distorts the monthly coverage table.

This writes six files into `outputs/`, all sharing one stem
`graph_20250101T000000_20251201T000000_all_<UTC timestamp>Z`:

- `<stem>.graphml` — the street graph
- `<stem>_segments.graphml` — the segment graph the models use
- `<stem>_crash_segments.csv` — crashes matched to segments
- `<stem>_segment_metrics.csv`, `<stem>_crashes.parquet`, `<stem>_map.html`

The timestamp is generated at run time, so resolve the stem instead of typing it:

```bash
GRAPH=$(ls -t outputs/graph_20250101T000000_20251201T000000_all_*_segments.graphml | head -1)
CRASHES="${GRAPH%_segments.graphml}_crash_segments.csv"
echo "$GRAPH"; echo "$CRASHES"      # both must exist
```

Keep `$GRAPH` and `$CRASHES` set for the next step, or re-run those two lines in
each new shell.

## 5. Main run

```bash
uv run python scripts/midpoint.py \
    --speeds data/parquet/chicago-speeds \
    --graph "$GRAPH" \
    --crashes "$CRASHES" \
    --window 2025-01-01 2025-12-01 \
    --static-features length_m degree betweenness \
    --epochs 20 --patience 4 --seed 42 --controls 5 \
    --device auto \
    --processed data/processed-expanded \
    --out outputs/expanded
```

`--processed data/processed-expanded` is **required**. Its default is
`data/processed`, which holds the frozen pilot tables; the script now refuses to
overwrite them, so leaving it off produces an error rather than silent damage.

**On a retry**, that same guard will stop you, because the first attempt already
wrote the tables. Add `--overwrite-processed`, and use a fresh `--out`: the
script also refuses to write into a directory that already holds a
`summary.json`, so last run's CSVs can never sit beside this run's JSON.

This trains seven models (`persistence`, `historical_mean`, `mlp`, `gcn`, `sage`,
`tgru`, `stgcn`), estimates crash-segment effects against matched controls, and
writes figures. On CPU this is roughly nine hours; the two temporal models are
most of it, and the GPU should cut them sharply.

Writes to `outputs/expanded/`:

```
summary.json              results.csv             NOTES.md
coverage_funnel.csv       monthly_coverage.csv    neighbour_coverage.csv
effect_estimate.csv       events.csv              effects_crash.csv
effects_control.csv       coverage.png            model_comparison.png
crash_effect.png          crash_trajectories.png
```

## 6. Event study

Reads the joined tables the previous step wrote, so it needs no GPU and takes
about a minute.

```bash
uv run python scripts/event_study.py \
    --data data/processed-expanded \
    --hops 2 --controls 5 --seed 42 \
    --out outputs/event_study-expanded
```

Writes `summary.json`, `effect_by_hop.csv`, `effect_against_placebo.csv`,
`effect_fixed_cohort.csv`, `effect_excluding_overlaps.csv`, `pre_trends.csv`,
`covariate_balance.csv`, `neighbour_coverage.csv`, `effects_by_cell.csv`,
`hop_effects.png`.

**Read `effect_against_placebo.csv` before quoting any number from it.** On the
pilot the 1- and 2-hop rows had placebos 60% or more of their effects, which
means those rows could not be separated from pre-existing differences. If the
expanded sample changes that, it is a finding worth flagging immediately.

## 7. Semi-synthetic benchmark

```bash
uv run python scripts/benchmark.py \
    --data data/processed-expanded \
    --events 200 --controls 5 \
    --epochs 20 --patience 4 --seed 42 \
    --device auto \
    --out outputs/synthetic-expanded
```

Writes `summary.json`, `effect_recovery.csv`, `forecast.csv`, `truth.csv`,
`events.csv`, `controls.csv`. This retrains all seven models, so budget
similarly to step 5.

## 8. Zero-speed sensitivity

A separate run, not a replacement. It treats an observed `0.0` mph as missing;
the main run keeps zeros, which is the pilot convention.

```bash
uv run python scripts/midpoint.py \
    --data data/processed-expanded \
    --static-features length_m degree betweenness \
    --epochs 20 --patience 4 --seed 42 --controls 5 \
    --device auto --drop-zero-speed \
    --out outputs/expanded-nozero
```

It reads the already-joined tables, so it does not touch `--processed` and cannot
disturb step 5. It writes to `outputs/expanded-nozero/`.

Expect **13 files, not 14**: `coverage_funnel.csv` is only written when
`--crashes` is given, and the `--data` path has no crash audit CSV to build it
from. `summary.json` records `"coverage_funnel": null`. That is correct, not a
failed job.

This flag changes more than the outcome: it also moves the model's inputs, the
normalisation constants and two figures. `docs/experiments.md` records what, and
the comparison should be read with that in mind.

## 9. What to send back

The whole of each output directory, plus the commit hash from step 1 and the
device line from step 2:

```
outputs/expanded/
outputs/event_study-expanded/
outputs/synthetic-expanded/
outputs/expanded-nozero/
```

Also send any `slurm-*.out` logs. If a job dies, send the log before retrying;
a crash after six hours of training usually says something about the data.

## 10. Rules

- Never write to `outputs/midpoint/` or `data/processed/`.
- One run, one output directory. Do not re-run into a directory that already has
  results; stale files from a previous run read as current.
- Do not change filtering, window or seed arguments to get better-looking
  numbers. If a count looks wrong, report it.
- `--seed` will not redraw the controls. The control match is deterministic, so
  repeating a run under a new seed gives the same control set and only moves the
  model initialisation. To vary the control draw, use
  `scripts/event_study.py --random-controls`.
- `--device auto` falls back to CPU silently if CUDA is not visible on the node.
  Every `summary.json` now records the device actually used; check it before
  trusting a fast-looking run.
- If a run fails or a number looks surprising, say so before running the next
  thing. Surprising counts usually mean a data problem, and running three more
  jobs on the same bad input wastes the cluster time.
