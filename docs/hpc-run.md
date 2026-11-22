# HPC run handoff

Final-results runs on Rivanna. Two people: Kyle stages the data locally, the
cluster operator submits the jobs. Replace `CHANGE_ME` placeholders before use.

## The plan

Two archives, each its own run. Submit A first and do not block it on B.

| Run | Window | Rows | `sbatch --mem` | `--time` |
| --- | --- | ---: | ---: | ---: |
| **A** required | `2024-06-11 2026-05-01` | 86.9M | 48G | 18:00:00 |
| **B** secondary | `2018-02-28 2023-09-07` | 274.9M | 128G | 36:00:00 |

A nine-month hole (2023-09-07 to 2024-06-10) separates the archives, so they
stay in separate runs and no event window straddles it. `docs/data.md` has the
archive details. Memory is the usual failure: `data.align_grid` is row-count
driven and runs before any GPU work, so an under-requested job dies early.

## 1. Local, by Kyle

Fetch on a machine with internet, ship Parquet. One request cannot carry a
multi-year span, so this goes month by month and reuses months already on disk.

```bash
bash scripts/fetch_speeds_history.sh 2024-06-11 2026-05-01      # run A
uv run python scripts/convert_to_parquet.py \
    data/chicago-speeds-history.csv --kind speeds --overwrite
cat data/parquet/chicago-speeds/_conversion.json                # rows must NOT be 50,000,000
```

A round 50,000,000 means a truncated extract. Crashes come from
`bash scripts/fetch_data.sh --crashes-only` after editing its `$where` bounds to
the run window. Repeat with `2018-02-28 2023-09-07` for run B.

Then copy these to the same relative paths on Rivanna:

| Path | Size |
| --- | ---: |
| `data/parquet/chicago-speeds/` | ~135 MB (A), ~430 MB (B) |
| `data/parquet/chicago-crashes/` | ~90 MB |
| `data/chicago_road_network.graphml` | 41 MB |
| `data/speed_road_network.graphml` | 4.5 MB |
| `data/speed_roads.parquet` | 184 KB |
| `data/speed_segments.parquet` | 32 KB |
| `data/chicago-traffic-segments.json` | 372 KB |

## 2. On Rivanna: set up and verify

```bash
cd CHANGE_ME_PROJECT_DIR
git checkout main && git pull && git rev-parse --short HEAD   # record this
uv venv --python 3.12 .venv && uv sync --locked
```

Four gates. Stop on any failure rather than submitting.

```bash
# a. inputs arrived intact
shasum -a 256 data/chicago_road_network.graphml data/speed_road_network.graphml \
  data/speed_roads.parquet data/speed_segments.parquet data/chicago-traffic-segments.json
# expect these first 16 hex:
#   08cd229161a9efd1  chicago_road_network.graphml
#   9798972d9cc099de  speed_road_network.graphml
#   47bc6f99e0bbe3ab  speed_roads.parquet
#   bc4925f48a1f6d63  speed_segments.parquet
#   ebd729ace619bbea  chicago-traffic-segments.json

# b. road network is the real one
uv run python -c "
import pandas as pd; r = pd.read_parquet('data/speed_roads.parquet')
print(len(r), 'edges;', r.segment_id.nunique(), 'segments')"
# expect: 6249 edges; 881 segments

# c. code is healthy
uv run ruff check . && uv run pytest -q          # expect 69 passed

# d. GPU works
uv run python -m traffic_gml.mock --out data/mock
uv run python -m traffic_gml.train --data data/mock --model stgcn --epochs 2 --device auto
```

## 3. Submit

Set `--account` in `slurm/experiment.slurm` first, then:

```bash
sbatch --mem=48G  --time=18:00:00 \
  --export=ALL,RUN=runA,START=2024-06-11,END=2026-05-01 slurm/experiment.slurm

# only after A has produced a complete outputs/runA/
sbatch --mem=128G --time=36:00:00 \
  --export=ALL,RUN=runB,START=2018-02-28,END=2023-09-07 slurm/experiment.slurm
```

One job runs the whole chain: graph, forecasting and effects, event study,
semi-synthetic benchmark, zero-speed sensitivity. Monitor with

```bash
squeue -u $USER
sacct -j CHANGE_ME_JOBID --format=JobID,State,Elapsed,MaxRSS,ReqMem
tail -f logs/traffic-CHANGE_ME_JOBID.out
```

`MaxRSS` against `ReqMem` is the first thing to check if a job dies.

**On a retry**, the run already wrote its tables, so add
`--overwrite-processed` to the two `scripts/experiment.py` calls and change
`RUN` (for example `runA2`) so nothing writes into a used output directory.

## 4. Results

Per run: `outputs/runA/`, `outputs/event_study-runA/`,
`outputs/synthetic-runA/`, `outputs/runA-nozero/`. The main directory holds

```
summary.json          results.csv           NOTES.md
coverage_funnel.csv   monthly_coverage.csv  neighbour_coverage.csv
effect_estimate.csv   events.csv            effects_crash.csv
effects_control.csv   coverage.png          model_comparison.png
crash_effect.png      crash_trajectories.png
```

The `-nozero` directory has 13 of those, not 14: `coverage_funnel.csv` needs
`--crashes`, and `summary.json` records `"coverage_funnel": null`. That is
correct, not a failure.

Send back every directory, the commit hash from section 2, and all
`logs/*.out`. Before quoting any number, read
`event_study-*/effect_against_placebo.csv`: on the pilot the 1- and 2-hop rows
had placebos 60% or more of their effects, meaning those rows could not be
separated from pre-existing differences. A change there is worth flagging
immediately.

## Rules

- Never write to `outputs/midpoint/` or `data/processed/`. They are the frozen
  pilot every recorded result rests on.
- Never download on the cluster. Every input is copied in; compute nodes
  generally have no outbound HTTP. A pipeline that starts fetching has a
  missing input, and the fix is the file, not the network.
- Never run `scripts/prepare_roads.py` or `scripts/download_street_graph.py`.
  OpenStreetMap is live, so a rebuild today yields different node IDs than the
  2026-11-08 download the pilot used, which silently breaks every
  pilot-versus-expanded comparison.
- One run, one `RUN` tag. Stale files from a previous run read as current.
- Do not change filtering, window or seed arguments to improve the numbers. If
  a count looks wrong, report it.
- `--seed` does not redraw the controls; the match is deterministic. Use
  `scripts/event_study.py --random-controls` to vary the control draw.
- `--device auto` falls back to CPU silently. Every `summary.json` records the
  device actually used; check it before trusting a fast run.
- If a job fails or a number surprises you, say so before submitting the next
  one. Surprising counts usually mean a data problem.
