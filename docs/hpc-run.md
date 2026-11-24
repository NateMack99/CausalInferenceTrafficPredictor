# HPC run handoff

Final-results runs on Rivanna. Everything happens on the cluster except one
46 MB copy of the road-network files. Replace `CHANGE_ME` before use.

## The plan

Two published archives, each its own run. Submit A first; do not block it on B.

| Run | Window | Speed rows | `--mem` | `--time` |
| --- | --- | ---: | ---: | ---: |
| **A** required | `2024-06-11 2026-05-01` | 101.1M | 48G | 18:00:00 |
| **B** secondary | `2018-02-28 2023-09-07` | 274.9M | 128G | 36:00:00 |

Both windows are `[start, end)`, so run B's `2023-09-07` includes all of
2023-09-06, the archive's last day. Each window spans its whole archive: the
counts above are `count(*)` from the portal, not the column metadata, which
understates `4g9f-3jbs` by 16% because that archive was still being appended to
when the cache was built.

Run A opens mid-month on purpose, because the archive itself starts at
2024-06-11T13:01. That does not distort the monthly report:
`evaluate.monthly_coverage` emits a `bins` column per month and computes
`observed_pct` against that month's own bins, so a short month is visible
rather than misleading. Trimming to whole interior months would drop about 7%
of the window to tidy a diagnostic table.

A nine-month hole (2023-09-07 to 2024-06-10) separates the archives, so they
stay in separate runs and no event window straddles it. `docs/data.md` has the
details. Memory is the usual failure: `data.align_grid` is row-count driven and
runs before any GPU work, so an under-requested job dies in the first minutes.

## 1. Code and environment

```bash
cd CHANGE_ME_PROJECT_DIR
git checkout main && git pull
git rev-parse --short HEAD          # record this; it ships with the results
uv venv --python 3.12 .venv && uv sync --locked
mkdir -p logs            # Slurm opens the log file before the job runs
```

Both job files request `--account=nssac_students`. Change the
`#SBATCH --account=` line of `slurm/fetch.slurm` and `slurm/experiment.slurm`
if your allocation differs; a job submitted without a valid one is rejected
immediately.

## 2. Copy the road network from Kyle

Four reference files plus the segment geometry, 46 MB total. These are the only
inputs that cannot be downloaded, because regenerating them would change the
road network out from under the frozen pilot.

```bash
# from Kyle's machine
scp data/chicago_road_network.graphml data/speed_road_network.graphml \
    data/speed_roads.parquet data/speed_segments.parquet \
    data/chicago-traffic-segments.json \
    CHANGE_ME_USER@rivanna:CHANGE_ME_PROJECT_DIR/data/
```

Verify on the cluster. This either prints five `OK` lines or names the bad file:

```bash
shasum -a 256 -c <<'EOF'
08cd229161a9efd196bab5d9abf5c98a90d624aea723e1c7b1dc2106643e8f1e  data/chicago_road_network.graphml
9798972d9cc099de3417d30e8ce73e036c5da852365c180ac440e5171cf0da6b  data/speed_road_network.graphml
47bc6f99e0bbe3ab63083d316418662a0f3fe3066aadab0accf3c58cae79ec10  data/speed_roads.parquet
bc4925f48a1f6d63244f0e1758d9dbd133822112eb5ce7a8fefbfa9e918baaf9  data/speed_segments.parquet
ebd729ace619bbea1d276ffcf7603de993da415e3fad4c208fd0ddd9afe9c696  data/chicago-traffic-segments.json
EOF
```

## 3. Stage the data

```bash
sbatch --export=ALL,START=2024-06-11,END=2026-05-01 slurm/fetch.slurm
```

Speeds come down one month per request and crashes in one; a rerun reuses
months already on disk, so a timed-out job can just be resubmitted. The
intermediate CSV is about 3.5 GB for run A and 9 GB for run B and can be
deleted afterwards. If the portal rate-limits, put a Socrata app token in `data/.env` as
`CHICAGO_DATA_APP_TOKEN=...` and resubmit; it is not required.

When it finishes, `rows` in the printed manifest must **not** be exactly
50,000,000. That is the old single-request cap and means a truncated extract.

## 4. Check before submitting

On the login node:

```bash
uv run python -c "
import pandas as pd; r = pd.read_parquet('data/speed_roads.parquet')
print(len(r), 'edges;', r.segment_id.nunique(), 'segments')"   # expect 6249 edges; 881 segments

uv run ruff check . && uv run pytest -q                        # expect 69 passed
```

Then the GPU smoke test, which only proves anything **inside an allocation**:

```bash
salloc -A CHANGE_ME_ALLOCATION -p gpu --gres=gpu:1 -c 4 --mem=16G -t 00:30:00
uv run python -m traffic_gml.mock --out data/mock
uv run python -m traffic_gml.train --data data/mock --model stgcn --epochs 2 --device auto
exit
```

Record the device line it prints. `--device auto` falls back to CPU silently, so
a run that prints `cpu` here means the allocation has no visible GPU. Stop on
any failure rather than submitting. The job itself asserts CUDA before doing
any work, so a misallocated run fails fast rather than quietly taking days.

## 5. Submit

```bash
sbatch --mem=48G --time=18:00:00 \
  --export=ALL,RUN=runA,START=2024-06-11,END=2026-05-01 slurm/experiment.slurm
```

One job runs the whole chain: graph, forecasting and effects, event study,
semi-synthetic benchmark, zero-speed sensitivity.

```bash
squeue -u $USER
tail -f logs/traffic-CHANGE_ME_JOBID.out
sacct -j CHANGE_ME_JOBID --format=JobID,State,Elapsed,MaxRSS,ReqMem
```

`MaxRSS` against `ReqMem` is the first thing to check if a job dies.

**If a job is killed**, resubmit the same command. Each of the five stages is
skipped once its own `summary.json` exists, so a resubmission resumes where the
previous attempt stopped and replaces only the partial stage. Nothing needs
editing, and `RUN` should stay the same.

For run B, repeat steps 3 and 5 with `START=2018-02-28 END=2023-09-07`,
`RUN=runB`, `--mem=128G` and `--time=36:00:00`.

## 6. Send back

Four directories per run, plus the commit hash from step 1 and all `logs/*.out`:

```
outputs/runA/  outputs/event_study-runA/  outputs/synthetic-runA/  outputs/runA-nozero/
```

`outputs/runA/` should hold 14 files (`summary.json`, `results.csv`, `NOTES.md`,
six CSVs, four PNGs). The `-nozero` directory holds 13, not 14:
`coverage_funnel.csv` needs `--crashes`, and `summary.json` records
`"coverage_funnel": null`. That is correct, not a failure.

Before quoting any number, read `event_study-runA/effect_against_placebo.csv`.
On the pilot the 1- and 2-hop rows had placebos 60% or more of their effects,
meaning those rows could not be separated from pre-existing differences. A
change there is worth flagging immediately.

## Rules

- Never run `scripts/prepare_roads.py` or `scripts/download_street_graph.py`.
  OpenStreetMap is live, so a rebuild today yields different node IDs than the
  2026-11-08 download the pilot used, silently breaking every
  pilot-versus-expanded comparison. Step 2 is the only way these files arrive.
- `outputs/` and `data/processed*` are gitignored, so a fresh clone has neither
  the frozen pilot nor any results. Everything you produce is untracked and has
  to be copied back (step 6).
- One run, one `RUN` tag. Stale files from a previous run read as current.
- Do not change filtering, window or seed arguments to improve the numbers. If
  a count looks wrong, report it.
- `--seed` does not redraw the controls; the match is deterministic. Use
  `scripts/event_study.py --random-controls` to vary the control draw.
- `--device auto` falls back to CPU silently. Every `summary.json` records the
  device actually used; check it before trusting a fast run.
- If a job fails or a number surprises you, say so before submitting the next
  one. Surprising counts usually mean a data problem.
