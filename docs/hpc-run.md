# Rivanna runbook

One command queues the whole Run A matrix. This page is the copy-paste
procedure, written so it can be followed without knowing the pipeline.

Nothing here assumes your username, paths, allocation or partitions. Section 0
lists every value you supply and how to find it.

## 0. Values you supply

| Value | How to find it | Used as |
| --- | --- | --- |
| Username | `whoami` on the cluster | in `ssh` |
| Project directory | `pwd` after you clone | in `cd` |
| Allocation | `sacctmgr show assoc user=$USER format=Account -nP \| sort -u` | `ACCOUNT` |
| CPU partition | `sinfo -s` and pick a general-purpose one | `CPU_PARTITION` |
| GPU partition | `sinfo -o '%P %G' \| grep gpu` | `GPU_PARTITION` |
| GPU type | the `%G` column above, e.g. `gpu:a100:1` | `GPU_GRES` |

The repository defaults to `ACCOUNT=nssac_students`, `CPU_PARTITION=standard`,
`GPU_PARTITION=interactive`, `GPU_GRES=gpu:rtx_3090:1`, with GPU jobs asking for
`GPU_TIME=12:00:00` and `GPU_MEM=64G` because `interactive` caps wall time at
twelve hours. **Check these against the commands above before your first
submission.** They are the values committed to this repository, not values
verified for your account. Export whatever differs:

```bash
export ACCOUNT=your_allocation
export CPU_PARTITION=your_cpu_partition
export GPU_PARTITION=interactive
export GPU_GRES=gpu:rtx_3090:1
```

Put those lines in `~/.bashrc` so every later step inherits them.

You do not have to get this right by inspection. Preflight asks Slurm itself,
with `sbatch --test-only`, whether it would accept the CPU and GPU shapes, and
refuses to queue anything if either is rejected. A wrong partition or an
over-limit wall time stops the submission before a single job is queued, and
prints Slurm's own error.

Other knobs, all optional: `DATA_DIR` (default `data`), `GPU_TIME`, `GPU_MEM`,
`CPU_MEM` (default 128G, for the stages that read the joined tables),
`SMALL_MEM` (default 8G, for the stages that only read checksums and CSVs),
`CPU_TIME`, `PREPARE_TIME`, `GPU_CONCURRENCY` (default 2), `MIN_FREE_GB`
(default 200).

### If the queue wait is long

Two levers, both free. First, the allocation: fair-share differs per account
and a lower one waits longer. List yours and pick a well-ranked one you are
authorised to use on this project.

```bash
allocations           # or: sacctmgr show assoc user=$USER format=Account -nP | sort -u
export ACCOUNT=your_allocation
```

Second, the partition. `qlist` shows free cores, the pending backlog and the
wall-time cap per partition. `standard` carries a large backlog; `interactive`
is usually near-empty but caps wall time at twelve hours, which every CPU stage
here fits.

```bash
qlist
export CPU_PARTITION=interactive
```

Preflight will reject this if `PREPARE_TIME` exceeds the cap or the partition
limits how many jobs you may hold, before anything is queued.

## 1. Log in

```bash
ssh YOUR_USERNAME@rivanna.hpc.virginia.edu
```

## 2. Get the code there

First time:

```bash
git clone https://github.com/NateMack99/CausalInferenceTrafficPredictor.git
cd CausalInferenceTrafficPredictor
```

Every later time:

```bash
cd ~/CausalInferenceTrafficPredictor     # wherever you cloned it
git checkout main && git pull
git rev-parse --short HEAD               # note this; it ships with the results
```

## 3. Environment and the frozen road network

```bash
module load uv/0.13.5
uv venv --python 3.12 .venv && uv sync --locked
```

Five road-network files cannot be downloaded, because OpenStreetMap is live and
rebuilding them today would change node ids out from under Run A. Copy them
from the machine that holds them:

```bash
# run this on the machine that has data/, not on the cluster
scp data/chicago_road_network.graphml data/speed_road_network.graphml \
    data/speed_roads.parquet data/speed_segments.parquet \
    data/chicago-traffic-segments.json \
    YOUR_USERNAME@rivanna.hpc.virginia.edu:PATH_TO_PROJECT/data/
```

Verify on the cluster. This prints five `OK` lines or names the bad file:

```bash
shasum -a 256 -c <<'EOF'
08cd229161a9efd196bab5d9abf5c98a90d624aea723e1c7b1dc2106643e8f1e  data/chicago_road_network.graphml
9798972d9cc099de3417d30e8ce73e036c5da852365c180ac440e5171cf0da6b  data/speed_road_network.graphml
47bc6f99e0bbe3ab63083d316418662a0f3fe3066aadab0accf3c58cae79ec10  data/speed_roads.parquet
bc4925f48a1f6d63244f0e1758d9dbd133822112eb5ce7a8fefbfa9e918baaf9  data/speed_segments.parquet
ebd729ace619bbea1d276ffcf7603de993da415e3fad4c208fd0ddd9afe9c696  data/chicago-traffic-segments.json
EOF
```

`prepare` reads these where they are. If they are absent it rebuilds them from
live OSM and prints a warning; results from such a run are not comparable with
Run A, because node ids differ.

These files are gitignored and the boundary polygon they were clipped to was
never saved, so they cannot be rebuilt identically. The checksums above are the
record; copy the files, do not regenerate them.

Optional: a Socrata app token avoids portal rate limits during the download.

```bash
echo 'CHICAGO_DATA_APP_TOKEN=your_token' > data/.env
```

## 3b. Datasets you already have

The pipeline reads the converted datasets where they are. Nothing is copied,
and there is no symlink to set up.

```
data/parquet/chicago-speeds-history/
data/parquet/chicago-crashes/
```

Point `DATA_DIR` elsewhere if yours live somewhere else:

```bash
export DATA_DIR=/scratch/$USER/traffic-data
```

Check what will be reused before submitting:

```bash
uv run python scripts/check_dataset.py \
    --speeds "${DATA_DIR:-data}/parquet/chicago-speeds-history" \
    --crashes "${DATA_DIR:-data}/parquet/chicago-crashes"
```

A dataset counts as reusable only when its `_conversion.json` is present,
records rows, and its recorded file count matches the parquet files on disk, so
a half-copied dataset is refused rather than silently half-read. Anything that
fails is downloaded into `DATA_DIR`; a dataset that passes is never rewritten.
Preflight prints the same verdict, so a surprise download is visible before any
job is queued, and `prepare` records what it resolved in
`outputs/hpc/NAME/work/data-provenance.json`.

## 4. Check the code, then smoke-test

Lint and the offline tests run here, on the login node, in under a minute. They
used to be a queued job, which meant a scheduling wait before the pipeline
could even start.

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

A failure here is a code problem; fix it before submitting.

### Smoke test

Checks the whole pipeline on mock data in a few minutes, including the GPU.

```bash
bash scripts/hpc/submit.sh --name smoke --smoke
```

Watch it, then confirm it wrote an index:

```bash
bash scripts/hpc/status.sh outputs/hpc/smoke/manifest-latest.tsv
cat outputs/hpc/smoke/run/smoke/paper/result_index.csv
```

A failure here is a configuration problem. Fix it before step 5.

## 5. Submit the full pipeline

```bash
bash scripts/hpc/submit.sh --name runA
```

That is the whole submission. It runs preflight, then queues 16 jobs with
Slurm dependencies and prints the manifest path. Nothing else needs submitting.

Preflight refuses to queue anything if the partitions are wrong, the road
network is missing, or the filesystem has under 200 GB free. Submit from a
filesystem with room; `/scratch` usually has it.

## 6. Monitor

```bash
bash scripts/hpc/status.sh outputs/hpc/runA/manifest-latest.tsv
```

It prints queued and running jobs, the state of every submitted stage, which
stages are already recorded complete, and the tail of the preparation log.

Logs are one file per stage under `outputs/hpc/runA/logs/`:

```bash
ls outputs/hpc/runA/logs/
tail -f outputs/hpc/runA/logs/prepare-*.log
```

## 7. Did a stage succeed?

Three independent ways, in increasing detail:

```bash
# 1. completion markers: a stage writes one only after it finishes
ls outputs/hpc/runA/run/.done/

# 2. Slurm's own record, including exit codes and memory used
sacct --starttime today --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS

# 3. per-job provenance, including the exact command and its exit code
cat outputs/hpc/runA/logs/meta/JOBID/exit-code.txt
cat outputs/hpc/runA/logs/meta/JOBID/command.sh
```

`MaxRSS` against the requested memory is the first thing to check when a job
dies without a Python traceback.

## 8. Recover from a failure or a time limit

Resubmit the same run. Completed stages are skipped, so only unfinished work
is queued again:

```bash
bash scripts/hpc/submit.sh --name runA --resume
```

This is safe to repeat. A stage records its marker only on success, so a job
killed at its wall-time limit simply runs again. Nothing needs editing.

If one stage keeps failing, read its log, fix the cause, and resume: the rest
of the matrix is unaffected because each stage is guarded separately.

To force a completed stage to run again, delete its marker first:

```bash
rm outputs/hpc/runA/run/.done/graph-ablation-3
bash scripts/hpc/submit.sh --name runA --resume
```

## 9. Collect the results

The `aggregate` stage writes everything the paper needs into one directory:

```bash
ls outputs/hpc/runA/run/paper/
cat outputs/hpc/runA/run/paper/result_index.csv      # which file feeds which table
cat outputs/hpc/runA/run/paper/manifest.json         # what finished, what is missing
```

`result_index.csv` has one row per manuscript table or figure, the file that
supplies it, and whether it is `ready` or `missing`. A `missing` row means that
stage has not finished; no number is ever invented for it.

Copy the bundle back:

```bash
# from your own machine
scp -r YOUR_USERNAME@rivanna.hpc.virginia.edu:PATH_TO_PROJECT/outputs/hpc/runA/run/paper ./
```

Send the commit hash from step 2 along with it. Before quoting any effect,
read `effect_against_placebo.csv`: on the pilot the 1- and 2-hop rows had
placebos 60% or more of their effects, meaning those rows could not be
separated from pre-existing differences.

## What the pipeline runs

| Stage | Resource | What it produces |
| --- | --- | --- |
| `prepare` | CPU | downloads speeds and crashes, builds the graph, joins the tables |
| `audit` | CPU, 8G | coverage funnel, monthly coverage, input checksums |
| `main` | GPU | seven forecasters and the matched-residual effect |
| `event` | CPU | matched event study, placebo, availability, attrition |
| `benchmark` | GPU | Benchmark 0 known-effect recovery |
| `report` | CPU, 8G | first aggregation; fails if a primary stage is missing |
| `treatment` | GPU x3 | exposure-conditioned MLP and GNN against the residual estimator |
| `graph-ablation` | GPU x12 | real, self-only and rewired topologies at three seeds |
| `matching` | CPU x6 | four control-matching rules, including target-road matching |
| `main-nozero`, `event-nozero`, `benchmark-nozero` | mixed | zero-speed sensitivity |
| `synthetic-scenarios` | GPU x9 | synthetic null, plateau and congested placement at three seeds |
| `forecast-seeds` | GPU x8 | seed spread for the headline forecast table |
| `aggregate` | CPU, 8G | paper tables, the ablation figure, the result index |
| `finish` | CPU, 8G | final status summary |

Every experiment stage depends only on `audit`, which writes the checksum they
all verify, so the scheduler decides the order and one stage failing never
blocks another. `report` is the only stage that reads other stages' outputs.
Expect most of the matrix to sit pending at once. GPU arrays run two at a time;
raise it with `GPU_CONCURRENCY=4` if your allocation allows, or drop it to 1 if
`interactive` limits how many jobs you may hold.

**If a stage runs out of the twelve hours**, it leaves no marker and `--resume`
runs it again, which only helps if it can finish next time. `main` and
`benchmark` are the likeliest to be tight, because each trains several models
in one process with no mid-run checkpoint. If one keeps hitting the limit, move
just that attempt to a partition with a longer ceiling and resume; every
finished stage is skipped:

```bash
GPU_PARTITION=your_long_partition GPU_GRES=gpu:a100:1 GPU_TIME=24:00:00 \
  bash scripts/hpc/submit.sh --name runA --resume
```

## Why this is several jobs and not one

One `sbatch` job cannot hold the matrix. The GPU work alone is 36 tasks, and
`interactive` allows twelve hours per job, so the matrix cannot fit in one job
however it is arranged; a single job would also hold a GPU idle through every
CPU stage. `submit.sh` is
therefore one command that queues a dependency graph, which is the closest
practical equivalent and needs no manual coordination between stages.

Submitting from the login node rather than from inside a batch job is also
deliberate: preflight then fails in a second instead of after a queue wait.

## Rules

- Never run `scripts/prepare_roads.py` or `scripts/download_street_graph.py`
  by hand. Step 3 is how the road network arrives.
- `outputs/` and `data/processed*` are gitignored. Everything you produce is
  untracked and has to be copied back.
- One run, one `--name`. Stale files from a previous run read as current.
- Do not change filtering, window or seed arguments to improve a number. If a
  count looks wrong, report it.
- `--seed` does not redraw the controls; the match is deterministic. Use
  `scripts/event_study.py --random-controls` to vary the control draw.
- Every `summary.json` records the device actually used. Check it before
  trusting a fast run.
- Run B (`2018-02-28` to `2023-09-07`, 274.9M speed rows) is optional and must
  not delay Run A. A nine-month gap separates the two archives, so no event
  window straddles them and they stay separate runs.
