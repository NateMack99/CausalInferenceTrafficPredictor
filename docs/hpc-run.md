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

The repository currently defaults to `ACCOUNT=nssac_students`,
`CPU_PARTITION=standard`, `GPU_PARTITION=gpu`, `GPU_GRES=gpu:a100:1`. **Check
them against the commands above before your first submission.** They are the
values already committed to this repository, not values anyone verified for
your account. Export whatever differs:

```bash
export ACCOUNT=your_allocation
export CPU_PARTITION=your_cpu_partition
export GPU_PARTITION=your_gpu_partition
export GPU_GRES=gpu:a100:1
```

Put those four lines in `~/.bashrc` so every later step inherits them.

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

`prepare` copies these into the run directory. If they are absent it rebuilds
them from live OSM and prints a warning; results from such a run are not
comparable with Run A.

Optional: a Socrata app token avoids portal rate limits during the download.

```bash
echo 'CHICAGO_DATA_APP_TOKEN=your_token' > data/.env
```

## 4. Smoke test

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

That is the whole submission. It runs preflight, then queues 17 jobs with
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
| `verify` | CPU | lint and the offline test suite |
| `prepare` | CPU | downloads speeds and crashes, builds the graph, joins the tables |
| `audit` | CPU | coverage funnel, monthly coverage, input checksums |
| `main` | GPU | seven forecasters and the matched-residual effect |
| `event` | CPU | matched event study, placebo, availability, attrition |
| `benchmark` | GPU | Benchmark 0 known-effect recovery |
| `report` | CPU | first aggregation; fails if a primary stage is missing |
| `treatment` | GPU x3 | exposure-conditioned MLP and GNN against the residual estimator |
| `graph-ablation` | GPU x12 | real, self-only and rewired topologies at three seeds |
| `matching` | CPU x6 | four control-matching rules, including target-road matching |
| `main-nozero`, `event-nozero`, `benchmark-nozero` | mixed | zero-speed sensitivity |
| `synthetic-scenarios` | GPU x9 | synthetic null, plateau and congested placement at three seeds |
| `forecast-seeds` | GPU x8 | seed spread for the headline forecast table |
| `aggregate` | CPU | paper tables, the ablation figure, the result index |
| `finish` | CPU | final status summary |

Primary results are queued first, so a wall-time loss costs the least
important work. GPU arrays run two at a time; raise it with
`GPU_CONCURRENCY=4` if your allocation allows.

## Why this is several jobs and not one

One `sbatch` job cannot hold the matrix. The GPU work alone is 36 tasks, each
allowed up to 24 hours, which exceeds any single job's wall-time limit; a
single job would also hold a GPU idle through every CPU stage. `submit.sh` is
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
