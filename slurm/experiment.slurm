#!/bin/bash
#SBATCH --job-name=traffic
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# One run, end to end: graph, forecasting and effects, event study, benchmark,
# zero-speed sensitivity. `mkdir -p logs` before the first submission, because
# Slurm opens the output file before this script runs. Then:
#
#   sbatch --mem=48G  --time=18:00:00 --export=ALL,RUN=runA,START=2024-06-11,END=2026-05-01 slurm/experiment.slurm
#   sbatch --mem=128G --time=36:00:00 --export=ALL,RUN=runB,START=2018-02-28,END=2023-09-07 slurm/experiment.slurm
#
# Resubmitting is safe and is the way to recover from a walltime kill: each
# stage is skipped once its own summary.json exists, so a resubmission picks up
# where the previous attempt stopped. Add --gres=gpu:a100:1 to pin a faster
# card at the cost of queue time.
#
# Memory is the usual failure: data.align_grid is row-count driven and runs
# before any GPU work, so an under-requested job dies in the first minutes.
set -euo pipefail

: "${RUN:?set RUN, e.g. RUN=runA}"
: "${START:?set START, e.g. START=2024-06-11}"
: "${END:?set END, e.g. END=2026-05-01}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
PROCESSED="data/processed-$RUN"
STEM="graph_${START//-/}T000000_${END//-/}T000000_all"

echo "run=$RUN window=$START..$END commit=$(git rev-parse --short HEAD) node=$(hostname)"
# Assert rather than print: --device auto would otherwise fall back to CPU and
# turn a misallocated job into a run that looks fine but takes days.
uv run python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device visible in this allocation'
print('cuda', torch.__version__, torch.cuda.get_device_name(0))"

if ls outputs/${STEM}_*_segments.graphml >/dev/null 2>&1; then
    echo "skip 1/5 graph: outputs/${STEM}_* already built"
else
    echo "== 1/5 graph =="
    uv run python scripts/build_graph.py --no-fetch \
        --crash-window "$START" "$END" --speed-window "$START" "$END"
fi

GRAPH=$(ls -t outputs/${STEM}_*_segments.graphml | head -1)
CRASHES="${GRAPH%_segments.graphml}_crash_segments.csv"
test -s "$GRAPH" && test -s "$CRASHES"
echo "graph=$GRAPH"

if [[ -f "outputs/$RUN/summary.json" ]]; then
    echo "skip 2/5 forecasting and effects: outputs/$RUN already complete"
else
    echo "== 2/5 forecasting and effects =="
    # This run owns $PROCESSED, so replacing a partial copy from a killed
    # attempt is safe. The frozen data/processed is never $PROCESSED.
    OVERWRITE=()
    [[ -d "$PROCESSED" ]] && OVERWRITE=(--overwrite-processed)
    uv run python scripts/experiment.py \
        --speeds data/parquet/chicago-speeds --graph "$GRAPH" --crashes "$CRASHES" \
        --window "$START" "$END" \
        --static-features length_m degree betweenness \
        --epochs 20 --patience 4 --seed 42 --controls 5 --device auto \
        --processed "$PROCESSED" --out "outputs/$RUN" ${OVERWRITE[@]+"${OVERWRITE[@]}"}
fi

if [[ -f "outputs/event_study-$RUN/summary.json" ]]; then
    echo "skip 3/5 event study: already complete"
else
    echo "== 3/5 event study =="
    uv run python scripts/event_study.py --data "$PROCESSED" \
        --hops 2 --controls 5 --seed 42 --out "outputs/event_study-$RUN"
fi

if [[ -f "outputs/synthetic-$RUN/summary.json" ]]; then
    echo "skip 4/5 benchmark: already complete"
else
    echo "== 4/5 semi-synthetic benchmark =="
    uv run python scripts/benchmark.py --data "$PROCESSED" \
        --events 200 --controls 5 --epochs 20 --patience 4 --seed 42 --device auto \
        --out "outputs/synthetic-$RUN"
fi

if [[ -f "outputs/$RUN-nozero/summary.json" ]]; then
    echo "skip 5/5 zero-speed sensitivity: already complete"
else
    echo "== 5/5 zero-speed sensitivity =="
    uv run python scripts/experiment.py --data "$PROCESSED" \
        --static-features length_m degree betweenness \
        --epochs 20 --patience 4 --seed 42 --controls 5 --device auto --drop-zero-speed \
        --out "outputs/$RUN-nozero"
fi

echo "done $RUN"
