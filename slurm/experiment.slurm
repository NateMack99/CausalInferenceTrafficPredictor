#!/bin/bash
#SBATCH --job-name=traffic
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# One run, end to end: graph, forecasting and effects, event study, benchmark,
# zero-speed sensitivity. Set --account, then submit per run (CLI flags win
# over the directives above):
#
#   sbatch --mem=48G  --time=18:00:00 --export=ALL,RUN=runA,START=2024-06-11,END=2026-05-01 slurm/experiment.slurm
#   sbatch --mem=128G --time=36:00:00 --export=ALL,RUN=runB,START=2018-02-28,END=2023-09-07 slurm/experiment.slurm
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
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== 1/5 graph =="
uv run python scripts/build_graph.py --no-fetch \
    --crash-window "$START" "$END" --speed-window "$START" "$END"

GRAPH=$(ls -t outputs/${STEM}_*_segments.graphml | head -1)
CRASHES="${GRAPH%_segments.graphml}_crash_segments.csv"
test -s "$GRAPH" && test -s "$CRASHES"
echo "graph=$GRAPH"

echo "== 2/5 forecasting and effects =="
uv run python scripts/experiment.py \
    --speeds data/parquet/chicago-speeds --graph "$GRAPH" --crashes "$CRASHES" \
    --window "$START" "$END" \
    --static-features length_m degree betweenness \
    --epochs 20 --patience 4 --seed 42 --controls 5 --device auto \
    --processed "$PROCESSED" --out "outputs/$RUN"

echo "== 3/5 event study =="
uv run python scripts/event_study.py --data "$PROCESSED" \
    --hops 2 --controls 5 --seed 42 --out "outputs/event_study-$RUN"

echo "== 4/5 semi-synthetic benchmark =="
uv run python scripts/benchmark.py --data "$PROCESSED" \
    --events 200 --controls 5 --epochs 20 --patience 4 --seed 42 --device auto \
    --out "outputs/synthetic-$RUN"

echo "== 5/5 zero-speed sensitivity =="
uv run python scripts/experiment.py --data "$PROCESSED" \
    --static-features length_m degree betweenness \
    --epochs 20 --patience 4 --seed 42 --controls 5 --device auto --drop-zero-speed \
    --out "outputs/$RUN-nozero"

echo "done $RUN"
