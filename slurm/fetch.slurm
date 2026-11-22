#!/bin/bash
#SBATCH --job-name=traffic-fetch
#SBATCH --account=CHANGE_ME
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Stage one run's speeds and crashes. No GPU. Submit per run:
#
#   sbatch --export=ALL,START=2024-06-11,END=2026-05-01 slurm/fetch.slurm
#   sbatch --export=ALL,START=2018-02-28,END=2023-09-07 slurm/fetch.slurm
#
# Downloads go month by month and a rerun reuses whatever is already on disk,
# so a timed-out job can simply be resubmitted. The intermediate CSV is large
# (about 3 GB for run A, 9 GB for run B) and can be deleted after conversion.
set -euo pipefail

: "${START:?set START, e.g. START=2024-06-11}"
: "${END:?set END, e.g. END=2026-05-01}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "== speeds $START..$END =="
bash scripts/fetch_speeds_history.sh "$START" "$END"
uv run python scripts/convert_to_parquet.py \
    data/chicago-speeds-history.csv --kind speeds --overwrite

echo "== crashes $START..$END =="
CRASH_START="$START" CRASH_END="$END" bash scripts/fetch_data.sh --crashes-only

echo "== manifest =="
cat data/parquet/chicago-speeds/_conversion.json
echo
echo "If rows is exactly 50000000 the extract is truncated; report it."
echo "Monthly chunks are in data/speed-chunks and the combined CSV in"
echo "data/chicago-speeds-history.csv; both can be deleted now."
