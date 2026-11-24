#!/bin/bash
# Supply resources with sbatch; see docs/hpc-run.md.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
meta_root=$1
shift
run_meta="$meta_root/$SLURM_JOB_ID"
mkdir -p "$meta_root"
mkdir "$run_meta"
run_meta=$(realpath "$run_meta")
trap 'status=$?; printf "%s\n" "$status" > "$run_meta/exit-code.txt"' EXIT
module load uv/0.13.5
export UV_CACHE_DIR="$PWD/cache/uv"
export UV_NO_SYNC=1
# Queued runs may use a frozen worktree with the shared locked environment.
# Prefer that worktree's source over the editable installation's original path.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$run_meta/matplotlib"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
date -u +%FT%TZ > "$run_meta/started-utc.txt"
hostname > "$run_meta/hostname.txt"
printf '%q ' "$@" > "$run_meta/command.sh"
printf '\n' >> "$run_meta/command.sh"
git rev-parse HEAD > "$run_meta/commit.txt"
git status --short > "$run_meta/git-status.txt"
git diff HEAD > "$run_meta/changes.patch"
sha256sum uv.lock > "$run_meta/lock.sha256"
tar -czf "$run_meta/source.tar.gz" src scripts tests docs pyproject.toml uv.lock README.md AGENTS.md
uv pip freeze > "$run_meta/packages.txt"
scontrol show job "$SLURM_JOB_ID" > "$run_meta/slurm.txt"
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    nvidia-smi > "$run_meta/nvidia-smi.txt"
fi
srun "$@"
