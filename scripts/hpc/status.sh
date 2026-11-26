#!/bin/bash
# Read-only status for this project's submitted job manifest.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
manifest=${1:-"$root/outputs/hpc/queue-current.tsv"}
if [[ ! -f "$manifest" ]]; then
    echo "No job manifest at $manifest" >&2
    exit 1
fi
ids=$(awk -F '\t' 'NR > 1 {print $2}' "$manifest" | paste -sd, -)
printf 'Active jobs (PD=pending, R=running):\n'
squeue --jobs="$ids" --format='%.20i %.26j %.2t %.12M %R' || true
printf '\nAll submitted stages:\n'
sacct --jobs="$ids" --allocations --format=JobID,JobName%26,State%24,ExitCode,Elapsed
printf '\nStages recorded complete (resubmitting skips these):\n'
run=$(dirname -- "$manifest")/run
if compgen -G "$run/.done/*" > /dev/null; then
    ls "$run/.done" | sort | sed 's/^/  /'
else
    printf '  none yet\n'
fi

printf '\nPreparation log:\n'
prep_log=$(awk -F '\t' '$1 == "prepare" {print $4}' "$manifest")
# The manifest keeps sbatch's own `%a`, which Slurm substituted when it opened
# the file, so the recorded path never exists. Glob over what it wrote instead.
tail -n 8 ${prep_log//%a/*} 2>/dev/null || true

# Slurm's own states decide this, so no stage count here can go stale.
states=$(sacct --jobs="$ids" --allocations --noheader --format=State%20 | awk 'NF {print $1}')
active=$(grep -cE 'RUNNING|PENDING|REQUEUED|SUSPENDED|CONFIGURING' <<<"$states" || true)
unfinished=$(grep -cvE 'COMPLETED|RUNNING|PENDING|REQUEUED|SUSPENDED|CONFIGURING' <<<"$states" || true)
index="$run/paper/result_index.csv"
printf '\nVerdict: '
if [[ -z "$states" ]]; then
    printf 'Slurm has no record of these jobs yet.\n'
elif (( active > 0 )); then
    printf '%s job(s) still active. Not finished.\n' "$active"
elif (( unfinished > 0 )); then
    printf '%s job(s) did not complete. Resubmit with --resume.\n' "$unfinished"
elif [[ ! -s "$index" ]]; then
    printf 'every job completed but no result index was written. Check the aggregate log.\n'
elif grep -q ',missing,' "$index"; then
    printf 'every job completed, but results are missing:\n'
    grep ',missing,' "$index" | sed 's/^/  /'
else
    printf 'DONE. Every job completed and every result is present.\n'
fi
