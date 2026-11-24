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
tail -n 8 "$prep_log" 2>/dev/null || true
