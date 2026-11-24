#!/bin/bash
# The one command that queues the whole Run A matrix.
#
#   bash scripts/hpc/submit.sh --name runA
#   bash scripts/hpc/submit.sh --name runA --resume     # after a failure
#   bash scripts/hpc/submit.sh --name smoke --smoke     # pipeline check only
#
# Submitting from the login node rather than from inside a batch job is
# deliberate: preflight then fails in a second instead of after a queue wait,
# and one batch job cannot hold the matrix anyway (see docs/hpc-run.md).
#
# Cluster settings come from the environment so nothing here assumes an
# allocation. docs/hpc-run.md says how to discover each one.
set -euo pipefail
ACCOUNT=${ACCOUNT:-nssac_students}
CPU_PARTITION=${CPU_PARTITION:-standard}
GPU_PARTITION=${GPU_PARTITION:-interactive}
GPU_GRES=${GPU_GRES:-gpu:rtx_3090:1}
GPU_CONCURRENCY=${GPU_CONCURRENCY:-2}
CPU_MEM=${CPU_MEM:-128G}
MIN_FREE_GB=${MIN_FREE_GB:-200}
GPU_MEM=${GPU_MEM:-64G}
# interactive caps wall time at 12 hours. A stage killed at the limit leaves no
# marker, so --resume simply runs it again.
GPU_TIME=${GPU_TIME:-12:00:00}
CPU_TIME=${CPU_TIME:-04:00:00}
PREPARE_TIME=${PREPARE_TIME:-12:00:00}

name=run-$(date -u +%Y%m%dT%H%M%SZ)
resume=no
smoke=no
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) name=$2; shift 2 ;;
        --resume) resume=yes; shift ;;
        --smoke) smoke=yes; shift ;;
        *) echo "usage: $0 [--name NAME] [--resume] [--smoke]" >&2; exit 2 ;;
    esac
done

root=outputs/hpc/$name
work=$root/work
run=$root/run
logs=$root/logs
if [[ -e "$root" && "$resume" == no ]]; then
    echo "$root exists. Pass --resume to queue only the stages that have not" >&2
    echo "finished, or --name NAME for a separate run." >&2
    exit 1
fi

echo "== preflight =="
for tool in sbatch squeue sacct; do
    command -v "$tool" >/dev/null || { echo "$tool not found; run this on the cluster" >&2; exit 1; }
done
test -f uv.lock || { echo "run from the project root" >&2; exit 1; }
test -f data/chicago-traffic-segments.json || {
    echo "data/chicago-traffic-segments.json missing; copy it over first" >&2; exit 1
}
# Ask Slurm whether it would accept each shape, rather than reimplementing its
# limits. A rejection mid-submission would leave half a dependency graph queued.
for shape in "cpu $CPU_PARTITION $CPU_MEM $PREPARE_TIME" "gpu $GPU_PARTITION $GPU_MEM $GPU_TIME"; do
    read -r kind partition memory duration <<<"$shape"
    gres=()
    [[ "$kind" != gpu ]] || gres=(--gres="$GPU_GRES")
    if ! sbatch --test-only --account="$ACCOUNT" --partition="$partition" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem="$memory" --time="$duration" \
        ${gres[@]+"${gres[@]}"} --wrap=true >/dev/null 2>"${TMPDIR:-/tmp}/traffic-preflight.$$"; then
        echo "Slurm would reject the $kind shape:" >&2
        sed 's/^/  /' "${TMPDIR:-/tmp}/traffic-preflight.$$" >&2
        echo "  account=$ACCOUNT partition=$partition mem=$memory time=$duration" >&2
        rm -f "${TMPDIR:-/tmp}/traffic-preflight.$$"
        exit 1
    fi
    rm -f "${TMPDIR:-/tmp}/traffic-preflight.$$"
done
free_gb=$(df -Pk . | awk 'NR == 2 {print int($4 / 1048576)}')
(( free_gb >= MIN_FREE_GB )) || {
    echo "only ${free_gb}G free here; Run A needs about ${MIN_FREE_GB}G." >&2
    echo "Submit from a larger filesystem, or set MIN_FREE_GB if you know better." >&2
    exit 1
}
mkdir -p "$run/.done" "$logs/meta"
echo "account=$ACCOUNT  cpu=$CPU_PARTITION  gpu=$GPU_PARTITION/$GPU_GRES"
echo "gpu time=$GPU_TIME mem=$GPU_MEM  free=${free_gb}G"

manifest=$root/manifest-$(date -u +%Y%m%dT%H%M%SZ).tsv
printf 'stage\tjob_id\tdependency\tlog\n' > "$manifest"

submit() {
    local stage=$1 kind=$2 duration=$3 dependencies=$4 array=${5:-}
    local extra=() command=() job partition=$CPU_PARTITION memory=$CPU_MEM
    if [[ "$kind" == gpu ]]; then
        partition=$GPU_PARTITION
        memory=$GPU_MEM
        extra+=(--gres="$GPU_GRES")
        # Cap only GPU arrays: that is the scarce resource, and a CPU array of
        # six short jobs has no reason to run two at a time.
        [[ -z "$array" ]] || array="$array%$GPU_CONCURRENCY"
    fi
    [[ -z "$array" ]] || extra+=(--array="$array")
    [[ -z "$dependencies" ]] || extra+=(--dependency="$dependencies" --kill-on-invalid-dep=yes)
    case "$stage" in
        verify) command=(bash scripts/hpc/verify.sh) ;;
        prepare) command=(bash scripts/hpc/prepare.sh "$work") ;;
        smoke) command=(bash scripts/hpc/smoke.sh "$run/smoke") ;;
        finish) command=(bash scripts/hpc/status.sh "$manifest") ;;
        *) command=(bash scripts/hpc/experiments.sh "$stage" "$work" "$run") ;;
    esac
    job=$(sbatch --parsable --account="$ACCOUNT" --partition="$partition" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem="$memory" --time="$duration" \
        --job-name="traffic-$stage" \
        --output="$logs/$stage-%A_%a.log" ${extra[@]+"${extra[@]}"} \
        scripts/hpc/run.sh "$logs/meta" "${command[@]}")
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s/%s-%s_%%a.log\n' \
        "$stage" "$job" "$dependencies" "$logs" "$stage" "$job" >> "$manifest"
    printf 'queued %-20s %s\n' "$stage" "$job" >&2
    printf '%s\n' "$job"
}

verify=$(submit verify cpu 00:30:00 "")
if [[ "$smoke" == yes ]]; then
    submit smoke gpu 00:30:00 "afterok:$verify" > /dev/null
    cp "$manifest" "$root/manifest-latest.tsv"
    printf '\nSmoke test queued. Status: bash scripts/hpc/status.sh %s\n' \
        "$root/manifest-latest.tsv"
    exit 0
fi

prepare=$(submit prepare cpu "$PREPARE_TIME" "afterok:$verify")
audit=$(submit audit cpu 00:15:00 "afterok:$prepare")

# Primary Run A results first, then the decisive paired comparisons, then
# sensitivity arms, so a wall-time loss costs the least important work.
main=$(submit main gpu "$GPU_TIME" "afterok:$audit")
event=$(submit event cpu "$CPU_TIME" "afterok:$audit")
benchmark=$(submit benchmark gpu "$GPU_TIME" "afterok:$audit")
report=$(submit report cpu 00:30:00 "afterok:$main:$event:$benchmark")

treatment=$(submit treatment gpu "$GPU_TIME" "afterok:$report" '0-2')
ablation=$(submit graph-ablation gpu "$GPU_TIME" "afterok:$report" '0-11')

matching=$(submit matching cpu "$CPU_TIME" "afterok:$report" '0-5')
zero_main=$(submit main-nozero gpu "$GPU_TIME" "afterok:$ablation")
zero_event=$(submit event-nozero cpu "$CPU_TIME" "afterok:$report")
zero_benchmark=$(submit benchmark-nozero gpu "$GPU_TIME" "afterok:$zero_main")
scenarios=$(submit synthetic-scenarios gpu "$GPU_TIME" "afterok:$zero_benchmark" '0-8')
seeds=$(submit forecast-seeds gpu "$GPU_TIME" "afterok:$scenarios" '0-7')

every="$audit:$main:$event:$benchmark:$report:$treatment:$ablation:$matching"
every="$every:$zero_main:$zero_event:$zero_benchmark:$scenarios:$seeds"
aggregate=$(submit aggregate cpu 00:30:00 "afterany:$every")
submit finish cpu 00:15:00 "afterany:$aggregate" > /dev/null

cp "$manifest" "$root/manifest-latest.tsv"
cat <<EOF

Queued the Run A matrix.
  manifest  $manifest
  outputs   $run
  logs      $logs
  status    bash scripts/hpc/status.sh $root/manifest-latest.tsv
  resume    bash scripts/hpc/submit.sh --name $name --resume
EOF
