#!/bin/bash
# Queue the fixed matrix with Slurm dependencies and at most two GPUs at once.
# Run from the frozen code worktree. Arguments are absolute paths and job IDs.
set -euo pipefail
work=$1
run=$2
manifest=$3
logs=$4
prepare=$5
verify=$6
[[ "$prepare" =~ ^[0-9]+$ && "$verify" =~ ^[0-9]+$ ]]
[[ ! -e "$manifest" ]] || { echo "Manifest exists; refusing duplicate submissions" >&2; exit 1; }
mkdir -p "$run" "$logs"
printf 'stage\tjob_id\tdependency\tlog\n' > "$manifest"
printf 'verify\t%s\tcompleted-or-running\t%s/verify-%s.log\n' "$verify" "$logs" "$verify" >> "$manifest"
printf 'prepare\t%s\tafterok:%s\t%s/prepare-%s.log\n' "$prepare" "$verify" "$logs" "$prepare" >> "$manifest"

submit() {
    local stage=$1 partition=$2 memory=$3 duration=$4 dependencies=$5 array=${6:-}
    local extra=() command=() job
    [[ "$partition" != gpu ]] || extra+=(--gres=gpu:a100:1)
    [[ -z "$array" ]] || extra+=(--array="$array")
    if [[ "$stage" == finish ]]; then
        command=(bash scripts/hpc/status.sh "$manifest")
    else
        command=(bash scripts/hpc/experiments.sh "$stage" "$work" "$run")
    fi
    job=$(sbatch --parsable --account=nssac_students --partition="$partition" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem="$memory" --time="$duration" \
        --job-name="traffic-$stage" --dependency="$dependencies" --kill-on-invalid-dep=yes \
        --output="$logs/$stage-%A_%a.log" ${extra[@]+"${extra[@]}"} \
        scripts/hpc/run.sh "$logs/meta" "${command[@]}")
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s/%s-%s_%%a.log\n' \
        "$stage" "$job" "$dependencies" "$logs" "$stage" "$job" >> "$manifest"
    printf 'Queued %s: %s\n' "$stage" "$job" >&2
    printf '%s\n' "$job"
}

audit=$(submit audit bii 4G 00:15:00 "afterok:$prepare")
main=$(submit main gpu 128G 24:00:00 "afterok:$audit")
event=$(submit event bii 128G 04:00:00 "afterok:$audit")
benchmark=$(submit benchmark gpu 128G 24:00:00 "afterok:$audit")
report=$(submit report bii 4G 00:15:00 "afterok:$main:$event:$benchmark")
zero_main=$(submit main-nozero gpu 128G 24:00:00 "afterok:$report")
zero_event=$(submit event-nozero bii 128G 04:00:00 "afterok:$report")
zero_benchmark=$(submit benchmark-nozero gpu 128G 24:00:00 "afterok:$report")
matching=$(submit matching bii 128G 04:00:00 "afterok:$report" '0-5%1')
seeds=$(submit forecast-seeds gpu 128G 24:00:00 "afterok:$zero_main:$zero_benchmark" '0-7%2')
graph=$(submit graph-ablation gpu 128G 24:00:00 "afterok:$seeds" '0-5%2')
scenarios=$(submit synthetic-scenarios gpu 128G 24:00:00 "afterok:$graph" '0-2%2')
submit finish bii 4G 00:15:00 \
    "afterany:$audit:$main:$event:$benchmark:$report:$zero_main:$zero_event:$zero_benchmark:$matching:$seeds:$graph:$scenarios" \
    > /dev/null
printf 'Job manifest: %s\n' "$manifest"
