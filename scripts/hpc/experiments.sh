#!/bin/bash
# Fixed commands for the submitted experiment matrix; no parameter search.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through scripts/hpc/run.sh}"
stage=$1
work=$2
run=$3
data="$work/processed"
window=(--data "$data" --history 12 --horizon 6 --stride 1 --exposure-hops 2)
features=(--static-features length_m degree betweenness)
training=(--epochs 20 --patience 4 --hidden 64 --batch-size 64 --device cuda)
benchmark=(--events 200 --controls 5 --magnitude 4 12 --hops 2 --hop-decay 0.45)
event=(--hops 2 --controls 5 --leads 3)
mkdir -p "$run"

if [[ "$stage" != audit ]]; then
    sha256sum -c "$run/processed.sha256"
fi

case "$stage" in
    audit)
        test -s "$work/audit/audit.json"
        sha256sum -c "$work/graph.sha256"
        for table in speeds crashes segments edges; do
            test -s "$data/$table.parquet"
        done
        sha256sum "$data/"*.parquet > "$run/processed.sha256"
        for name in audit.json coverage_funnel.csv monthly_coverage.csv neighbour_coverage.csv; do
            test -s "$work/audit/$name"
            printf '\n%s\n' "$name"
            cat "$work/audit/$name"
        done
        ;;
    main|main-nozero)
        extra=()
        [[ "$stage" != main-nozero ]] || extra=(--drop-zero-speed)
        uv run --no-sync python scripts/experiment.py "${window[@]}" "${features[@]}" \
            "${training[@]}" --seed 42 --controls 5 --effect-model gcn \
            "${extra[@]}" --out "$run/$stage-seed42"
        ;;
    event|event-nozero)
        extra=()
        [[ "$stage" != event-nozero ]] || extra=(--drop-zero-speed)
        uv run --no-sync python scripts/event_study.py "${window[@]}" \
            "${event[@]}" --match-hops 0 --seed 42 "${extra[@]}" --out "$run/$stage"
        ;;
    benchmark|benchmark-nozero)
        extra=()
        [[ "$stage" != benchmark-nozero ]] || extra=(--drop-zero-speed)
        uv run --no-sync python scripts/benchmark.py "${window[@]}" "${features[@]}" \
            "${training[@]}" "${benchmark[@]}" --shape decay --duration 2 \
            --placement uniform --seed 42 "${extra[@]}" --out "$run/$stage-seed42"
        ;;
    forecast-seeds)
        models=(mlp gcn tgru stgcn)
        task=${SLURM_ARRAY_TASK_ID:?}
        (( task >= 0 && task < 8 ))
        model=${models[task % 4]}
        seed=$((43 + task / 4))
        uv run --no-sync python -m traffic_gml.train "${window[@]}" "${features[@]}" \
            "${training[@]}" --model "$model" --seed "$seed" --out "$run/forecast-seeds"
        ;;
    graph-ablation)
        models=(mlp tgru gcn stgcn gcn stgcn)
        task=${SLURM_ARRAY_TASK_ID:?}
        (( task >= 0 && task < 6 ))
        model=${models[task]}
        graph=real
        (( task < 4 )) || graph=permuted
        uv run --no-sync python -m traffic_gml.train "${window[@]}" \
            --static-features length_m "${training[@]}" --model "$model" --seed 42 \
            --graph-ablation "$graph" --graph-seed 42 --out "$run/graph-length/$graph"
        ;;
    matching)
        task=${SLURM_ARRAY_TASK_ID:?}
        (( task >= 0 && task < 6 ))
        if (( task < 3 )); then
            extra=(--random-controls --seed "$((42 + task))")
            name="random-seed$((42 + task))"
        elif (( task < 5 )); then
            extra=(--match-hops "$((task - 2))" --seed 42)
            name="matched-hop$((task - 2))"
        else
            extra=(--match-target --seed 42)
            name=matched-target
        fi
        uv run --no-sync python scripts/event_study.py "${window[@]}" "${event[@]}" \
            "${extra[@]}" --out "$run/matching/$name"
        ;;
    synthetic-scenarios)
        task=${SLURM_ARRAY_TASK_ID:?}
        (( task >= 0 && task < 3 ))
        case "$task" in
            # Benchmark 1: a real crash-road effect and a true zero on both
            # outer rings, so any neighbour estimate here is spurious.
            0) extra=(--shape decay --duration 2 --placement uniform --hop-decay 0)
               name=null ;;
            1) extra=(--shape plateau --duration 3 --placement uniform)
               name=plateau ;;
            2) extra=(--shape decay --duration 2 --placement congested)
               name=congested ;;
        esac
        uv run --no-sync python scripts/benchmark.py "${window[@]}" "${features[@]}" \
            "${training[@]}" "${benchmark[@]}" --seed 42 "${extra[@]}" \
            --out "$run/synthetic-scenarios/$name"
        ;;
    report)
        for name in main-seed42 event benchmark-seed42; do
            test -s "$run/$name/summary.json"
        done
        {
            for file in \
                "$work/audit/audit.json" \
                "$work/audit/monthly_coverage.csv" \
                "$work/audit/neighbour_coverage.csv" \
                "$run/main-seed42/results.csv" \
                "$run/main-seed42/effect_estimate.csv" \
                "$run/event/effect_against_placebo.csv" \
                "$run/event/covariate_balance.csv" \
                "$run/event/effect_fixed_cohort.csv" \
                "$run/benchmark-seed42/forecast.csv" \
                "$run/benchmark-seed42/effect_recovery.csv"; do
                printf '\n%s\n' "$file"
                cat "$file"
            done
        } > "$run/primary-diagnostics.txt"
        cat "$run/primary-diagnostics.txt"
        ;;
    *) echo "Unknown experiment stage: $stage" >&2; exit 2 ;;
esac
