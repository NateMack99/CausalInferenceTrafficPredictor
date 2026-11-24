#!/bin/bash
# Fixed commands for the submitted Run A matrix; no parameter search.
#
# A completed stage records a marker under $run/.done, so resubmitting the same
# run directory skips finished work instead of repeating or overwriting it.
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
task=${SLURM_ARRAY_TASK_ID:-}
marker="$run/.done/$stage${task:+-$task}"
mkdir -p "$run/.done"

# Aggregation re-reads every stage, so it must never be skipped or marked.
aggregating=no
[[ "$stage" != report && "$stage" != aggregate ]] || aggregating=yes
if [[ "$aggregating" == no && -f "$marker" ]]; then
    echo "skip $stage${task:+ task $task}: marker $marker exists"
    exit 0
fi
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
        (( task >= 0 && task < 8 ))
        model=${models[task % 4]}
        seed=$((43 + task / 4))
        uv run --no-sync python -m traffic_gml.train "${window[@]}" "${features[@]}" \
            "${training[@]}" --model "$model" --seed "$seed" --out "$run/forecast-seeds"
        ;;
    graph-ablation)
        # Three topologies on one architecture at three seeds is the principal
        # comparison; stgcn repeats it once to check the spatiotemporal pair.
        # Graph-derived static features stay off, as docs/experiments.md requires.
        arms=(
            "gcn real 42" "gcn none 42" "gcn permuted 42"
            "gcn real 43" "gcn none 43" "gcn permuted 43"
            "gcn real 44" "gcn none 44" "gcn permuted 44"
            "stgcn real 42" "stgcn none 42" "stgcn permuted 42"
        )
        (( task >= 0 && task < ${#arms[@]} ))
        read -r model graph seed <<<"${arms[task]}"
        # benchmark.py, not train.py: the two-panel result needs forecast error
        # and known-effect recovery from the same arm on the same interventions.
        uv run --no-sync python scripts/benchmark.py "${window[@]}" \
            --static-features length_m "${training[@]}" "${benchmark[@]}" \
            --models "$model" --seed "$seed" --shape decay --duration 2 \
            --placement uniform --graph-ablation "$graph" --graph-seed 42 \
            --out "$run/graph-ablation/$model-$graph-seed$seed"
        ;;
    treatment)
        # Both arms get the same neighbourhood summary, so the only difference
        # is message passing. Factual supervision only; see docs/methodology.md.
        models=(treatment_mlp treatment_gcn)
        (( task >= 0 && task < 6 ))
        model=${models[task % 2]}
        seed=$((42 + task / 2))
        uv run --no-sync python -m traffic_gml.train "${window[@]}" "${features[@]}" \
            "${training[@]}" --neighbour-hops 1 --model "$model" --seed "$seed" \
            --out "$run/treatment"
        ;;
    matching)
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
        # Benchmark 1 is --hop-decay 0: a real crash-road effect and a true zero
        # on both outer rings, so any neighbour estimate there is spurious.
        arms=(
            "null 42 --hop-decay 0 --shape decay --duration 2 --placement uniform"
            "null 43 --hop-decay 0 --shape decay --duration 2 --placement uniform"
            "null 44 --hop-decay 0 --shape decay --duration 2 --placement uniform"
            "plateau 42 --shape plateau --duration 3 --placement uniform"
            "plateau 43 --shape plateau --duration 3 --placement uniform"
            "plateau 44 --shape plateau --duration 3 --placement uniform"
            "congested 42 --shape decay --duration 2 --placement congested"
            "congested 43 --shape decay --duration 2 --placement congested"
            "congested 44 --shape decay --duration 2 --placement congested"
        )
        (( task >= 0 && task < ${#arms[@]} ))
        read -r name seed rest <<<"${arms[task]}"
        read -r -a extra <<<"$rest"
        uv run --no-sync python scripts/benchmark.py "${window[@]}" "${features[@]}" \
            "${training[@]}" "${benchmark[@]}" --seed "$seed" "${extra[@]}" \
            --out "$run/synthetic-scenarios/$name-seed$seed"
        ;;
    report)
        for name in main-seed42 event benchmark-seed42; do
            test -s "$run/$name/summary.json"
        done
        uv run --no-sync python scripts/aggregate.py --run "$run" --work "$work" \
            --out "$run/paper" --require main-seed42 event benchmark-seed42
        ;;
    aggregate)
        uv run --no-sync python scripts/aggregate.py --run "$run" --work "$work" \
            --out "$run/paper"
        ;;
    *) echo "Unknown experiment stage: $stage" >&2; exit 2 ;;
esac

[[ "$aggregating" == yes ]] || touch "$marker"
