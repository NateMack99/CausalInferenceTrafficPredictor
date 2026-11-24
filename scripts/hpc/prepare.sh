#!/bin/bash
# Fixed primary data specification; reruns reuse completed inputs only.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a CPU allocation}"
work=$1
data="$work/data"
graph_dir="$work/graph"
mkdir -p "$data" "$graph_dir"
cp -n data/chicago-traffic-segments.json "$data/"
if [[ ! -f "$data/parquet/chicago-speeds-history/_conversion.json" ]]; then
    uv run --no-sync python scripts/fetch_window.py --data-dir "$data" \
        --window 2024-07-01 2026-04-01 --kinds speeds
fi
# Include the preceding day so a boundary crash can still contaminate history.
if [[ ! -f "$data/parquet/chicago-crashes/_conversion.json" ]]; then
    uv run --no-sync python scripts/fetch_window.py --data-dir "$data" \
        --window 2024-06-30 2026-04-01 --kinds crashes
fi
if [[ ! -f "$data/chicago_road_network.graphml" ]]; then
    if ! uv run --no-sync python scripts/download_street_graph.py --data-dir "$data"; then
        uv run --no-sync python scripts/download_street_graph.py --data-dir "$data" \
            --overpass-url https://overpass.private.coffee/api
    fi
fi
if [[ ! -f "$data/speed_road_network.graphml" ]]; then
    uv run --no-sync python scripts/prepare_roads.py --data-dir "$data"
fi
shopt -s nullglob
graphs=("$graph_dir/"*_segments.graphml)
if [[ ${#graphs[@]} -eq 0 ]]; then
    uv run --no-sync python scripts/build_graph.py --no-fetch --data-dir "$data" \
        --output-dir "$graph_dir" --crash-window 2024-06-30 2026-04-01 \
        --speed-window 2024-07-01 2026-04-01
    graphs=("$graph_dir/"*_segments.graphml)
fi
[[ ${#graphs[@]} -eq 1 ]] || { echo "Expected exactly one segment graph" >&2; exit 1; }
graph=${graphs[0]}
crashes="${graph%_segments.graphml}_crash_segments.csv"
sha256sum "$graph" "$crashes" > "$work/graph.sha256"
uv run --no-sync python scripts/experiment.py --data-only \
    --speeds "$data/parquet/chicago-speeds-history" --graph "$graph" --crashes "$crashes" \
    --window 2024-07-01 2026-04-01 --static-features length_m degree betweenness \
    --history 12 --horizon 6 --exposure-hops 2 --stride 1 --controls 5 --seed 42 \
    --processed "$work/processed" --out "$work/audit"
