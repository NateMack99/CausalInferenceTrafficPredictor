#!/bin/bash
# Fixed primary data specification; reruns reuse completed inputs only.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a CPU allocation}"
work=$1
# The datasets are read where they already live. Nothing is copied or linked,
# and a fetch only writes a dataset that is absent, never over a valid one.
data=${2:-data}
graph_dir="$work/graph"
speeds="$data/parquet/chicago-speeds-history"
crashes="$data/parquet/chicago-crashes"
mkdir -p "$work" "$graph_dir"

valid() { uv run --no-sync python scripts/check_dataset.py "$@" >/dev/null 2>&1; }

if ! valid --speeds "$speeds"; then
    uv run --no-sync python scripts/fetch_window.py --data-dir "$data" \
        --window 2024-07-01 2026-04-01 --kinds speeds
fi
# Include the preceding day so a boundary crash can still contaminate history.
if ! valid --crashes "$crashes"; then
    uv run --no-sync python scripts/fetch_window.py --data-dir "$data" \
        --window 2024-06-30 2026-04-01 --kinds crashes
fi
uv run --no-sync python scripts/check_dataset.py --speeds "$speeds" --crashes "$crashes" \
    --record "$work/data-provenance.json"

if [[ ! -f "$data/chicago_road_network.graphml" ]]; then
    echo "WARNING: no frozen road network in $data; rebuilding from live OSM." >&2
    echo "Node ids will differ from Run A. See docs/hpc-run.md step 3." >&2
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
crash_segments="${graph%_segments.graphml}_crash_segments.csv"
sha256sum "$graph" "$crash_segments" > "$work/graph.sha256"
uv run --no-sync python scripts/experiment.py --data-only \
    --speeds "$speeds" --graph "$graph" --crashes "$crash_segments" \
    --window 2024-07-01 2026-04-01 --static-features length_m degree betweenness \
    --history 12 --horizon 6 --exposure-hops 2 --stride 1 --controls 5 --seed 42 \
    --processed "$work/processed" --out "$work/audit"
