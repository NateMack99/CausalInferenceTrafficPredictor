#!/bin/bash
# Recover the road network without a live Overpass query.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a CPU allocation}"
data=$1
boundary=$2
base=https://download.bbbike.org/osm/bbbike/Chicago
mkdir -p "$data/osm-extract"
extract="$data/osm-extract"
if [[ ! -f "$extract/Chicago.osm.gz" ]]; then
    curl --fail --location --retry 3 --max-time 120 \
        "$base/CHECKSUM.txt" -o "$extract/publisher-checksums.txt"
    curl --fail --location --retry 3 --connect-timeout 30 --max-time 1800 \
        "$base/Chicago.osm.gz" -o "$extract/Chicago.osm.gz.part"
    gzip -t "$extract/Chicago.osm.gz.part"
    expected=$(awk '$2 == "Chicago.osm.gz" {print $1}' "$extract/publisher-checksums.txt")
    [[ "$expected" =~ ^[0-9a-f]{32}$ ]]
    printf '%s  %s\n' "$expected" "$extract/Chicago.osm.gz.part" | md5sum --check -
    mv "$extract/Chicago.osm.gz.part" "$extract/Chicago.osm.gz"
fi
sha256sum "$extract/Chicago.osm.gz" > "$extract/source.sha256"
cp "$boundary" "$extract/chicago-boundary.geojson"
uv run --no-sync python scripts/download_street_graph.py --data-dir "$data" \
    --osm-xml "$extract/Chicago.osm.gz" --boundary "$extract/chicago-boundary.geojson" \
    --source-url "$base/Chicago.osm.gz" --buffer-m 100
