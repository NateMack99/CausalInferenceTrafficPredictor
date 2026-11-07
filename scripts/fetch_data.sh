#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
cd "$DATA_DIR"

source "$DATA_DIR/.env"
if [[ -n "${PYTHON_PATH:-}" ]]; then
    FETCH_PYTHON="$PYTHON_PATH"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    FETCH_PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    echo "Run bash scripts/setup.sh first." >&2
    exit 1
fi
if ! "$FETCH_PYTHON" -c 'import pandas, pyarrow' >/dev/null 2>&1; then
    echo "Run uv sync --locked --python 3.12 before fetching." >&2
    exit 1
fi
if [[ "${1:-}" != "--crashes-only" ]]; then
curl --fail --show-error --compressed \
  --header "X-App-Token: $CHICAGO_DATA_APP_TOKEN" \
  --json '{"query":"SELECT time, segment_id, speed WHERE time >= '\''2025-01-01T00:00:00'\'' AND time < '\''2026-01-01T00:00:00'\''"}' \
  'https://data.cityofchicago.org/api/v3/views/4g9f-3jbs/export.csv' \
  --output chicago-speeds.csv
"$FETCH_PYTHON" "$SCRIPT_DIR/convert_to_parquet.py" chicago-speeds.csv --kind speeds --overwrite
fi
curl --fail --show-error --compressed \
  --header "X-App-Token: $CHICAGO_DATA_APP_TOKEN" \
  --json @- \
  --output chicago-crashes.csv \
  'https://data.cityofchicago.org/api/v3/views/85ca-t3if/export.csv' <<'JSON'
{"query":"SELECT crash_record_id, crash_date, crash_type, damage, street_no, street_direction, street_name, latitude, longitude WHERE crash_date >= '2025-01-01T00:00:00' AND crash_date < '2026-01-01T00:00:00'"}
JSON
"$FETCH_PYTHON" "$SCRIPT_DIR/convert_to_parquet.py" chicago-crashes.csv --kind crashes --overwrite
if [[ "${1:-}" != "--crashes-only" ]]; then
    curl --fail --show-error --compressed \
      'https://data.cityofchicago.org/resource/n4j6-wkkf.json?$limit=5000' \
      --output chicago-traffic-segments.json
    "$FETCH_PYTHON" "$SCRIPT_DIR/download_street_graph.py"
    "$FETCH_PYTHON" "$SCRIPT_DIR/prepare_roads.py"
fi
