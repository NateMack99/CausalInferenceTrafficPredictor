#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
cd "$DATA_DIR"

if [[ -f "$DATA_DIR/.env" ]]; then
  source "$DATA_DIR/.env"
fi
TOKEN_ARGS=()
if [[ -n "${CHICAGO_DATA_APP_TOKEN:-}" ]]; then
  TOKEN_ARGS=(--header "X-App-Token: $CHICAGO_DATA_APP_TOKEN")
fi
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
curl --fail --show-error --compressed --retry 3 --retry-delay 5 --retry-all-errors \
  ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} \
  'https://data.cityofchicago.org/resource/4g9f-3jbs.csv?$limit=50000000&$select=time%2Csegment_id%2Cspeed&$where=time%20between%20%272025-01-01T00%3A00%3A00%27%20and%20%272026-01-01T00%3A00%3A00%27' \
  --output chicago-speeds.csv
"$FETCH_PYTHON" "$SCRIPT_DIR/convert_to_parquet.py" chicago-speeds.csv --kind speeds --overwrite
fi
curl --fail --show-error --compressed --retry 3 --retry-delay 5 --retry-all-errors \
  ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} \
  --output chicago-crashes.csv \
  'https://data.cityofchicago.org/resource/85ca-t3if.csv?$limit=50000000&$select=crash_record_id%2Ccrash_date%2Ccrash_type%2Cdamage%2Cstreet_no%2Cstreet_direction%2Cstreet_name%2Clatitude%2Clongitude&$where=crash_date%20between%20%272025-01-01T00%3A00%3A00%27%20and%20%272026-01-01T00%3A00%3A00%27'
"$FETCH_PYTHON" "$SCRIPT_DIR/convert_to_parquet.py" chicago-crashes.csv --kind crashes --overwrite
if [[ "${1:-}" != "--crashes-only" ]]; then
    curl --fail --show-error --compressed --retry 3 --retry-delay 5 --retry-all-errors \
      'https://data.cityofchicago.org/resource/n4j6-wkkf.json?$limit=5000' \
      --output chicago-traffic-segments.json
    "$FETCH_PYTHON" "$SCRIPT_DIR/download_street_graph.py"
    "$FETCH_PYTHON" "$SCRIPT_DIR/prepare_roads.py"
fi
