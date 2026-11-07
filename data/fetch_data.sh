#!/bin/bash

DATA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source $DATA_DIR/.env
curl --fail --show-error \
  --header "X-App-Token: $CHICAGO_DATA_APP_TOKEN" \
  --json '{"query":"SELECT time, segment_id, speed WHERE time >= '\''2025-01-01T00:00:00'\'' AND time < '\''2026-01-01T00:00:00'\''"}' \
  'https://data.cityofchicago.org/api/v3/views/4g9f-3jbs/export.csv' \
  --output chicago-speeds.csv
curl --fail --show-error \
  --header "X-App-Token: $CHICAGO_DATA_APP_TOKEN" \
  --json @- \
  --output chicago-crashes.csv \
  'https://data.cityofchicago.org/api/v3/views/85ca-t3if/export.csv' <<'JSON'
{"query":"SELECT crash_date, crash_type, damage, street_no, street_direction, street_name WHERE crash_date >= '2025-01-01T00:00:00' AND crash_date < '2026-01-01T00:00:00'"}
JSON
${PYTHON_PATH:-$(which python)} $DATA_DIR/download_street_graph.py