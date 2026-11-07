#!/bin/bash

source .env
curl --fail --show-error \
  --header "X-App-Token: $SOCRATA_APP_TOKEN" \
  --json '{"query":"SELECT time, segment_id, speed WHERE time >= '\''2025-01-01T00:00:00'\'' AND time < '\''2026-01-01T00:00:00'\''"}' \
  'https://data.cityofchicago.org/api/v3/views/4g9f-3jbs/export.csv' \
  --output chicago-speeds.csv
curl --fail --show-error \
  --header "X-App-Token: $SOCRATA_APP_TOKEN" \
  --json '{"query":"SELECT * WHERE time >= '\''2025-01-01T00:00:00'\'' AND time < '\''2026-01-01T00:00:00'\''"}' \
  'https://data.cityofchicago.org/api/v3/views/85ca-t3if/export.csv' \
  --output chicago-crashes.csv
# wget https://data.cityofchicago.org/api/v3/views/85ca-t3if/query.json
# mv query.json crashes.json
