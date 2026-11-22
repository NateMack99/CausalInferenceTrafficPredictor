#!/bin/bash
# Fetch the Chicago segment speed history for a date range into one CSV.
#
# The portal splits this series across two datasets separated by a nine-month
# hole (docs/data.md), and no single request can carry the whole history. Months
# are fetched one at a time so a failure costs one month, and a rerun reuses the
# months already on disk. Converting the result is a separate, explicit step so
# this can never replace a good dataset on its own.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 START_DATE END_DATE [OUT_CSV]" >&2
    echo "  YYYY-MM-DD; END_DATE is exclusive" >&2
    exit 2
fi
START="$1"
END="$2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
OUT="${3:-$DATA_DIR/chicago-speeds-history.csv}"
CHUNKS="$DATA_DIR/speed-chunks"

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

mkdir -p "$CHUNKS"

# Era spans below are the *bulk* data, verified against the portal on
# 2026-11-22 -- not the dataset titles and not their min/max metadata. Both
# eras carry one stray 1047-row snapshot outside their real coverage
# (2023-09-08 and 2024-03-05); trusting those would request nine empty months.
plan=$("$FETCH_PYTHON" - "$START" "$END" <<'PY'
import datetime
import sys

ERAS = (
    ("sxs8-h27x", datetime.date(2018, 2, 28), datetime.date(2023, 9, 7)),
    ("4g9f-3jbs", datetime.date(2024, 6, 11), datetime.date(2026, 5, 1)),
)
start, end = (datetime.date.fromisoformat(a) for a in sys.argv[1:3])
if start >= end:
    raise SystemExit("START_DATE must be before END_DATE")
found = False
for dataset, era_low, era_high in ERAS:
    low, high = max(start, era_low), min(end, era_high)
    if low >= high:
        continue
    found = True
    cursor = low
    while cursor < high:
        following = (cursor.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        print(f"{dataset} {cursor} {min(following, high)}")
        cursor = following
if not found:
    raise SystemExit(f"No published segment data between {start} and {end}")
PY
)

while read -r dataset low high; do
    chunk="$CHUNKS/speeds_${low}.csv"
    if [[ -s "$chunk" ]] && [[ $(wc -l <"$chunk") -gt 1 ]]; then
        echo "have  $low -> $high  $dataset  ($(($(wc -l <"$chunk") - 1)) rows)"
        continue
    fi
    echo "get   $low -> $high  $dataset"
    curl --fail --show-error --compressed --retry 5 --retry-delay 10 --retry-all-errors \
        ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} \
        --output "$chunk.part" \
        "https://data.cityofchicago.org/resource/$dataset.csv?\$limit=10000000&\$select=time%2Csegment_id%2Cspeed&\$where=time%20%3E=%20%27${low}T00:00:00%27%20AND%20time%20%3C%20%27${high}T00:00:00%27"
    mv "$chunk.part" "$chunk"
done <<<"$plan"

echo "Combining into $OUT ..."
first=1
: >"$OUT.part"
while read -r dataset low high; do
    chunk="$CHUNKS/speeds_${low}.csv"
    [[ -s "$chunk" ]] || continue
    if [[ $first -eq 1 ]]; then
        cat "$chunk" >>"$OUT.part"
        first=0
    else
        tail -n +2 "$chunk" >>"$OUT.part"
    fi
done <<<"$plan"
mv "$OUT.part" "$OUT"
echo "Wrote $OUT ($(($(wc -l <"$OUT") - 1)) data rows)"
echo
echo "Next, convert it (this replaces data/parquet/chicago-speeds):"
echo "  uv run python scripts/convert_to_parquet.py $OUT --kind speeds --overwrite"
echo "Monthly chunks stay in $CHUNKS; delete them once the conversion succeeds."
