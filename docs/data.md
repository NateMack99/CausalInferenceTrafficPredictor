# Data

## Sources

- [Chicago Traffic Crashes](https://data.cityofchicago.org/d/85ca-t3if): crash times, locations, and details.
- [Chicago Traffic Tracker — Historical Congestion Estimates by Segment](https://data.cityofchicago.org/d/4g9f-3jbs): speed estimates by road segment and 10-minute interval.
- Segment geometry: the portal's traffic-segment metadata (`chicago-traffic-segments.json`).
- Street network: OpenStreetMap via OSMnx (`roads.download_street_graph`). Record the
  download date with any result built from it.

## Coverage constraints

These bound the study period and the study area; check them before choosing either.

- The speed series estimates **arterial** segment speeds from CTA bus GPS traces. It does
  not cover every street, so a crash on an unmonitored street has no outcome to measure.
- The Traffic Tracker datasets were deprecated and the final published records are dated
  **2026-04-30**. Crashes therefore have to come from the period that overlaps the
  available speed history; there are no speeds after it.
- A missing 10-minute interval means **no estimate was published**. It is not a stopped
  road and not a zero speed, and it must never be treated as either.
- Coverage varies widely by segment, so select a dense, connected study area (see
  [Coverage subsets and map previews](#coverage-subsets-and-map-previews)) instead of
  assuming the whole city.

## Conventions

- Use one stable road-segment ID across speed data, graph nodes, and crashes.
- Keep the portal's naive local timestamps end to end. Rush hour is a local phenomenon, and
  `data.local_time` treats a naive column as already local; only tz-aware input is converted.
- Use 10-minute bins; record binning and matching tolerances.
- Store speeds in mph, with an explicit missing-observation mask.
- Keep an event ID for each crash and exclude overlapping events when required.
- Split by time/event before fitting transformations to avoid leakage.

## Collection and storage

Set up the environment first (`bash scripts/setup.sh`; see the README). Chicago's Socrata
API does not document Parquet as an export format, so fetching downloads CSVs and then
calls `data.csv_to_parquet` to write compressed, date-partitioned Parquet.

```bash
bash scripts/fetch_data.sh                  # Speeds, crashes, street graph, road reference
bash scripts/fetch_data.sh --crashes-only   # Crashes and their Parquet conversion
```

Convert a CSV already on disk without fetching again:

```bash
uv run python scripts/convert_to_parquet.py data/chicago-speeds.csv --kind speeds
uv run python scripts/convert_to_parquet.py data/chicago-crashes.csv --kind crashes
```

`fetch_data.sh` requests a fixed calendar range from the portal; edit the `$where` bounds
in the script to move the study period, keeping it inside the available speed history.
`--overwrite` refreshes an existing conversion, and fetching passes it automatically.
Replacement is staged: a completed conversion replaces the previous dataset including
obsolete partitions, a failed one leaves it intact.

Converted datasets live in `data/parquet/chicago-speeds/` and
`data/parquet/chicago-crashes/`, partitioned as `date=2025-01-01/`. `_conversion.json`
records row counts, the timestamp range, and the source file. Read a time range with
`data.load_speed_dataset(path, (start, end))`, which prunes whole `date` partitions before
opening any file; `[start, end)` in the dataset's own naive local time.

A full fetch also writes the speed-road reference: `speed_roads.parquet` (OSM
edge-to-segment mapping), `speed_segments.parquet` (per-segment coverage and observed
maxima), `speed_road_network.graphml` (the speed-observed starter subgraph), and the
readable 80%-coverage list `data/dense_80pct_segments.csv`. Rebuild them with
`uv run python scripts/prepare_roads.py`. Edge-to-segment matching requires a matching
street name, a heading within 45°, and a 40-metre corridor; coverage uses 144 nominal
bins per day.

Sources: [Socrata export formats](https://support.socrata.com/hc/en-us/articles/43491231777047-Using-the-SODA3-API),
[Arrow partitioned datasets](https://arrow.apache.org/docs/python/dataset.html).

## Processed tables

The model contract. Built by `events.processed_tables` from the collection and graph
outputs, written to `data/processed/` by `events.write_processed`, and read back by
`data.load_tables`:

| Table | Required columns |
|---|---|
| `speeds.parquet` | `segment_id`, `time`, `speed` |
| `crashes.parquet` | `crash_record_id`, `crash_time`, `segment_id` |
| `segments.parquet` | `segment_id`, plus optional `length_m`, `degree`, `betweenness`, `speed_mph` |
| `edges.parquet` | `src`, `dst` |

A missing speed must arrive as NaN — not the `-1` Chicago publishes, and not a zero. `time`
is read as naive local, matching what `csv_to_parquet` writes; handing over UTC wall-clock
instead would put every rush-hour feature hours out of phase. Crashes need a confident match
to one segment; drop the rest upstream rather than guessing. An edge naming a segment absent
from `segments.parquet` is dropped, while a segment with no speed rows is kept as a node with
an all-missing series.

The partitioned files in `data/parquet/` are **staging data**, not this contract: they keep
the portal's raw column names and wall-clock timestamps, and `events.processed_tables`
builds the four tables above from them.

## Coverage subsets and map previews

An exploratory view of one window, used to check coverage and crash matching. The GraphML
here describes **OSM road edges**; it is not the road-as-node PyG representation the models
consume, which `events.build_windows` builds from the processed tables.

```python
from traffic_gml.graph import build_graph
from traffic_gml.plot import plot_graph

G = build_graph(
    crash_window=("2025-01-01 12:00", "2025-01-01 13:00"),
    speed_window=("2025-01-01 12:00", "2025-01-01 13:00"),
    only_80_percent=True,  # False uses all speed-observed roads
)
plot_graph(G)
```

Both windows are `[start, end)` in the portal's naive local time. The GraphML, a
crash-assignment table, and the PNG go to `outputs/`, with the window timestamps in their
filenames.

`only_80_percent=True` keeps edges whose traffic segment has `coverage_fraction >= 0.8`,
where coverage is `distinct intervals with a valid speed / (calendar days × 144)`:
negative and missing speeds do not count, zero does, and duplicates inside one interval
count once. For a full year that is at least 42,048 of 52,560 nominal intervals. The
denominator is nominal and does not adjust for daylight-saving changes. `False` includes
every road with any valid speed.

Coverage and observed maxima are computed over the **full reference period**, not the
window you request, so an included road can still have missing speeds inside that window —
and any reference used in a prediction experiment has to be rebuilt from training data
only, or it leaks.

Congestion is `1 - mean speed / observed maximum`, clipped to `[0, 1]`: a visualization
proxy, not a model input and not an effect estimate. Purple marks crashes, gray means no
speed in the window. Crash direction remains unresolved.

## Local paths

- `data/`: unchanged downloaded files (`chicago-speeds.csv`, `chicago-crashes.csv`,
  `chicago-traffic-segments.json`, the street-network GraphML) and the speed-road reference.
- `data/parquet/`: converted, partitioned staging datasets.
- `data/processed/`: the four contract tables above.
- `outputs/`: figures, metrics, and checkpoints.

These directories are excluded from Git. Document download commands, source dates, study
area, counts, and cleaning decisions in the data-preparation PR.
