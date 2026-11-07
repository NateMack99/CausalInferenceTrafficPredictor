# Data

## Public sources

- [Chicago Traffic Crashes](https://data.cityofchicago.org/d/85ca-t3if): event times, locations, and crash details.
- [Chicago Historical Traffic Speeds](https://data.cityofchicago.org/d/4g9f-3jbs): speed observations by road segment and time; check coverage before selecting a study window.
- Road-segment geometry/reference data: record the exact source and version used in PR #1/#2.

## Conventions

- Use one stable road-segment ID across speed data, graph nodes, and crashes.
- Convert input timestamps to UTC; retain original local timestamps when needed for validation.
- Use 10-minute bins; record binning and matching tolerances.
- Store speeds in mph, with an explicit missing-observation mask.
- Keep an event ID for each crash and exclude overlapping events when required.
- Split by time/event before fitting transformations to avoid leakage.

## Processed tables

Written by PR #1/#2 into `data/processed/`, read by `traffic_gml.events.load_tables`:

| Table | Required columns |
|---|---|
| `speeds.parquet` | `segment_id`, `time`, `speed` |
| `crashes.parquet` | `crash_record_id`, `crash_time`, `segment_id` |
| `segments.parquet` | `segment_id`, plus optional `length`, `degree`, `betweenness` |
| `edges.parquet` | `src`, `dst` |

A missing speed must arrive as NaN — not the `-1` Chicago publishes, and not a zero. `time` is read as UTC even when tz-naive, so delivering local wall time puts every rush-hour feature hours out of phase. Crashes need a confident match to one segment; drop the rest upstream rather than guessing. An edge naming a segment absent from `segments.parquet` is dropped, while a segment with no speed rows is kept as a node with an all-missing series.

## Local paths

- `data/raw/`: unchanged downloaded files.
- `data/processed/`: cleaned, aligned Parquet tables.
- `outputs/`: figures, metrics, and checkpoints.

These directories are excluded from Git. Document download commands, source dates, study area, counts, and cleaning decisions in the data-preparation PR.

## Collection and map previews

Run `bash scripts/setup.sh` to sync dependencies, configure the portal token, and
optionally fetch data. See [collection commands](data_collection.md) and
[window graph previews](window_graphs.md). The partitioned files in `data/parquet/`
retain portal wall-clock timestamps; they are staging data, **not** the UTC
`data/processed/` model contract above. Window GraphML files describe OSM road
edges and are not yet the road-as-node PyG representation used by the models.
