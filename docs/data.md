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

## Local paths

- `data/raw/`: unchanged downloaded files.
- `data/processed/`: cleaned, aligned Parquet tables.
- `outputs/`: figures, metrics, and checkpoints.

These directories are excluded from Git. Document download commands, source dates, study area, counts, and cleaning decisions in the data-preparation PR.
