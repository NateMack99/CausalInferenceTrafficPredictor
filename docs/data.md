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

## Which speed dataset to use

The portal splits one segment-level series across three datasets. Checked
2026-11-22 against the Socrata metadata and per-day samples:

| Dataset | Real bulk span | Rows | Verdict |
|---|---|---:|---|
| [`4g9f-3jbs`](https://data.cityofchicago.org/d/4g9f-3jbs) | 2024-06-11 to 2026-04-30 | 101.1M | **Primary** (run A). |
| [`sxs8-h27x`](https://data.cityofchicago.org/d/sxs8-h27x) | 2018-02-28 to 2023-09-06 | 274.9M | **Secondary** (run B). |
| [`77hq-huss`](https://data.cityofchicago.org/d/77hq-huss) | 2011-08-02 to 2018-05-03 | 19.6M | **Rejected**, see below. |

Row counts are `count(*)` from the portal. Do not use the column metadata's
`non_null`: it understates `4g9f-3jbs` as 86.9M because that archive was still
being appended to when the cache was built. `sxs8-h27x`, static since 2025,
matches.

Those are the spans of the *bulk* data and are not what the dataset titles or
the `min(time)`/`max(time)` metadata report. Each modern archive carries one
stray 1047-row snapshot outside its real coverage, on 2023-09-08 and
2024-03-05, which makes the published extremes look nine months wider than the
data actually is. `sxs8-h27x` really stops on 2023-09-06 (a part day), and
`4g9f-3jbs` really starts at 2024-06-11T13:01. Requesting the metadata range
returns nine months of empty results; `scripts/fetch_speeds_history.sh` encodes
the verified spans instead.

`sxs8-h27x` is a true continuation, not a parallel series: same `time` /
`segment_id` / `speed` columns, same `-1` missing sentinel, and the same **1047
segment IDs with byte-identical geometry** (street, from/to, heading, length,
start coordinate) as the era we use. Observed-bin coverage holds at 55-70% from
2018 to 2026 with no drift, so the graph and `speed_roads.parquet` carry over
unchanged. Its three extra `:@computed_region_*` columns are not selected.

Outages are the one thing to check before using a sub-window of it. A full month
runs about 4.5M rows; June 2021 holds 2.53M, so roughly 44% of that month is
gone, including whole days such as 2021-06-09, and March 2021 is down about 19%.
Every other month sampled from 2018 to 2023 is within normal range. Missing bins
arrive as NaN and fail the usual eligibility screen, so they cost sample size
rather than correctness, but do not read a monthly series across June 2021.

`77hq-huss` is rejected on three counts: a different schema (`last_update` and
`traffic`, no `speed`, no geometry or street names); **2015, 2016 and 2017 are
absent entirely** (rows land only in 2013, 2014 and 2018, with 2011-2012 a
rounding error); and its substantive years predate citywide crash reporting, so
there is almost nothing to pair speeds with.

Two further limits bound any expansion:

- **A nine-month hole**, 2023-09-07 to 2024-06-10, that no segment-level
  dataset covers. The portal's other Traffic Tracker datasets are region-level.
  The two archives are therefore analysed as separate runs rather than one
  panel, so no event window straddles it.
- **Crash history effectively starts in 2018.** Chicago's crash table runs from
  2013 but reaches citywide coverage only as the e-crash rollout completes:
  2 rows in 2013, 6 in 2014, 9,834 in 2015, 44,297 in 2016, 83,786 in 2017, then
  110-119k a year from 2018. Pre-2018 speeds have no usable crash sample.

So the usable total is 7.4 years of actual observation, 361.8M speed rows,
split 5.52 years before the hole and 1.88 years after it. `docs/hpc-run.md`
section 4 carries the run plan, the per-run memory requests and the staging
commands.

**A single request truncates silently.** `scripts/fetch_data.sh` asks for
`$limit=50000000` in one CSV, and calendar 2025 alone holds 53.6M rows, so the
extract we first built stops mid-December with no error; `_conversion.json`
records the round `"rows": 50000000` as the only clue. Use
`scripts/fetch_speeds_history.sh START END` for anything wider: it fetches one
month per request, reuses months already on disk, skips the hole, and selects
the archive per month. Check `_conversion.json` afterwards and treat a round
50,000,000 as a truncated extract.

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
