# Data storage

Chicago's Socrata API does not document Parquet as an export format. Fetching
downloads CSVs and then calls `traffic_gml.data.csv_to_parquet` to
create compressed, date-partitioned Parquet datasets.

Run `bash scripts/setup.sh` first to sync the locked Python environment.

```bash
bash scripts/fetch_data.sh                  # Speeds, crashes, and street graph
bash scripts/fetch_data.sh --crashes-only   # Crashes and their Parquet conversion
```

Convert existing downloads without fetching again:

```bash
uv run python scripts/convert_to_parquet.py data/chicago-speeds.csv --kind speeds
uv run python scripts/convert_to_parquet.py data/chicago-crashes.csv --kind crashes
```

Use `--overwrite` to refresh an existing conversion. A completed conversion
replaces the previous dataset, including obsolete partitions; a failed conversion
preserves it. Fetching uses this option automatically.

Datasets are stored under `data/parquet/chicago-speeds/` and
`data/parquet/chicago-crashes/`, with partitions such as
`date=2025-01-01/`. `_conversion.json` records row counts, timestamp range, and
source file. The converter streams CSV chunks rather than loading the full file.

Full fetching also saves `speed_roads.parquet` (edge-to-segment mapping),
`speed_segments.parquet` (availability and observed maxima), and
`speed_road_network.graphml` (the speed-observed starter subgraph).
Rebuild them with `uv run python scripts/prepare_roads.py`. Matching uses street names,
direction, and a 40-meter corridor tolerance; coverage uses 144 nominal bins/day.

Generated graphs, plots, and crash assignments go in `outputs/`.
The readable 80% coverage list is `data/dense_80pct_segments.csv`.

### Example
Query a one-hour window with partition pruning:

```python
from datetime import datetime
import pyarrow.dataset as ds

speeds = ds.dataset("data/parquet/chicago-speeds", format="parquet", partitioning="hive")
window = speeds.to_table(
    filter=(
        (ds.field("date") == "2025-01-01")
        & (ds.field("timestamp") >= datetime(2025, 1, 1, 12))
        & (ds.field("timestamp") < datetime(2025, 1, 1, 13))
    )
)
```

Sources: [Socrata export formats](https://support.socrata.com/hc/en-us/articles/43491231777047-Using-the-SODA3-API),
[Arrow partitioned datasets](https://arrow.apache.org/docs/python/dataset.html).
