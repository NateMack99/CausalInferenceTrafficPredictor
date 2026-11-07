# Scripts

Add small `argparse` entry points here as the corresponding roadmap PRs land.
Keep reusable logic in `src/traffic_gml/`.

- `bash scripts/setup.sh`: sync the Python environment, verify a portal token, and optionally fetch data.
- `bash scripts/fetch_data.sh [--crashes-only]`: fetch CSVs and prepare Parquet/reference files.
- `uv run python scripts/convert_to_parquet.py --help`: convert existing CSVs.
- `uv run python scripts/prepare_roads.py`: rebuild the speed-road reference.
- `uv run python scripts/build_graph.py --help`: build a window graph.
- `uv run python scripts/plot_graph.py [--graph PATH] [--show]`: plot a saved graph or an example.
