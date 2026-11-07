"""Write mock processed tables for running the models without real data.

uv run python scripts/make_mock_data.py --out data/mock
"""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_gml.mock import make_mock


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/mock"))
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--crashes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    tables = make_mock(args.rows, args.cols, args.days, args.crashes, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(args.out / f"{name}.parquet", index=False)
    print(f"wrote {args.out}: " + ", ".join(f"{len(t):,} {n}" for n, t in tables.items()))


if __name__ == "__main__":
    main()
