"""Mock processed tables, so the models can run before the real data lands.

A development fixture, not a research artefact, and deliberately not PR #8's
`synthetic.py`. Nothing here is evidence about Chicago traffic: the roads are
coupled only through the injected spillover, which crash-free masking then
removes from training, so it carries almost no spatial signal for a graph
model to find. Do not read a graph-versus-no-graph result off it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FREQ, local_time


def make_mock(
    rows: int = 6,
    cols: int = 6,
    days: int = 60,
    crashes: int = 200,
    missing_rate: float = 0.15,
    shock: float = 8.0,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """A grid of one-way roads, a diurnal speed series with gaps, and crashes."""
    rng = np.random.default_rng(seed)
    ids = np.arange(rows * cols, dtype="int64") + 1
    node = ids.reshape(rows, cols)
    links = [(node[r, c], node[r, c + 1]) for r in range(rows) for c in range(cols - 1)]
    links += [(node[r, c], node[r + 1, c]) for r in range(rows - 1) for c in range(cols)]
    edges = pd.DataFrame(links, columns=["src", "dst"]).astype("int64")

    times = pd.date_range("2025-01-06", periods=days * 144, freq=FREQ)
    local = local_time(times)
    hour = np.asarray(local.hour) + np.asarray(local.minute) / 60
    rush = np.exp(-0.5 * ((hour - 8) / 1.4) ** 2) + np.exp(-0.5 * ((hour - 17.5) / 1.6) ** 2)
    weekday = (np.asarray(local.day_of_week) < 5).astype("float32")
    dip = (weekday * rush)[:, None]
    values = rng.uniform(22, 34, ids.size) - rng.uniform(6, 14, ids.size) * dip

    noise = rng.normal(0, 1.2, values.shape)
    for t in range(1, len(times)):
        noise[t] += 0.7 * noise[t - 1]
    values = values + noise

    neighbours: dict[int, list[int]] = {int(i): [] for i in range(ids.size)}
    for src, dst in edges.to_numpy() - 1:
        neighbours[int(src)].append(int(dst))
        neighbours[int(dst)].append(int(src))
    crash_road = rng.integers(0, ids.size, crashes)
    crash_bin = rng.integers(12, len(times) - 12, crashes)
    for road, bin_ in zip(crash_road, crash_bin, strict=True):
        for step in range(6):
            hit = shock * 0.7**step
            values[bin_ + step, road] -= hit
            values[bin_ + step, neighbours[int(road)]] -= hit * 0.45

    frame = pd.DataFrame(np.clip(values, 2, 45).astype("float32"), index=times, columns=ids)
    speeds = (
        frame.where(rng.random(values.shape) >= missing_rate)
        .rename_axis(index="time", columns="segment_id")
        .reset_index()
        .melt(id_vars="time", var_name="segment_id", value_name="speed")
        .dropna(subset=["speed"])
        .sort_values(["time", "segment_id"], ignore_index=True)
    )
    speeds["segment_id"] = speeds["segment_id"].astype("int64")

    return {
        "speeds": speeds,
        "crashes": pd.DataFrame(
            {
                "crash_record_id": [f"MOCK{i:05d}" for i in range(crashes)],
                "crash_time": times[crash_bin],
                "segment_id": ids[crash_road],
            }
        ).sort_values("crash_time", ignore_index=True),
        "segments": pd.DataFrame(
            {
                "segment_id": ids,
                "length_m": np.linspace(120.0, 400.0, ids.size).astype("float32"),
                "degree": np.array([len(neighbours[i]) for i in range(ids.size)], dtype="int64"),
            }
        ),
        "edges": edges,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write mock processed tables.")
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
