"""Run the whole pipeline end to end and write the midpoint figures and tables.

Joins the speed, crash and road-graph outputs into the model contract, trains
every model on identical splits, estimates crash effects against matched
accident-free controls, and writes figures plus a results table.

    uv run python scripts/midpoint.py --speeds data/parquet/chicago-speeds \
        --graph outputs/<stem>_segments.graphml \
        --crashes outputs/<stem>_crash_segments.csv \
        --window 2025-03-01 2025-05-01

With --data instead, it reads already-joined tables (for example data/mock).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_gml.data import load_tables
from traffic_gml.evaluate import (
    comparison_figure,
    coverage_figure,
    effect_figure,
    effects,
    evaluate,
)
from traffic_gml.events import build_windows, event_datasets, processed_tables, write_processed
from traffic_gml.models import BASELINES, CONVS, build_model
from traffic_gml.train import DEFAULTS, data_version, fit, set_seed

MODELS = (*BASELINES, *CONVS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("inputs (either --data, or all three pipeline paths)")
    source.add_argument("--data", type=Path, help="already-joined contract tables")
    source.add_argument("--speeds", type=Path, help="date-partitioned speeds dataset")
    source.add_argument("--graph", type=Path, help="*_segments.graphml from segments.analyze")
    source.add_argument("--crashes", type=Path, help="*_crash_segments.csv beside it")
    source.add_argument("--window", nargs=2, metavar=("START", "END"), help="naive local bounds")

    parser.add_argument("--out", type=Path, default=Path("outputs/midpoint"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--exposure-hops", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--static-features", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    parser.add_argument("--controls", type=int, default=1, help="controls per crash")
    parser.add_argument("--effect-model", default="gcn", choices=MODELS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.data and not (args.speeds and args.graph and args.crashes):
        parser.error("pass --data, or all of --speeds, --graph and --crashes")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.data:
        tables = load_tables(args.data)
        print(f"read joined tables from {args.data}")
    else:
        window = tuple(args.window) if args.window else None
        tables = processed_tables(args.speeds, args.graph, args.crashes, window)
        write_processed(tables, args.processed)
        print(f"joined the pipeline into {args.processed}")
    print("  " + "  ".join(f"{name}={len(table):,}" for name, table in tables.items()))

    sets, edge_index = build_windows(
        tables,
        history=args.history,
        horizon=args.horizon,
        exposure_hops=args.exposure_hops,
        stride=args.stride,
        static_features=args.static_features,
    )
    train = sets["train"]
    print(
        f"{train.num_roads} roads, {edge_index.size(1):,} directed arcs, "
        + "  ".join(f"{name}={len(data):,} windows" for name, data in sets.items())
    )
    coverage = coverage_figure(train, args.out / "coverage.png")

    metrics, fitted = {}, {}
    for name in MODELS:
        set_seed(args.seed)
        model = build_model(name, train, hidden=args.hidden)
        fit(model, sets, edge_index, epochs=args.epochs, patience=args.patience, verbose=False)
        metrics[name] = evaluate(model, sets["test"], edge_index)
        fitted[name] = model
        row = metrics[name]
        print(f"  {name:16s} test MAE {row['mae']:6.3f}  RMSE {row['rmse']:6.3f}")

    segment_ids = np.sort(tables["segments"]["segment_id"].unique())
    table, control_table, event, control = event_datasets(
        train, tables["crashes"], segment_ids, controls=args.controls, seed=args.seed
    )
    model = fitted[args.effect_model]
    keys = ["origin", "road", "crash_record_id", "segment_id"]
    event_effects = effects(model, event, edge_index).merge(table[keys], on=["origin", "road"])
    control_effects = effects(model, control, edge_index).merge(
        control_table[keys], on=["origin", "road"]
    )
    print(
        f"{len(table):,} crashes with a crash-free history and {len(control_table):,} "
        f"held-out controls; effects from {args.effect_model}"
    )
    for minutes, value in event_effects.groupby("minutes").effect.mean().items():
        matched = control_effects.loc[control_effects.minutes == minutes, "effect"].mean()
        n = int((event_effects.minutes == minutes).sum())
        print(
            f"  {minutes - 10:3d}-{minutes:3d} min  crash {value:+6.2f} mph"
            f"   control {matched:+6.2f} mph   (n={n:,})"
        )

    results = pd.DataFrame(
        [
            {
                "model": name,
                "mae": values["mae"],
                "rmse": values["rmse"],
                **{f"mae_{k}min": v for k, v in values["mae_by_minutes"].items()},
            }
            for name, values in metrics.items()
        ]
    ).sort_values("mae")
    results.to_csv(args.out / "results.csv", index=False)
    table.to_csv(args.out / "events.csv", index=False)
    event_effects.assign(window="crash").to_csv(args.out / "effects_crash.csv", index=False)
    control_effects.assign(window="control").to_csv(args.out / "effects_control.csv", index=False)
    figures = [
        coverage,
        comparison_figure(metrics, args.out / "model_comparison.png"),
        effect_figure(event_effects, control_effects, args.out / "crash_effect.png"),
    ]
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "data": data_version(tables, args.data or args.speeds),
                "roads": train.num_roads,
                "edges": int(edge_index.size(1)),
                "windows": {name: len(data) for name, data in sets.items()},
                "crashes": len(table),
                "control_windows": len(control_table),
                "effect_model": args.effect_model,
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    names = ", ".join(path.name for path in figures)
    print(f"wrote {names} and results.csv to {args.out}")
    return metrics


if __name__ == "__main__":
    main()
