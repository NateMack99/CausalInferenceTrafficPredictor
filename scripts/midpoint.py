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
    coverage_funnel,
    effect_figure,
    effects,
    evaluate,
    grid_coverage,
    placebo_corrected,
    trajectory_figure,
)
from traffic_gml.events import build_windows, event_datasets, processed_tables, write_processed
from traffic_gml.models import BASELINES, CONVS, build_model
from traffic_gml.train import DEFAULTS, code_version, data_version, fit, set_seed

MODELS = (*BASELINES, *CONVS)


def write_notes(args, metrics: dict, estimate: pd.DataFrame, coverage: dict) -> str:
    """Caveats that must travel with the numbers, not sit in a docstring.

    Everything here is a constraint on how these outputs may be read. A reader
    who opens `results.csv` alone sees a leaderboard; the proposal is explicit
    that lower forecast error is not evidence of lower causal bias.
    """
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["mae"])
    gap = metrics["gcn"]["mae"] - metrics["mlp"]["mae"]
    pooled = estimate.effect.mean()
    return f"""# How to read these outputs

Run: `{args.window[0]} to {args.window[1]}`, seed {args.seed}, one seed only.

## results.csv is a forecast leaderboard, not a causal ranking

It sorts by MAE, so `{ranked[0][0]}` appears first. That is forecast accuracy on
crash-free windows and says nothing about which model estimates crash effects
better. The proposal is explicit that lower prediction error is not evidence of
lower causal bias, and effect accuracy is not measured here at all -- that needs
the semi-synthetic benchmark (PR #8).

Only `gcn` against `mlp` isolates the graph: they share a trunk and parameter
shapes and differ only in having edges. That gap is {gap:+.3f} mph on a single
seed. `sage` has roughly twice the per-layer parameters, so `sage` beating `mlp`
mixes the graph with model capacity.

## The effect estimate is crash minus matched control

`effects_crash.csv` is a raw forecast residual and carries the model's own bias.
The control arm measures that bias on windows where nothing happened, so the
estimate is the difference, in `effect_estimate.csv`: {pooled:+.2f} mph pooled.
Effects are measured **on the crash segment only** -- no neighbour or hop-distance
effect is computed yet, so nothing here speaks to spillover.

## Study area

All speed-observed monitored segments, not the >=80% coverage subset. That subset
is 67 segments in 10 components, largest 22, which is too fragmented for a graph
study; the full monitored network is 881 segments with 90% in one component. The
cost is coverage: {coverage:.1%} of grid cells observed. See
coverage_funnel.csv for retention at every filtering stage.

## Intervals

Event-level only while each crash contributes one row per bin, and not clustered
by crash. They are not valid once neighbour roads enter.
"""


def segment_ids_of(tables: dict[str, pd.DataFrame]) -> np.ndarray:
    return np.sort(tables["segments"]["segment_id"].unique())


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
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "auto", "cuda", "mps"],
        help="auto picks cuda, then mps, then cpu",
    )
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
    coverage = grid_coverage(train)
    print(f"  speed grid observed on {coverage:.1%} of [bin, road] cells")
    funnel = None
    if args.crashes:
        funnel = coverage_funnel(
            pd.read_csv(args.crashes), train, segment_ids_of(tables), edge_index
        )
        funnel.to_csv(args.out / "coverage_funnel.csv", index=False)
        print("  crash funnel:")
        for row in funnel.itertuples():
            print(f"    {row.crashes:7,}  {row.retained_pct:5.1f}% kept  {row.stage}")
    else:
        print("  no crash funnel: pass --crashes <*_crash_segments.csv> for the coverage report")
    coverage_path = coverage_figure(train, args.out / "coverage.png")

    metrics, fitted = {}, {}
    for name in MODELS:
        set_seed(args.seed)
        model = build_model(name, train, hidden=args.hidden)
        print(f"  training {name}", flush=True)
        fit(
            model,
            sets,
            edge_index,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
        )
        metrics[name] = evaluate(model, sets["test"], edge_index, device=args.device)
        fitted[name] = model
        row = metrics[name]
        print(f"  {name:16s} test MAE {row['mae']:6.3f}  RMSE {row['rmse']:6.3f}")

    segment_ids = segment_ids_of(tables)
    table, control_table, event, control = event_datasets(
        train, tables["crashes"], segment_ids, controls=args.controls, seed=args.seed
    )
    model = fitted[args.effect_model]
    keys = ["origin", "road", "crash_record_id", "segment_id"]
    event_effects = effects(model, event, edge_index, device=args.device).merge(
        table[keys], on=["origin", "road"]
    )
    control_effects = effects(model, control, edge_index, device=args.device).merge(
        control_table[keys], on=["origin", "road"]
    )
    print(
        f"{len(table):,} crashes with a crash-free history and {len(control_table):,} "
        f"held-out controls; effects from {args.effect_model}"
    )
    estimate = placebo_corrected(event_effects, control_effects)
    print("  crash minus matched control, on the crash segment only:")
    for row in estimate.itertuples():
        star = " *" if row.low * row.high > 0 else "  "
        print(
            f"  {row.minutes - 10:3d}-{row.minutes:3d} min  crash {row.crash:+6.2f}"
            f"   control {row.control:+6.2f}   effect {row.effect:+6.2f} mph"
            f"  [{row.low:+5.2f}, {row.high:+5.2f}]{star} n={row.n:,}"
        )
    estimate.to_csv(args.out / "effect_estimate.csv", index=False)

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
        coverage_path,
        trajectory_figure(train, table, control_table, args.out / "crash_trajectories.png"),
        comparison_figure(metrics, args.out / "model_comparison.png"),
        effect_figure(event_effects, control_effects, args.out / "crash_effect.png"),
    ]
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "code": code_version(),
                "data": data_version(tables, args.data or args.speeds),
                "roads": train.num_roads,
                "edges": int(edge_index.size(1)),
                "windows": {name: len(data) for name, data in sets.items()},
                "crashes": len(table),
                "control_windows": len(control_table),
                "grid_coverage": round(coverage, 4),
                "coverage_funnel": (funnel.to_dict("records") if funnel is not None else None),
                "effect_mph": estimate.set_index("minutes").effect.round(3).to_dict(),
                "effect_model": args.effect_model,
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    (args.out / "NOTES.md").write_text(write_notes(args, metrics, estimate, coverage))
    names = ", ".join(path.name for path in figures)
    print(f"wrote {names} and results.csv to {args.out}")
    return metrics


if __name__ == "__main__":
    main()
