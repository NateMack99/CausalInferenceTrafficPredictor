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
    monthly_coverage,
    neighbour_coverage,
    placebo_corrected,
    trajectory_figure,
)
from traffic_gml.events import (
    ABLATIONS,
    GRAPH_FEATURES,
    ablate_edges,
    build_windows,
    event_datasets,
    processed_tables,
    write_processed,
)
from traffic_gml.models import BASELINES, CONVS, TEMPORAL, build_model, resolve_device
from traffic_gml.train import DEFAULTS, code_version, data_version, fit, set_seed

MODELS = (*BASELINES, *CONVS, *TEMPORAL)


def ablation_warning(args) -> str:
    """The loudest caveat of all, when the graph a run used was not the real one.

    `write_notes` is the file a reader opens instead of `summary.json`, and
    without this an ablated run's notes still claim the gcn-mlp gap isolates
    the graph and still point at `neighbour_coverage.csv` as the
    spillover-feasibility answer. Neither is true for that run.
    """
    if args.graph_ablation == "real":
        return ""
    what = {
        "permuted": "the node labels were shuffled, so the topology and degree sequence "
        "are the real ones but no road has its own neighbours",
        "none": "every edge was dropped",
    }[args.graph_ablation]
    return f"""> **This run did not use the real road graph.** `--graph-ablation
> {args.graph_ablation}`: {what}. The forecast numbers below are an ablation
> arm, not a result about Chicago, and the gcn-mlp sentence does not describe
> the real network. The coverage and neighbour tables are still computed on the
> real graph, so those remain comparable across arms.

"""


def write_notes(args, metrics: dict, estimate: pd.DataFrame, coverage: dict) -> str:
    """Caveats that must travel with the numbers, not sit in a docstring.

    Everything here is a constraint on how these outputs may be read. A reader
    who opens `results.csv` alone sees a leaderboard; the proposal is explicit
    that lower forecast error is not evidence of lower causal bias.
    """
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["mae"])
    gap = metrics["gcn"]["mae"] - metrics["mlp"]["mae"]
    temporal_gap = metrics["stgcn"]["mae"] - metrics["tgru"]["mae"]
    pooled = estimate.effect.mean()
    # `--window` is None on the `--data` path, which made this line crash at the
    # very last step of an otherwise finished run. Pre-existing; fixed here
    # because this function is being changed anyway.
    span = f"{args.window[0]} to {args.window[1]}" if args.window else "joined tables"
    return (
        ablation_warning(args)
        + f"""# How to read these outputs

Run: `{span}`, seed {args.seed}, one seed only.

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

`stgcn` against `tgru` is the same controlled pair for the spatiotemporal
models, both a GRU down the history and differing only in the edges:
{temporal_gap:+.3f} mph. Comparing `stgcn` with `mlp` instead mixes the graph
with the temporal encoder and is not a graph result.

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

## Intervals, and what they do not cover

Event-level only while each crash contributes one row per bin, and not clustered
by crash. They are not valid once neighbour roads enter.

They also say nothing about the control draw. Redrawing which accident-free
windows serve as controls has moved per-bin estimates by up to 0.16 mph and
changed how many bins exclude zero, while the crash arm moved under 0.03 mph --
so the control arm, not the crash arm, is the noise. Matching yields about two
usable controls per crash over two months, which caps what `--controls` can buy;
a longer span is the fix. Read the pooled estimate and the shape across horizons,
not the per-bin stars from one draw.

## Spillover feasibility

`neighbour_coverage.csv` answers whether the main research question is reachable:
per hop ring, how many events have a connected road observed at all, over half
the post bins, and over every post bin with a history to forecast from. The last
column is the one that gates a spillover study. Missingness that is flat across
the horizon means immediate and delayed effects stay separable; missingness that
grows with time since the crash would not.
"""
    )


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
    parser.add_argument(
        "--overwrite-processed",
        action="store_true",
        help="allow the join to replace existing tables in --processed",
    )
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--exposure-hops", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--static-features", nargs="*", default=[])
    parser.add_argument(
        "--graph-ablation",
        default="real",
        choices=list(ABLATIONS),
        help="`permuted` scrambles which road is which at equal degree; `none` drops edges",
    )
    parser.add_argument(
        "--graph-seed", type=int, help="seed for the ablation only; defaults to --seed"
    )
    parser.add_argument(
        "--drop-zero-speed",
        action="store_true",
        help="treat an observed 0.0 mph as missing (the midpoint run kept zeros)",
    )
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
    if args.data and (args.speeds or args.graph or args.crashes):
        # Accepting both silently ignored the pipeline paths while still reading
        # --crashes for the funnel, which built it from a different population.
        parser.error("--data replaces --speeds/--graph/--crashes; pass one or the other")
    if args.data and args.window:
        # --window only applies to the join, but NOTES.md printed it as the run's
        # span either way, so the one file written for humans named a period that
        # was never applied.
        parser.error("--window applies to the pipeline path only, not to --data")
    for path in (args.data, args.speeds, args.graph, args.crashes):
        if path and not path.exists():
            parser.error(f"{path} does not exist")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    # A rerun into a used directory leaves last run's CSVs beside this run's
    # JSON, and nothing marks them stale. It cost me a wrong measurement once.
    if (args.out / "summary.json").exists():
        raise SystemExit(f"{args.out} already holds a run; use a new --out directory")
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.data:
        tables = load_tables(args.data)
        print(f"read joined tables from {args.data}")
    else:
        window = tuple(args.window) if args.window else None
        # Checked before the join, not after: on the expanded window that join
        # reads a year of speeds, so a requeued run used to sit through all of
        # it only to abort. The default --processed is the frozen March-April
        # set every recorded result rests on.
        existing = sorted(args.processed.glob("*.parquet"))
        if existing and not args.overwrite_processed:
            raise SystemExit(
                f"{args.processed} already holds {len(existing)} tables. Joining a "
                "different window would replace them and silently change what every "
                "later --data run reads. Pass --processed <new dir> for a new window, "
                "or --overwrite-processed to replace these (the usual answer on a retry)."
            )
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
        drop_zero_speed=args.drop_zero_speed,
    )
    train = sets["train"]
    # `model_edges`, not `edge_index`: the coverage funnel and the neighbour
    # observability table describe the *real* road network, and shadowing the
    # variable made them describe the ablated one instead -- under `none` every
    # hop-ring count came out zero, so the file that answers "is spillover
    # reachable" claimed no neighbour is ever observable.
    graph_seed = args.seed if args.graph_seed is None else args.graph_seed
    model_edges = ablate_edges(edge_index, train.num_roads, args.graph_ablation, graph_seed)
    if args.graph_ablation != "real":
        print(
            f"graph ablation: {args.graph_ablation} (seed {graph_seed}, "
            f"{model_edges.size(1)} arcs); coverage tables still describe the real graph"
        )
        leaked = [c for c in args.static_features if c in GRAPH_FEATURES]
        if leaked:
            print(
                f"  warning: {leaked} are graph-derived and are NOT permuted, so they "
                "describe a graph this run does not have. Use --static-features length_m."
            )
    print(
        f"{train.num_roads} roads, {edge_index.size(1):,} directed arcs, "
        + "  ".join(f"{name}={len(data):,} windows" for name, data in sets.items())
    )
    coverage = grid_coverage(train)
    print(f"  speed grid observed on {coverage:.1%} of [bin, road] cells")
    coverage_path = coverage_figure(train, args.out / "coverage.png")

    metrics, fitted = {}, {}
    for name in MODELS:
        set_seed(args.seed)
        model = build_model(name, train, hidden=args.hidden)
        print(f"  training {name}", flush=True)
        fit(
            model,
            sets,
            model_edges,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
        )
        metrics[name] = evaluate(model, sets["test"], model_edges, device=args.device)
        fitted[name] = model
        row = metrics[name]
        print(f"  {name:16s} test MAE {row['mae']:6.3f}  RMSE {row['rmse']:6.3f}")

    # Written here, not at the end: everything below is cheap but can raise, and
    # an exception after the training loop used to discard the whole GPU budget
    # along with the one table that needed it.
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

    segment_ids = segment_ids_of(tables)
    table, control_table, event, control = event_datasets(
        train, tables["crashes"], segment_ids, controls=args.controls, seed=args.seed
    )
    funnel = None
    if args.crashes:
        funnel = coverage_funnel(pd.read_csv(args.crashes), train, segment_ids, edge_index, table)
        funnel.to_csv(args.out / "coverage_funnel.csv", index=False)
        print("  crash funnel:")
        for row in funnel.itertuples():
            print(f"    {row.crashes:7,}  {row.retained_pct:5.1f}% kept  {row.stage}")
    else:
        print("  no crash funnel: pass --crashes <*_crash_segments.csv> for the coverage report")

    nearby = neighbour_coverage(train, edge_index, table, hops=args.exposure_hops)
    nearby.to_csv(args.out / "neighbour_coverage.csv", index=False)
    monthly = monthly_coverage(train, tables["crashes"], table, control_table)
    monthly.to_csv(args.out / "monthly_coverage.csv", index=False)
    print("  coverage, matches and control supply by month:")
    print(monthly.to_string(index=False).replace("\n", "\n  "))
    print(f"  neighbour observability (events with such a neighbour, of {len(table):,}):")
    for row in nearby.itertuples():
        print(
            f"    {row.hop}-hop  any bin {row.any_post_bin:6,}  "
            f">=half {row.half_post_bins:6,}  full trajectory {row.full_trajectory:6,}  "
            f"({row.full_trajectory / max(len(table), 1):.1%})"
        )

    model = fitted[args.effect_model]
    # One key row per window, and `validate` so a future many-to-many merge
    # fails loudly instead of silently duplicating effect rows.
    keys = ["origin", "road", "crash_record_id", "segment_id"]
    event_keys = table[keys].drop_duplicates(subset=["origin", "road"])
    control_keys = control_table[keys].drop_duplicates(subset=["origin", "road"])
    event_effects = effects(model, event, model_edges, device=args.device).merge(
        event_keys, on=["origin", "road"], validate="m:1"
    )
    control_effects = effects(model, control, model_edges, device=args.device).merge(
        control_keys, on=["origin", "road"], validate="m:1"
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
                "device": str(resolve_device(args.device)),
                "code": code_version(),
                "data": data_version(tables, args.data or args.speeds),
                "roads": train.num_roads,
                "edges": int(model_edges.size(1)),
                "windows": {name: len(data) for name, data in sets.items()},
                "crashes": len(table),
                "control_windows": len(control_table),
                "grid_coverage": round(coverage, 4),
                "coverage_funnel": (funnel.to_dict("records") if funnel is not None else None),
                "neighbour_coverage": nearby.to_dict("records"),
                "monthly_coverage": monthly.to_dict("records"),
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
