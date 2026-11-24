"""Score known-effect recovery on the semi-synthetic benchmark (PR #9).

Injects known disruptions into held-out crash-free windows, runs every
forecaster through the *same* estimator the study uses on real crashes, and
reports how well each recovers the injected effect by hop ring and horizon
step. Forecast error is reported in a separate table, because a model can win
one and lose the other.

    uv run python scripts/benchmark.py --data data/processed --out outputs/benchmark

Two estimators are scored on the same cells, on purpose:

- `corrected` subtracts the forecaster's bias measured on matched untreated
  control windows. This is the estimator available for a real crash.
- `raw` is the bare residual. Its error is the untreated forecast error
  algebraically, so it is a diagnostic, not independent causal validation.

The arm deliberately *not* scored here is the untreated-outcome difference:
both arms share their inputs, so the prediction cancels and the result is the
injected truth itself. See `docs/methodology.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traffic_gml.data import load_tables
from traffic_gml.evaluate import control_corrected, effect_recovery, effects, evaluate
from traffic_gml.events import build_windows, fitted_cells, processed_tables, write_processed
from traffic_gml.models import BASELINES, CONVS, TEMPORAL, build_model, resolve_device
from traffic_gml.synthetic import PLACEMENTS, SHAPES, inject, matched_controls, treated_windows
from traffic_gml.train import DEFAULTS, code_version, data_version, fit, set_seed

MODELS = (*BASELINES, *CONVS, *TEMPORAL)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("inputs (either --data, or all three pipeline paths)")
    source.add_argument("--data", type=Path, help="already-joined contract tables")
    source.add_argument("--speeds", type=Path, help="date-partitioned speeds dataset")
    source.add_argument("--graph", type=Path, help="*_segments.graphml from segments.analyze")
    source.add_argument("--crashes", type=Path, help="*_crash_segments.csv beside it")
    source.add_argument("--window", nargs=2, metavar=("START", "END"), help="naive local bounds")

    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))

    windows = parser.add_argument_group("windows")
    windows.add_argument("--history", type=int, default=12)
    windows.add_argument("--horizon", type=int, default=6)
    windows.add_argument("--exposure-hops", type=int, default=2)
    windows.add_argument(
        "--neighbour-hops",
        type=int,
        default=0,
        help="neighbourhood-mean speed channel, in graph hops; 0 is off",
    )
    windows.add_argument("--stride", type=int, default=1)
    windows.add_argument("--static-features", nargs="*", default=[])
    windows.add_argument("--drop-zero-speed", action="store_true")

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument("--events", type=int, default=200)
    benchmark.add_argument("--controls", type=int, default=5, help="controls per event road")
    benchmark.add_argument(
        "--magnitude", type=float, nargs=2, default=(4.0, 12.0), metavar=("LOW", "HIGH")
    )
    benchmark.add_argument("--hops", type=int, default=2, help="rings the disruption reaches")
    benchmark.add_argument("--hop-decay", type=float, default=0.45)
    benchmark.add_argument("--shape", default="decay", choices=list(SHAPES))
    benchmark.add_argument("--duration", type=float, default=2.0, help="in bins")
    benchmark.add_argument("--placement", default="uniform", choices=list(PLACEMENTS))

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    optimisation.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    optimisation.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    optimisation.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    optimisation.add_argument("--seed", type=int, default=42)
    optimisation.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])

    args = parser.parse_args(argv)
    if args.data is None and not (args.speeds and args.graph and args.crashes):
        parser.error("pass --data, or all of --speeds, --graph and --crashes")
    return args


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.data is not None:
        tables = load_tables(args.data)
        source = args.data
    else:
        window = tuple(args.window) if args.window else None
        tables = processed_tables(args.speeds, args.graph, args.crashes, window)
        write_processed(tables, args.processed)
        source = args.processed

    sets, edge_index = build_windows(
        tables,
        history=args.history,
        horizon=args.horizon,
        exposure_hops=args.exposure_hops,
        neighbour_hops=args.neighbour_hops,
        stride=args.stride,
        static_features=args.static_features,
        drop_zero_speed=args.drop_zero_speed,
    )
    train, test = sets["train"], sets["test"]
    print(
        f"{train.num_roads} roads, {edge_index.size(1):,} arcs, "
        + "  ".join(f"{n}={len(d):,}" for n, d in sets.items())
    )

    # Inject into the held-out split only: a disruption in a window the model
    # trained on would score memorisation rather than recovery.
    disruptions = inject(
        test,
        edge_index,
        events=args.events,
        magnitude=tuple(args.magnitude),
        hops=args.hops,
        hop_decay=args.hop_decay,
        shape=args.shape,
        duration=args.duration,
        placement=args.placement,
        seed=args.seed,
    )
    fitted = fitted_cells(train)
    pairs, control_windows = matched_controls(
        test, disruptions, fitted, controls=args.controls, seed=args.seed
    )
    treated = treated_windows(test, disruptions)
    identity = disruptions.events[["event_id", "origin"]]
    # Per (event, road): `pairs` has a row per (event, road, control), so
    # dividing by events would multiply by the roads in each event's rings.
    event_roads = max(len(pairs.drop_duplicates(subset=["event_id", "road"])), 1)
    per_road = len(pairs) / event_roads
    print(
        f"injected {disruptions.settings['events']} of {args.events} requested events; "
        f"{len(disruptions.truth):,} labelled cells, "
        f"{per_road:.2f} controls per event road over {event_roads:,} roads"
    )

    recovery, forecast = [], {}
    for name in args.models:
        # Re-seed per model, as `scripts/experiment.py` does. Seeding once would
        # leave each model's initialisation and shuffling dependent on what ran
        # before it, so `--models gcn` and the full list would disagree and the
        # recorded seed would reproduce neither.
        set_seed(args.seed)
        model = build_model(name, train, hidden=args.hidden)
        fit(
            model,
            sets,
            edge_index,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
            verbose=False,
        )
        forecast[name] = evaluate(model, test, edge_index, args.batch_size, args.device)

        event = effects(model, treated, edge_index, args.batch_size, args.device).merge(
            identity, on="origin"
        )
        control = effects(model, control_windows, edge_index, args.batch_size, args.device)
        corrected = control_corrected(event, control, pairs)

        # Both estimators are scored on the *same* cells. `control_corrected`
        # drops a cell whose matched control is missing, and that loss is
        # hop-dependent, so deriving `raw` from the full event frame instead
        # would compare bias correction against a different, differently
        # populated sample and read the gap as the correction's doing.
        scored = {
            "corrected": corrected,
            "raw": corrected.assign(estimated=corrected["effect"]),
        }
        # Not `tables`: that name holds the data tables this run was built from.
        scores = {}
        for estimator, frame in scored.items():
            scores[estimator] = effect_recovery(frame, disruptions.truth)
            recovery.append(scores[estimator].assign(model=name, estimator=estimator))

        pooled = scores["corrected"].query("group == 'pooled'").iloc[0]
        hops = scores["corrected"].query("group == 'hop'").sort_values("value")
        by_hop = "  ".join(f"h{int(r.value)} {r.effect_mae:5.3f}" for r in hops.itertuples())
        print(
            f"{name:16s} forecast MAE {forecast[name]['mae']:6.3f} mph   "
            f"effect MAE pooled {pooled.effect_mae:6.3f}  "
            f"bias {pooled.bias_mph:+.3f} [{pooled.bias_low:+.3f}, {pooled.bias_high:+.3f}]"
            f"   {by_hop}"
        )

    recovery = pd.concat(recovery, ignore_index=True)
    columns = ["model", "estimator", "group", "value"]
    recovery = recovery[columns + [c for c in recovery.columns if c not in columns]]
    recovery.to_csv(args.out / "effect_recovery.csv", index=False)
    flat = [
        {"model": name, **{k: v for k, v in metrics.items() if k != "mae_by_minutes"}}
        for name, metrics in forecast.items()
    ]
    pd.DataFrame(flat).sort_values("mae").to_csv(args.out / "forecast.csv", index=False)
    disruptions.truth.to_csv(args.out / "truth.csv", index=False)
    disruptions.events.to_csv(args.out / "events.csv", index=False)
    pairs.to_csv(args.out / "controls.csv", index=False)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "device": str(resolve_device(args.device)),
                "code": code_version(),
                "data": data_version(tables, Path(source)),
                "benchmark": disruptions.settings,
                "controls_per_event_road": round(per_road, 2),
                "event_roads": int(event_roads),
                "control_windows": int(len(control_windows)),
                "labelled_cells": int(len(disruptions.truth)),
                "cells_before_correction": int(len(event)),
                "cells_after_correction": int(len(corrected)),
                "forecast": forecast,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"wrote {args.out}")
    return recovery


if __name__ == "__main__":
    main()
