"""Matched event study: direct and neighbouring-road crash effects (PR #7).

Estimates crash effects by road distance and time with **no forecaster**, by
differencing observed speeds against each window's own pre-crash level and
then against matched crash-free control windows. It is the interpretable
baseline the graph models have to beat, and it runs in seconds.

    uv run python scripts/event_study.py --data data/processed --out outputs/event_study

Everything it writes is observational and assumption-dependent. It is reported
separately from the semi-synthetic known-effect results in
`scripts/benchmark.py`, because one is an estimate and the other is a check
against ground truth; `docs/methodology.md` sets out the difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_gml.causal import (
    against_placebo,
    covariate_balance,
    event_study,
    fixed_cohort,
    pre_trends,
    reported_cells,
    summarise,
)
from traffic_gml.data import load_tables
from traffic_gml.evaluate import hop_effect_figure, neighbour_coverage
from traffic_gml.events import build_windows, event_datasets, processed_tables, write_processed
from traffic_gml.train import code_version, data_version, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("inputs (either --data, or all three pipeline paths)")
    source.add_argument("--data", type=Path, help="already-joined contract tables")
    source.add_argument("--speeds", type=Path, help="date-partitioned speeds dataset")
    source.add_argument("--graph", type=Path, help="*_segments.graphml from segments.analyze")
    source.add_argument("--crashes", type=Path, help="*_crash_segments.csv beside it")
    source.add_argument("--window", nargs=2, metavar=("START", "END"), help="naive local bounds")

    parser.add_argument("--out", type=Path, default=Path("outputs/event_study"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--exposure-hops", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--static-features", nargs="*", default=[])
    parser.add_argument("--drop-zero-speed", action="store_true")
    parser.add_argument("--hops", type=int, default=2, help="rings to estimate effects on")
    parser.add_argument("--controls", type=int, default=5, help="controls per crash")
    parser.add_argument("--leads", type=int, default=3, help="pre-trend placebo bins")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.data is None and not (args.speeds and args.graph and args.crashes):
        parser.error("pass --data, or all of --speeds, --graph and --crashes")
    if args.hops > args.exposure_hops:
        # Past the exposure radius the event arm stops being crash-dirty, so the
        # control arm's clean-post filter starts applying to both arms alike and
        # the distance profile gets a step at that radius which is the filter,
        # not traffic.
        parser.error(
            f"--hops {args.hops} exceeds --exposure-hops {args.exposure_hops}; rings beyond "
            "the masking radius are filtered differently and their effects are not comparable"
        )
    return args


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.data is not None:
        tables, source = load_tables(args.data), args.data
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
        stride=args.stride,
        static_features=args.static_features,
        drop_zero_speed=args.drop_zero_speed,
    )
    train = sets["train"]
    segment_ids = np.sort(tables["segments"]["segment_id"].unique())
    # `held_out=False`: that rule keeps controls out of cells a *forecaster* was
    # fitted on, and this estimator has no forecaster. Leaving it on pushed every
    # control into the weeks after its event (mean +3.3 weeks, 88% later) and cut
    # supply to 1.9 per crash, because the only admissible cells sat outside the
    # training span. Waiving it gives 2,720 events at 4.7 controls each, a
    # symmetric offset, and better pre-event balance on the frozen data.
    table, controls, _, _ = event_datasets(
        train,
        tables["crashes"],
        segment_ids,
        controls=args.controls,
        seed=args.seed,
        held_out=False,
    )
    events = table.rename(columns={"crash_record_id": "event_id"})
    control = controls.rename(columns={"crash_record_id": "event_id"})
    print(f"{len(events):,} events, {len(control):,} control pairs, {train.num_roads} roads")

    reported = reported_cells(tables["crashes"], train.times, segment_ids)
    did = event_study(train, events, control, edge_index, hops=args.hops, reported=reported)
    available = summarise(did)
    cohort = fixed_cohort(did)
    overlapping = did["other_crash"].mean()
    print(
        f"{len(did):,} estimated cells over {did.event_id.nunique():,} events; "
        f"{overlapping:.1%} carry a second reported crash; "
        f"fixed cohort keeps {cohort.event_id.nunique():,} events"
    )

    tables_out = {
        "effect_by_hop": available,
        "effect_excluding_overlaps": summarise(did[~did["other_crash"]]),
        "effect_fixed_cohort": summarise(cohort) if len(cohort) else pd.DataFrame(),
        "pre_trends": pre_trends(
            train, events, control, edge_index, hops=args.hops, leads=args.leads
        ),
        "covariate_balance": covariate_balance(
            train, events, control, tables["segments"], segment_ids
        ),
        "neighbour_coverage": neighbour_coverage(train, edge_index, table, hops=args.hops),
    }
    for name, frame in tables_out.items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
    did.to_csv(args.out / "effects_by_cell.csv", index=False)
    figure = hop_effect_figure(available, args.out / "hop_effects.png")

    checked = against_placebo(available, tables_out["pre_trends"])
    tables_out["effect_against_placebo"] = checked
    checked.to_csv(args.out / "effect_against_placebo.csv", index=False)

    print("\neffect by hop ring, event-clustered 95% intervals:")
    print(available.to_string(index=False))
    print(
        "\nAGAINST THE PLACEBO -- read this before quoting any row above. The placebo\n"
        "estimates something the crash cannot have caused, so a share near 1 means the\n"
        "arms were already diverging and the row is not a crash effect:"
    )
    print(checked.to_string(index=False))
    worst = checked.loc[checked["placebo_share"].idxmax()] if len(checked) else None
    if worst is not None and worst["placebo_share"] >= 0.33:
        print(
            f"  -> hop {int(worst['hop'])} at {int(worst['minutes'])} min has a placebo "
            f"{worst['placebo_share']:.0%} of its effect. Do not report it as spillover."
        )
    print(
        f"\nfixed cohort ({cohort.event_id.nunique():,} of {did.event_id.nunique():,} events "
        "observed at every hop and step) -- better identified, weaker:"
    )
    print(tables_out["effect_fixed_cohort"].to_string(index=False))
    print("\ncovariate balance on the crash road (read `gap` in mph, not std_diff):")
    print(tables_out["covariate_balance"].to_string(index=False))

    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "code": code_version(),
                "data": data_version(tables, Path(source)),
                "events": int(did.event_id.nunique()),
                "cells": int(len(did)),
                "controls_per_event": round(len(control) / max(len(events), 1), 2),
                "share_with_second_crash": round(float(overlapping), 4),
                "fixed_cohort_events": int(cohort.event_id.nunique()) if len(cohort) else 0,
                "effect_by_hop": available.to_dict("records"),
                "effect_against_placebo": checked.to_dict("records"),
                "effect_fixed_cohort": tables_out["effect_fixed_cohort"].to_dict("records"),
                "covariate_balance": tables_out["covariate_balance"].to_dict("records"),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"\nwrote {args.out} and {figure.name}")
    return available


if __name__ == "__main__":
    main()
