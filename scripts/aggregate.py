"""Collect the Run A tables the manuscript needs and index them.

A stage that has not run is reported as missing. Nothing here computes a
result; it copies what the stages wrote and joins the array stages into one
table per comparison, so the paper reads numbers from saved outputs only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: real first: it is the reference arm the other two are read against.
TOPOLOGIES = ("real", "none", "permuted")

#: manuscript label, what it is, and the file the stages write it to.
SINGLE = (
    ("tab:funnel", "crash-to-road selection funnel", "work", "audit/coverage_funnel.csv"),
    ("tab:funnel", "monthly coverage and control supply", "work", "audit/monthly_coverage.csv"),
    ("tab:forecast", "seven-model held-out forecast error", "run", "main-seed42/results.csv"),
    ("fig:eventcurves", "0/1/2-hop effects, available case", "run", "event/effect_by_hop.csv"),
    ("fig:eventcurves", "fixed-event cohort", "run", "event/effect_fixed_cohort.csv"),
    (
        "tab:diagnostics",
        "effect beside the pre-trend placebo",
        "run",
        "event/effect_against_placebo.csv",
    ),
    ("tab:diagnostics", "pre-event covariate balance", "run", "event/covariate_balance.csv"),
    ("tab:diagnostics", "speed availability by arm, hop and step", "run", "event/availability.csv"),
    ("tab:diagnostics", "cells dropped for want of a control", "run", "event/attrition.csv"),
    (
        "tab:diagnostics",
        "effect excluding overlapping crashes",
        "run",
        "event/effect_excluding_overlaps.csv",
    ),
    (
        "tab:synthetic",
        "known-effect recovery, Benchmark 0",
        "run",
        "benchmark-seed42/effect_recovery.csv",
    ),
    (
        "tab:diagnostics",
        "zero-speed sensitivity, event study",
        "run",
        "event-nozero/effect_by_hop.csv",
    ),
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def ablation_table(run: Path) -> pd.DataFrame:
    """One row per (architecture, topology, seed) with both error kinds."""
    rows = []
    for arm in sorted((run / "graph-ablation").glob("*/summary.json")):
        settings = json.loads(arm.read_text())["args"]
        forecast = pd.read_csv(arm.parent / "forecast.csv")
        recovery = pd.read_csv(arm.parent / "effect_recovery.csv")
        pooled = recovery.loc[(recovery.estimator == "corrected") & (recovery.group == "pooled")]
        for model in forecast.model:
            hit = pooled.loc[pooled.model == model]
            rows.append(
                {
                    "model": model,
                    "topology": settings["graph_ablation"],
                    "seed": int(settings["seed"]),
                    "forecast_mae": float(forecast.loc[forecast.model == model, "mae"].iloc[0]),
                    "effect_mae": float(hit.effect_mae.iloc[0]) if len(hit) else None,
                    "effect_bias": float(hit.bias_mph.iloc[0]) if len(hit) else None,
                }
            )
    frame = pd.DataFrame(rows)
    return (
        frame.sort_values(["model", "topology", "seed"], ignore_index=True) if len(frame) else frame
    )


def treatment_table(run: Path) -> pd.DataFrame:
    """Exposure-conditioned arms. Forecast error only: scoring their head
    contrast against injected truth is not implemented, so no effect column is
    invented here."""
    rows = []
    for arm in sorted((run / "treatment").glob("*/metrics.json")):
        metrics = json.loads(arm.read_text())
        settings = json.loads((arm.parent / "args.json").read_text())
        rows.append(
            {
                "model": settings["model"],
                "seed": int(settings["seed"]),
                "neighbour_hops": int(settings["neighbour_hops"]),
                "forecast_mae": metrics["splits"]["test"]["mae"],
                "forecast_rmse": metrics["splits"]["test"]["rmse"],
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["model", "seed"], ignore_index=True) if len(frame) else frame


def ablation_figure(table: pd.DataFrame, path: Path) -> None:
    """Forecast error beside known-effect recovery, the pair RQ2 turns on."""
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for axis, column, label in zip(
        axes,
        ("forecast_mae", "effect_mae"),
        ("Forecast MAE (mph)", "Effect MAE (mph)"),
        strict=True,
    ):
        shown = table.dropna(subset=[column])
        for marker, (model, group) in zip("os^D", shown.groupby("model"), strict=False):
            spread = group.groupby("topology")[column].agg(["mean", "std", "size"])
            spread = spread.reindex([t for t in TOPOLOGIES if t in spread.index])
            axis.errorbar(
                range(len(spread)),
                spread["mean"],
                yerr=spread["std"].fillna(0.0),
                marker=marker,
                capsize=3,
                label=f"{model} (n={int(spread['size'].max())})",
            )
            axis.set_xticks(range(len(spread)), spread.index)
        axis.set_ylabel(label)
        axis.set_xlabel("message-passing topology")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require", nargs="*", default=[], help="stages that must be present")
    args = parser.parse_args()
    roots = {"run": args.run, "work": args.work}
    args.out.mkdir(parents=True, exist_ok=True)

    index = []
    for label, description, root, relative in SINGLE:
        source = roots[root] / relative
        present = source.is_file() and source.stat().st_size > 0
        if present:
            frame = pd.read_csv(source)
            name = relative.replace("/", "__")
            frame.to_csv(args.out / name, index=False)
        index.append(
            {
                "manuscript": label,
                "description": description,
                "source": str(source),
                "status": "ready" if present else "missing",
                "rows": len(frame) if present else 0,
                "sha256": _digest(source) if present else "",
            }
        )

    for name, builder, label, description in (
        ("graph_ablation.csv", ablation_table, "tab:causalmodels", "real/self-only/rewired arms"),
        ("treatment.csv", treatment_table, "tab:causalmodels", "exposure-conditioned arms"),
    ):
        frame = builder(args.run)
        target = args.out / name
        if len(frame):
            frame.to_csv(target, index=False)
        index.append(
            {
                "manuscript": label,
                "description": description,
                "source": str(target),
                "status": "ready" if len(frame) else "missing",
                "rows": len(frame),
                "sha256": _digest(target) if len(frame) else "",
            }
        )
        if name == "graph_ablation.csv" and len(frame):
            ablation_figure(frame, args.out / "fig_ablation.png")

    index = pd.DataFrame(index)
    index.to_csv(args.out / "result_index.csv", index=False)
    done = (
        sorted(p.name for p in (args.run / ".done").glob("*"))
        if (args.run / ".done").is_dir()
        else []
    )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "generated_utc": pd.Timestamp.now("UTC").isoformat(),
                "run": str(args.run),
                "completed_stages": done,
                "ready": int((index.status == "ready").sum()),
                "missing": sorted(index.loc[index.status == "missing", "source"]),
            },
            indent=2,
        )
        + "\n"
    )
    print(index.to_string(index=False))
    print(f"\n{len(done)} stage markers; wrote {args.out}")
    absent = [stage for stage in args.require if stage not in " ".join(done)]
    if absent:
        raise SystemExit(f"required stages missing: {absent}")


if __name__ == "__main__":
    main()
