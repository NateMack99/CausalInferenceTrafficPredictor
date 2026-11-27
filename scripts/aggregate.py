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
        "how long each ring stays detectably negative",
        "run",
        "event/effect_duration.csv",
    ),
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
    ("tab:diagnostics", "zero-speed sensitivity, forecast", "run", "main-nozero/results.csv"),
    (
        "tab:diagnostics",
        "zero-speed sensitivity, known-effect recovery",
        "run",
        "benchmark-nozero/effect_recovery.csv",
    ),
)

#: `synthetic-scenarios` directory prefix -> the scenario the manuscript names.
SCENARIOS = {"null": "B1 direct-only", "plateau": "B2 plateau", "congested": "B3 congested"}


def _datasets(work: Path) -> dict:
    """What `prepare` resolved and validated, so the paper records its inputs."""
    record = work / "data-provenance.json"
    return json.loads(record.read_text()) if record.is_file() else {}


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
    """Exposure-conditioned arms beside the residual estimator they are tested
    against, scored on the same injected interventions."""
    rows = []
    for arm in sorted((run / "treatment").glob("*/effect_recovery.csv")):
        recovery = pd.read_csv(arm)
        forecast = pd.read_csv(arm.parent / "forecast.csv")
        seed = json.loads((arm.parent / "summary.json").read_text())["args"]["seed"]
        pooled = recovery.loc[recovery.group == "pooled"]
        for model in forecast.model:
            hit = pooled.loc[pooled.model == model]
            rows.append(
                {
                    "model": model,
                    "estimator": hit.estimator.iloc[0] if len(hit) else None,
                    "seed": int(seed),
                    "forecast_mae": float(forecast.loc[forecast.model == model, "mae"].iloc[0]),
                    "effect_mae": float(hit.effect_mae.iloc[0]) if len(hit) else None,
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["model", "seed"], ignore_index=True) if len(frame) else frame


def spread_table(run: Path) -> pd.DataFrame:
    """One row per relabelling, kept apart from `graph_ablation.csv`.

    Its `real` baseline is that table's real arm at the same training seed,
    because `real` ignores the relabelling seed. Folding these rows into the
    ablation table instead would redefine what its `permuted` group means.
    """
    rows = []
    for arm in sorted((run / "topology-spread").glob("*/summary.json")):
        settings = json.loads(arm.read_text())["args"]
        recovery = pd.read_csv(arm.parent / "effect_recovery.csv")
        pooled = recovery.loc[(recovery.estimator == "corrected") & (recovery.group == "pooled")]
        forecast = pd.read_csv(arm.parent / "forecast.csv")
        for model in forecast.model:
            hit = pooled.loc[pooled.model == model]
            rows.append(
                {
                    "model": model,
                    "graph_seed": int(settings["graph_seed"]),
                    "seed": int(settings["seed"]),
                    "forecast_mae": float(forecast.loc[forecast.model == model, "mae"].iloc[0]),
                    "effect_mae": float(hit.effect_mae.iloc[0]) if len(hit) else None,
                    "effect_bias": float(hit.bias_mph.iloc[0]) if len(hit) else None,
                }
            )
    frame = pd.DataFrame(rows)
    if not len(frame):
        return frame
    return frame.sort_values(["model", "seed", "graph_seed"], ignore_index=True)


def _zero_mae(arm: Path) -> dict[tuple[str, str], float]:
    """MAE of predicting no effect at all, on the cells the arm scored.

    The injected signal averages about 1.5 mph against a forecast noise floor
    near 3 mph, so an estimator that shrinks every prediction to zero scores
    well on absolute error while recovering nothing. Without this column a
    reader cannot tell those two cases apart, and the exposure heads sit much
    closer to it than to the truth.
    """
    truth = pd.read_csv(arm / "truth.csv")
    error = truth["true_effect"].abs()
    rows = {("pooled", "all"): float(error.mean())}
    for group, keys in (("hop", truth["hop"]), ("minutes", truth["minutes"])):
        rows.update({(group, str(k)): float(v) for k, v in error.groupby(keys).mean().items()})
    return rows


def scenario_table(run: Path) -> pd.DataFrame:
    """Known-effect recovery across all four injected scenarios in one table.

    B1 is the one that cannot be read off B0: its outer rings carry a true
    zero, so any neighbour estimate there is the estimator inventing spillover.
    """
    arms = [("B0 decay", run / "benchmark-seed42")]
    arms += [
        (SCENARIOS[arm.name.split("-seed")[0]], arm)
        for arm in sorted((run / "synthetic-scenarios").glob("*-seed*"))
        if arm.name.split("-seed")[0] in SCENARIOS
    ]
    frames = []
    for scenario, arm in arms:
        if not (arm / "effect_recovery.csv").is_file():
            continue
        frame = pd.read_csv(arm / "effect_recovery.csv")
        frame.insert(0, "scenario", scenario)
        frame.insert(1, "seed", int(json.loads((arm / "summary.json").read_text())["args"]["seed"]))
        zero = _zero_mae(arm) if (arm / "truth.csv").is_file() else {}
        frame["zero_mae"] = [
            round(zero[(g, str(v))], 3) if (g, str(v)) in zero else None
            for g, v in zip(frame["group"], frame["value"], strict=True)
        ]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(["scenario", "seed", "model", "estimator"], ignore_index=True)


def matching_table(run: Path) -> pd.DataFrame:
    """Every control-drawing rule on one table, the reported rule first.

    The reported estimate is one of six admissible rules. Collected together,
    the spread across rules is the honest uncertainty on the matching design,
    and it is not inside any single arm's interval.
    """
    arms = [("matched pre-event, 0 hops (reported)", run / "event")]
    arms += [(arm.name, arm) for arm in sorted((run / "matching").glob("*")) if arm.is_dir()]
    frames = []
    for rule, arm in arms:
        if not (arm / "effect_by_hop.csv").is_file():
            continue
        frame = pd.read_csv(arm / "effect_by_hop.csv")
        frame.insert(0, "rule", rule)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def seed_table(run: Path) -> pd.DataFrame:
    """Held-out forecast error per architecture and training seed.

    Topology differences are reported as paired within-seed contrasts, so this
    is what says whether those contrasts are larger than ordinary seed noise.
    """
    rows = []
    for record in sorted((run / "forecast-seeds").glob("*-seed*/metrics.json")):
        model, seed = record.parent.name.rsplit("-seed", 1)
        splits = json.loads(record.read_text())["splits"]
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "test_mae": round(float(splits["test"]["mae"]), 4),
                "test_rmse": round(float(splits["test"]["rmse"]), 4),
                "val_mae": round(float(splits["val"]["mae"]), 4),
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
        ("scenarios.csv", scenario_table, "tab:synthetic", "B0-B3 recovery beside the zero null"),
        ("matching.csv", matching_table, "tab:diagnostics", "all six control-drawing rules"),
        ("forecast_seeds.csv", seed_table, "tab:forecast", "forecast error by training seed"),
        ("topology_spread.csv", spread_table, "fig:topology", "ten independent relabellings"),
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
                "datasets": _datasets(args.work),
                "ready": int((index.status == "ready").sum()),
                "missing": sorted(index.loc[index.status == "missing", "source"]),
            },
            indent=2,
        )
        + "\n"
    )
    print(index.to_string(index=False))
    print(f"\n{len(done)} stage markers; wrote {args.out}")


if __name__ == "__main__":
    main()
