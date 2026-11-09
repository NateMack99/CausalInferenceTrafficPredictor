"""Prediction metrics.

Speed-forecast MAE and RMSE over observed, crash-free targets, in mph and per
horizon step. Errors accumulate in running sums, so a year of roads never has
to fit in memory at once.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- after the Agg backend is set
from torch.utils.data import DataLoader

from .data import BIN_MINUTES
from .events import Windows, crash_index
from .graph import neighbourhood
from .models import on_device, predict, resolve_device


def masked_error(pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    """Error at masked-in cells, zero elsewhere. Selected rather than multiplied:
    a NaN at a masked-out cell would otherwise poison the whole sum."""
    return torch.where(mask, pred - y, torch.zeros((), dtype=pred.dtype))


def masked_loss(pred: Tensor, y: Tensor, mask: Tensor, power: int = 1) -> Tensor:
    """Mean absolute (`power=1`) or squared (`power=2`) error over masked-in cells."""
    count = mask.sum()
    error = masked_error(pred, y, mask).abs().pow(power)
    return error.sum() / count if count else (pred * 0).sum()


#: --loss name -> exponent for `masked_loss`.
LOSSES = {"mae": 1, "mse": 2}


@torch.no_grad()
def evaluate(
    model, data: Windows, edge_index: Tensor, batch_size: int = 64, device: str = "cpu"
) -> dict:
    """Forecast MAE and RMSE in mph, overall and per horizon step."""
    where = resolve_device(device) if isinstance(model, torch.nn.Module) else torch.device("cpu")
    if isinstance(model, torch.nn.Module):
        model.eval().to(where)
    edge_index = edge_index.to(where)
    absolute, square, count = (torch.zeros(data.horizon, dtype=torch.float64) for _ in range(3))
    for batch in DataLoader(data, batch_size=batch_size):
        batch = on_device(batch, where)
        out = predict(model, batch, edge_index)
        # Accumulate in float64 on the CPU: MPS has no float64 at all.
        error = (masked_error(out, batch["y"], batch["y_mask"]) * data.std).cpu().double()
        absolute += error.abs().sum(dim=(0, 2))
        square += error.pow(2).sum(dim=(0, 2))
        count += batch["y_mask"].cpu().sum(dim=(0, 2))
    total = count.sum()
    if not total:
        raise ValueError(
            "no observed crash-free targets to evaluate on; lower --exposure-hops "
            "or check that the speeds table has observations on these roads"
        )
    # None, not 0.0, for a horizon step with no targets: they are not the same thing.
    per_step = (absolute / count.clamp(min=1)).tolist()
    return {
        "mae": float(absolute.sum() / total),
        "rmse": float((square.sum() / total).sqrt()),
        "mae_by_minutes": {
            str((step + 1) * BIN_MINUTES): round(value, 3) if count[step] else None
            for step, value in enumerate(per_step)
        },
        "n_targets": int(total),
    }


@torch.no_grad()
def effects(
    model, data: Windows, edge_index: Tensor, batch_size: int = 64, device: str = "cpu"
) -> pd.DataFrame:
    """Observed minus predicted no-accident speed, per window, road and step.

    Negative means slower than the counterfactual. One row per unmasked target,
    with `effect` and `observed` in mph. Forecast accuracy alone does not make
    this causal: `docs/methodology.md` sets out what else has to hold.
    """
    run_on = resolve_device(device) if isinstance(model, torch.nn.Module) else torch.device("cpu")
    if isinstance(model, torch.nn.Module):
        model.eval().to(run_on)
    edge_index = edge_index.to(run_on)
    frames = []
    for batch in DataLoader(data, batch_size=batch_size):
        batch = on_device(batch, run_on)
        prediction = predict(model, batch, edge_index).cpu()
        batch = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in batch.items()}
        mask = batch["y_mask"]
        where = mask.nonzero()
        frames.append(
            pd.DataFrame(
                {
                    "origin": batch["origin"][where[:, 0]].numpy(),
                    "minutes": (where[:, 1].numpy() + 1) * BIN_MINUTES,
                    "road": where[:, 2].numpy(),
                    "observed": (batch["y"] * data.std + data.mean)[mask].numpy(),
                    "predicted": (prediction * data.std + data.mean)[mask].numpy(),
                    "effect": ((batch["y"] - prediction) * data.std)[mask].numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


#: Validated categorical slots (light surface); markers carry identity too, so
#: the three low-contrast hues are never colour-alone. See the dataviz palette.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARKERS = ("o", "s", "^", "D", "v")
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _axes(width: float = 7.0, height: float = 4.0, columns: int = 1):
    figure, axes = plt.subplots(1, columns, figsize=(width, height), facecolor=SURFACE)
    for axis in np.atleast_1d(axes):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color="#e6e5e1", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#d6d5d0")
        axis.tick_params(colors=MUTED, labelsize=9)
    return figure, axes


def _save(figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def coverage_figure(data: Windows, path: str | Path) -> Path:
    """Speed coverage over the study period, and across roads."""
    observed = np.isfinite(data.values.numpy())
    days = pd.Series(observed.mean(axis=1), index=data.times).resample("D").mean()
    figure, (left, right) = _axes(width=10.0, columns=2)
    left.plot(days.index, days.to_numpy(), color=SERIES[0], linewidth=2)
    left.set_title("Observed speed bins per day", color=INK, fontsize=11, loc="left")
    left.set_ylabel("fraction observed", color=MUTED, fontsize=9)
    left.set_ylim(0, 1)
    figure.autofmt_xdate()
    right.hist(observed.mean(axis=0), bins=24, color=SERIES[0], edgecolor=SURFACE)
    right.set_title("Roads by coverage", color=INK, fontsize=11, loc="left")
    right.set_xlabel("fraction of bins observed", color=MUTED, fontsize=9)
    right.set_ylabel("roads", color=MUTED, fontsize=9)
    return _save(figure, path)


def comparison_figure(metrics: dict[str, dict], path: str | Path) -> Path:
    """Forecast MAE by horizon, one line per model."""
    figure, axis = _axes()
    for i, (name, values) in enumerate(metrics.items()):
        steps = {int(k): v for k, v in values["mae_by_minutes"].items() if v is not None}
        axis.plot(
            list(steps),
            list(steps.values()),
            label=name,
            color=SERIES[i % len(SERIES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=5,
            linewidth=2,
        )
    axis.set_title("No-accident forecast error by horizon", color=INK, fontsize=11, loc="left")
    axis.set_xlabel("minutes ahead", color=MUTED, fontsize=9)
    axis.set_ylabel("MAE (mph)", color=MUTED, fontsize=9)
    axis.legend(frameon=False, labelcolor=INK, fontsize=9)
    return _save(figure, path)


def effect_figure(event: pd.DataFrame, control: pd.DataFrame, path: str | Path) -> Path:
    """Observed minus predicted speed after a crash, against matched controls."""
    figure, axis = _axes()
    axis.axhline(0, color="#b9b8b2", linewidth=1)
    for i, (name, frame) in enumerate((("crash windows", event), ("matched controls", control))):
        by_step = frame.groupby("minutes").effect.agg(["mean", "sem"])
        axis.errorbar(
            by_step.index,
            by_step["mean"],
            yerr=1.96 * by_step["sem"].fillna(0),
            label=name,
            color=SERIES[i],
            marker=MARKERS[i],
            markersize=5,
            linewidth=2,
            capsize=3,
        )
    axis.set_title(
        "Speed effect after a crash, with 95% intervals", color=INK, fontsize=11, loc="left"
    )
    axis.set_xlabel("minutes after the crash (bin end)", color=MUTED, fontsize=9)
    axis.set_ylabel("observed − predicted (mph)", color=MUTED, fontsize=9)
    axis.legend(frameon=False, labelcolor=INK, fontsize=9)
    return _save(figure, path)


def placebo_corrected(event: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Crash arm minus matched-control arm, per horizon bin, with a 95% interval.

    This is the estimate the proposal defines: the control arm is not a second
    finding, it is the forecaster's own bias on windows where nothing happened,
    and it has to be subtracted rather than displayed beside the crash arm.
    The interval is event-level only while each crash contributes one row per
    bin; add clustering before trusting it with neighbour roads in play.
    """
    rows = []
    for minutes in sorted(event.minutes.unique()):
        a = event.loc[event.minutes == minutes, "effect"]
        b = control.loc[control.minutes == minutes, "effect"]
        effect = a.mean() - b.mean()
        spread = 1.96 * float(np.hypot(a.sem(), b.sem()))
        rows.append(
            {
                "minutes": int(minutes),
                "crash": round(a.mean(), 3),
                "control": round(b.mean(), 3),
                "effect": round(effect, 3),
                "low": round(effect - spread, 3),
                "high": round(effect + spread, 3),
                "n": int(len(a)),
            }
        )
    return pd.DataFrame(rows)


def coverage_funnel(
    audit: pd.DataFrame,
    data: Windows,
    segment_ids: np.ndarray,
    edge_index: Tensor,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """How many crashes survive each filter, with retention at every stage.

    A large raw crash table does not mean a usable event study. The row that
    matters for sample size is the analysis sample -- the crashes the effect
    estimate is actually computed on. The two rows below it describe a stricter
    reference subset *within* that sample, so every stage is nested and the
    percentages compose.

    Reading the last row as the sample size is the mistake this layout exists
    to prevent: requiring every bin observed is far stricter than the estimator
    is, because a masked target simply contributes nothing to that bin.

    Stages one to four are counted from `audit`, the full crash table that
    `segments.analyze` writes with a `match_status` per row. Counting them from
    the joined `crashes` table instead would mix two populations: that table is
    already filtered to matched crashes, so a funnel built from both would
    report more survivors than the stage above it.
    """
    matched = audit.loc[audit["match_status"] == "matched"].copy()
    when = matched["crash_time"] if "crash_time" in matched else matched["timestamp"]
    matched["crash_time"] = pd.to_datetime(when)
    matched = matched.loc[matched["segment_id"].notna()]
    matched["segment_id"] = matched["segment_id"].astype("int64")

    observed = np.isfinite(data.values.numpy())
    history, horizon = data.history, data.horizon
    reach = neighbourhood(edge_index, len(segment_ids), 1)

    bins, _, on_grid = crash_index(matched, data.times, segment_ids)
    in_grid = 0
    for crash_bin, known in zip(bins, on_grid, strict=True):
        origin = int(crash_bin) - 1
        if known and origin - history + 1 >= 0 and origin + horizon < len(data.times):
            in_grid += 1

    # Counted over the analysis sample, so these stages nest inside it.
    complete = with_neighbours = 0
    for origin, road in zip(events["origin"], events["road"], strict=True):
        origin, road = int(origin), int(road)
        pre = observed[origin - history + 1 : origin + 1, road]
        post = observed[origin + 1 : origin + 1 + horizon, road]
        if not (pre.all() and post.all()):
            continue
        complete += 1
        others = [n for n in reach[road].indices if n != road]
        if others and observed[origin + 1 : origin + 1 + horizon, others].any():
            with_neighbours += 1

    stages = [
        ("crashes in the study period", len(audit)),
        ("with usable coordinates", int((audit.match_status != "invalid_coordinates").sum())),
        ("matched to a monitored segment", len(matched)),
        ("on a segment with speed data, window fits", in_grid),
        ("crash-free history and a held-out control (analysis sample)", len(events)),
        ("of those, every pre and post bin observed", complete),
        ("and a connected road observed too", with_neighbours),
    ]
    first = stages[0][1] or 1
    return pd.DataFrame(
        [
            {
                "stage": name,
                "crashes": count,
                "retained_pct": round(100 * count / first, 1),
                "step_pct": round(100 * count / (stages[i - 1][1] or 1), 1) if i else 100.0,
            }
            for i, (name, count) in enumerate(stages)
        ]
    )


def neighbour_coverage(
    data: Windows, edge_index: Tensor, events: pd.DataFrame, hops: int = 2
) -> pd.DataFrame:
    """Can spillover be measured? Neighbour observability per hop, per event.

    The funnel's last row only asks whether *any* connected road was observed
    in *any* bin. Estimating how congestion spreads needs more than that: the
    neighbour has to be observed across the bins the disruption would travel
    through, and have a history to forecast it from. One row per hop ring, with
    how many events clear progressively stricter bars.

    `full_trajectory` is the one that gates a spillover study. Rings are
    exclusive, so the 2-hop row describes roads two hops away and no nearer.
    """
    observed = np.isfinite(data.values.numpy())
    history, horizon = data.history, data.horizon
    reach = [neighbourhood(edge_index, data.num_roads, k) for k in range(hops + 1)]

    rows = []
    for hop in range(1, hops + 1):
        counts = {"any": 0, "half": 0, "full": 0, "trajectory": 0, "in_graph": 0, "usable": 0}
        for origin, road in zip(events["origin"], events["road"], strict=True):
            origin, road = int(origin), int(road)
            inner = set(reach[hop - 1][road].indices)
            ring = np.array([n for n in reach[hop][road].indices if n not in inner], dtype=int)
            if not len(ring):
                continue
            counts["in_graph"] += 1
            post = observed[origin + 1 : origin + 1 + horizon][:, ring].sum(axis=0)
            has_history = observed[origin - history + 1 : origin + 1][:, ring].any(axis=0)
            counts["any"] += bool((post >= 1).any())
            counts["half"] += bool((post >= horizon / 2).any())
            counts["full"] += bool((post == horizon).any())
            usable = (post == horizon) & has_history
            counts["trajectory"] += bool(usable.any())
            counts["usable"] += int(usable.sum())
        rows.append(
            {
                "hop": hop,
                "events": len(events),
                "with_neighbour_in_graph": counts["in_graph"],
                "any_post_bin": counts["any"],
                "half_post_bins": counts["half"],
                "all_post_bins": counts["full"],
                "full_trajectory": counts["trajectory"],
                "usable_per_event": round(counts["usable"] / max(len(events), 1), 2),
            }
        )
    return pd.DataFrame(rows)


def grid_coverage(data: Windows) -> float:
    """Fraction of the `[bins, roads]` speed grid that was actually observed."""
    return float(np.isfinite(data.values.numpy()).mean())


def trajectory_figure(
    data: Windows,
    events: pd.DataFrame,
    controls: pd.DataFrame,
    path: str | Path,
    count: int = 3,
) -> Path:
    """Observed speed around real crashes, against each one's matched control.

    Panels are the earliest fully observed crash windows, in time order -- not
    the deepest dips. Ranking by the size of the drop would manufacture the
    result this figure is meant to be neutral about, and ranking by how many
    bins happen to be observed just picks the busiest road. The last panel
    averages every fully observed event, and is the only panel with enough
    windows behind it to read a shape off.

    This shows that crashes align with speed histories. A dip is not an effect
    estimate: rush hour, weather or an earlier incident could produce the same
    picture, which is what the counterfactual model is for.
    """
    values = data.values.numpy()
    history, horizon = data.history, data.horizon
    span = np.arange(-history + 1, horizon + 1) * BIN_MINUTES

    def trace(origin: int, road: int) -> np.ndarray:
        return values[origin - history + 1 : origin + 1 + horizon, road]

    first_control = controls.groupby("crash_record_id", as_index=False).origin.first()
    paired = events.merge(first_control, on="crash_record_id", suffixes=("", "_control"))
    if paired.empty:
        raise ValueError("no crash has a matched control to plot against")
    complete = [np.isfinite(trace(r.origin, r.road)).all() for r in paired.itertuples()]
    # Prefer crashes whose own window has no gaps; fall back rather than fail, since
    # on sparse coverage an unbroken 18-bin window can be genuinely rare.
    whole = paired.loc[complete].sort_values("crash_time")
    partial = not len(whole)
    if partial:
        whole = paired.sort_values("crash_time")

    picked = whole.head(count)
    figure, axes = _axes(width=4.2 * (len(picked) + 1), columns=len(picked) + 1)
    axes = np.atleast_1d(axes)
    arms = (("crash", SERIES[0], MARKERS[0]), ("matched control", SERIES[1], MARKERS[1]))

    for axis, row in zip(axes, picked.itertuples(), strict=False):
        axis.axvline(0, color="#b9b8b2", linewidth=1)
        for origin, (label, colour, marker) in zip(
            (row.origin, row.origin_control), arms, strict=True
        ):
            axis.plot(
                span,
                trace(origin, row.road),
                label=label,
                color=colour,
                marker=marker,
                markersize=4,
                linewidth=2,
            )
        axis.set_title(
            f"segment {row.segment_id} · {row.crash_time:%d %b %H:%M}",
            color=INK,
            fontsize=10,
            loc="left",
        )
        axis.set_xlabel("minutes from the crash", color=MUTED, fontsize=9)

    mean_axis = axes[-1]
    mean_axis.axvline(0, color="#b9b8b2", linewidth=1)
    for column, (label, colour, marker) in zip(("origin", "origin_control"), arms, strict=True):
        stack = np.stack([trace(int(o), r) for o, r in zip(whole[column], whole.road, strict=True)])
        seen = np.isfinite(stack).sum(0).clip(min=1)
        mean = np.nanmean(stack, axis=0)
        band = 1.96 * np.nan_to_num(np.nanstd(stack, axis=0, ddof=1)) / np.sqrt(seen)
        mean_axis.plot(
            span, mean, label=label, color=colour, marker=marker, markersize=4, linewidth=2
        )
        mean_axis.fill_between(
            span, mean - band, mean + band, color=colour, alpha=0.18, linewidth=0
        )
    label = "events" if partial else "fully observed events"
    mean_axis.set_title(f"mean of {len(whole):,} {label}", color=INK, fontsize=10, loc="left")
    mean_axis.set_xlabel("minutes from the crash", color=MUTED, fontsize=9)
    mean_axis.legend(frameon=False, labelcolor=INK, fontsize=9)
    axes[0].set_ylabel("observed speed (mph)", color=MUTED, fontsize=9)
    return _save(figure, path)
