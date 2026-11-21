"""Matched event study: crash effects with no forecaster at all (PR #7).

The model-based estimator in `evaluate.effects` predicts a no-crash speed and
reads the effect off the residual. This module answers the same question
without a model, by differencing observed speeds twice:

    did = (crash road after - its own pre-crash level)
        - (matched control after - its own pre-control level)

The first difference removes each road's own speed level, so a slow road is
not mistaken for a crash effect. The second removes whatever the time of week
does to every road alike, which is what the matched control is for. Nothing
here is fitted, so nothing here can leak: it is the interpretable baseline the
graph models have to beat, and a sanity check on them.

What it does **not** do is make the estimate causal. It controls for the
road's own level, the slot of the week, and road identity. It does not control
for weather, for why this road had a crash and the control week did not, or
for a second crash arriving on a neighbour road during the window -- which is
counted and reported rather than assumed away. See `docs/methodology.md`.

Hop rings are the exclusive rings of `events.hop_rings`, so hop 0 is the crash
road, hop 1 the roads adjacent to it and no nearer, and so on.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from torch import Tensor

from .data import BIN_MINUTES
from .events import Windows, crash_index, hop_rings


def reported_cells(
    crashes: pd.DataFrame, times: pd.DatetimeIndex, segment_ids: np.ndarray
) -> np.ndarray:
    """`[bins, roads]` bool: where a crash was *reported*, with no hop spread.

    `Windows.clean` marks everything a crash contaminates, including the index
    crash's own wake, so it cannot distinguish "this crash" from "another
    crash". This marks only report sites, which is what identifies a second
    crash landing on a neighbour road inside the window.
    """
    bins, roads, on_grid = crash_index(crashes, times, segment_ids)
    cells = np.zeros((len(times), len(segment_ids)), dtype=bool)
    inside = on_grid & (bins >= 0)
    cells[bins[inside], roads[inside]] = True
    return cells


def arm_table(
    reference: Windows,
    frame: pd.DataFrame,
    edge_index: Tensor,
    hops: int = 2,
    reported: np.ndarray | None = None,
    require_clean_post: bool = False,
) -> pd.DataFrame:
    """Observed speed minus each window's own pre-event baseline, per ring road.

    `frame` needs `event_id`, `road` and `origin`; `road` is the crash road, and
    the rings are expanded from it, so a control window inherits the rings of
    the event it was matched to.

    A ring road is skipped unless its **pre-event span is crash-free** and has
    at least one observed speed: a depressed or empty baseline makes the first
    difference meaningless. `require_clean_post` additionally demands a
    crash-free post span, which is what the control arm needs -- a control is
    only vetted on the crash road, so a neighbour of it may carry a real crash.
    The event arm must not use it, because the index crash dirties its own post
    span by construction.

    When `reported` is given, `other_crash` flags rows whose post span contains
    a crash reported on that same road at or after the origin, excluding the
    index crash's own cell. That is the overlapping-incident confound; it is
    returned as a column so a caller can report both the available-case and the
    clean-cohort estimate rather than silently choosing one.
    """
    values = reference.values.numpy()
    clean = reference.clean.numpy()
    history, horizon = reference.history, reference.horizon
    rings = hop_rings(edge_index, reference.num_roads, hops)

    rows: list[dict] = []
    for event_id, road, origin in zip(
        frame["event_id"], frame["road"], frame["origin"], strict=True
    ):
        origin, road = int(origin), int(road)
        pre = slice(origin - history + 1, origin + 1)
        post = slice(origin + 1, origin + 1 + horizon)
        for hop in range(hops + 1):
            ring = rings[hop][road]
            if hop:
                ring = np.setdiff1d(ring, rings[hop - 1][road])
            for other in ring:
                other = int(other)
                if not clean[pre, other].all():
                    continue
                if require_clean_post and not clean[post, other].all():
                    continue
                baseline = values[pre, other]
                baseline = baseline[np.isfinite(baseline)]
                if not baseline.size:
                    continue
                level = float(baseline.mean())
                second = False
                if reported is not None:
                    hits = reported[post, other].copy()
                    if hop == 0:
                        # The index crash is reported in the window's first bin
                        # on its own road; that one is the treatment, not a
                        # confound.
                        hits[0] = False
                    second = bool(hits.any())
                for step in range(horizon):
                    speed = values[origin + 1 + step, other]
                    if not np.isfinite(speed):
                        continue
                    rows.append(
                        {
                            "event_id": event_id,
                            "origin": origin,
                            "road": other,
                            "hop": hop,
                            "minutes": (step + 1) * BIN_MINUTES,
                            "observed": float(speed),
                            "baseline": level,
                            "delta": float(speed) - level,
                            "other_crash": second,
                        }
                    )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(
            "no window has a ring road with a crash-free, observed pre-event span; "
            "check the event table and --exposure-hops"
        )
    return table


def event_study(
    reference: Windows,
    events: pd.DataFrame,
    controls: pd.DataFrame,
    edge_index: Tensor,
    hops: int = 2,
    reported: np.ndarray | None = None,
) -> pd.DataFrame:
    """Matched difference-in-differences, one row per (event, road, step).

    `events` and `controls` are what `events.event_datasets` returns, with
    `crash_record_id` renamed to `event_id`. Controls are averaged within each
    (event, road, step) before differencing, so an event with five controls
    does not outweigh one with two.

    `did` is the estimate; `n_controls` says how many control windows stood
    behind it, and `other_crash` whether a second crash was reported on that
    road inside the window.
    """
    crash_arm = arm_table(reference, events, edge_index, hops, reported)
    control_arm = arm_table(
        reference, controls, edge_index, hops, reported, require_clean_post=True
    )
    pooled = control_arm.groupby(["event_id", "road", "hop", "minutes"], as_index=False).agg(
        control_delta=("delta", "mean"), n_controls=("delta", "size")
    )
    merged = crash_arm.merge(pooled, on=["event_id", "road", "hop", "minutes"])
    merged["did"] = merged["delta"] - merged["control_delta"]
    return merged


def summarise(did: pd.DataFrame, column: str = "did") -> pd.DataFrame:
    """By hop ring and horizon step, with **event-clustered** 95% intervals.

    Road-time cells inside one event are correlated, and there are many of them
    per event, so treating them as independent would shrink the interval by
    roughly the square root of that count and manufacture significance. Cells
    are averaged within an event first, and the interval comes from the spread
    across events -- so `events` is the real sample size, not `n`.
    """
    per_event = did.groupby(["hop", "minutes", "event_id"], as_index=False)[column].mean()
    rows = []
    for (hop, minutes), group in per_event.groupby(["hop", "minutes"]):
        cells = did[(did.hop == hop) & (did.minutes == minutes)]
        mean = float(group[column].mean())
        spread = float(group[column].sem()) if len(group) > 1 else float("nan")
        half = 1.96 * spread if np.isfinite(spread) else float("nan")
        rows.append(
            {
                "hop": int(hop),
                "minutes": int(minutes),
                "events": int(group.event_id.nunique()),
                "roads": int(cells.road.nunique()),
                "n": int(len(cells)),
                "effect": round(mean, 3),
                "low": round(mean - half, 3) if np.isfinite(half) else None,
                "high": round(mean + half, 3) if np.isfinite(half) else None,
            }
        )
    return pd.DataFrame(rows)


def fixed_cohort(did: pd.DataFrame) -> pd.DataFrame:
    """The events observed at every (hop, step) the table covers.

    An available-case estimate can show an effect fading with distance purely
    because the far rings are measured on a different, better-observed set of
    events. Restricting to a common cohort costs sample size and answers the
    distance question on one population. Report both; they are different
    estimands and the gap between them is itself a finding.
    """
    wanted = did.groupby(["hop", "minutes"]).ngroups
    seen = did.groupby("event_id")[["hop", "minutes"]].apply(
        lambda g: len(g.drop_duplicates()), include_groups=False
    )
    return did[did.event_id.isin(seen[seen == wanted].index)]


def pre_trends(
    reference: Windows,
    events: pd.DataFrame,
    controls: pd.DataFrame,
    edge_index: Tensor,
    hops: int = 2,
    leads: int = 3,
) -> pd.DataFrame:
    """Placebo difference-in-differences on the bins *before* the event.

    Shifts every origin back `leads` bins and runs the same estimator, so the
    "effect" it reports is one the crash cannot have caused. A flat, near-zero
    result supports the design; a sloped one says the arms were already
    diverging and the headline estimate is partly that divergence.

    `leads` bins of history are given up, so this needs `history > leads`.
    """
    if leads >= reference.history:
        raise ValueError(f"leads={leads} needs history greater than it, got {reference.history}")
    shifted = copy.copy(reference)
    shifted.history = reference.history - leads
    shifted.horizon = leads
    return summarise(
        event_study(
            shifted,
            events.assign(origin=events["origin"] - leads),
            controls.assign(origin=controls["origin"] - leads),
            edge_index,
            hops,
        )
    )


def covariate_balance(
    reference: Windows,
    events: pd.DataFrame,
    controls: pd.DataFrame,
    segments: pd.DataFrame,
    segment_ids: np.ndarray,
    columns: tuple[str, ...] = ("length_m", "degree", "betweenness"),
) -> pd.DataFrame:
    """Standardised differences between the arms, on the crash road.

    `std_diff` is the arm gap over the pooled standard deviation, so it is
    comparable across features. Conventionally anything under about 0.1 is
    well balanced. Road features balance by construction here -- a control is
    the same road -- so those rows are a check that the matching did what it
    claims, while the pre-event level and trend rows are the informative ones.

    Balance on these says nothing about weather, road works, or whatever made
    this road have a crash in this hour. Those remain unobserved.
    """
    values = reference.values.numpy()
    history = reference.history
    static = segments.set_index("segment_id").reindex(segment_ids)

    def describe(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for road, origin in zip(frame["road"], frame["origin"], strict=True):
            road, origin = int(road), int(origin)
            window = values[origin - history + 1 : origin + 1, road]
            half = history // 2
            early, late = window[:half], window[half:]
            early, late = early[np.isfinite(early)], late[np.isfinite(late)]
            row = {
                "pre_level": float(np.nanmean(window)) if np.isfinite(window).any() else np.nan,
                "pre_trend": (
                    float(late.mean() - early.mean()) if early.size and late.size else np.nan
                ),
            }
            for name in columns:
                row[name] = float(static[name].iloc[road]) if name in static.columns else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    crash, control = describe(events), describe(controls)
    rows = []
    for name in ["pre_level", "pre_trend", *columns]:
        a, b = crash[name].dropna(), control[name].dropna()
        if not a.size or not b.size:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append(
            {
                "feature": name,
                "crash_mean": round(float(a.mean()), 3),
                "control_mean": round(float(b.mean()), 3),
                "std_diff": round(float((a.mean() - b.mean()) / pooled), 3) if pooled else 0.0,
                "n_crash": int(a.size),
                "n_control": int(b.size),
            }
        )
    return pd.DataFrame(rows)
