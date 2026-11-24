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
from .events import Windows, crash_index, hop_rings, pre_event_stats
from .graph import neighbourhood


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    That asymmetry has a measured cost. On the frozen data the filter removes
    about 2.7% of control rows and they are the slow ones -- mean delta -0.55
    at 1 hop against -0.14 for the rows kept, on a lower pre-event baseline --
    so it keeps the calmer control weeks and pushes the estimate more
    negative, by up to 0.02 mph. The pre-trend placebo cannot detect this:
    almost every event-arm row has a dirty post span in the headline run and
    almost none does in the placebo run, so the two runs are filtered
    differently and the placebo is silent on exactly this bias.

    Returns the per-cell table and, by hop and step, eligible ring roads
    against observed targets: the missingness audit.

    When `reported` is given, `other_crash` flags rows whose post span contains
    a crash reported on that road **or any road adjacent to it**, excluding the
    index crash. That is the overlapping-incident confound; it is returned as a
    column so a caller can report both the available-case and the clean-cohort
    estimate rather than silently choosing one. The pre span needs no flag: a
    ring road is already skipped unless its own pre span is crash-free, and
    `clean` spreads `--exposure-hops` across the graph.
    """
    values = reference.values.numpy()
    clean = reference.clean.numpy()
    history, horizon = reference.history, reference.horizon
    rings = hop_rings(edge_index, reference.num_roads, hops)
    levels, trends = pre_event_stats(values, history)
    # Reports within one hop of each road, counted so the index crash's own
    # report can be subtracted exactly rather than masking a genuine second one.
    near_reports = None
    if reported is not None:
        adjacent = neighbourhood(edge_index, reference.num_roads, 1)
        near_reports = reported.astype(np.int32) @ adjacent

    rows: list[dict] = []
    eligible = np.zeros(hops + 1, dtype=np.int64)
    observed = np.zeros((hops + 1, horizon), dtype=np.int64)
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
                level = levels[origin, other]
                if not np.isfinite(level):
                    continue
                eligible[hop] += 1
                observed[hop] += np.isfinite(values[post, other])
                second = False
                if near_reports is not None:
                    hits = near_reports[post, other].copy()
                    if hop <= 1:
                        # The index crash sits within one hop of its own road and
                        # of its immediate neighbours, reported in the window's
                        # first bin. That one is the treatment, not a confound.
                        hits[0] -= 1
                    second = bool((hits > 0).any())
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
                            "baseline": float(level),
                            "trend": float(trends[origin, other]),
                            "delta": float(speed) - float(level),
                            "other_crash": second,
                        }
                    )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(
            "no window has a ring road with a crash-free, observed pre-event span; "
            "check the event table and --exposure-hops"
        )
    availability = pd.DataFrame(
        {
            "hop": np.repeat(np.arange(hops + 1), horizon),
            "minutes": np.tile((np.arange(horizon) + 1) * BIN_MINUTES, hops + 1),
            "eligible": np.repeat(eligible, horizon),
            "observed": observed.ravel(),
        }
    )
    return table, availability


def nearest_on_target(crash_arm: pd.DataFrame, control_arm: pd.DataFrame) -> pd.DataFrame:
    """Keep, per (event, ring road), the control window closest on that road's
    own pre-event level and trend.

    The draw in `events.event_datasets` matches the crash road, so the ring
    roads the hop effects are measured on are matched on nothing, and one
    control week cannot be closest for all of them at once.
    """
    keys = ["event_id", "road"]
    stats = ["baseline", "trend"]
    target = crash_arm.groupby(keys, as_index=False)[stats].first()
    candidates = control_arm.groupby([*keys, "origin"], as_index=False)[stats].first()
    gap = candidates.merge(target, on=keys, suffixes=("", "_crash"))
    # Scaled, or the ~25 mph level would swamp the ~0.1 mph trend.
    level_scale = max(float(crash_arm.baseline.std()), 1e-6)
    trend_scale = max(float(crash_arm.trend.std()), 1e-6)
    gap["distance"] = (
        abs(gap.baseline - gap.baseline_crash) / level_scale
        + abs(gap.trend - gap.trend_crash) / trend_scale
    )
    best = gap.loc[gap.groupby(keys).distance.idxmin(), [*keys, "origin"]]
    return control_arm.merge(best, on=[*keys, "origin"])


def event_study(
    reference: Windows,
    events: pd.DataFrame,
    controls: pd.DataFrame,
    edge_index: Tensor,
    hops: int = 2,
    reported: np.ndarray | None = None,
    match_target: bool = False,
) -> pd.DataFrame:
    """Matched difference-in-differences, one row per (event, road, step).

    `events` and `controls` are what `events.event_datasets` returns, with
    `crash_record_id` renamed to `event_id`. Controls are averaged within each
    (event, road, step) before differencing, so an event with five controls
    does not outweigh one with two.

    `did` is the estimate; `n_controls` says how many control windows stood
    behind it, and `other_crash` whether a second crash was reported on that
    road inside the window.

    `match_target` narrows each ring road to its own nearest control instead of
    averaging all of them; see `nearest_on_target`.
    """
    crash_arm, crash_seen = arm_table(reference, events, edge_index, hops, reported)
    # No `reported` for the control arm: `require_clean_post` already rejects
    # any road whose post span is dirty, and a reported crash is always inside
    # that mask, so the flag is False on every control row and is dropped by the
    # groupby regardless.
    control_arm, control_seen = arm_table(
        reference, controls, edge_index, hops, require_clean_post=True
    )
    if match_target:
        control_arm = nearest_on_target(crash_arm, control_arm)
    pooled = control_arm.groupby(["event_id", "road", "hop", "minutes"], as_index=False).agg(
        control_delta=("delta", "mean"), n_controls=("delta", "size")
    )
    keys = ["event_id", "road", "hop", "minutes"]
    merged = crash_arm.merge(pooled, on=keys, how="left", indicator=True)
    # A crash cell whose control did not survive is dropped, and the loss is
    # hop-dependent, so count it rather than letting the distance profile be
    # read across slightly different samples per ring.
    merged.attrs["dropped_by_hop"] = (
        merged.loc[merged["_merge"] == "left_only"].groupby("hop").size().to_dict()
    )
    merged = merged.loc[merged["_merge"] == "both"].drop(columns="_merge")
    merged["did"] = merged["delta"] - merged["control_delta"]
    # Records, not a frame: pandas carries `attrs` through merges and groupbys
    # and compares the values, which is ambiguous for a DataFrame.
    merged.attrs["availability"] = pd.concat(
        [crash_seen.assign(arm="crash"), control_seen.assign(arm="control")],
        ignore_index=True,
    ).to_dict("records")
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


def against_placebo(effect: pd.DataFrame, placebo: pd.DataFrame) -> pd.DataFrame:
    """The headline effect beside the placebo at the same hop and step.

    This is the table to read before quoting any effect, and the reason it is
    a table rather than two tables is that nobody compares two tables. The
    placebo estimates something the crash cannot have caused, so an effect of
    comparable size is not evidence of a crash effect -- it is evidence the
    arms were already diverging.

    `placebo_share` is the ratio `|placebo| / |effect|` and nothing more. It is
    unbounded, exceeds 1 whenever the placebo is larger than the estimate, and
    is **not** a probability or a p-value; do not report it as one. On the
    frozen March-April data the direct effect sits near 0.06 and the 1- and
    2-hop effects above 0.6, so the spillover rows are not reportable as crash
    effects there. Treat anything above roughly a third as unresolved, and read
    the share together with the interval: an effect whose interval already
    spans zero is not rescued by a small placebo.

    A large share means the estimate cannot be separated from pre-existing
    divergence. It does **not** mean the effect is zero.

    Rows are the horizon steps the placebo covers, which is `--leads` of them,
    so this is a check on the early horizon and not the whole profile.
    """
    merged = effect.merge(
        placebo[["hop", "minutes", "effect"]].rename(columns={"effect": "placebo"}),
        on=["hop", "minutes"],
    )
    # `inf`, not NaN, when the effect rounds to zero. An effect of -0.000 beside
    # a placebo of -0.018 is the most unidentified row there is, and dividing by
    # the rounded effect used to blank it -- so the table went quiet on exactly
    # the row it exists to flag, and `idxmax` skipped it when picking the worst.
    with np.errstate(divide="ignore", invalid="ignore"):
        share = merged["placebo"].abs() / merged["effect"].abs()
    merged["placebo_share"] = share.round(2)
    keep = ["hop", "minutes", "effect", "low", "high", "placebo", "placebo_share", "events"]
    return merged[[c for c in keep if c in merged.columns]]


def fixed_cohort(did: pd.DataFrame) -> pd.DataFrame:
    """The events observed at every (hop, step) the table covers.

    An available-case estimate can show an effect fading with distance purely
    because the far rings are measured on a different, better-observed set of
    events. Restricting to a common cohort costs sample size and narrows the
    distance question to one set of events. Report both; they are different
    estimands and the gap between them is itself a finding -- on the frozen
    data the cohort keeps under a fifth of events and most of its rows stop
    excluding zero.

    It fixes the **event** set and nothing else. The roads and the control
    windows still vary step by step, because a row is only emitted where that
    bin was observed: on the frozen data about half of (event, road, hop)
    cells draw on a different set of control windows at different steps. So a
    time profile read off either table is still a profile across changing
    road and control populations, not one population followed through time.
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

    Both frames need `event_id`, `road` and `origin`. The control arm is
    averaged **within event** before the arms are compared, because an event
    with five controls would otherwise count five times -- `summarise` and
    `event_study` weight events equally and this has to match them. On the
    frozen data row-weighting understated the pre-event level imbalance
    threefold (-0.30 against -0.10 event-weighted).

    `std_diff` is the arm gap over the pooled standard deviation. **Do not read
    the conventional "under 0.1 is balanced" bar here.** That convention
    assumes the effect is large next to the covariate spread; on the frozen
    data the pre-event trend differs between arms by -0.23 mph against a
    pooled SD of 3.37, which reports as -0.068 and looks balanced -- while the
    effect being estimated is 0.14 to 0.61 mph. A gap a tenth of an SD is
    larger than the whole result, so read `crash_mean` against `control_mean`
    in mph and compare that to the effect, not `std_diff` against 0.1.

    Road features balance by construction -- a control is the same road -- so
    those rows check that matching did what it claims. Balance on any of this
    says nothing about weather, road works, or whatever made this road have a
    crash in this hour; those stay unobserved.
    """
    if len(segment_ids) != reference.num_roads:
        raise ValueError("segment_ids does not match the grid these windows were built on")
    static = segments.set_index("segment_id").reindex(segment_ids)
    # The matcher's own statistics, not a second implementation of them. The
    # hand-rolled version split the history the other way round and diverged by
    # up to 11 mph at an odd `--history`, so the balance table would have
    # reported a quantity the matcher never optimised.
    level, trend = pre_event_stats(reference.values.numpy(), reference.history)

    def describe(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for event_id, road, origin in zip(
            frame["event_id"], frame["road"], frame["origin"], strict=True
        ):
            road, origin = int(road), int(origin)
            row = {
                "event_id": event_id,
                "pre_level": float(level[origin, road]),
                "pre_trend": float(trend[origin, road]),
            }
            for name in columns:
                row[name] = float(static[name].iloc[road]) if name in static.columns else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    crash = describe(events).groupby("event_id", as_index=False).mean(numeric_only=True)
    control = describe(controls).groupby("event_id", as_index=False).mean(numeric_only=True)

    rows = []
    for name in ["pre_level", "pre_trend", *columns]:
        if name not in crash.columns:
            continue
        a, b = crash[name].dropna(), control[name].dropna()
        if not a.size or not b.size:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append(
            {
                "feature": name,
                "crash_mean": round(float(a.mean()), 3),
                "control_mean": round(float(b.mean()), 3),
                "gap": round(float(a.mean() - b.mean()), 3),
                "std_diff": (
                    round(float((a.mean() - b.mean()) / pooled), 3)
                    if np.isfinite(pooled) and pooled
                    else None
                ),
                "n_crash": int(a.size),
                "n_control": int(b.size),
            }
        )
    return pd.DataFrame(rows)
