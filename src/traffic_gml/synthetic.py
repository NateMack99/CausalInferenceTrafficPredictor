"""Known disruptions injected into real crash-free windows (PR #8).

This is the benchmark the whole study rests on: recovering a known effect is
the evidence that the estimator works, and forecast accuracy is not. It is
deliberately **not** `mock.py`, which fabricates speeds. Here the speeds are
real and held out, and only the disruption is invented, so an estimator is
scored against a truth that exists.

The contract:

- Untreated outcomes are the real observed speeds, kept unmodified in
  `Disruptions.untreated`.
- Injection starts at `origin + 1`, so the whole `history` span is untouched
  and a forecaster reading the treated grid sees **exactly** the inputs it
  would have seen untreated. The injected effect is never an input.
- `Disruptions.truth` records the effect actually realised in the data, after
  the floor at 0 mph, not the effect that was requested. A label that
  disagrees with the array it describes would score every estimator wrong.
- Events never overlap, and are separated by their whole lag-and-horizon span
  across every road. A window's `x` is the full `[history, roads]` grid, so a
  nearby event's injection would otherwise arrive as this event's input. That
  caps how many events a split can hold, at roughly its length in bins over
  `history + horizon`; `settings` records how many were asked for and how many
  fit, and a short split silently yields fewer.

Nothing here estimates anything. Scoring recovery against `truth` is PR #9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .data import BIN_MINUTES
from .events import WEEK_SLOTS, Windows, hop_rings

#: Time profiles. Both are driven by one `duration`, in bins.
SHAPES = ("decay", "plateau")
#: How an event's road is chosen. `congested` is the selection stress test.
PLACEMENTS = ("uniform", "congested")
#: Speeds cannot go negative; the realised effect is recorded after this floor.
FLOOR = 0.0


@dataclass
class Disruptions:
    """An injected benchmark and its ground truth.

    `values` is the treated grid, `untreated` the real one it was copied from.
    `events` is one row per disruption with the arguments that produced it;
    `truth` is one row per observed affected cell, carrying the realised
    effect. `effect_profile` and `settings` record what was asked for, so a
    run is reproducible from this object alone.
    """

    values: np.ndarray
    untreated: np.ndarray
    events: pd.DataFrame
    truth: pd.DataFrame
    settings: dict


def effect_profile(shape: str, horizon: int, duration: float) -> np.ndarray:
    """Per-step multiplier on the hop-0 magnitude, length `horizon`.

    `decay` halves every `duration` bins, so the disruption fades; `plateau`
    holds full strength for `duration` bins and then stops. One parameter
    drives both, which keeps the family small enough to report in a table.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    step = np.arange(horizon, dtype="float64")
    if shape == "decay":
        return 0.5 ** (step / duration)
    if shape == "plateau":
        return (step < duration).astype("float64")
    raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")


def _pre_mean(series: np.ndarray) -> float:
    """Mean of the observed pre-event speeds; `inf` when none were observed.

    `inf` rather than `nan` so a candidate with no history sorts last and never
    reaches the congested pool, without a warning from an all-missing slice.
    """
    finite = series[np.isfinite(series)]
    return float(finite.mean()) if finite.size else float("inf")


def inject(
    reference: Windows,
    edge_index: Tensor,
    events: int = 200,
    magnitude: tuple[float, float] = (4.0, 12.0),
    hops: int = 2,
    hop_decay: float = 0.45,
    shape: str = "decay",
    duration: float = 2.0,
    placement: str = "uniform",
    seed: int = 0,
) -> Disruptions:
    """Inject `events` known slowdowns into `reference`'s real crash-free windows.

    Candidates are exactly the windows `reference` already marks usable, which
    is the same bar a real crash event has to clear: crash-free across its lag
    and target span, with an observed target. A candidate additionally needs an
    observed pre-event history, or there is nothing to forecast from.

    The slowdown is `magnitude` mph at the event road, `hop_decay ** hop` of
    that on each ring out to `hops`, shaped over the horizon by `shape` and
    `duration`. Pass `reference` as the **held-out** split: injecting into
    windows a forecaster trained on measures memorisation, not recovery.

    `placement="congested"` draws only from the third of candidates with the
    lowest pre-event mean speed, so disruptions land on already-slow roads.
    That is the selection pattern a matched estimator is supposed to survive,
    and comparing it against `uniform` is how the study finds out whether it
    does.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement {placement!r}; choose from {PLACEMENTS}")
    low, high = magnitude
    if low > high:
        raise ValueError(f"magnitude must be (low, high), got {magnitude}")

    rng = np.random.default_rng(seed)
    untreated = reference.values.numpy()
    values = untreated.copy()
    history, horizon = reference.history, reference.horizon
    observed = np.isfinite(untreated)
    # A ring cell must be crash-free as well as observed. The event road is
    # crash-free because it came from `usable`, but a ring road is not checked
    # by anything: contamination spreads `exposure_hops` from a real crash and
    # a ring sits up to `hops` from the event, so a crash three or four hops
    # out dirties the ring while leaving the event road clean. Injecting there
    # would stack a fake slowdown on a real one and label only the fake part,
    # and the control arm already demands `clean`, so the two arms would be
    # held to different bars -- biasing the very hop rows this exists to score.
    clean = reference.clean.numpy()
    origins = reference.origins.numpy()
    profile = effect_profile(shape, horizon, duration)

    rings = hop_rings(edge_index, reference.num_roads, hops)

    # Candidates are the reference's own usable (window, road) pairs.
    window_of, road_of = reference.usable.numpy().nonzero()
    if not len(window_of):
        raise ValueError("the reference split has no usable (window, road) pair to inject into")

    # Rank a subsample rather than every candidate: on real data `usable` holds
    # millions of pairs and only the order of a few thousand can matter.
    pool = rng.permutation(len(window_of))[: max(events * 50, events)]
    pre_mean = np.array(
        [
            _pre_mean(untreated[origins[w] - history + 1 : origins[w] + 1, r])
            for w, r in zip(window_of[pool], road_of[pool], strict=True)
        ]
    )
    if placement == "congested":
        slowest = np.argsort(pre_mean, kind="stable")[: max(len(pool) // 3, 1)]
        order = pool[rng.permutation(slowest)]
    else:
        order = pool

    # Reserved bins, across every road. A window's `x` carries the whole
    # `[history, roads]` grid, so an injection anywhere in another event's lag
    # span becomes one of that event's inputs. Separating events by their full
    # lag-and-horizon span is the one rule that rules this out everywhere,
    # rather than only outside the current model's receptive field.
    claimed = np.zeros(values.shape[0], dtype=bool)
    rows: list[dict] = []
    chosen: list[dict] = []

    for index in order:
        if len(chosen) == events:
            break
        window, road = int(window_of[index]), int(road_of[index])
        origin = int(origins[window])
        if not observed[origin - history + 1 : origin + 1, road].any():
            continue

        guard = slice(origin - history + 1, origin + 1 + horizon)
        if claimed[guard].any():
            continue
        claimed[guard] = True

        size = float(rng.uniform(low, high))
        event_id = f"SYN{len(chosen):05d}"
        for hop in range(hops + 1):
            # Exclusive rings, so a road carries one hop label and one effect.
            ring = rings[hop][road]
            if hop:
                ring = np.setdiff1d(ring, rings[hop - 1][road])
            if not len(ring):
                continue
            wanted = size * hop_decay**hop * profile
            for step, want in enumerate(wanted):
                bin_ = origin + 1 + step
                seen = ring[observed[bin_, ring] & clean[bin_, ring]]
                # `want == 0` is labelled, not skipped: the write is a no-op so
                # the grid is unchanged, but the truth row records an honest
                # zero. Skipping it left the `plateau` shape's off-period
                # unscored, so overshoot after the plateau was never penalised.
                if not len(seen) or want < 0:
                    continue
                before = untreated[bin_, seen]
                after = np.maximum(before - want, FLOOR)
                values[bin_, seen] = after
                rows += [
                    {
                        "event_id": event_id,
                        "origin": origin,
                        "road": int(r),
                        "hop": hop,
                        "minutes": (step + 1) * BIN_MINUTES,
                        "untreated": float(u),
                        "treated": float(v),
                        # Realised, not requested: the floor can clip a slowdown.
                        "true_effect": float(v - u),
                        "requested_effect": -float(want),
                    }
                    for r, u, v in zip(seen, before, after, strict=True)
                ]
        chosen.append(
            {
                "event_id": event_id,
                "origin": origin,
                "road": road,
                "magnitude": size,
                "shape": shape,
                "duration": duration,
                "hop_decay": hop_decay,
                "hops": hops,
                "placement": placement,
            }
        )

    if not chosen:
        raise ValueError(
            "no candidate window could take a disruption; lower --events or widen the split"
        )

    settings = {
        "events": len(chosen),
        "requested_events": events,
        "magnitude": [low, high],
        "hops": hops,
        "hop_decay": hop_decay,
        "shape": shape,
        "duration": duration,
        "placement": placement,
        "seed": seed,
        "floor_mph": FLOOR,
        "profile": [round(float(p), 6) for p in profile],
    }
    return Disruptions(
        values=values,
        untreated=untreated,
        events=pd.DataFrame(chosen),
        truth=pd.DataFrame(rows),
        settings=settings,
    )


def exposure_labels(disruptions: Disruptions) -> np.ndarray:
    """`[bins, roads]` exposure states for the injected events, as
    `events.EXPOSURES` indexes them: ring `hop` becomes label `hop + 1`.

    Lets an exposure-conditioned model be asked for the outcome under the
    injected exposure and under none, where the difference has a known answer.
    """
    truth = disruptions.truth
    grid = np.zeros(disruptions.values.shape, dtype=np.int8)
    bins = truth["origin"].to_numpy() + truth["minutes"].to_numpy() // BIN_MINUTES
    grid[bins, truth["road"].to_numpy()] = truth["hop"].to_numpy() + 1
    return grid


def treated_windows(reference: Windows, disruptions: Disruptions) -> Windows:
    """Event windows over the treated grid, scoring every road a disruption reached.

    The returned set shares `reference`'s normalisation statistics and clock
    features, so a model trained on `reference` can be applied directly.

    `usable` is per-road and broadcasts over the horizon, so a road that
    carries a known effect at one step is unmasked at every step. That can
    yield rows with no truth label -- a step where the cell was unobserved or
    crash-contaminated. `evaluate.effect_recovery` inner-joins on
    (event_id, road, minutes), so those rows are dropped before anything is
    scored; do not aggregate this frame without joining it to `truth` first.
    """
    events = disruptions.events
    usable = np.zeros((len(events), reference.num_roads), dtype=bool)
    position = {event: i for i, event in enumerate(events["event_id"])}
    truth = disruptions.truth
    usable[[position[e] for e in truth["event_id"]], truth["road"].to_numpy()] = True

    data = reference.at(events["origin"].to_numpy(), usable)
    data.values = torch.from_numpy(disruptions.values)
    return data


def matched_controls(
    reference: Windows,
    disruptions: Disruptions,
    fitted: np.ndarray,
    controls: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, Windows]:
    """Untreated control windows for every road a disruption reached.

    Held to the same rule as a real crash's control in `event_datasets`: the
    same road at the same slot of the week in another week, crash-free across
    its whole span, and never a cell in `fitted`.

    `fitted` is whatever the caller passes, normally
    `events.fitted_cells(train_set)` -- which covers cells a gradient step
    touched and **not** the validation period that early stopping selected on.
    A control one or more whole weeks from a held-out event frequently lands in
    validation, so this is the usual case rather than an edge one. Pass
    `fitted_cells(train) | fitted_cells(val)` to exclude those too, and expect
    control supply to fall sharply when you do.

    These windows read the **untreated** grid, so each is a genuine
    no-disruption window, and the correction they supply is the forecaster's
    own bias measured where nothing happened — the same quantity
    `evaluate.placebo_corrected` subtracts for a real crash. Unlike
    `untreated_windows`, a control does **not** share the event's inputs, so
    the prediction does not cancel and the comparison is not an identity.

    **The held-out span has to exceed a week.** A control sits at the same slot
    of the week, so its candidates are 1,008 bins away; if every sibling bin
    falls in the training split it is fitted and rejected, and the road gets no
    control at all. Injecting into a test split shorter than that leaves this
    function with nothing to draw, which is a property of the split, not a bug.

    Returns one row per (event, road, control origin), and the window set.
    """
    history, horizon = reference.history, reference.horizon
    values = disruptions.untreated
    clean = reference.clean.numpy()
    slot = reference.slot.numpy()
    n_bins = values.shape[0]
    rng = np.random.default_rng(seed)

    # One pass over the slot grid instead of one per (event, road).
    same_slot = {int(s): np.flatnonzero(slot == s) for s in np.unique(slot)}
    origin_of = dict(zip(disruptions.events["event_id"], disruptions.events["origin"], strict=True))

    pairs: list[dict] = []
    wanted = disruptions.truth[["event_id", "road"]].drop_duplicates()
    for event_id, road in zip(wanted["event_id"], wanted["road"], strict=True):
        origin, road = int(origin_of[event_id]), int(road)
        eligible = [
            int(t)
            for t in same_slot[int(slot[origin])]
            if abs(t - origin) >= history + horizon
            and t - history + 1 >= 0
            and t + horizon < n_bins
            and not fitted[t + 1 : t + 1 + horizon, road].any()
            and clean[t - history + 1 : t + horizon + 1, road].all()
            and np.isfinite(values[t + 1 : t + 1 + horizon, road]).any()
        ]
        if not eligible:
            continue
        chosen = rng.choice(eligible, size=min(controls, len(eligible)), replace=False)
        pairs += [
            {"event_id": event_id, "road": road, "origin": int(t)} for t in np.atleast_1d(chosen)
        ]

    table = pd.DataFrame(pairs)
    if table.empty:
        raise ValueError(
            "no disruption road has an eligible untreated control window. A control "
            "sits one or more whole weeks from its event, so if the held-out split is "
            f"shorter than a week every sibling bin is a fitted training cell. This "
            f"split spans {n_bins:,} bins with {WEEK_SLOTS:,} per week; widen the "
            "speed window, or inject into a longer held-out period."
        )

    # One window per distinct origin, unmasking every road wanted there. Roads
    # of one event share candidate bins, so a row per (origin, road) rebuilt
    # the same origin about seven times and every model paid for it again. The
    # mask does not reach the prediction -- `x_mask` is built from `clean` and
    # `isfinite`, never from `usable` -- so the rows this yields are identical.
    origins = np.unique(table["origin"].to_numpy())
    only = np.zeros((len(origins), reference.num_roads), dtype=bool)
    only[np.searchsorted(origins, table["origin"].to_numpy()), table["road"].to_numpy()] = True
    windows = reference.at(origins, only)
    windows.values = torch.from_numpy(values)
    return table, windows


def untreated_windows(reference: Windows, disruptions: Disruptions) -> Windows:
    """The same windows over the untouched grid: the no-disruption outcomes.

    **Do not difference these residuals against the treated ones and call the
    result an effect estimate.** Injection leaves `x` identical in both arms by
    construction, so a forecaster returns the same prediction for each, and

        (y_treated - ŷ) - (y_untreated - ŷ) = y_treated - y_untreated

    which is the injected truth itself. Effect MAE against `truth` then comes
    out at exactly 0.000 mph and has measured nothing — the prediction has
    cancelled out. This was checked, and it does.

    What these windows are for is the forecaster's own bias on the very cells
    the benchmark scores: `y_untreated - ŷ` is an oracle control, available
    here and never available for a real crash. PR #9 has to score the estimator
    the study actually has — the residual against the no-disruption forecast,
    bias-corrected with matched controls drawn the same way as for real crashes
    — and may use this arm only to show how much of the remaining error is
    forecast bias rather than estimator bias.
    """
    data = treated_windows(reference, disruptions)
    data.values = torch.from_numpy(disruptions.untreated)
    return data
