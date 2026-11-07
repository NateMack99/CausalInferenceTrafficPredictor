# Architecture

## Pipeline

Chicago crash + speed data → cleaned, aligned road observations → road graph + event/control windows → no-accident forecasting → causal effect estimation → semi-synthetic and real-world evaluation.

## Module boundaries

- `data.py`: retrieve, clean, align speeds and crashes.
- `graph.py`: build road-segment connectivity; map crashes to segments.
- `events.py`: build accident, control, and crash-free training windows.
- `models.py`: non-graph, GNN, and spatiotemporal forecasters.
- `train.py`: optimization, checkpointing, and forecasting.
- `causal.py`: counterfactual estimates and matched causal baselines.
- `synthetic.py`: inject known disruptions into real crash-free windows.
- `evaluate.py`: prediction metrics, treatment-effect metrics, and plots.

Create modules only when their PR requires them. Scripts call these functions; no duplicate logic in scripts.

### Status after PR #3

`events.py` (crash-free windows only; PR #4 adds accident and control windows), `models.py`, `train.py` and `evaluate.py` (forecast metrics only; PR #9 adds effect metrics) exist in minimal form. `data.py` and `graph.py` are deliberately **not** created, so PR #3 adds nothing to the files PR #1 and PR #2 are writing this sprint; the four pieces it needed — `load_tables`, `align_grid`, `build_edge_index`, `neighbourhood` — sit in `events.py` behind a scope note and move out when those land. `mock.py` writes fixture tables so the models run meanwhile; it is not PR #8's `synthetic.py`, which injects known effects into *real* crash-free windows.

## Shared model contract

| Item | Representation |
|---|---|
| Road index | Stable ID mapped to a fixed integer index |
| Time | UTC, aligned to 10-minute bins |
| Inputs `x` | `[batch, history, roads, features]` |
| Future targets `y` | `[batch, horizon, roads]` |
| Missing observations | Boolean mask with the same time/road dimensions |
| Road graph | PyG `edge_index` shaped `[2, edges]` |
| Events | Unique event ID, timestamp, road index, metadata |

Pre-event features may include recent speeds, time-of-day, and static road features. A forecaster must not see post-event outcomes when predicting the no-accident scenario.

PR #3 implements this, asserted in `tests/test_smoke.py`. A batch carries `x`, `x_mask`, `y`, `y_mask` and `origin` at the shapes above, `x` channels being `speed, tod_sin, tod_cos, dow_sin, dow_cos` then any static road columns. Speeds are normalised on crash-free training cells only, a missing speed is zero (the training mean) with a false mask, and a tz-naive `time` column is read as UTC with clock features derived in `America/Chicago`. The basic GNN flattens `history` into per-road features; the dimension stays separate so PR #6 can consume it directly.

Three deliberate asymmetries: message passing uses `edge_index` **as given**, so it is directed, while crash contamination spreads **undirected**, over-reaching being the safe error when excluding training data; a crash is assumed to contaminate for `horizon` bins after it is reported, which is an assumption and not a measurement; and `--layers` message-passing rounds read `--layers` hops of input, so `--exposure-hops` defaults to 2 to match the default 2-layer model. It stays a data flag rather than being derived from `--layers`, so that changing the architecture cannot silently change the splits; `train.py` warns when the two disagree.

Freeze schema names and shapes before PRs #4–6; update this document if the implementation differs.
