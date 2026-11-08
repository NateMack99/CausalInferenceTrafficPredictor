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
- `roads.py`: provision raw inputs; download the street network; build the speed-observed
  road reference.
- `segments.py`: segment graph, centrality, crash-to-segment matching, and the map.
- `plot.py`: window-graph congestion figures.
- `mock.py`: fixture tables in the processed schema, for development only.

Create modules only when their PR requires them. Scripts call these functions; no duplicate logic in scripts.

### Status

Ten of the twelve listed modules exist; `causal.py` and `synthetic.py` are not written. `data.py` retrieves, cleans and aligns; `graph.py` builds connectivity and the PyG `edge_index`; `events.py` joins the three streams (`processed_tables`) and builds crash-free training windows plus crash/control event windows; `models.py`, `train.py` and `evaluate.py` cover forecasting, optimisation and metrics. `evaluate.py` also holds the crash-effect estimate and the report figures.

`mock.py` writes fixture tables so the models run without the portal. It is not PR #8's `synthetic.py`, which injects known effects into *real* crash-free windows.

## Shared model contract

Left column is the name the course brief uses; right column is what the code calls it.

| Contract | In this repo |
|---|---|
| `road_ids` | `np.sort(segments.segment_id.unique())`; its position is the road index used by every tensor and by `edge_index` |
| `edge_index` | PyG `[2, edges]`, built by `graph.build_edge_index`, directed **as given** |
| `x` | `[batch, history, roads, features]`, batch key `x` |
| `y` | `[batch, horizon, roads]`, batch key `y` |
| `mask` | batch keys `x_mask` and `y_mask`, each matching its tensor's time/road axes |
| `event_id` | `crashes.crash_record_id` |
| `event_time` | `crashes.crash_time`, naive local, floored to a 10-minute bin |
| Time | Naive local (America/Chicago), aligned to 10-minute bins (`FREQ`) |

Pre-event features may include recent speeds, time-of-day, and static road features. A forecaster must not see post-event outcomes when predicting the no-accident scenario.

PR #3 implements this, asserted in `tests/test_smoke.py`. A batch carries `x`, `x_mask`, `y`, `y_mask` and `origin` at the shapes above, `x` channels being `speed, tod_sin, tod_cos, dow_sin, dow_cos` then any static road columns. Speeds are normalised on crash-free training cells only, a missing speed is zero (the training mean) with a false mask, and a naive `time` column is read as local, which is what the portal publishes and `csv_to_parquet` preserves. The basic GNN flattens `history` into per-road features; the dimension stays separate so PR #6 can consume it directly.

Three deliberate asymmetries: message passing uses `edge_index` **as given**, so it is directed, while crash contamination spreads **undirected**, over-reaching being the safe error when excluding training data; a crash is assumed to contaminate for `horizon` bins after it is reported, which is an assumption and not a measurement; and `--layers` message-passing rounds read `--layers` hops of input, so `--exposure-hops` defaults to 2 to match the default 2-layer model. It stays a data flag rather than being derived from `--layers`, so that changing the architecture cannot silently change the splits; `train.py` warns when masking is narrower than the receptive field.

Freeze schema names and shapes before PRs #4–6; update this document if the implementation differs.
