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

## Shared model contract (proposed)

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

Freeze schema names and shapes before PRs #4–6; update this document if the implementation differs.
