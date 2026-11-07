# Methodology

## Research question

How much do traffic crashes change speed on the affected road and connected roads, and does graph learning improve estimates of these spillovers?

## Counterfactual model

Train a non-graph baseline, a GCN/GraphSAGE baseline, and a **spatiotemporal GNN** on crash-free windows. Given the road graph and **pre-event** speed history, predict future speeds under normal/no-crash conditions. Optimize forecasting MSE or MAE.

For a crash event, estimate the speed effect for each road and horizon as:

`estimated effect = observed post-crash speed − predicted no-crash speed`

Negative values mean slower traffic. Forecast accuracy alone does not establish a causal effect.

## Causal comparisons

Use matched accident-free windows and a matched difference-in-differences/event-study baseline. Check pre-trends, placebo events, simultaneous crashes, selection bias, and uncertainty grouped by crash event. Report assumptions and limitations.

## Semi-synthetic validation (core)

Start with held-out real crash-free windows; inject known slowdowns that vary by graph distance, direction, magnitude, and duration. Compare estimated effects with the known effects, using effect MAE/RMSE. Do not train on the benchmark test windows.

## Main experiments

- Speed forecasting MAE/RMSE on held-out crash-free windows.
- Semi-synthetic treatment-effect MAE/RMSE.
- Real-event effects by hop distance and time after the crash.
- Graph/no-graph and rewired-edge ablations; temporal and history-length ablations.
- Matching sensitivity, placebo checks, and event-clustered uncertainty.
