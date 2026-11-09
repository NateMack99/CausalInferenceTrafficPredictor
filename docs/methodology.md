# Methodology

## Research question

How much do traffic crashes change speed on the affected road and on connected roads, and
does graph learning improve estimates of these spillovers?

## Counterfactual estimator

Train a forecaster on **accident-free target windows** using **only pre-event
information** — recent speeds, time-of-day/day-of-week, static road features, and the road
graph — to predict the speed that would have held with no accident, `ŷ⁽⁰⁾`. For each
affected road and horizon step, the estimated effect is

```
τ̂ = y_observed − ŷ⁽⁰⁾
```

A negative `τ̂` is a slowdown **relative to the estimated no-accident condition**. A
forecaster never sees post-event outcomes, and crash-contaminated cells are kept out of
training targets, the historical-mean table, and the normalisation statistics
(`events.build_windows`).

`τ̂` alone is not a causal effect. It also carries the forecaster's own bias, so the
reported estimate subtracts the same quantity measured on matched accident-free control
windows that the model did not train on:

```
effect = mean τ̂ (crash windows) − mean τ̂ (matched control windows)
```

Forecast error and effect error are different quantities and are always reported
separately.

## What is implemented

PR #3 plus `scripts/midpoint.py`:

- Forecasters `persistence`, `historical_mean`, `mlp`, `gcn`, `sage` (`models.py`), all
  trained on identical splits with the same seed. `mlp` is `gcn` without edges, so that
  pair isolates the graph; `sage` has more parameters per layer.
- Crash-free window construction with chronological, gap-separated train/val/test splits
  and crash masking that spreads `--exposure-hops` across the graph and `--horizon` bins
  forward (`events.py`). Ten-minute bins; see `train.py --help` for the window flags.
- Forecast MAE/RMSE in mph, overall and per horizon step (`evaluate.evaluate`).
- `τ̂` per window, road and step (`evaluate.effects`), and the crash-minus-matched-control
  estimate with a 95% interval (`evaluate.placebo_corrected`).
- A nested coverage funnel ending at the analysis sample, and per-hop neighbour
  observability for that sample (`evaluate.coverage_funnel`, `evaluate.neighbour_coverage`).
  The funnel's stricter rows sit *inside* the analysis sample: requiring every bin observed
  is much stricter than the estimator is, since a masked target contributes nothing to its
  bin, so reading the last row as the sample size understates it several-fold.

Limits of what exists: effects are measured on the crash segment only, intervals are
event-level and not clustered, and a crash is assumed to contaminate for `horizon` bins
after it is reported — an assumption, not a measurement.

The estimate is also sensitive to **which** accident-free windows are drawn as controls.
Redrawing them moved per-bin estimates by up to 0.16 mph and changed how many bins
excluded zero, while the crash arm stayed within 0.03 mph. Control supply is the binding
constraint: matching on the same road and week slot, crash-free and never trained on,
yields about two candidates per crash over a two-month window, so `--controls` above that
buys little and a longer span buys more than a larger flag. Per-bin significance should not
be read as a finding from a single draw; report the pooled estimate and the horizon shape,
and treat the control draw as a source of uncertainty the interval does not cover.

No results are reported in this document; the figures and tables a run produces are run
outputs, not validated effects.

## What is planned

Not yet written (see `PROJECT_ROADMAP.md` for the sprint each belongs to):

- **Spatiotemporal GNN** — message passing plus a GRU or temporal convolution, trained on
  the same accident-free windows and compared on the same splits (PR #6).
- **Matched causal baseline** — matched difference-in-differences / event study as a
  non-graph comparison, with pre-trend checks (`causal.py`, PR #7).
- **Semi-synthetic benchmark** — defined below (`synthetic.py`, PR #8).
- **Effect evaluation, robustness, ablations** — effect-recovery scoring, the checks
  listed under *Assumptions and limits*, and graph/no-graph, rewired-edge, history-length
  and horizon ablations (PRs #9–#12).

## Semi-synthetic validation plan (core)

This is the plan, not a validated result. Nothing below has been run.

1. Take **real accident-free windows**, held out from training.
2. Inject **known** slowdowns on a chosen road and propagate them across connected roads,
   varying magnitude, duration, graph distance, and direction.
3. Save the unmodified speeds and the injected effects as ground truth.
4. Give the counterfactual predictor only pre-event inputs from the injected series. The
   injected effects are **never** revealed to it, and it is never trained on benchmark
   test windows.
5. Score **effect recovery** with effect MAE/RMSE against the known effects, reported
   **separately** from forecasting MAE/RMSE, and compare non-graph, basic GNN, and
   spatiotemporal estimates on the same windows.

Recovering known injected effects is the main evidence that the estimator works; good
forecasting accuracy on its own is not.

## Assumptions and limits

These effects are observational, so they are assumption-dependent:

- Matched accident-free windows have to approximate the speeds that would have occurred
  without the crash. They adjust for observed congestion patterns, not every hidden cause.
- Overlapping or nearby simultaneous crashes have to be excluded or flagged.
- Pre-event trends, placebo windows, and selection into which crashes are usable all need
  checking before any estimate is read as causal.
- Uncertainty has to be clustered by crash event once neighbour roads are in play.

Report these alongside every estimate. The research rules the code is held to are in
[AGENTS.md](../AGENTS.md).
