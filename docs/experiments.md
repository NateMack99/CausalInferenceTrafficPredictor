# Experiments

Shared settings, the exclusion rules, and the frozen midpoint baseline. Change a
convention here and every comparison across runs breaks, so change it deliberately.

## Shared settings

Agreed defaults. A run that departs from them has to say so in its write-up.

| Setting | Value | Where |
| --- | --- | --- |
| Bin width | 10 minutes | `data.FREQ` |
| History `L` | 12 bins (2 h) | `--history` |
| Horizon `H` | 6 bins (1 h) | `--horizon` |
| Stride | 1 bin | `--stride` |
| Crash masking | 2 graph hops, `H` bins forward | `--exposure-hops` |
| Splits | chronological 70/15/15, gap `ceil((L+H)/stride)` windows | `events.build_windows` |
| Seed | 42 | `--seed` |
| Static features | `length_m degree betweenness` | `--static-features` |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | `train.fit` |
| Loss | masked L1 on normalised speeds | `--loss mae` |
| Metrics | masked MAE and RMSE in mph, overall and per horizon step | `evaluate.evaluate` |
| Controls per crash | 5 requested (about 1.9 available) | `--controls` |
| Effect model | `gcn` | `--effect-model` |

Conventions that are not flags:

- **Timestamps are naive local time** (America/Chicago), never UTC. See `docs/data.md`.
- **Splits are chronological, never random.** Windows overlap at stride 1, so a
  random split leaks neighbouring targets into training.
- **Missing speeds are masked, never imputed.** A masked target contributes
  nothing to the loss or to any metric. Do not fill and then score.
- **Normalisation statistics come from the training split only.**
- **A control is the same road at the same slot of the week in another week**,
  crash-free across its whole span, and never a window the model trained on.
- **Forecast error and effect error are reported separately.** Lower MAE is not
  evidence of lower causal bias.

## Exclusion rules

Two filtering stages. The first reduces the raw crash table to crashes that sit on
the monitored network (`evaluate.coverage_funnel`); the second builds analysable
events (`events.event_datasets`). Counts are from the frozen baseline below.

### Stage 1 — onto the monitored network

| Stage | Crashes | Kept |
| --- | ---: | ---: |
| Crashes in the study period | 17,658 | 100.0% |
| With usable coordinates | 17,479 | 99.0% |
| Matched to a monitored segment | 3,668 | 20.8% |
| On a segment with speed data, window fits | 3,666 | 20.8% |

The 79% loss at segment matching is **scope, not error**: the speed feed covers
monitored arterials, not every Chicago street.

### Stage 2 — into the analysis sample

Applied in this order, each to what survived the last:

| Removed | Rule | Why it exists |
| ---: | --- | --- |
| 0 | crash segment not in the speed grid | no outcome series to forecast |
| 2 | window runs off the start or end of the series | needs `L` bins before and `H` after |
| 15 | another crash already covers this bin and road | one disturbance, counted once |
| **693** | an earlier crash contaminates the pre-crash history | a depressed history biases the forecast it is made from |
| 171 | no observed speed anywhere in the pre-crash history | nothing to forecast from |
| 258 | no crash-free, held-out, same-slot control exists | without a control the residual is not bias-corrected |
| | **analysis sample** | **2,529** |

The dominant loss is the contaminated pre-crash history: 693 crashes, 19% of those
entering stage 2. It is a direct consequence of masking 2 hops wide and `H` bins
forward — a generous assumption chosen so contamination errs toward exclusion.
Measuring the real contamination duration would recover some of these.

**The analysis sample is 2,529.** Requiring *every* pre and post bin observed
leaves 297, a strict subset reported for reference. Do not quote 297 as the sample
size: the estimator masks unobserved targets rather than requiring complete series.

## Frozen midpoint baseline

The November 9, 2026 midpoint review. Preserved so later work has a fixed point
to compare against.

- **Code:** `b7bcfee` (`Report neighbour coverage and nest the funnel`). The run's
  `summary.json` records `3f141b4-dirty` because the code was not yet committed when
  it ran; the content is identical to `b7bcfee`.
- **Data:** `data/parquet/chicago-speeds`, 7,699,161 speed rows spanning
  `2025-03-01 00:01:03` to `2025-04-30 23:50:25`; 3,668 matched crashes; 879
  segments; 4,726 directed arcs.
- **Graph:** `graph_20250301T000000_20250501T000000_all_20261108T203320189921Z`.

```
uv run python scripts/midpoint.py \
  --speeds data/parquet/chicago-speeds \
  --graph outputs/<stamp>_segments.graphml \
  --crashes outputs/<stamp>_crash_segments.csv \
  --window 2025-03-01 2025-05-01 \
  --static-features length_m degree betweenness \
  --epochs 20 --patience 4 --seed 42 --controls 5 \
  --out outputs/midpoint
```

Held-out forecast error (identical splits, 6,118/1,296/1,316 windows, 3,033,666
test targets):

| Model | MAE | RMSE |
| --- | ---: | ---: |
| sage | 3.385 | 4.680 |
| gcn | 3.427 | 4.709 |
| mlp | 3.448 | 4.741 |
| historical_mean | 3.546 | 4.873 |
| persistence | 4.557 | 6.232 |

Crash-segment effect, crash arm minus matched control, 2,529 events against 4,813
control windows: pooled **−0.43 mph**, largest at 20–30 minutes (**−0.641**,
nominal 95% interval [−0.94, −0.35]).

Spillover feasibility: of 2,529 events, 2,437 (96.4%) have a 1-hop neighbour
observed in at least 3 of 6 post bins; 1,459 (57.7%) have one observed in every
post bin with a history to forecast from, and 1,628 (64.4%) at 2 hops.

**What this baseline does not establish:** the controlled gcn–mlp gap is 0.022 mph
on one seed, so no graph advantage is shown; effects are crash-segment only, so
nothing here measures spillover; and the estimate moves with the control draw
(pooled −0.34 to −0.43 across three draws). See `docs/methodology.md`.
