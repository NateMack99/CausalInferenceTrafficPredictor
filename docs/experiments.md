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
| Crash masking | 2 graph hops, `H` bins forward, on `x` **and** `y` | `--exposure-hops` |
| Zero speeds | kept (midpoint convention) | `--drop-zero-speed` |
| Splits | chronological 70/15/15, gap `ceil((L+H)/stride)` windows | `events.build_windows` |
| Seed | 42 | `--seed` |
| Static features | `length_m degree betweenness` | `--static-features` |
| Neighbour summary | off | `--neighbour-hops` |
| Graph ablation | `real` | `--graph-ablation` |
| Ablation seed | follows `--seed` | `--graph-seed` |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | `train.fit` |
| Loss | masked L1 on normalised speeds | `--loss mae` |
| Metrics | masked MAE and RMSE in mph, overall and per horizon step | `evaluate.evaluate` |
| Controls per crash | 5 requested (about 1.9 available) | `--controls` |
| Effect model | `gcn` | `--effect-model` |
| Benchmark events | 200 | `benchmark.py --events` |
| Benchmark controls | 5 per event road | `benchmark.py --controls` |
| Injected rings | 2, decaying 0.45 per hop | `--hops`, `--hop-decay` |
| Synthetic null | `--hop-decay 0`, rings still labelled | `--hop-decay` |
| Injected time shape | `decay`, half-life 2 bins | `--shape`, `--duration` |
| Event placement | `uniform` | `--placement` |
| Event-study rings | 2, must not exceed `--exposure-hops` | `event_study.py --hops` |
| Event-study controls | 5 per crash, held-out rule waived | `event_study.py --controls` |
| Control matching | pre-event level and trend | `match_pre_event` |
| Pre-trend placebo | 3 lead bins | `--leads` |

Conventions that are not flags:

- **An ablation arm drops the graph-derived static features.** `degree` and
  `betweenness` are not permuted, so under `--graph-ablation permuted` they
  describe a graph the model does not have. Run those arms with
  `--static-features length_m`; the entry points warn otherwise.
- **One permutation is not a permutation test.** `--graph-seed` draws another
  without also redrawing the model initialisation or the control windows, which
  `--seed` would. A real-against-permuted gap from a single draw has no noise
  floor, and `docs/methodology.md` already records that redrawing controls alone
  moves per-bin estimates by up to 0.16 mph. Report the gap as a single draw, or
  average over several `--graph-seed` values.
- **Coverage tables always describe the real graph.** The funnel and the
  neighbour-observability table are computed on the real `edge_index` even in an
  ablation arm, so hop-ring sample sizes stay comparable across arms.

- **The synthetic null needs no separate generator.** `--hop-decay 0` keeps
  the crash-road effect and sets every outer ring to exactly zero, and because
  `inject` labels a zero-effect ring rather than skipping it, rings 1 and 2
  still arrive as truth rows. Any neighbour effect recovered against them is
  spurious by construction, which is what Benchmark 1 measures. `FLOOR` is
  0.0 mph, so a zero-effect write cannot clip an observed speed.
- **The neighbour-mean channel is graph-derived, so an ablation arm must turn
  it off.** `--neighbour-hops k` appends one input channel holding the mean
  speed over each road's `k`-hop neighbourhood, built from the real graph and
  from masked cells only. It is the fair-baseline input the contract asks for:
  without it `mlp` and `tgru` see only their own road, so a `gcn`-`mlp` gap
  mixes connectivity with one model simply being able to see its neighbours.
  Default off, and **no run has used it yet**.
- **A control window's road must be crash-free within the same 2 undirected
  hops**, across its whole lag-and-horizon span, not merely on its own road.
  Control admissibility reads the same `crash_cells` grid that masks training
  cells, so the control no-exposure radius always equals `--exposure-hops`.

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
- **A benchmark control may sit in the validation period.** `fitted_cells(train)` covers
  cells a gradient step touched, not the cells early stopping selected on, and a control
  one or more whole weeks from a held-out event often lands in validation. Pass
  `fitted_cells(train) | fitted_cells(val)` to exclude them, and expect control supply to
  fall sharply.

## Eligibility

Three different questions, often confused because one funnel reports them. A
crash can be an eligible **event** while most of its cells are ineligible
**outcomes**, and a window can be an eligible **control** for one road and not
another.

| Eligibility | Asks | Rule | Where |
| --- | --- | --- | --- |
| Crash event | can this crash be studied? | matched to a monitored segment, on the speed grid, `history` bins before and `horizon` after fit, pre-crash history crash-free, at least one observed bin in that history, first crash in its (bin, road) | `events.event_datasets` |
| Observed outcome | is this cell scored? | the speed is observed and the road is unmasked for that window | `Windows.__getitem__`, `y_mask` |
| Control match | can this window stand in? | same road, same slot of the week, at least `history + horizon` bins away, in bounds, crash-free across its whole span, an observed target, and for a model-based estimate not a fitted cell | `events.event_datasets` |

Two consequences worth stating, because both have been misread:

- **A masked outcome does not shrink the event sample.** The estimator masks
  unobserved targets rather than requiring complete series, so requiring every
  bin observed gives a far smaller number that is not the sample size. The
  funnel nests the strict rows inside the analysis sample to make this visible.
- **The pre-history threshold is one observed bin, not a full history.** The
  requirement is that the lag span is crash-free and has something in it, not
  that it is complete.

Exposure and masking: a crash contaminates its own road and every road within
`--exposure-hops` (2) for `--horizon` (6) bins after it is reported, undirected,
and that mask applies to **inputs and targets alike**. Over-reaching is the
safe error when excluding training data. The forward extent is an assumption,
not a measurement, and the crash-time rounding recorded below means the
backward extent is not exact either.

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

## Matched event study, March–April 2025 (PR #7)

The first real-data result since the midpoint. Run on the frozen
`data/processed` set with `--hops 2 --controls 5`, seed 42, no forecaster
involved. **The direct effect survives its checks; the spillover does not.**

`event_datasets` is called with `held_out=False`, because that rule keeps
controls out of cells a *forecaster* was fitted on and this estimator has
none. Leaving it on cost more than it sounds:

| | events | controls/event | control offset | pre-trend balance |
| --- | ---: | ---: | ---: | ---: |
| `held_out=True` | 2,527 | 1.90 | **+3.29 weeks**, 88% later | −0.068 |
| `held_out=False` | 2,720 | 4.72 | −0.15 weeks, 48% later | −0.047 |

Effect by hop ring, event-clustered 95% intervals, against the pre-trend
placebo at the same step:

| hop | effect at 10 min | interval | placebo | placebo share |
| ---: | ---: | --- | ---: | ---: |
| 0 | −0.255 | [−0.498, −0.012] | −0.069 | 0.27 |
| 1 | −0.137 | [−0.278, +0.004] | −0.090 | **0.66** |
| 2 | −0.146 | [−0.265, −0.027] | −0.086 | **0.59** |

All six hop-0 rows exclude zero and their placebo shares run 0.17–0.27, so the
crash-road effect is between −0.26 and −0.68 mph and is not explained by
pre-existing divergence. The hop-1 and hop-2 rows are a different matter: their
placebos are 59–66% of the estimates at 10 minutes, and hop 2 at 30 minutes has
a placebo **575%** of its effect. `covariate_balance` agrees — the crash arm's
pre-window trend is −0.315 mph against the control arm's −0.136, a gap of
−0.179, which is larger than the entire hop-1 estimate.

**No spillover number from this table may be written up as evidence that
congestion spreads across the graph.** What it supports is that crash roads slow
and that the measured neighbour difference is, at this sample and specification,
indistinguishable from arms that were already diverging.

Sample: 2,720 events, 12,827 control pairs, 879 roads, 146,007 estimated cells
over 2,713 events. 0.6% of cells carry a second reported crash on the same road
inside the window. The fixed cohort — events observed at every hop and step — is
671 of 2,713 (24.7%), and most of its rows stop excluding zero.

Controls are matched on pre-event speed level and trend rather than drawn at
random. That was done to close the pre-event divergence above, and it half
worked. On the crash road it worked well: the placebo at 10 minutes falls from
-0.069 to -0.012 and its share of the effect from 0.27 to **0.06**, so the
direct effect is cleanly identified. The cost is attenuation at the earliest
bins, where the 10- and 20-minute rows now span zero; the effect survives at 30
to 60 minutes at -0.30 to -0.57 mph.

It did **not** rescue the spillover. At 1 hop the placebo share rises from 0.66
to **1.59**, so the placebo is now larger than the effect. The reason is
mechanical: matching is computed on the **crash road's** pre-event state, and
the 1- and 2-hop effects are measured on neighbour roads whose pre-event state
nothing matches on. Matching the neighbourhood rather than the crash road is
the obvious next step and is not implemented.

Matching the **neighbourhood** instead of the crash road was tried next, since
the hop effects are measured on neighbour roads. `match_hops` widens the match
to the mean pre-event level and trend over a road's k-hop neighbourhood. It
moves the problem around and does not solve it. Placebo share at 10 minutes,
same 2,720 events throughout, only the control draw changes:

| matched on | hop 0 | hop 1 | hop 2 | flag |
| --- | ---: | ---: | ---: | --- |
| crash road (default) | **0.04** | 1.55 | 0.69 | none |
| 1-hop neighbourhood | 0.31 | 1.39 | 0.84 | `--match-hops 1` |
| 2-hop neighbourhood | 0.23 | 0.74 | 0.60 | `--match-hops 2` |
| random draw | 0.27 | 0.66 | 0.59 | `--random-controls` |

Each row is one `scripts/event_study.py --data data/processed --hops 2
--controls 5` run with the flag shown. The random row is kept because PR #10
asks for a matching-rule sensitivity, not because it is a better answer: it
gives a larger 1-hop number and a worse direct effect.

**No specification brings a spillover row under 0.59**, and the one that helps
the spillover most is the one that spoils the direct effect. `placebo_share` is
the ratio `|placebo| / |effect|`, nothing more: it is unbounded, it exceeds 1
whenever the placebo is larger than the estimate, and it is **not** a
probability or a p-value. Read it beside the interval, never instead of it. Widening the match
trades balance on the crash road for balance on its neighbours, because a
single control week cannot be closest to the crash road and to thirty other
roads at once. The default stays `match_hops=0`: it gives the cleanest estimate
of the effect that *is* reportable, and the wider settings are kept as the
sensitivity check rather than as a better answer.

Read together, these three rows say the spillover estimate is **inconclusive,
not zero**. They show the control-matching design cannot separate a
neighbouring-road effect from pre-existing differences between the arms, under
any of the four rules tried. They do not show that crashes leave neighbouring
roads unaffected: a true effect could be present and still be unidentifiable
with these controls. The reason is not sample size, since every row above rests
on the same 2,720 events. Report the magnitude and direction as uncertain.

The old random draw is kept as `match_pre_event=False` for the matching-rule
sensitivity check PR #10 asks for. It is not the default, and it should not be
chosen because it makes an effect larger: it gives a bigger 1-hop number and a
worse placebo.

Known limits of this table, beyond the placebo: the road and control sets vary
step by step, so a time profile is read across changing populations; the
control arm's crash-free post-span filter is one-sided and keeps the calmer
control weeks, worth up to 0.02 mph; and the crash-time rounding recorded below
attenuates everything here.

## Run A provenance

`PROJECT_ROADMAP.md` logs commit `40f672e` for Run A. That commit **cannot** run
the job matrix: `scripts/hpc/*` called `scripts/midpoint.py`, which the same
merge had renamed to `scripts/experiment.py`, and the runner had neither
`--data-only`, which `prepare.sh` depends on, nor `--batch-size`, which the
`main` stage passes. The code that actually ran therefore differed from the
logged commit in at least those three ways, all now repaired.

The authoritative snapshot of any run is not its commit but the metadata
`scripts/hpc/run.sh` writes beside it: `commit.txt`, `git-status.txt`,
`changes.patch`, `source.tar.gz`, `lock.sha256`, `packages.txt`, `slurm.txt` and
`command.sh`. Read those before reproducing Run A, and do not claim the commit
alone reproduces it until a rerun has been compared against the saved outputs.

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
uv run python scripts/experiment.py \
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

**Known issues found after this baseline was frozen** (none rerun; the first two
are immaterial at this scale, the third is not):

- Control admissibility marked window *origins* rather than the cells a window
  was fitted on, so 22 of 4,813 control pairs shared a fitted cell. Fixed; the
  pooled estimate moves by 0.001 mph and the sample becomes 2,527 events / 4,791
  controls.
- `x` was not masked for crash contamination, only `y`. **Fixed**: `x_mask` is now
  `isfinite(lags) & clean[lag]`, so a contaminated input is masked and zeroed like a
  missing one. This one is not cosmetic — only the graph models aggregate
  neighbours, so the contamination reached `gcn`/`sage` and never `mlp`, which
  confounds the gcn–mlp gap the study reads the graph effect off. The frozen
  numbers above predate the fix, but only the edge-using models need rerunning:
  `crash_free_roads` already requires a scored road's whole lag and target span
  clean, so `x_mask` is unchanged on that road, and `persistence`,
  `historical_mean` and `mlp` never read another road. Those three are
  unaffected to the bit; `gcn`, `sage` and `stgcn` move.
- 98 crash rows and 66 control rows have an observed speed of exactly 0.0 mph and
  carry **31% of the pooled effect**: without them it is −0.296, not −0.430.
  Whether 0.0 is a sentinel or real standstill is still unresolved, but it is now
  switchable with `--drop-zero-speed` (default off, keeping this baseline
  reproducible). Deciding it needs the portal's own definition, not a rerun.
  **The flag is not a clean target-side ablation**: a dropped zero also leaves
  the model's inputs (an ex-zero lag cell becomes mask-off, filled with the
  training mean), the normalisation statistics (mean 24.00 → 24.09, std 5.98 →
  5.79), the historical-mean table, and the report figures. A flag-on against
  flag-off comparison is therefore a slightly different forecaster on a
  slightly different sample, not only a different outcome convention. Two
  measured consequences: 698 more (slot, road) cells lose every crash-free
  observation and fall back to the 24.09 mph overall mean, which turns a
  reliable standstill slot into free flow for `historical_mean`; and
  `trajectory_figure`'s all-18-bins-observed filter starts correlating with the
  outcome, dropping 11 of 182 paired events and flattening the mean panel from
  16.13 to 16.60 mph. Window counts and the event/control samples do not move.
- **Crash times are rounded by whoever filed them.** 39.3% of the 3,668 matched
  crash stamps fall on a quarter hour and 14.8% on the hour, against 6.7% and
  1.7% under uniform minutes, with every second zero. A crash rounded up by a
  few minutes puts post-crash congestion in the last pre-crash input bin on the
  crash road itself, which no masking rule can detect because every rule reads
  the same rounded stamp. This attenuates `τ̂`, so **−0.43 mph is a lower bound
  on this count**, and it is the one open issue here that could move the
  headline rather than its third decimal. Unquantified; it needs either a
  finer event time or a sensitivity check that shifts the origin back a bin.

- **The corrected estimate weighted control cells, not crashes.**
  `evaluate.placebo_corrected` averaged every control cell, so a crash that
  drew five matched controls counted five times against one that drew one, and
  the interval treated those cells as independent. Both arms now average within
  a crash first. The crash arm is unchanged to the bit; on the mock fixture the
  control arm moves up to 0.085 mph and the effect up to 0.045 mph. **Every
  corrected estimate above predates this**, and the real-data effects are -0.26
  to -0.68 mph, so a shift that size is not negligible against them. The output
  gains an `events` column; `n` still counts cells.

**What this baseline does not establish:** the controlled gcn–mlp gap is 0.022 mph
on one seed, so no graph advantage is shown; effects are crash-segment only, so
nothing here measures spillover; and the estimate moves with the control draw
(pooled −0.34 to −0.43 across three draws). See `docs/methodology.md`.
