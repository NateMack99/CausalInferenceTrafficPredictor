# Run A validation and publication readiness

Written 2026-11-26 against the bundle in `raw_results/` and the code at `816c07a`.
Every number here was recomputed from the CSVs or read out of the executed stage
commands. Nothing was taken from the earlier notes in this folder.

## 1. What exists

The Run A matrix completed all 45 stage markers. The experiments that ran, and
where their outputs stand:

| Arms | What it gives | In the bundle |
|---|---|---|
| `main`, `main-nozero` | 7-model held-out forecast error | main only |
| `event`, `event-nozero` | matched event study, 8 diagnostic tables | yes |
| `benchmark`, `benchmark-nozero` | B0 known-effect recovery, 9 models | benchmark only |
| `graph-ablation` (12) | real / self-only / relabelled, 3 seeds + stgcn | yes |
| `treatment` (3) | exposure-conditioned arms, 3 seeds | yes |
| `synthetic-scenarios` (9) | **B1, B2, B3 at 3 seeds each** | **no** |
| `matching` (6) | five alternative control-drawing rules | **no** |
| `forecast-seeds` (8) | 4 architectures at 2 extra seeds | **no** |

**The single largest finding of this review is that B1, B2 and B3 were already
run.** So were the alternative matching arms and the seed sweep. They are sitting
in `outputs/hpc/runA/run/` on Rivanna. `scripts/aggregate.py` never collected
them, so they never reached the paper and the earlier notes concluded they were
missing experiments. They are not. No retraining is required to report them.

That gap is now fixed in `scripts/aggregate.py`, which gained `scenario_table`,
`matching_table`, `seed_table` and `spread_table`, plus the two `nozero` arms.

## 2. Provenance audit

**The bundle is internally consistent and matches the executed commands.** The
window is 2024-07-01 to 2026-04-01 with a 12-bin history, 6-bin horizon, stride 1
and 2 exposure hops; training is 20 epochs, patience 4, hidden 64, batch 64;
matching is 5 controls per crash. These are read from
`scripts/hpc/experiments.sh`, and `tests/test_config.py` parses that file through
the real argument parsers so the settings cannot drift unnoticed.

Checks run and passed:

- Every mean quoted in the earlier `RESULTS_VALIDATION.md` reproduces exactly
  from `graph_ablation.csv` and the event-study files. Its arithmetic is right.
- `30,118` in the funnel against `30,315` in the event study is **not** a
  discrepancy. The audit funnel requires a held-out control. `scripts/event_study.py`
  waives that rule deliberately, with the reason in the code: keeping it pushed
  88% of controls after their event and cut supply to 1.9 per crash. Different
  eligibility, both correct.
- `benchmark-seed42` reports treatment GCN effect MAE 1.210 while `treatment.csv`
  seed 42 reports 1.312. Also not a discrepancy. The `treatment` stage passes
  `--neighbour-hops 1` and the `benchmark` stage does not. The earlier notes
  flagged this as unexplained.

One real gap remains: the run recorded `"code": "60c62f8-dirty"`, so no commit
identifies the executable. The working-tree diff was saved to
`outputs/hpc/runA/logs/meta/23468232/changes.patch` and should be read before
submission.

**The three earlier markdown files in this folder describe a package that does
not exist.** They reference `traffic_spillovers/causal.py`, `results.py`,
`experiments.py` and a branch `37fefd8` with a 28-arm matrix. This repository has
`src/traffic_gml/` and a 45-stage matrix. Their *arithmetic* on the supplied CSVs
is sound and their cautions are mostly right, but their provenance claims are not
about this codebase. `make_figures.py` also read a `raw_results/` directory that
was never shipped, so the figures could not be regenerated as delivered. That
directory now exists and the script runs.

## 3. The graph-ablation effect metric

This was the question to settle, and the answer matters more than expected.

`scripts/aggregate.py` takes `estimator == "corrected"` for the ablation table, so
the reported effect MAE is matched-control-corrected, not raw. The reviewer's
additive identity does not apply to it directly.

It applies exactly to the `raw` estimator. `synthetic.py` injects from `origin+1`
onward, so the forecaster's inputs are identical in both arms. With `effect = y -
ŷ` the raw error is `(τ + ε) - τ = ε`, the ordinary forecast residual. The
codebase already documents this in `untreated_windows`. Raw effect MAE therefore
carries no information beyond forecast MAE and must never be read as evidence
about counterfactual estimation.

For `corrected`, the error is `ε_treated - ε_control`, a difference of two
forecast residuals. Not identical to forecast error, but a direct function of it.
The data confirm the consequence:

| | mean added MAE vs real graph |
|---|---|
| forecast | **+0.0301** |
| corrected effect | **+0.0285** |

over the eight paired topology contrasts. The two gains are the same size. The
paper's central question is whether correct topology helps counterfactual
estimation *beyond* forecasting, and on this metric the answer is no.

The one quantity that is not a function of forecast accuracy is signed bias,
because systematic forecaster error cancels between the two residuals. There the
real graph does lead: mean absolute pooled bias 0.091 (real), 0.105 (self-only),
0.115 (relabelled) over three GCN seeds. Three seeds and one relabelling cannot
establish it, which is what the new `topology-spread` stage is for.

## 4. The no-effect null

Every injected effect is non-positive by construction: `after = max(before - want,
0)` and `true_effect = after - before`. So the MAE of estimating no effect equals
the realized magnitude under the same weighting, and it is readable from the
`true_mph` column already in the bundle.

B0, seed 42, mph:

| | hop 0 | hop 1 | hop 2 | pooled |
|---|---|---|---|---|
| cells | 871 | 4,434 | 6,643 | 11,948 |
| **estimate no effect** | **3.955** | **1.815** | **0.812** | **1.520** |
| GCN matched residual | 3.170 | 3.255 | 3.171 | 3.202 |
| GCN raw residual | 2.907 | 3.004 | 2.981 | 2.984 |
| Treatment GCN | 3.501 | 1.589 | 0.656 | 1.210 |
| Treatment MLP | 3.560 | 1.594 | 0.641 | 1.208 |

Both residual estimators are beaten by predicting zero. At two hops the claim is
weighting-free: the injected effect there is magnitude (at most 12) times
`0.45^2` times a profile bounded by 1, so the null cannot exceed 2.43 mph while
the residual estimators record 2.98 and 3.17. Those 6,643 cells are 56% of the
scored sample.

The exposure heads do beat the null, by about 20% pooled, and they do it by
shrinking: they recover 14% of the injected magnitude (-0.22 estimated against
-1.52 realized). Low absolute error and near-zero prediction are the same
behaviour when the signal is half the noise.

Topology differences of 0.03 mph are being read off a metric whose best
performer loses to a constant. That is the finding the paper now leads with.

## 5. What changed in the code

- `scripts/aggregate.py`: collects the scenario, matching, seed and spread arms;
  adds a `zero_mae` column computed from each arm's saved `truth.csv`.
- `src/traffic_gml/evaluate.py`: `scored_cells` exposes the estimate-truth join
  that `effect_recovery` already did internally.
- `scripts/benchmark.py`: writes `cells.csv`, one row per scored cell with
  `model, estimator, event_id, road, hop, minutes, true_effect, estimated`. This
  is what makes a common-support comparison possible without retraining, now and
  for any future question about weighting.
- `scripts/hpc/experiments.sh` and `submit.sh`: new `topology-spread` stage, 30
  array tasks, 10 independent relabellings at 3 training seeds. A new stage name,
  not an extension of `graph-ablation`, so Run A's existing task markers keep
  pointing at the arms that produced them.
- `tests/test_hpc.py`: one test covering the new collectors and the zero null.

`uv run ruff check .`, `ruff format --check .` and `pytest -q` (104 tests) pass.

## 6. What to run, in order of value

Nothing below has been launched.

**(a) Re-aggregate. No GPU, no retraining, ~5 minutes.** This alone surfaces
B1, B2, B3, the five matching arms, the seed sweep, both nozero arms and the zero
null.

```
cd ~/CausalInferenceTrafficPredictor && git pull
uv run --no-sync python scripts/aggregate.py \
    --run outputs/hpc/runA/run --work outputs/hpc/runA/work \
    --out outputs/hpc/runA/run/paper
```

**(b) Ten relabellings. 30 GPU arms, about 20 min each, ~5 h wall at
`GPU_CONCURRENCY=2`.** Needs a fresh run directory because Run A's markers would
skip it.

```
bash scripts/hpc/submit.sh --name runB
```

That queues the whole matrix again. To queue only the new stage, submit it
directly against the existing Run A work directory and leave Run A's outputs
alone.

**(c) Common-support recomputation. 13 GPU arms** (`benchmark`, `treatment` x3,
`synthetic-scenarios` x9), only if the paired head-vs-residual ranking is wanted.
These rerun because `cells.csv` did not exist when Run A ran. Everything else in
Run A stays valid and must not be overwritten.

## 7. Where the paper stands

Revised `paper/main.tex`: abstract, topology results, estimator results,
discussion and conclusion rewritten around the two findings above. Four figures
regenerate from `raw_results/` with `python paper/figures/make_figures.py`.
Figure 1 is rebuilt from real data (the frozen network, the real funnel, and the
signal-against-noise scale). Figure 2 gained the decomposition panel. Figure 3
gained the placebo panel. Figure 4 is now recovery against the null. No LaTeX
toolchain is installed locally, so the PDF was not rebuilt: run `pdflatex main.tex`
twice in `paper/`.

Three `\pending` markers remain, all for work that has run and needs collecting
(B1-B3, the matching arms) or needs `cells.csv` (common support).

## 8. Honest assessment

**Originality.** Low as an architecture contribution, which is fine, and the
paper does not claim one. Moderate as an evaluation contribution. The specific
result worth publishing is that a matched-residual effect metric built on
forecasts is not an independent axis of evidence and loses to a no-effect null on
most cells. That is a real methodological warning with a clean demonstration.

**Technical validity.** Good. The pipeline guards leakage carefully, the
injection contract is sound, the diagnostics are unusually complete for a course
project, and the code documents its own failure modes.

**Causal interpretation.** Appropriately hedged. The manuscript never claims
identification from observational matching.

**Experimental completeness.** The weak point until (a) is run. One relabelling
is not enough for the headline comparison, and the B1 null-spillover test is the
most load-bearing missing table.

**Statistical robustness.** Weak. Three seeds, nominal event-level intervals, no
clustering on roads or time blocks, no multiplicity adjustment. The topology
differences are 0.03 mph against a seed standard deviation of 0.003, which is a
real signal for forecasting, but the effect-metric version of it is not
independent evidence.

**Reproducibility.** Strong except for the `-dirty` code version.

**Venue fit.** As it stands, a workshop paper. A negative or null result, clearly
demonstrated, is publishable at a causal-ML or graph-learning workshop. After (a)
and (b), with B1 false-spillover numbers and a distribution over ten wrong
graphs, the central claim would be properly supported either way and the paper
would be competitive for a specialized venue. It is not a main-track conference
paper, and padding it with more experiments would not make it one. The
contribution is the controlled evaluation and the null baseline.
