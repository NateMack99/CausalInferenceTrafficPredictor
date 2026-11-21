# Chicago Traffic Accident Spillovers — Project Roadmap

**Course:** CS 6501 Graph Machine Learning · Fall 2026  
**Team:** Nate, Maddox, Kyle  
**Final deadline:** December 9, 2026  
**Plan:** 4 sprints · 12 planned PRs · Sprint 1 complete · Remaining owners unassigned  
**Roadmap updated:** November 19, 2026

> **Status note:** Verified against the repository on November 19, 2026: no commit has landed since the midpoint (`0c1d5c9`), so PRs #4–#12 are all still open and the November 17 checkpoint was missed. Sprint 1 is green (Ruff clean, 31 tests pass) and `docs/experiments.md` matches `outputs/midpoint/summary.json`. Of the three known issues recorded there, control admissibility is fixed in code; **`x` crash-masking and the 0 mph convention are still open**.

## Research goal

Estimate how reported crashes are associated with changes in traffic speeds **on the crash road and on connected roads over time**. Test whether road connectivity and temporal graph learning improve **no-crash counterfactual estimation**, not merely ordinary forecasting.

**Core design:** Build a road-segment graph; train a no-crash forecaster on appropriately screened untreated windows; use matched, out-of-sample controls to correct residual bias; estimate effects by road distance and time; and evaluate the complete estimator on **semi-synthetic disruptions with known effects**. Observational results are conditional on the quality of matching and causal assumptions.

**What this project is not:** Causal-DAG discovery, a new structural causal model, a latent-confounder VAE, or policy optimization. No NOTEARS, DAG-GNN, DECI, or CEVAE implementation is planned.

## Engineering rules

- **One useful outcome per PR**, not one PR per function. The 12 PRs are a plan, not a quota.
- **Reuse the current codebase.** Reusable logic lives in `src/traffic_gml/`; `scripts/` holds small `argparse` entry points. Add modules only when their logic exists.
- **Keep tooling minimal:** `uv`, `pyproject.toml`, PyTorch/PyG, NetworkX, Ruff, pytest. No Hydra, MLflow, causal-discovery framework, or custom experiment platform.
- **Preserve strict interfaces:** stable segment IDs, time-grid conventions, graph indices, observation masks, event IDs, and shared evaluation splits.
- **Protect identification:** pre-event inputs only for a crash counterfactual; untreated training windows must be screened for exposure in both **input histories and targets**; controls must be genuinely out-of-sample.
- **Make fair comparisons:** same held-out periods, target masks, and event cohorts where possible; compare predictive accuracy, semi-synthetic effect recovery, and real-event estimates separately.
- **Write only useful documentation:** `README.md` for running, `AGENTS.md` for coding, `docs/data.md` for measurement rules, and `docs/methodology.md` / `docs/experiments.md` for the estimand, assumptions, decisions, and run records.
- **Paper:** Overleaf remains the LaTeX source of truth; the repository may contain a convenience copy.

---

## Sprint 1 — Data, Graph & Midpoint Baseline

**November 6–9 · 3 PRs · Complete**  
**Goal:** Establish the real-data pipeline and credible preliminary results.

### PR #1 — Traffic Data — Nate
**Purpose:** Prepare Chicago crash reports and monitored speed observations.

- [x] Download, inspect, and clean crash and speed datasets.
- [x] Choose an initial study period and document missing observations.
- [x] Save reusable segment-by-time speed data and crash records.
- [x] Report data coverage and filtering counts.

### PR #2 — Road Graph — Maddox
**Purpose:** Define physical road connectivity and locate reported crashes.

- [x] Build and inspect the NetworkX road-segment graph.
- [x] Match crashes to monitored segments with consistent IDs.
- [x] Calculate basic graph properties and create a network visualization.

### PR #3 — Forecasting Baselines — Kyle
**Purpose:** Establish models and metrics for later comparisons.

- [x] Set up PyTorch, PyG, `uv`, and the training entry point.
- [x] Implement simple baselines, MLP, GCN, and GraphSAGE.
- [x] Train and evaluate models on common splits with masked MAE/RMSE.

**Shared completed work:** Joined crashes, speeds, and graph data; constructed crash/control windows; produced midpoint coverage, model, and preliminary crash-road effect results; saved run provenance and known limitations.

**Frozen reference:** March–April 2025; **2,529** analysis crash events; **879** modeled roads; **53.8%** observed speed-grid cells. Real neighboring-road effects, graph-specific causal improvements, and known-effect recovery were **not** established at midpoint. Keep the old results as a reference rather than overwriting them.

---

## Sprint 2 — Data Validity, Control Design & Spatiotemporal Modeling

**Original target: November 10–17 · PRs #4–#6 · Verify progress / carry forward unfinished work**  
**Goal:** Fix the known data and evaluation problems before making new causal claims; train one spatiotemporal model in parallel.

### PR #4 — Data Quality & Expanded Sample
**Purpose:** Retain useful observations without hiding measurement problems or relaxing causal eligibility arbitrarily.

- [ ] Expand the study period using the historical data already available; report coverage, matches, and control candidates by month before full retraining. *(The by-month table is implemented as `evaluate.monthly_coverage` and written by `scripts/midpoint.py`; the expanded run itself is still pending.)*
- [ ] Audit the meaning of **0 mph** in the published data and compare analyses retaining versus excluding zeros; preserve the midpoint convention as a reference. *(`--drop-zero-speed` exists and defaults to off; the two analyses have not been run and the portal's definition of 0.0 is still unconfirmed.)*
- [ ] Separate crash-event eligibility, observed-outcome eligibility, and eligibility for a particular matched-control method.
- [ ] Compare practical pre-event observation thresholds; use masks for missing values and never score imputed outcomes as observations.
- [ ] Test plausible contamination durations and hop distances, including the cost of excluding overlapping incidents; **do not choose a rule because it makes the effect larger**.
- [ ] Recompute 0/1/2-hop observation coverage and document sample attrition with the expanded period.

**Done when:** Expanded data and each major filtering decision have reproducible coverage and sensitivity results.

### PR #5 — Leakage-Safe Training & Comparable Controls
**Purpose:** Make untreated forecasts and matched controls genuinely out of sample and comparable to crash events.

- [ ] Confirm or fix remaining crash-to-segment bugs and preserve regression tests; verify the completed fitted-target and wall-clock safeguards.
- [x] Exclude or mask **crash-exposed input histories as well as training targets**, based on the chosen event exposure rule; test this on small examples.
- [ ] Implement a simple **time-blocked out-of-fold (OOF)** prediction scheme, with an embargo for overlapping windows, so historical crash/control periods can be evaluated without fitting on their targets.
- [ ] Choose matched controls using **only pre-event** speed level/trend and calendar/road features; report control counts, overlap, pre-event balance, and calendar alignment.
- [ ] Make control selection reproducible; separate variation from model-training seeds and control selections without building a tracking framework.
- [ ] Record a primary analysis specification in `docs/experiments.md` before assessing its effect estimates.

**Done when:** Leakage checks pass; crash and control predictions are out of sample; matching quality and control-supply improvements can be inspected.

### PR #6 — Spatiotemporal No-Crash Forecaster
**Purpose:** Test whether jointly modeling road connectivity and time improves no-crash traffic predictions.

- [x] Implement **one** modest spatiotemporal architecture: existing graph message passing plus a GRU *or* temporal convolution.
- [x] Reuse the existing data tensors, masks, model interface, and training loop; avoid creating a second pipeline.
- [ ] Train on valid untreated windows to predict observed speeds 10–60 minutes ahead using masked forecasting loss. *(Runs reproducibly on the mock fixture only; no real-data run yet.)*
- [ ] Compare with current baselines and a **matched temporal no-edge version** using identical held-out periods, features, and evaluation masks. *(`tgru` is the matched no-edge twin at equal parameter count and `scripts/midpoint.py` trains both; results not yet produced.)*
- [ ] Report per-horizon MAE/RMSE and a few training seeds; start small and use Rivanna only if profiling justifies it.

**Done when:** The spatiotemporal and matched non-graph temporal models run reproducibly on the same clean evaluation sample.

**Sprint 2 deliverable:** Defensible data filters, leakage-safe controls, and a trained spatiotemporal model. Start a tiny synthetic-injection smoke test as soon as the event-window format is stable; do not delay the full benchmark until the final sprint.

---

## Sprint 3 — Matched Causal Analysis, Semi-Synthetic Validation & First Spillovers

**November 18–26 · PRs #7–#9 · Current planned research phase**  
**Goal:** Obtain the main research results early enough to diagnose them before the presentation.

### PR #7 — Matched Event Study & Initial Neighbor Effects
**Purpose:** Estimate direct and neighboring-road effects using an interpretable non-GNN comparison.

- [ ] Implement a matched event-study or difference-in-differences comparison using the validated controls.
- [ ] Show **pre-event trends and covariate balance**; explain which factors are controlled for and which may remain unobserved.
- [ ] Estimate preliminary **0-, 1-, and 2-hop** speed differences over time using a clear event/exposure definition.
- [ ] Report the number of events and observed roads at each hop and horizon; compare a common-event cohort with available cases.
- [ ] Keep these observational estimates distinct from simulated ground-truth results.

**Done when:** A non-GNN matched analysis produces reproducible direct and first-pass neighbor-effect curves with honest assumptions.

### PR #8 — Semi-Synthetic Network-Effect Benchmark
**Purpose:** Check the full effect-estimation procedure against known injected disruptions, as recommended by the mentor.

- [x] Start from **held-out, crash-free** real traffic windows and retain their original, observed speeds as untreated outcomes.
- [x] Inject known direct and 1-/2-hop slowdowns over time using a small, documented family of effect shapes.
- [x] Vary effect size, duration, and spatial spread; include both simple event placement and **pre-congestion-dependent** placement to test selection/matching sensitivity.
- [x] Preserve true untreated outcomes, treated outcomes, true effects, event identities, masks, and the random seed.
- [x] Add deterministic tests for injection and labels; **never reveal injected post-event effects to the forecaster**.

**Done when:** Every model and causal baseline can run on the *same* reproducible, known-effect test scenarios.

### PR #9 — Counterfactual Effect Evaluation
**Purpose:** Determine whether the entire matched counterfactual procedure reliably estimates direct and spillover effects.

- [ ] Apply non-graph, GCN, GraphSAGE, and spatiotemporal forecasters to the **same held-out event sets**; use only information available before the event.
- [ ] Compute the real-event residual-plus-matched-control estimate and its synthetic analogue using the same estimator logic.
- [ ] Evaluate **average effect bias, 0/1/2-hop and time-profile recovery, and uncertainty coverage** against known synthetic effects; compare by selection and missingness scenario.
- [ ] Report ordinary forecast MAE/RMSE separately. **Sanity check:** for an injected additive effect, unadjusted individual-effect error equals untreated forecast error algebraically; do not present that metric alone as independent causal validation.
- [ ] Compare effect estimates across forecasters, and fixed-cohort versus available-case estimates. Include **first real 1-/2-hop results** here rather than waiting until Sprint 4.

**Done when:** There is a reproducible table of known-effect recovery plus initial real direct/neighbor estimates, with modeling and causal claims clearly separated.

**Sprint 3 deliverable:** Initial direct/spillover results, a matched causal baseline, and a semi-synthetic benchmark that evaluates more than standard forecasting error.

---

## Sprint 4 — Robustness, Graph Ablations & Final Research Outputs

**November 27–December 9 · PRs #10–#12 · Unassigned**  
**Goal:** Verify the main findings and finish the paper without expanding the model zoo.

### PR #10 — Causal Assumptions & Robustness
**Purpose:** Assess how strongly the results depend on matching, sample selection, and uncertainty assumptions.

- [ ] Test pre-event differences and placebo event times.
- [ ] Vary matching rules and control sets; measure **control-selection variance separately from training-seed variance**.
- [ ] Check zero-speed handling, missingness, overlapping accidents, exposure distances, and contamination windows.
- [ ] Use event-level/bootstrap uncertainty appropriate for correlated observations and multiple horizons; avoid counting road-time targets as independent crashes.
- [ ] Document confounding, limited overlap, observation-process bias, and the difference between association and an identified causal effect.

**Done when:** The primary results include interpretable uncertainty and sensitivity/diagnostic findings, including negative results.

### PR #11 — Spillover Patterns & Controlled Graph Tests
**Purpose:** Establish whether neighboring-road results depend on real road structure.

- [ ] Refine 0-/1-/2-hop effects over time with a common-event comparison; test upstream/downstream distinctions only if road direction is trustworthy.
- [ ] Compare the **real graph**, removed/rewired edges, and the **matched-capacity temporal no-edge model** while holding samples and inputs fixed. *(`--graph-ablation real|permuted|none` is implemented on both entry points, with `mlp`/`tgru` as the matched no-edge models; the comparison itself still needs a run.)*
- [ ] Evaluate both untreated forecasting and semi-synthetic **effect-profile recovery** under these graph ablations.
- [ ] Report separate hop-ring sample sizes; do not overinterpret apparent distance differences from different event populations.
- [ ] If time permits, add **one** preregistered subgroup analysis (e.g., pre-crash congestion or crash severity), not a large exploratory grid.

**Done when:** The study can state what road connectivity contributes, or that no reliable benefit was detected.

### PR #12 — Final Comparisons, Figures & Reproducibility
**Purpose:** Deliver the minimum complete evidence needed to support the final paper.

- [ ] Run only a small number of motivated history-window, temporal-encoder, or node-feature ablations.
- [ ] Compare models on paired held-out time blocks/events, not on millions of overlapping targets treated as independent.
- [ ] Generate final tables/plots: data coverage; forecasting; matched direct and neighbor effects; known-effect recovery; graph ablations; robustness.
- [ ] Re-run and verify the selected commands, configurations, data versions, and reported results; retain the frozen midpoint reference.
- [ ] Complete the presentation and ACM paper (Overleaf source of truth); document limitations and reproducibility commands in the README.

**Done when:** Results are reproducible, the paper/presentation agree with the code, and no unsupported causal claim remains.

**Sprint 4 deliverable:** Validated direct/spillover analysis, controlled graph comparison, reproducible research outputs, and the final paper.

---

## Deadlines and sequencing

| Date | Deliverable |
| --- | --- |
| **Nov 9** | Midpoint review and frozen baseline (complete) |
| **Nov 17** | Original Sprint 2 checkpoint; carry forward unverified work immediately |
| **Nov 26** | Semi-synthetic benchmark, matched baseline, and first real 0/1/2-hop results |
| **Nov 29** | Main results and presentation figures selected |
| **Nov 30–Dec 7** | Project presentations |
| **Dec 9** | Final ACM paper and reproducible repository |

### Execution priorities as of November 19

1. **First:** Confirm what already landed in PRs #4–#6. Fix any remaining input contamination, control leakage/calendar mismatch, or zero-speed ambiguity before trusting new causal results.
2. **In parallel:** Train the spatiotemporal GNN on a small clean sample and build the semi-synthetic injection fixture.
3. **Before Nov 26:** Get a matched real-event baseline, known-effect evaluation, and at least one 1-hop effect plot. Expand to 2 hops if the samples support it.
4. **Afterward:** Graph ablations, placebo/uncertainty, and targeted sensitivity; prioritize these over extra neural architectures or broad hyperparameter searches.

**Scope guardrail:** If time is tight, prefer **one correct ST-GNN + one fair non-graph temporal baseline + one clean matched estimator + one sound semi-synthetic benchmark**. Additional models and optional heterogeneity analyses are secondary.
