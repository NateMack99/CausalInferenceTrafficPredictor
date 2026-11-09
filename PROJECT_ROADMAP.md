# Chicago Traffic Accident Spillovers — Project Roadmap

**Course:** CS 6501 Graph Machine Learning · Fall 2026  
**Team:** Nate, Maddox, Kyle  
**Timeline:** November 6 – December 9, 2026  
**Plan:** 4 sprints · 12 planned PRs · Sprint 1 complete · Future work unassigned

## Research goal

Estimate how crashes affect speeds on the crash road and connected roads over time. Compare non-graph, graph, and spatiotemporal graph models to determine whether road-network information improves **counterfactual effect estimation**, not just traffic forecasting.

**Approach:** Train no-accident speed forecasters on appropriate crash-free data, estimate crash-associated effects relative to untreated predictions and matched controls, and validate effect recovery on **semi-synthetic data with known ground truth**. Real-world causal claims remain conditional on the matching and identification assumptions.

## Development conventions

- **One PR per useful research deliverable.** These 12 PRs are a plan, not a quota; split only if work blocks teammates or becomes difficult to review.
- **Build on the existing code.** Keep reusable logic in `src/traffic_gml/` and scripts as small `argparse` entry points. Reuse existing loaders, splits, and metrics; do not introduce a second pipeline.
- **Keep tooling lean.** `uv`, `pyproject.toml`, Ruff, pytest, and simple saved experiment arguments/results are enough. No new configuration, tracking, or orchestration framework unless there is a demonstrated need.
- **Preserve methodological correctness.** No post-event inputs for no-crash forecasts, no contaminated untreated training examples, no in-sample controls, and no imputed observations treated as measured targets.
- **Make comparisons fair.** Use shared held-out periods, masks, and event cohorts. Distinguish forecasting error, known-effect recovery, and observational effect estimates.
- **Document important decisions, not every function.** Keep data rules in `docs/data.md`, research decisions and run provenance in `docs/experiments.md` / `docs/methodology.md`, and coding conventions in `AGENTS.md`.

---

## Sprint 1 — Initial Setup & Midpoint Review

**November 6–9 · 3 PRs · Complete**  
**Goal:** Establish real data, a road graph, baseline models, and preliminary results.

### PR #1 — Traffic Data — Nate

**Purpose:** Prepare crash and traffic-speed data for downstream experiments.

- [x] Download Chicago crash and historical traffic-speed datasets.
- [x] Inspect columns, missing values, and time coverage.
- [x] Select an initial study period/area with usable speed coverage.
- [x] Clean data and save CSV/Parquet files.
- [x] Provide speeds indexed by road segment and timestamp.

**Done:** Clean data and coverage information are available.

### PR #2 — Road Graph — Maddox

**Purpose:** Represent the monitored roads and link crashes to road segments.

- [x] Build the road-segment graph with NetworkX.
- [x] Connect segments through navigable intersections.
- [x] Match crashes to monitored road segments.
- [x] Compute degree and betweenness centrality.
- [x] Visualize roads and matched crashes.

**Done:** Road graph and crash-to-segment mappings are available.

### PR #3 — Baselines & Modeling Setup — Kyle

**Purpose:** Establish reusable forecasting models and evaluation code.

- [x] Set up Python, PyTorch, PyTorch Geometric, and project dependencies.
- [x] Implement simple forecasting baselines.
- [x] Implement GCN and GraphSAGE models.
- [x] Define common model inputs and forecast targets.
- [x] Implement training and MAE/RMSE evaluation.

**Done:** Baseline and GNN forecasting experiments run on real data.

### Shared midpoint deliverables

- [x] Integrate crash, speed, and graph data with consistent IDs and times.
- [x] Build crash and matched crash-free windows.
- [x] Train/evaluate models on common splits and create coverage figures.
- [x] Prepare the midpoint report.
- [x] Record the baseline and its known limitations in `docs/experiments.md`.

**Reference midpoint result:** March–April 2025, 2,529 analysis events, 879 modeled road segments, 53.8% speed-grid coverage, and preliminary _crash-road-only_ effects. The controlled GCN–MLP forecasting gap is small; no real spillover or causal recovery result is established yet. Preserve this run for comparison rather than silently replacing it.

---

## Sprint 2 — Data Quality, Controls & Spatiotemporal Modeling

**November 10–17 · 3 PRs · Unassigned**  
**Goal:** Resolve known data/evaluation risks, expand usable observations, and train the primary spatiotemporal model.

### PR #4 — Data Quality & Expansion

**Purpose:** Increase usable data without weakening the validity of event selection.

- [ ] Extend the study period using the historical data already on disk; report monthly coverage, events, and potential controls before retraining everything.
- [ ] Investigate whether `0 mph` is a valid observation or a data sentinel; report **both** conventions as a sensitivity analysis without silently changing the baseline.
- [ ] Separate general event eligibility from eligibility for a particular matched-control estimator.
- [ ] Compare partial-window coverage rules (for example, 1/3/6 observed pre-event bins); keep observed-outcome masks and a strict reference subset.
- [ ] Test reasonable crash-contamination durations and hop distances; report how many events each rule excludes (including the previously excluded pre-history events).
- [ ] Recheck 1-hop/2-hop observation coverage and data-quality differences across the enlarged sample.

**Done when:** Expanded, versioned data has documented filtering rules, retention counts, and zero-speed/missingness sensitivity results.

### PR #5 — Leakage-Safe Controls & Experiment Protocol

**Purpose:** Make untreated comparisons credible, reproducible, and comparable across calendar periods.

- [ ] Resolve or explicitly document the two previously identified crash-matching/segment bugs, with regression tests for any fixes.
- [ ] Prevent crash-contaminated **input histories** as well as target windows from entering untreated training examples; test the receptive-field/exposure assumptions.
- [ ] Preserve the already-fixed fitted-target exclusion and raw wall-clock timestamp checks with automated tests.
- [ ] Replace unstable random control selection with reproducible, pre-event similarity-based matching; report control supply and pre-event balance.
- [ ] Address the training/control calendar mismatch using leakage-safe time-blocked out-of-fold predictions or another documented, out-of-sample design; never admit in-sample controls.
- [ ] Define the primary splits, seeds, matching rules, and metrics once; support comparing runs without building a new experiment framework.

**Done when:** Controls are out of sample, crash/control periods are comparable, leakage checks pass, and control-selection sensitivity is measurable.

### PR #6 — Spatiotemporal Counterfactual Forecaster

**Purpose:** Model future no-crash traffic using road connectivity and recent temporal dynamics.

- [ ] Add a small spatiotemporal GNN: graph message passing plus a GRU **or** temporal convolution (start with one architecture).
- [ ] Train only on eligible untreated windows using masked forecasting loss and pre-event inputs.
- [ ] Forecast 10–60 minutes ahead on the established data interface.
- [ ] Compare with MLP, GCN, GraphSAGE, and simple baselines on identical held-out periods, target masks, and sample definitions.
- [ ] Save model settings, training seed, runtime, and per-horizon MAE/RMSE; smoke-test on small data before scaling on Rivanna if needed.

**Done when:** A trained spatiotemporal model produces reproducible held-out no-accident forecasts and a fair baseline comparison.

**Sprint 2 deliverable:** A more reliable expanded dataset, leakage-safe comparison design, and a working spatiotemporal model. **If the data interface stabilizes early, start a tiny known-effect injection smoke test now**; the full benchmark remains PR #8.

---

## Sprint 3 — Causal Comparisons, Known-Effect Validation & First Spillovers

**November 18–26 · 3 PRs · Unassigned**  
**Goal:** Produce the main research results _before_ the presentation window.

### PR #7 — Matched Causal Baseline & Initial Real Spillovers

**Purpose:** Provide an interpretable non-GNN effect estimate and an early answer to the network question.

- [ ] Implement a matched event-study or difference-in-differences baseline using pre-treatment matching and checks.
- [ ] Examine pre-event trends and treatment/control balance.
- [ ] Estimate crash-road effects and **first-pass 1-hop/2-hop effects** over time.
- [ ] Report sample sizes for each ring/horizon and a common-event comparison where possible.
- [ ] Save results separately from forecasting residual-based estimates.

**Done when:** We have a reproducible causal baseline and preliminary observed spillover curves, without claiming they establish causal propagation.

### PR #8 — Semi-Synthetic Accident Benchmark

**Purpose:** Test effect recovery against known ground truth, as requested by the mentor.

- [ ] Use held-out, crash-free real traffic windows as untreated reference trajectories.
- [ ] Inject known shocks at an incident road and selected connected roads over time, using simple, documented propagation rules.
- [ ] Vary magnitude, duration, and hop distance; add direction only if graph directions are reliable.
- [ ] Preserve untreated speeds, injected treated speeds, observation masks, and exact true effects.
- [ ] Add small deterministic tests for injection, masking, and effect-label correctness.

**Done when:** The same benchmark can evaluate every model against identical known effects, without giving forecasters access to the injected effect.

### PR #9 — Counterfactual Effect Recovery & Model Comparison

**Purpose:** Determine whether better forecasts yield better estimates of accident effects.

- [ ] Apply each forecaster to held-out real crash events and semi-synthetic events using pre-event information only.
- [ ] Compute model-based effects and the matched-control adjustment where applicable.
- [ ] Report effect MAE/RMSE (and bias if useful) against semi-synthetic ground truth, **separately** from forecasting MAE/RMSE.
- [ ] Compare effect estimates across MLP, GCN, GraphSAGE, spatiotemporal GNN, and simple baselines; check sensitivity to forecaster choice.
- [ ] Compare available-case estimates with a fixed cohort observed at all horizons; keep crash/control samples aligned.

**Done when:** There is a common results table for known-effect recovery and preliminary real-world direct and neighboring-road estimates.

**Sprint 3 deliverable:** Primary causal comparisons, semi-synthetic ground-truth results, and first real spillover figures, ready to refine before the presentation.

---

## Sprint 4 — Robustness, Ablations & Final Deliverables

**November 27 – December 9 · 3 PRs · Unassigned**  
**Goal:** Stress-test the main results and finish the presentation and paper.

### PR #10 — Causal Robustness & Uncertainty

**Purpose:** Show which findings survive changes in reasonable analysis choices.

- [ ] Run crash-free placebo events and pre-trend diagnostics.
- [ ] Quantify variation across model-training seeds, control selections, and matching rules **separately**.
- [ ] Check sensitivity to zero-speed handling, missingness, contamination windows, and overlapping crashes.
- [ ] Estimate event-level uncertainty (clustered or event-bootstrap); do not treat repeated roads/horizons as independent events.
- [ ] State the assumptions and limits of causal interpretation.

**Done when:** Main effect findings include uncertainty, placebo evidence, and a clear sensitivity summary.

### PR #11 — Network Spillover Analysis & Graph Ablations

**Purpose:** Test whether effect patterns depend on network position and whether real road structure helps.

- [ ] Refine 0-hop, 1-hop, and 2-hop effects across time using a common event cohort alongside available-case results.
- [ ] Compare upstream/downstream effects where direction is trustworthy.
- [ ] Test original versus removed/rewired edges and a matched-capacity non-graph comparator.
- [ ] Report **effect recovery** as well as forecasting performance for the graph ablations.
- [ ] Explore severity, time-of-day, baseline congestion, or centrality heterogeneity **only if the main tests are complete**.

**Done when:** The network findings and graph contribution are supported by explicit comparisons, including negative or null results.

### PR #12 — Final Model Checks & Research Outputs

**Purpose:** Finish the few analyses needed for a reproducible and interpretable final paper.

- [ ] Run targeted temporal/history-length and feature ablations; skip exhaustive architecture searches.
- [ ] Compare models on paired held-out periods and event-level units, not as 3 million independent speed targets.
- [ ] Prepare final forecasting, effect-recovery, spillover, and sensitivity tables/figures.
- [ ] Verify the commands and recorded settings reproduce the selected results.
- [ ] Keep the repository clean; summarize limitations and lessons from unsuccessful experiments.

**Done when:** The results required by the paper and presentation are final, reproducible, and reviewed.

**Sprint 4 deliverable:** Robustness and ablation results, presentation, and final paper. Overleaf remains the source of truth for LaTeX; the Git copy is for convenience.

---

## Milestones

| Date             | Expected deliverable                                                             |
| ---------------- | -------------------------------------------------------------------------------- |
| **Nov 9**        | Midpoint report and frozen baseline                                              |
| **Nov 17**       | Expanded/validated data, safer controls, spatiotemporal forecasts                |
| **Nov 26**       | Matched baseline, semi-synthetic effect recovery, initial real spillover results |
| **Nov 29**       | Main tables and presentation figures ready                                       |
| **Nov 30–Dec 7** | Presentation                                                                     |
| **Dec 9**        | Final ACM-format paper and reproducible code                                     |

## Dependencies and priorities

1. **Correctness before claims:** PRs #4–5 establish the data and evaluation rules that final causal comparisons use. PR #6 can be developed in parallel on existing sample data.
2. **Ground truth early:** Start a simple semi-synthetic fixture as soon as the shared input format is stable; finish the benchmark and effect recovery in PRs #8–9.
3. **Spillovers before presentation:** Produce first real 1-hop/2-hop estimates in PR #7, leaving detailed graph analysis and ablations for PR #11.
4. **Be selective:** Reliable controls, effect recovery, and graph-specific evidence take priority over extra architectures, complex infrastructure, or optional heterogeneity plots.

**PR rule:** Each PR should have a clear output, tests for any new correctness-critical behavior, and a short note on how it was validated. Merge continuously; do not hold unrelated work for an entire sprint.
