# Chicago Traffic Accident Spillovers — Project Roadmap

**Course:** CS 6501 Graph Machine Learning · Fall 2026  
**Team:** Nate, Maddox, Kyle  
**Final paper:** December 9, 2026 · **Presentation:** November 30–December 7  
**Updated:** November 21, 2026  
**Plan:** 12 planned PRs across four original sprint phases; **development and large-scale experiment runs now proceed in separate, nonblocking tracks**.

## Research goal

Estimate how reported traffic crashes affect speeds on the incident road and connected roads over time. Compare matched causal estimates, non-graph forecasting models, and graph-based counterfactual forecasters to test whether road connectivity improves **effect estimation**, not only prediction.

**Core approach:** Use Chicago crash reports, ten-minute road-speed observations, and the road-segment graph. Train on appropriately screened untreated windows; predict no-crash speeds using pre-event information; compare crash residuals with leakage-safe matched controls; and evaluate the complete estimator on semi-synthetic network shocks with known effects.

## Execution strategy — code first, experiments in parallel when safe

**The development track should never wait for a full GPU run.** Every new component must pass a small CPU end-to-end test before the expensive run. A teammate can run independent jobs on HPC as soon as their inputs are stable. The expensive multi-seed and graph-ablation sweeps happen **after** the estimator and scoring interfaces are frozen.

| When | Development track (local/agent work) | HPC / experiment track (teammate) | Paper track |
| --- | --- | --- | --- |
| **Nov 21–25** | Finish PRs #4–#9 interfaces and correctness-critical code; integrate small tests | Prepare environment and data; run expanded **data-only** coverage audit; run small CUDA smoke test | Preserve pilot results; prepare final table/figure slots |
| **Nov 26–29** | Fix blocking bugs only; finish scoring/analysis integration | **Main full-window runs:** forecasting, matched effects, semi-synthetic recovery, first 0/1/2-hop results | Fill results as outputs arrive; prepare presentation |
| **Nov 30–Dec 5** | Focused diagnostics and analysis code | **Targeted robustness:** a few seeds, selected graph ablations, zero-speed/control sensitivity | Presentation; interpret results and revise discussion |
| **Dec 6–9** | Reproducibility fixes only | Essential reruns only | Final ACM paper, references, result verification |

**Release gates:**

1. **Before the main experiment run:** crash matching and time checks pass; input *and* target contamination rules are applied; all controls and event predictions are out-of-sample; matched-control estimator works on a small sample; semi-synthetic scorer recovers a hand-constructed known effect.
2. **Before the graph-ablation/seed sweep:** complete a single-seed main run and inspect data counts, missingness, control balance, effect signs, and synthetic recovery. Lock the primary settings and choose only a few justified comparisons.
3. **Before submission:** figures and paper numbers trace to saved configurations, inputs, code revision, and outputs. Never replace the frozen midpoint run silently.

> **Status convention:** Sprint 1 is confirmed complete. PRs #4–#9 may have partial implementations or open branches; check the current repository before marking individual tasks done. The checkboxes below denote remaining acceptance criteria, not a claim that no code exists.

---

## Sprint 1 — Data, graph, and midpoint baseline (complete)

**November 6–9 · PRs #1–#3**

### PR #1 — Traffic Data — Nate ✓

- [x] Download and clean Chicago crash and speed data; inspect coverage and missingness.
- [x] Produce reusable segment-by-time speeds and crash records.

### PR #2 — Road Graph — Maddox ✓

- [x] Build the road-segment network, match crash locations to segments, and inspect connectivity.
- [x] Provide graph statistics and visualization.

### PR #3 — Forecasting Baselines — Kyle ✓

- [x] Establish shared data tensors, training loop, masks, and MAE/RMSE.
- [x] Run persistence, historical mean, MLP, GCN, and GraphSAGE baselines.

**Frozen reference:** The March–April 2025 midpoint pilot included 2,529 eligible crash events and 879 modeled road segments. Its direct-effect results are preliminary. Keep `outputs/midpoint/` and its provenance intact; corrected or expanded results belong in separate directories.

---

## Sprint 2 carryover — Finish core data and model components

**Originally November 10–17 · PRs #4–#6 · Carry over only unfinished acceptance criteria**

### PR #4 — Data Quality & Expanded Sample

**Purpose:** Prepare a larger, well-defined analysis sample. **This PR does not require full model retraining to close its data-work portion.**

- [ ] Run the existing pipeline's expanded-window **data/coverage diagnostics**; report matched events, missingness, candidate controls, attrition, and 0/1/2-hop feasibility. Add a simple monthly breakdown only if absent. *(Monthly breakdown shipped as `evaluate.monthly_coverage`; funnel, attrition and hop feasibility already exist. Awaiting the expanded run.)*
- [ ] Preserve the original zero-speed convention; investigate its semantics and define a retaining-vs-excluding sensitivity comparison for the final runs. *(`--drop-zero-speed` defaults off, so the convention is preserved, and `docs/experiments.md` defines the comparison and its side effects. The portal's meaning for 0.0 is still unconfirmed.)*
- [x] Distinguish crash-event, observed-outcome, and control-matching eligibility. Document pre-history thresholds and exposure/masking choices.
- [ ] Save reusable expanded inputs and coverage outputs without overwriting the frozen pilot.

**Done when:** Expanded data and the inclusion rules are documented and load correctly on a small sample. **GPU forecasting is not a blocker.**

### PR #5 — Leakage-Safe Evaluation & Matched Controls

**Purpose:** Provide one trustworthy event/control evaluation interface, reused by all forecasters.

- [x] Confirm the known crash-matching and wall-clock/fitted-target leakage fixes; preserve regression tests for each confirmed bug.
- [x] Screen **input histories as well as forecast targets** for crash exposure in untreated training windows, using an explicit neighborhood/time rule.
- [ ] Implement or finalize leakage-safe **time-blocked out-of-fold** predictions (or a documented equivalent), including a gap for overlapping forecast windows.
- [x] Match controls on pre-event speed level/trend and appropriate road/calendar features; record eligibility, pre-event balance, and reproducible selection. *(Matching is deterministic, so a matched run does not move with the seed. It cleans up the crash road (placebo share 0.27 → 0.06) but does not rescue the spillover, because it matches the crash road's pre-event state and the hop effects are measured on neighbours. See `docs/experiments.md`.)*
- [ ] Use one documented primary analysis specification; keep model seeds and control-selection variation separate.

**Done when:** Small synthetic and real-data tests establish that evaluated outcomes were not fitted on, control histories are valid, and the shared estimator receives consistent event/control inputs.

### PR #6 — Spatiotemporal Counterfactual Forecaster

**Purpose:** Add the temporal graph comparison using the existing training pipeline.

- [x] Finish **one primary** spatial-message-passing + temporal model (GRU *or* temporal convolution). Keep the other only if already implemented and useful; avoid architecture proliferation.
- [x] Provide a comparable temporal no-edge baseline using the same histories/features where practical.
- [x] Reuse existing masks, splits, `argparse` scripts, and metrics; predict 10–60-minute horizons.
- [x] Pass a small CPU or short-GPU training/evaluation smoke test and save settings and per-horizon metrics.

**Done when:** Both main comparators train and score correctly on a small held-out sample. **Full-window GPU training is deferred to the experiment track.**

---

## Sprint 3 — Complete the causal and semi-synthetic pipeline

**November 18–26 · PRs #7–#9 · Development priority through November 25**

### PR #7 — Matched Event Study & First Neighbor Effects

**Purpose:** Provide the interpretable non-GNN comparison and an initial network-effect estimate.

- [x] Implement the matched event-study / difference-in-differences summary on the established event/control data.
- [x] Report pre-event trends and balance and produce effects for crash roads, 1-hop neighbors, and 2-hop neighbors.
- [x] Report counts by hop and horizon and a common-event cohort alongside available cases.
- [x] Add focused tests for event-time alignment, sign, and control weighting.

**Done when:** A small real-data run produces reproducible event-time and hop-level outputs. Full-data estimates can be run by the HPC teammate.

### PR #8 — Semi-Synthetic Network-Effect Benchmark

**Purpose:** Provide known treatment effects, using observed crash-free traffic as the untreated reference.

- [x] Inject documented slowdown patterns into held-out, crash-free windows at 0/1/2 hops; keep original speeds, injected observations, true effects, IDs, and masks.
- [x] Start with a few deterministic shock shapes; test nonrandom, pre-congestion-dependent placement only after the basic benchmark works.
- [x] Test injection algebra, missing-data behavior, exposure labeling, and no post-event feature leakage.

**Done when:** A deterministic fixture has exactly known effects and is accepted by the regular evaluation interface. **No full GPU training required.**

### PR #9 — Effect Scoring & Cross-Model Evaluation

**Purpose:** Turn counterfactual predictions and the semi-synthetic truth table into research results.

- [ ] Finish the **scoring runner** that reads saved predictions, matching assignments, and known-effect labels; reuse one estimator for synthetic and real events. *(`scripts/benchmark.py` scores synthetic events end to end, but computes predictions in-process rather than reading saved ones, and the real-event path still lives in `scripts/midpoint.py`.)*
- [x] Score average-effect bias, temporal/0–2-hop recovery, and uncertainty where supported; compare forecasters on the same held-out event set. *(Bias now carries an event-clustered 95% interval; MAE/RMSE stay cell-level error magnitudes.)*
- [x] Keep forecasting MAE/RMSE separate from effect recovery. For additive injection, unadjusted individual-effect error algebraically equals untreated forecast error; do not treat that alone as independent causal validation.
- [x] Smoke-test the runner with a hand-checked example and a tiny end-to-end case.

**Done when:** A known-effect input produces a correct, reproducible recovery table **without needing a full model-training run**.

**Development milestone (Nov 25–26):** Merge or stabilize PRs #4–#9 and hand one tested commit/configuration to the HPC teammate. The code may continue receiving correctness fixes, but avoid changing the evaluation specification during the main runs.

---

## Independent HPC / experiment track — teammate-owned

**Ownership:** Assign one teammate to operate Rivanna and return outputs; the developer/agents continue PRs #5–#9 without waiting. Use existing scripts rather than building an experiment orchestration framework.

### Can start immediately (nonblocking)

- [ ] Sync the current working branch, environment, and gitignored data; verify CUDA, memory, and output permissions.
- [ ] Build the graph for the expanded historical window; run **data-only** coverage, attrition, and control-supply reporting if the existing pipeline supports it.
- [ ] Verify a short GPU smoke run using a stable model branch and record device, code revision, arguments, and runtime.
- [ ] Save aggregate/monthly coverage outputs for the paper. The expanded sample information can be useful before the final model is trained.

### Run after the code/interface freeze (target Nov 26–29)

- [ ] Execute the primary single-seed expanded-window comparison: existing forecasters plus the chosen spatiotemporal model and matched temporal baseline.
- [ ] Run PR #7's direct/neighbor event estimates and PR #9's semi-synthetic known-effect recovery on the same frozen setup.
- [ ] Save `summary.json`, forecasting metrics, event/control sample counts, effect tables, synthetic recovery tables, figures, logs, arguments, and git commit in a **new** output directory.
- [ ] Flag failed jobs or surprising data counts immediately; do not change filtering decisions independently to obtain more favorable results.

### Run only after main results are inspected (target Nov 30–Dec 5)

- [ ] Repeat selected models with a few seeds (e.g., 3), keeping matching draws distinct from model seeds.
- [ ] Run **one** graph-structure ablation (real vs removed/rewired edges) and the matched no-edge comparator.
- [ ] Run high-value robustness checks: pre-event placebo, matched-control sensitivity, and the zero-speed convention if the data interpretation remains ambiguous.
- [ ] Skip optional hyperparameter/architecture grids if the primary results, presentation, or paper need attention.

**HPC handoff rule:** Every run receives (1) an agreed commit, (2) one exact command/argument set, (3) input location, (4) output destination, and (5) the expected result files. Keep runs in `outputs/expanded/`, `outputs/synthetic/`, or uniquely named subdirectories; never reuse `outputs/midpoint/`.

---

## Sprint 4 — Analyze, verify, and finish the final project

**November 27–December 9 · PRs #10–#12**

### PR #10 — Causal Diagnostics & Uncertainty

**Purpose:** Test the credibility and sensitivity of the primary estimates.

- [ ] Evaluate placebo periods, pre-event trends, matching balance, and the effect of alternate control draws.
- [ ] Quantify uncertainty at the crash/event level; account for correlated neighboring roads and overlapping windows.
- [ ] Summarize sensitivity to zeros, contamination windows, missingness, and concurrent crashes using **targeted** reruns.

**Done when:** The main real-event conclusions are accompanied by appropriate uncertainty and limitations, rather than an exhaustive sensitivity grid.

### PR #11 — Graph Contribution & Spillover Analysis

**Purpose:** Explain whether real road structure adds value to direct and neighboring effects.

- [ ] Analyze 0/1/2-hop profiles over time using common-event comparisons and report sample sizes.
- [ ] Compare graph vs matched no-edge models and a selected removed/rewired-edge ablation on common data.
- [ ] Interpret **both** forecasting differences and semi-synthetic effect-recovery differences; optional upstream/downstream or subgroup analysis only if time permits.

**Done when:** The paper can support a measured claim about the value—or limited value—of road connectivity.

### PR #12 — Final Tables, Reproducibility & Paper

**Purpose:** Produce a complete, concise ACM-style submission.

- [ ] Create final tables/plots for sample selection, forecasting, known-effect recovery, real spillovers, and robustness. Each display should support a specific claim.
- [ ] Verify every reported number against a saved output/configuration and distinguish the **frozen midpoint** from the corrected expanded analysis.
- [ ] Complete the presentation and the final paper. Overleaf remains the authoritative LaTeX source.
- [ ] Keep a short reproducibility command list and document material limitations; avoid adding new infrastructure.

**Done when:** The paper, slides, and repository agree, and another team member can reproduce the key results.

---

## Critical path and scope guardrails

**Required for the main claims:** trusted untreated data, leakage-safe predictions/controls, one trained spatiotemporal model with a fair no-edge comparator, a matched real-event estimate, the semi-synthetic scorer, initial 0/1/2-hop results, and uncertainty/graph checks.

**Keep minimal:** a single training/evaluation pipeline; simple `argparse` scripts; reusable functions in `src/traffic_gml/`; `uv` + `pyproject.toml`; Ruff and pytest; existing documentation. Add a new module only for distinct logic already needed by a PR. A full MLOps system, multiple causal model families, and broad automated tuning are out of scope.

**If time becomes tight, prioritize in this order:**

1. PR #9 scorer + PR #7 matched estimator working on the PR #8 truth table and a small real sample.
2. One full expanded-window model run and initial direct/1-hop results **before November 30**.
3. Synthetic known-effect recovery and a fair spatiotemporal vs no-edge comparison.
4. Event-level uncertainty, one graph ablation, and a small number of seeds.
5. Optional 2-hop refinement, heterogeneity, and extra architecture comparisons.

**Final deadlines:** Main code integration **Nov 25–26** · First presentation-ready results **Nov 29** · Presentations **Nov 30–Dec 7** · Final paper **Dec 9**.
