# Chicago Traffic Accident Spillovers — Project Roadmap

**Course:** CS 6501 Graph Machine Learning · Fall 2026  
**Team:** Nate, Maddox, Kyle  
**Timeline:** November 6 – December 9, 2026  
**Plan:** 4 sprints · 12 planned pull requests (PRs)

## Research goal

Estimate how traffic accidents change road speeds **on the affected road and on connected roads over time**. Compare non-graph models, a basic GNN, and a spatiotemporal GNN to determine whether learning road-network structure improves counterfactual traffic forecasting and accident-effect estimation.

**Core approach:** Train models on accident-free traffic windows to forecast expected *no-accident* speeds from pre-event information. At real accidents, compare observed speeds against those predictions to estimate effects. Use matched causal methods as a separate baseline, and prioritize **semi-synthetic experiments with known effects** to test estimation accuracy. Forecasting accuracy alone does **not** establish causality.

---

## Sprint 1 — Initial Setup & Midpoint Review

**November 6–9 · 3 PRs · Complete**  
**Goal:** Collect data, build the road graph, and establish baseline modeling code.

### PR #1 — Traffic Data — Nate

**Purpose:** Collect and prepare the real traffic data needed to study how accidents affect road speeds.

**Tasks**
- [x] Download Chicago crash and traffic-speed datasets.
- [x] Inspect columns, missing values, and time coverage.
- [x] Choose a study area with good speed coverage.
- [x] Clean data and save CSV/Parquet files.
- [x] Create a table of road speeds by segment and timestamp.

**Done when:** Cleaned crash and speed datasets are available for the rest of the project.

### PR #2 — Road Graph — Maddox

**Purpose:** Represent Chicago roads as a network so models can learn how disruptions spread between connected road segments.

**Tasks**
- [x] Build a road-segment graph using NetworkX.
- [x] Connect segments that share navigable intersections.
- [x] Match accidents to road segments.
- [x] Compute degree and betweenness centrality.
- [x] Create a map showing roads and accident locations.

**Done when:** A usable road graph exists, with accidents mapped to graph nodes.

### PR #3 — Baselines & Modeling Setup — Kyle

**Purpose:** Establish reusable training code and simple models to compare with the later spatiotemporal GNN.

**Tasks**
- [x] Set up Python, PyTorch, and PyTorch Geometric.
- [x] Implement a simple speed-prediction baseline.
- [x] Implement a basic GCN or GraphSAGE.
- [x] Define model inputs and prediction targets.
- [x] Set up training scripts and MAE/RMSE evaluation.

**Done when:** A baseline and basic GNN can run on sample data and produce predictions.

### Shared midpoint tasks

- [x] Agree on consistent road IDs, timestamps, and data interfaces.
- [x] Join the available traffic, accident, and road graph data.
- [x] Create example before/after accident windows and accident-free controls.
- [x] Run initial baseline/GNN experiments if integration is ready.
- [x] Prepare data coverage visualizations and the **November 9 midpoint report**.

**Sprint deliverable:** Initial datasets, road network, runnable model code, and midpoint review. **Delivered**, plus an integrated end-to-end pipeline ahead of schedule: the three-way join, crash/control event windows, five forecasters on identical splits, the placebo-corrected effect estimate, and coverage reporting. The frozen baseline is recorded in [experiments.md](docs/experiments.md).

---

## Sprint 2 — Data Expansion & Spatiotemporal Modeling

**November 10–17 · 3 PRs · Unassigned**  
**Goal:** Lift the sample and control-supply limits the midpoint exposed, then train the spatiotemporal model. The end-to-end pipeline itself landed in Sprint 1.

### PR #4 — Data Expansion & Control Matching

**Purpose:** Remove the sample-size and control-supply limits that cap the midpoint estimates.

*The original join/window/split tasks were delivered during Sprint 1 integration; see [experiments.md](docs/experiments.md).*

**Tasks**
- [ ] Extend the study period beyond March–April 2025. **No download needed:** 338 days of speeds (2025-01-01 to 2025-12-06, 78 MB) and 109,121 crashes are already local. The midpoint used 61 days and 17,658 crashes, so this is a `--window` change.
- [ ] Re-measure control supply; only ~1.9 of 5 requested controls per crash exist over two months. Matching needs same-road, same-week-slot candidates, so ~8 candidate weeks becomes ~48 over a full year.
- [ ] Budget the compute: ~5.5x the data at ~20 min per CPU run makes a GPU run worth setting up (`--device cuda` on Rivanna).
- [ ] Loosen or tier the control-matching rule (nearby week slots, comparable roads) and test sensitivity.
- [ ] Measure how long speeds stay depressed after a crash instead of assuming `horizon` bins.
- [ ] Recover events lost to a contaminated pre-crash history (693 of 3,651, the largest single exclusion).

**Done when:** The analysis sample and the controls per crash both grow, and the effect estimate's sensitivity to the control draw is quantified rather than assumed.

### PR #5 — Experiment Protocol & Leakage Guards

**Purpose:** Make results comparable across runs and prove the splits are clean.

*Graph/feature alignment and the PyG tensor contract were delivered in Sprint 1.*

**Tasks**
- [ ] Report every estimate over multiple seeds and multiple control draws.
- [ ] Add a comparison script for results across models, seeds, and matching configurations.
- [ ] Add regression tests that fail on temporal, event, or control leakage.
- [ ] **Decide what a 0.0 mph speed means.** 0 mph is 15x more common than 1 mph (16,623 vs 1,135 rows), which looks like a sentinel, but zeros are isolated single bins across 821 of 879 segments, which looks real. `docs/data.md` currently counts zero as a valid observation. 98 crash rows and 66 control rows at exactly 0.0 carry **31% of the pooled effect** (-0.430 with, -0.296 without). Resolve before any effect estimate is published.
- [ ] Mask crash-contaminated cells in `x`, not just `y`. About 18% of usable training targets read a crash-affected road inside the receptive field, and `--exposure-hops` cannot fix it (see `docs/architecture.md`).
- [ ] Free the control pool from the last 30% of the calendar. Any training-period candidate is rejected by construction, so controls come from val+test while 71% of crash rows come from the training period; restricting the crash arm to the same period moves the estimate -0.430 to -0.466. Rolling-origin or k-fold training would remove the asymmetry.
- [ ] Decide whether monitored segments joined by unmonitored streets should be adjacent; 84 roads (9.6%) sit outside the largest component.
- [ ] Specify the semi-synthetic benchmark before building it: injection, ground-truth storage, recovery scoring.

**Done when:** Any two runs can be compared on the same splits, and leakage is tested rather than argued.

### PR #6 — Spatiotemporal Counterfactual Forecasting

**Purpose:** Build the main model, combining road connectivity and past traffic speeds to forecast expected traffic without an accident.

**Tasks**
- [ ] Implement a spatiotemporal GNN using message passing plus a GRU or temporal convolution.
- [ ] Train on accident-free windows to predict future speeds.
- [ ] Use forecasting MSE/MAE as the training objective.
- [ ] Predict multiple horizons (10, 20, and 30 minutes; optionally 60).
- [ ] Compare with non-graph and basic GNN baselines on the same splits.

**Done when:** Trained models produce expected no-accident speed forecasts on held-out data.

**Sprint deliverable:** A larger sample with better-supplied controls, a reproducible experiment protocol, and spatiotemporal forecasting compared against the frozen midpoint baseline.

---

## Sprint 3 — Causal Estimation & Semi-Synthetic Evaluation

**November 18–26 · 3 PRs · Unassigned**  
**Goal:** Estimate crash-related speed effects and measure effect recovery against known ground truth.

### PR #7 — Matched Causal Baseline

**Purpose:** Provide a traditional causal inference comparison that does not depend on graph learning.

**Tasks**
- [ ] Match accident periods with comparable accident-free periods.
- [ ] Implement matched difference-in-differences or event-study estimation.
- [ ] Estimate speed effects on crash roads and nearby roads.
- [ ] Check pre-accident speed trends.
- [ ] Save estimates for comparison with GNN-based methods.

**Done when:** A reproducible non-GNN causal baseline produces estimates for real crash events.

### PR #8 — Semi-Synthetic Accident Benchmark

**Purpose:** Create realistic traffic scenarios with **known** accident effects so we can check whether our methods recover them correctly.

**Tasks**
- [ ] Select real accident-free traffic windows as baseline conditions.
- [ ] Inject simulated speed reductions on selected roads.
- [ ] Simulate disruptions spreading across connected roads over time.
- [ ] Vary effect magnitude, duration, network distance, and direction.
- [ ] Save unmodified speeds and known effects as ground truth.

**Done when:** A reproducible benchmark provides treated observations, untreated reference speeds, and known effect labels.

### PR #9 — Counterfactual Effect Evaluation

**Purpose:** Connect speed forecasting to causal analysis and test whether models estimate effects accurately—not just future speeds.

**Tasks**
- [ ] Generate no-accident forecasts for real and semi-synthetic events using only pre-event inputs.
- [ ] Estimate effects as **observed speed − predicted no-accident speed**.
- [ ] Evaluate semi-synthetic effect recovery using treatment-effect MAE/RMSE.
- [ ] Compare non-graph, basic GNN, and spatiotemporal GNN estimates.
- [ ] Report forecasting accuracy separately from treatment-effect accuracy.
- [ ] Check the effect estimate is not forecaster-dependent; `gcn` was chosen while `sage` forecast better.
- [ ] Compare available-case against a fixed cohort observed at every horizon; per-bin `n` varies 1,782–1,846, so sample composition shifts with horizon.

**Done when:** Models can be compared on recovery of known effects as well as speed forecasting performance.

**Sprint deliverable:** Real accident-effect estimates, a ground-truth semi-synthetic benchmark, and initial causal evaluation results.

---

## Sprint 4 — Spillovers, Robustness & Final Experiments

**November 27 – December 9 · 3 PRs · Unassigned**  
**Goal:** Determine whether network learning helps, test robustness, and finalize paper-ready results.

### PR #10 — Causal Robustness

**Purpose:** Check whether estimated accident effects are credible or might reflect ordinary traffic patterns, confounding, or model bias.

**Tasks**
- [ ] Run placebo tests during periods without accidents.
- [ ] Check pre-accident trends and forecast bias.
- [ ] Test sensitivity to matching choices and overlapping accidents.
- [ ] Estimate uncertainty using accident-level clustering or bootstrap.
- [ ] Document assumptions, limitations, and threats to causal identification.

**Done when:** We can explain the reliability and limitations of our effect estimates.

### PR #11 — Network Spillovers & Graph Ablations

**Purpose:** Answer the main network question: how disruption spreads to nearby roads and whether the graph improves effect estimation.

**Tasks**
- [ ] Compare effects on crash roads, 1-hop neighbors, and 2-hop neighbors.
- [ ] Analyze how effects change over time.
- [ ] Compare upstream and downstream propagation where feasible.
- [ ] Test original, shuffled/removed-edge, and no-graph settings.
- [ ] Examine whether road centrality relates to disruption severity.
- [ ] Test effect heterogeneity by crash severity, time of day, and pre-crash congestion; a pooled mean can hide this.
- [ ] Budget for the smaller spillover sample: 1,459 events have a usable 1-hop trajectory, 1,628 at 2 hops, against 2,529 for direct effects.

**Done when:** We have spillover results and evidence about the value of graph connectivity.

### PR #12 — Model Ablations & Final Evaluation

**Purpose:** Identify which modeling choices improve performance and prepare reproducible results for the final paper.

**Tasks**
- [ ] Compare non-graph, basic GNN, and spatiotemporal GNN models.
- [ ] Test different history lengths and prediction horizons.
- [ ] Test temporal-component and node-feature ablations.
- [ ] Summarize speed forecasting results separately from causal effect recovery.
- [ ] Generate final tables, figures, and commands to reproduce experiments.
- [ ] Do not treat the 3.03M road-time test targets as independent; windows overlap and roads correlate. Use paired comparisons on identical test periods.

**Done when:** Main experiments are complete and outputs are ready for the final report.

**Sprint deliverable:** Spillover analysis, graph and model ablations, robustness results, figures, and reproducible experiments.

---

## Milestones & shared deliverables

| Date | Deliverable |
| --- | --- |
| **Nov 9** | Midpoint review: initial data, graph, models, and progress report |
| **Nov 17** | Integrated data pipeline and spatiotemporal forecasts |
| **Nov 26** | Causal estimates and semi-synthetic effect-recovery evaluation |
| **Nov 29** | Main experiments and presentation materials ready |
| **Nov 30 – Dec 7** | Project presentation |
| **Dec 9** | Final ACM-format paper and reproducible repository |

The **paper and presentation are shared responsibilities**, not separate code PRs. **Overleaf is the source of truth for LaTeX**; the Git repository may include a convenience copy.

## Integration & PR conventions

- **One PR per meaningful deliverable.** The 12 PRs are a planning target, not a strict limit; split a PR if it becomes too large or blocks the team.
- **Integrate continuously.** Do not wait until the end of a sprint to merge compatible work.
- **Prioritize PR #8 early** if the event-window format stabilizes ahead of schedule. The semi-synthetic benchmark is central to the mentor's feedback.
- Everything else about how a PR is written and checked lives in [AGENTS.md](AGENTS.md).

## Research checks

See [methodology.md](docs/methodology.md) for the forecasting objective, the effect estimator, the core comparison, ground-truth validation, and the causal caveats.
