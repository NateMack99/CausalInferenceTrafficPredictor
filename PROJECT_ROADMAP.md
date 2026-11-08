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

**November 6–9 · 3 PRs · Assigned**  
**Goal:** Collect data, build the road graph, and establish baseline modeling code.

### PR #1 — Traffic Data — Nate

**Purpose:** Collect and prepare the real traffic data needed to study how accidents affect road speeds.

**Tasks**
- [ ] Download Chicago crash and traffic-speed datasets.
- [ ] Inspect columns, missing values, and time coverage.
- [ ] Choose a study area with good speed coverage.
- [ ] Clean data and save CSV/Parquet files.
- [ ] Create a table of road speeds by segment and timestamp.

**Done when:** Cleaned crash and speed datasets are available for the rest of the project.

### PR #2 — Road Graph — Maddox

**Purpose:** Represent Chicago roads as a network so models can learn how disruptions spread between connected road segments.

**Tasks**
- [ ] Build a road-segment graph using NetworkX.
- [ ] Connect segments that share navigable intersections.
- [ ] Match accidents to road segments.
- [ ] Compute degree and betweenness centrality.
- [ ] Create a map showing roads and accident locations.

**Done when:** A usable road graph exists, with accidents mapped to graph nodes.

### PR #3 — Baselines & Modeling Setup — Kyle

**Purpose:** Establish reusable training code and simple models to compare with the later spatiotemporal GNN.

**Tasks**
- [ ] Set up Python, PyTorch, and PyTorch Geometric.
- [ ] Implement a simple speed-prediction baseline.
- [ ] Implement a basic GCN or GraphSAGE.
- [ ] Define model inputs and prediction targets.
- [ ] Set up training scripts and MAE/RMSE evaluation.

**Done when:** A baseline and basic GNN can run on sample data and produce predictions.

### Shared midpoint tasks

- [ ] Agree on consistent road IDs, timestamps, and data interfaces.
- [ ] Join the available traffic, accident, and road graph data.
- [ ] Create example before/after accident windows and accident-free controls.
- [ ] Run initial baseline/GNN experiments if integration is ready.
- [ ] Prepare data coverage visualizations and the **November 9 midpoint report**.

**Sprint deliverable:** Initial datasets, road network, runnable model code, and midpoint review. The fully integrated pipeline is completed in Sprint 2.

---

## Sprint 2 — Integration & Spatiotemporal Modeling

**November 10–17 · 3 PRs · Unassigned**  
**Goal:** Build an end-to-end pipeline for forecasting what traffic speeds would be without accidents.

### PR #4 — Causal Dataset Pipeline

**Purpose:** Turn traffic and crash records into consistent examples for model training and accident-effect estimation.

**Tasks**
- [ ] Join traffic speeds and accidents using consistent road IDs and timestamps.
- [ ] Create speed windows before and after accidents.
- [ ] Identify comparable accident-free windows for training and controls.
- [ ] Create train/validation/test splits without temporal or event leakage.
- [ ] Handle missing observations and overlapping accidents.

**Done when:** Accident and accident-free windows can be loaded reliably for experiments.

### PR #5 — Graph Integration & Model Features

**Purpose:** Link traffic observations to the road graph so models can use speed histories from connected roads.

**Tasks**
- [ ] Align graph nodes with traffic-speed and accident records.
- [ ] Identify crash roads and their 1-hop and 2-hop neighbors.
- [ ] Prepare historical speed features and observation masks.
- [ ] Convert road connections and features into PyTorch Geometric tensors.
- [ ] Verify that graph edges and feature indices match.

**Done when:** The shared dataset can be passed directly to graph models.

### PR #6 — Spatiotemporal Counterfactual Forecasting

**Purpose:** Build the main model, combining road connectivity and past traffic speeds to forecast expected traffic without an accident.

**Tasks**
- [ ] Implement a spatiotemporal GNN using message passing plus a GRU or temporal convolution.
- [ ] Train on accident-free windows to predict future speeds.
- [ ] Use forecasting MSE/MAE as the training objective.
- [ ] Predict multiple horizons (10, 20, and 30 minutes; optionally 60).
- [ ] Compare with non-graph and basic GNN baselines on the same splits.

**Done when:** Trained models produce expected no-accident speed forecasts on held-out data.

**Sprint deliverable:** An integrated dataset and working non-graph, basic GNN, and spatiotemporal forecasting experiments.

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

**Done when:** We have spillover results and evidence about the value of graph connectivity.

### PR #12 — Model Ablations & Final Evaluation

**Purpose:** Identify which modeling choices improve performance and prepare reproducible results for the final paper.

**Tasks**
- [ ] Compare non-graph, basic GNN, and spatiotemporal GNN models.
- [ ] Test different history lengths and prediction horizons.
- [ ] Test temporal-component and node-feature ablations.
- [ ] Summarize speed forecasting results separately from causal effect recovery.
- [ ] Generate final tables, figures, and commands to reproduce experiments.

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
