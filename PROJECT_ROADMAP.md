# Chicago Traffic Accident Spillovers — Ultimate Research Roadmap

**Course:** CS 6501 Graph Machine Learning, Fall 2026  
**Project:** Causal Graph Learning for Counterfactual Estimation of Traffic Accident Spillovers  
**Version:** November 24, 2026 | **Presentation:** November 30–December 7 | **Final ACM paper:** December 9  
**Execution:** One person coordinates automated coding, analysis, and Rivanna jobs. No manual annotation, hand labeling, or manual crash-by-crash inspection is required.

> **Central goal:** Determine how reported crashes relate to changes in speeds on the crash road and connected roads, and whether graph information improves **counterfactual effect estimation** beyond comparable non-graph methods. We will test these questions without presuming that GNNs win or that observed neighbor slowdowns are caused by crashes.

## 1. The three research questions

**RQ1 — Direct and neighboring-road effects.** How do estimated speed changes vary over the first hour after a crash, across the incident road, 1-hop roads, and 2-hop roads? Do the findings survive target-road pretrend, placebo, missingness, concurrent-crash, and fixed-cohort checks? Where the road data support it, does upstream/downstream direction matter?

**RQ2 — Value of the road graph.** Does real connectivity improve (a) held-out speed forecasting and (b) recovery of **known** treatment effects, compared with equally informed non-graph models, self-only graph models, and a falsified/rewired graph?

**RQ3 — Treatment-aware learning.** Does directly predicting factual and counterfactual outcomes with exposure-conditioned MLP/GNN heads improve effect recovery relative to the existing untreated-forecast-plus-matched-residual method? Is any improvement due to the graph rather than simply to treatment conditioning, neighborhood features, or extra capacity?

**Meaning of success:** A scientifically interpretable answer to these questions, including negative or mixed results, with explicit assumptions and reproducible numerical evidence. Forecasting MAE and causal effect error must never be treated as interchangeable.

## 2. Completed foundation — freeze, do not repeat

**Run A data window:** `[2024-06-11, 2026-05-01)` using Chicago segment speeds `4g9f-3jbs` and crash reports `85ca-t3if`. **Run B:** historical `sxs8-h27x`, `[2018-02-28, 2023-09-07)`, separate and optional.

- [x] Convert **101,138,608** raw speed observations and **207,612** crash reports to partitioned Parquet.
- [x] Build the monitored Chicago graph, match crashes, and retain frozen road-node identities.
- [x] Model **881** segments; **4,744** directed message-passing arcs; **44,108** geospatially matched crashes.
- [x] Produce **32,479** eligible events for primary matched-residual analysis; **57.1%** of road/time cells observed.
- [x] Train seven forecasting methods: persistence, historical mean, MLP, temporal GRU, GCN, GraphSAGE, STGCN.
- [x] Estimate initial 0-/1-/2-hop event-study contrasts, temporal placebo, event-clustered intervals, and **10,162-event** common cohort.
- [x] Evaluate the original 200-event additive semi-synthetic benchmark and a zero-speed sensitivity run.
- [x] Complete Run A's five Rivanna stages; GPU compatibility and dataset-path failures were fixed.

### Frozen reference numbers (not predetermined final conclusions)

| Metric | Completed Run A |
|---|---:|
| Forecast MAE: historical mean / MLP / temporal GRU | 3.151 / 3.102 / 3.067 mph |
| Forecast MAE: GCN / GraphSAGE / STGCN | 3.072 / 3.036 / 3.029 mph |
| Event-study contrast at 30 minutes, 0 / 1 / 2 hops | −0.523 / −0.106 / −0.056 mph |
| 1-hop at 10 minutes: contrast / placebo | −0.098 / −0.069 mph |
| Original additive synthetic effect MAE: historical mean / STGCN | 3.247 / 3.288 mph |

**Interpretation:** These are the outputs of one initial analysis. Neighboring-road contrasts are not automatically causal. The original additive synthetic benchmark can make individual effect error reduce algebraically to untreated forecasting error; stronger tests are needed. Do not choose later models, controls, or benchmarks to favor the current ranking.

**Preserve unchanged:** `outputs/midpoint/`, `outputs/runA/`, `outputs/event_study-runA/`, `outputs/synthetic-runA/`, `outputs/runA-nozero/`, `data/processed-runA/`, the Run A Parquet datasets, and the five graph/reference files. All extensions write to new names.

## 3. Experimental contract (P0) — met

The contract is `docs/experiments.md` (settings, eligibility, exclusions) and
`docs/methodology.md` (estimator and limits). No third spec file and no config
file: `scripts/hpc/experiments.sh` holds the matrix settings, and every run
records its own in `summary.json` or `audit.json`. The offline suite enforces
the clauses below.

- [x] **Provenance:** `scripts/hpc/run.sh` saves the commit, diff,
  `source.tar.gz`, lock hash, packages, Slurm arguments and seeds per job. Run
  A's logged commit `40f672e` cannot run the matrix, so that per-job snapshot is
  the record rather than the commit. Confirming reproduction needs a rerun.
- [x] **Splits:** chronological 70/15/15 with a `ceil((L+H)/stride)` embargo;
  synthetic train and test interventions disjoint.
- [ ] **Exposure:** rings from the real graph, undirected; a control's
  no-exposure radius equals `--exposure-hops`. **Open:** two crashes with
  overlapping rings leave a road's hop label ambiguous. Workstream A owns it,
  because choosing the rule needs the comparison run.
- [x] **Same target:** an ablation changes message passing only.
- [x] **Pre-treatment inputs:** `x_mask` masks contaminated and missing inputs
  alike; injection never enters the input history.
- [x] **Fair graph baseline:** `--neighbour-hops k` gives any model the masked
  mean speed over its k-hop neighbourhood. Off by default and unused so far, so
  every recorded `gcn`-`mlp` gap is still an auxiliary baseline.
- [x] **Inference:** crashes equal-weighted, intervals event-clustered.
- [x] **No manual labeling.**

**Gate passed.** The CPU fixture is the existing offline suite, which covers
exposure assignment, input leakage, paired masks and effect alignment.

## 4. Automated workstreams — all valuable, not all equally urgent

A coding agent can implement each workstream independently against the frozen interface. Complete CPU/unit tests before launching its large GPU run. The workstreams are **work units, not people or team assignments**.

### A. Observational identification and falsification — **required**

**Purpose:** Assess whether apparent neighboring-road slowdowns are distinguishable from pre-existing/shared congestion.

- [ ] Re-evaluate **the target neighbor's own** pre-crash speed level and trend, not only the crash road's; save balance before and after new control selection.
- [ ] Add target-road-specific matched or reweighted controls based on pre-event speed/trend, hour/day/season and comparable road conditions; document overlap and attrition.
- [ ] Add 2–3 pre-event pseudo-crash/placebo times and one unrelated/unexposed-road comparison where logically valid; preserve sign, CI, and placebo-to-effect magnitude ratio as a diagnostic, **not** a probability.
- [ ] Exclude or flag concurrent reported crashes on target/adjacent roads during the full input/outcome window; compare with original sample.
- [ ] Report **available-case and fixed-event-cohort** 0/1/2-hop trajectories with event-level 95% intervals, per-horizon events, roads, controls, and denominators.
- [ ] Audit speed availability by exposed/control status, hop, and event-time bin; test whether crash exposure predicts missingness. Keep observed `0 mph` separate from unavailable readings.
- [ ] Document a stopping rule: do not repeatedly tune matching until placebo effects look small; keep both original and revised specifications.

**Deliverables:** `outputs/identification-runA/` with balance/attrition tables, `hop_effects.csv`, placebo results, missingness diagnostics and 1–2 publication-ready figures.

### B. Graph-structure ablation — **required**

**Purpose:** Separate real topology from added features/capacity.

- [ ] Run one matched architecture under **real edges**, **self-only/no neighbor messages**, and **one fixed degree-preserving or otherwise documented rewired topology**.
- [ ] Keep event exposure/labels on the original network, same model budget, features, seeds, splits, masks and target population; change only message-passing structure.
- [ ] Compare test forecast MAE/RMSE and synthetic **effect** MAE/bias by hop and horizon; include paired event bootstrap differences where applicable.
- [ ] Add at least three training seeds for the principal comparison if jobs finish comfortably; report seed spread separately from sampling uncertainty.

**Deliverables:** `outputs/graph-ablation-runA/`, paired results table and a **two-panel figure: forecast error vs. effect recovery**.

### C. Treatment-aware outcomes: fair MLP vs. GNN — **required main extension**

**Purpose:** Test explicit intervention conditioning rather than only forecasts of untreated speed.

- [x] Generate automated factual exposure labels `none/own/1-hop/2-hop` and the corresponding observed speed targets; reject ambiguous simultaneous exposure or use a documented priority rule. `events.exposure_labels`; concurrent exposure resolves to the nearer crash.
- [x] Implement a simple exposure-conditioned MLP and graph encoder (GCN or STGCN), each with shared pre-treatment representation and exposure-specific head/embedding; supervise **only the observed factual outcome**. `models.TreatmentAware` as `treatment_mlp` and `treatment_gcn`, trained on `Windows.factual()`.
- [x] Give the MLP the same pre-event neighborhood information in aggregated form (`--neighbour-hops`); keep optimization budget, training sample, masks, horizons and exposure coding aligned.
- [ ] At evaluation, predict the same road/horizon **with the specified exposure and with no exposure**; score their difference against synthetic ground truth. Compare with the original residual-based estimator. `evaluate.exposure_effects` provides the readout; the scoring needs the run.
- [ ] Check exposure overlap, head utilization, masking, training/evaluation leaks, and small synthetic examples with analytically known answers before large training.
- [ ] Use observational outputs as a robustness comparison, **not** to tune methods to yield desired causal signs. Two potential-outcome heads do not alone resolve unmeasured confounding.

**Deliverables:** `outputs/treatment-aware-runA/` with shared-sample metrics, model comparisons and tests. Limit architectures to one MLP and one GNN unless a genuine implementation problem warrants adjustment.

### D. Stronger semi-synthetic benchmarks — **required**

**Purpose:** Test effect estimation when both potential outcomes and network exposure are known.

- [ ] Keep original **Benchmark 0** (200-event additive injection) for continuity; clearly state its forecast-error identity.
- [x] **Benchmark 1: synthetic null.** Direct crash-road slowdown but **zero true 1/2-hop effect**; measure spurious neighbor estimates and their intervals. `--hop-decay 0`, no new generator; queued as `synthetic-scenarios` task 0. **Awaiting its run.**
- [ ] **Benchmark 2: propagation.** Known 0/1/2-hop decay, temporal onset/duration, and at least two prespecified intervention magnitudes; measure effects by hop and horizon.
- [ ] **Benchmark 3: confounded assignment (valuable).** Compare random placement with assignment dependent only on pre-crash congestion; quantify bias and overlap. Treat as sensitivity, not proof of real-data identification.
- [ ] Use at least three **synthetic generation** seeds where feasible; never conflate those with training seeds.
- [ ] Preserve `Y(0)`, `Y(a)`, treatment label, ground-truth `tau`, event/road/time IDs and observation masks. All models score the **same** held-out interventions.
- [ ] Report effect MAE/RMSE, average bias, estimated-vs-true trajectories, false neighbor spillover under the null, and event-bootstrap intervals. Avoid pooled-only metrics.

**Deliverables:** `outputs/synthetic-extended-runA/` with clean truth tables and two concise benchmark figures/tables.

### E. Directed and heterogeneous effects — **targeted, automated**

**Purpose:** Restore the original proposal's traffic-specific questions about *where* and *how long* disruptions spread.

- [ ] If directed mapping and turn feasibility can be verified programmatically, classify eligible neighbors as upstream/downstream; compare event-time profiles and report ambiguous/omitted edges. If direction is unreliable, explicitly report it as **not assessed**, not a null effect.
- [ ] Compute a simple effect-duration measure (e.g., time windows of detectable negative contrasts) **only after** the placebo and missingness checks are stable.
- [ ] Predeclare **at most two** heterogeneity dimensions: pre-crash congestion and either reliable crash damage/severity or road centrality. Keep subgroup definitions fixed, show counts and uncertainty, and treat these as exploratory rather than multiple-confirmed causal findings.
- [ ] Optionally test a directed-distance exposure rule against plain hop count without changing the core paper's estimand; declare any new estimand separately.

**Deliverables:** `outputs/heterogeneity-runA/` with one compact table/plot. No manual severity annotation.

### F. Extra replication and external data — **background only, never on critical path**

- [ ] **Run B historical replication:** queue the existing resumable download and separate run only after primary Run A outputs are safe; `sxs8-h27x`, `[2018-02-28, 2023-09-07)`, no overlap with the 2023–2024 data gap. The historical/COVID distribution differs, so report as cross-era validation, not a pooled training extension by default.
- [ ] **Weather, construction, closures:** only if official fields can be automatically geotemporally joined with audited coverage and no manual map/label cleanup. Otherwise document as possible unobserved confounders.
- [ ] **Doubly robust adjustment:** add only if diagnostics reveal sufficient overlap and a credible nuisance-model interface; it must use held-out nuisance predictions. It is not a substitute for temporal placebos.

**Defer beyond course paper:** learned exposure mapping, GAT/graph Transformer families, hypergraph frameworks, spatial transfer, vulnerability rankings presented as causal, broad hyperparameter sweeps, custom MLOps infrastructure.

## 5. Dependency-aware execution plan

| Window | What to run/finish | Required check |
|---|---|---|
| **Nov 24–25** | P0 source snapshot, event/target contract, paper skeleton; launch CPU analyses of A/E | Identical IDs/targets, documented provenance |
| **Nov 25–27** | A controls/placebos/missingness; implement B/C/D behind shared interfaces; smoke-test | No post-treatment leakage; meaningful pretrend/overlap reports |
| **Nov 27–29** | GPU jobs for B and C; D null/decay benchmarks; preview figures/slides | Primary results frozen for presentation; unfinished work labeled |
| **Nov 30–Dec 5** | Presentation, finish C/D if pending, 3-seed comparisons, E direction/heterogeneity | Use only validated outputs; no auto-claim based on p-value |
| **Dec 6–7** | Lock main experiments and regenerate paper tables automatically | Checksums, cross-run consistency, captions and denominators |
| **Dec 8–9** | ACM paper edit, exact-reference review, reproducibility test, final PDF | No new unreviewed experiments; 8-page format |

**Parallelism:** Implementation of A–E may run in separate branches/worktrees, but all experiments must consume P0 contracts. Use one integration point for shared code and separate job/output names; avoid conflicting writes. Do not submit redundant 101M-row downloads or overwrite frozen outputs.

**Stop rule:** The course paper should still be submitted if an ambitious extension is incomplete. Only claim a completed experiment when the code, saved outputs and evaluation contract pass checks. Never fill a pending table with expected values.

## 6. Clear acceptance criteria and analysis decisions

| Claim we might test | Evidence needed before writing it |
|---|---|
| “Real graph improves forecasting” | Paired real-vs-self-only test, equal inputs/splits/targets, seeds or uncertainty |
| “Graph improves causal effect recovery” | Better **known-effect** error on the same synthetic events, not just speed MAE |
| “Treatment-aware model helps” | Factual-only supervised GNN/MLP on same data plus synthetic true-effect comparison |
| “There is a neighboring-road causal effect” | Target-road pretrends/placebos, feasible controls, concurrent-exposure robustness, stable cohorts and uncertainty; otherwise report an association and limitations |
| “Effects spread upstream/downstream” | Verified directional road exposure and enough eligible events; not merely an undirected graph neighborhood |
| “Effects are heterogeneous” | Prespecified subgroup, sufficient support and uncertainty, no post-hoc sign chasing |
| “Results generalize to historical era” | Independent Run B replication and discussion of data/regime differences |

**Default reporting:** state observational estimands as matched crash-associated speed contrasts. Stronger causal language is conditional on clearly defended assumptions; semi-synthetic recovery is not proof those assumptions hold for real crashes.

## 7. Figures and tables the paper needs

| Artifact | Source | Status |
|---|---|---|
| **Table 1:** dataset, matching funnel, graph and coverage | Run A + P0 | Baseline ready, verify |
| **Table 2:** seven model forecast performance | Run A | Complete; later add graph controls |
| **Figure 1:** road graph and study design / exposure rings | Existing graph/reference + generated schematic | Ready to render automatically |
| **Figure 2:** 0/1/2-hop trajectories, original vs revised controls, fixed cohort and placebo | A | Initial results; final update pending |
| **Table 3:** pretrend/placebo/missingness and overlapping-crash sensitivity | A | Pending |
| **Figure 3:** real/self-only/rewired forecasting and effect errors | B | Pending |
| **Table 4:** treatment-aware MLP/GNN and residual-method effect recovery | C + D | Pending |
| **Figure 4 / Table 5:** synthetic null and known decay by hop/horizon | D | Pending |
| **Optional:** directional and congestion/severity subgroup | E | Conditional |

**Space discipline:** Only the most informative subset goes into the eight-page ACM paper. Additional results can stay in repository files or an appendix if permitted. The argument should never depend on a figure that is still pending.

## 8. Reproducibility, automation, and quality checks

- [ ] Automated scripts produce all labels/cohorts/figures/tables; no hand-assigned crash outcomes or counterfactuals.
- [ ] Every output folder contains frozen configuration, source revision or code snapshot, data manifests, seeds, sample counts and flags.
- [ ] Row IDs, speeds, outcomes, treatment exposures and graphs are unchanged across paired methods except for the single dimension intentionally ablated.
- [ ] Test splits exclude overlapping future targets and contaminated control histories; event-clustered uncertainty treats roads/horizons as dependent.
- [ ] Synthetic ground truth is defined independently of which model happens to perform best; evaluate on held-out interventions.
- [ ] Clear denominator for every result (crashes, unique events, roads, road-time cells, matched controls). Separate full, observed, and fixed cohorts.
- [ ] Missing readings are masked; `0 mph` sensitivity is explicitly a changed measurement convention and possibly changed population.
- [ ] All paper numbers are read from saved machine-readable outputs. Generate a result index showing which source file populates which manuscript table/figure.
- [ ] Compare the final paper to the original proposal's three questions and two hypotheses; explain changes as methodological refinements rather than invented new goals.
- [ ] Final draft stays **balanced**: mark existing figures as completed and pending comparisons as pending, then write final interpretation after results arrive.

## 9. Final decisions

**Keep:** Strong neighbor-specific identification checks; equal-information graph controls; topology ablation; one treatment-aware MLP and one GNN; null/decay/confounding synthetic cases; a few paired seeds; automatic missingness, directional and predeclared heterogeneity diagnostics.

**Queue only if safe:** Historical Run B, vetted weather/roadworks join, held-out doubly robust comparison, alternative directed exposure rule.

**Skip for December 9:** New architecture families, manually curated labels, learned exposure maps, exhaustive tuning, novel causal-graph discovery claims and large infrastructure rewrites.

**Final research standard:** Compare graph and non-graph methods fairly, estimate known effects when truth is available, and show exactly what the observational data do and do not establish about traffic crash spillovers.
