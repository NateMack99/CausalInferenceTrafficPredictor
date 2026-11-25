# STRICT REIMPLEMENTATION DIRECTIVE — TRAFFIC SPILLOVERS

Temporary working spec. Delete once the reimplementation replaces the original
project tree.

Paper: **Graph Learning for Counterfactual Estimation of Traffic Accident
Spillovers**

**Objective: produce the smallest, clearest, scientifically correct
implementation of our existing research.**

This is not a cleanup of the current architecture. You may move, merge, replace,
or delete any existing implementation, entry point, abstraction, or file that is
not necessary. We do not simply reorganize 10,000 lines into fewer files: we
remove unnecessary behavior, reimplement the essential research directly, and
eliminate entire execution paths.

The primary quality metric is **human comprehension**. The repository should
resemble lean published research code, not a production ML platform.

Baseline inventory: approximately **10,701 lines** of maintained code, tests,
and technical documentation (excluding the paper, lockfile, and reference data).
Target: **3,000–4,000 maintained lines** — not a hard ceiling, but the standard
against which every remaining function must justify itself.

Study the implementation style of Graph WaveNet, MTGNN, SparseTSF, FITS,
CycleNet, iTransformer, SGFormer, Dragonnet, and Network Deconfounder. Copy
their strengths: direct execution, straightforward PyTorch, plain Python
functions, minimal interfaces, little infrastructure. Do not copy their
weaknesses: unclear data provenance, manual path editing, unreproducible
experimental settings.

## 1. Non-negotiable scientific scope

Preserve:

- RQ1: observational direct, 1-hop, and 2-hop crash effects, including matching,
  pretrends, placebos, balance, overlap, masking, and recovery-duration
  diagnostics.
- RQ2: fair graph versus non-graph comparisons, including required real,
  no-message-passing, and rewired topology ablations.
- RQ3: treatment-aware versus baseline counterfactual estimation and
  known-effect evaluation.
- All required Workstreams A–D, including frozen sensitivity and semi-synthetic
  variants that contribute to the final paper.
- Required forecasting baselines, graph models, and treatment-aware models.
- All figures, tables, diagnostics, and result-index entries actually required by
  the paper.

Do not preserve abandoned exploratory experiments, optional future studies,
unused visualization tools, or generalized model support.

Do not introduce weather, construction, Run B, 3-hop outcomes, or other deferred
scope.

## 2. Exact target organization

```
traffic-spillovers/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── run.py
├── data.py
├── graph.py
├── events.py
├── models.py
├── causal.py
├── synthetic.py
├── results.py
│
├── hpc/
│   ├── submit.sh
│   └── job.sh
│
├── test_research.py
│
├── data/
│   └── chicago-traffic-segments.json
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
└── outputs/               # Git-ignored
```

Eight Python implementation files, two HPC scripts, one main test file.

No `src/`, `scripts/`, `docs/`, `utils/`, `config/`, package hierarchies,
registries, generic experiment classes, or compatibility layers.

The exact names and count are a preferred design, not permission to create giant
modules. A second test file or a separate training module is acceptable only if
it genuinely improves comprehension. A separate file is justified only if it
contains a distinct, substantial responsibility and makes the code easier to
understand.

Do not mechanically merge existing files into enormous modules. Reimplement the
necessary operations directly.

## 3. Strict LOC budget by file

| File | Target LOC | Required responsibility |
| --- | --- | --- |
| `run.py` | 200–300 | CLI, fixed experiments, execution |
| `data.py` | 250–400 | Download, convert, load, align |
| `graph.py` | 350–550 | Road graph, spatial matching, hop rings, topology |
| `events.py` | 350–550 | Windows, masks, controls, splits, exposure |
| `models.py` | 240–380 | Required models and shared training |
| `causal.py` | 280–420 | Matching, DiD, placebos, inference |
| `synthetic.py` | 180–320 | Known effects, injection, truth |
| `results.py` | 180–300 | Metrics, tables, figures, result index |
| `hpc/submit.sh` | 130–210 | Dependency-managed Slurm matrix and resume |
| `hpc/job.sh` | 30–60 | Run exact source and record provenance |
| `test_research.py` | 350–550 | Essential correctness and end-to-end checks |
| `README.md` | 100–180 | Reproduction instructions |
| Small configuration files | 50–80 | `pyproject.toml`, `.gitignore` |

Any file that substantially exceeds its target requires a specific explanation.
**Never compress scientifically necessary logic to satisfy LOC accounting.**

## 4. Exact file responsibilities

### `data.py`

Own all necessary data acquisition, CSV/Parquet conversion, source validation,
timestamps, loading, and alignment.

There must be one supported historical-data acquisition path.

Preserve both Chicago speed archives, their coverage constraints, the crash
records, chunked data handling, and the exact data contract.

Delete obsolete alternative downloaders, interactive setup helpers, unnecessary
fetch wrappers, duplicate conversion interfaces, and repeated internal data
validation.

### `graph.py`

Own required OSM road preparation, road-segment identity, crash-to-road
matching, monitored graph construction, neighborhoods, and topology variants.

There must be one canonical construction of the frozen road graph and one
definition of graph hops.

Delete obsolete exploratory mapping, unused graph exports, redundant network
builders, optional visualization modes, and alternative graph-construction
workflows not needed to reconstruct the study.

Keep offline OSM extraction only if it is genuinely required for reproducing the
frozen graph.

### `events.py`

Own chronological splitting, embargoes, history/horizon windows, normalization,
masks, exposure labels, crash eligibility, and matched-control window
construction.

This is the scientific core. It must be transparent and correct.

Use one definition of contamination, one definition of exposure rings, and one
authoritative implementation of window eligibility.

Avoid elaborate dataset interfaces and unnecessary mutable state. A PyTorch
Dataset class is acceptable if it materially simplifies training.

### `models.py`

Implement only the actual required model families and a straightforward shared
training loop. Use PyTorch and PyG directly.

Keep baseline, graph, non-graph, temporal, and treatment-aware models necessary
for the frozen comparisons.

Remove generic model registries, trainer frameworks, configurable architecture
systems, redundant wrapper classes, and speculative extensibility.

Do not remove model capacity controls or shared training behavior needed for
fair comparisons.

### `causal.py`

Own the observational estimators and their essential diagnostics: matched
difference-in-differences, pretrends, placebos, balance, fixed cohorts, effect
duration, event-level weighting, and confidence intervals.

Keep estimator definitions explicit. Do not hide causal assumptions in generic
statistic helpers.

Do not turn observational results into unjustified causal claims.

### `synthetic.py`

Own known-effect injection, actual realized ground truth, exposure-consistent
interventions, synthetic controls, and required synthetic variants.

Remove generic simulation infrastructure and unused intervention families.

Preserve the exact timing, spatial propagation, floor clipping, null scenarios,
and known-effect scoring contract.

### `results.py`

Own required metrics, plots, paper tables, aggregation, and result indexing.

Delete figures that are not used by the paper or required diagnostics.

Never fabricate or silently substitute missing results. Incomplete outputs must
remain identifiable as incomplete.

### `run.py`

The single public Python entry point, using minimal standard-library `argparse`.

Expose the necessary actions: prepare, observational event study,
forecasting/model experiments, synthetic benchmarks, reporting, and smoke
testing.

All frozen paper experiment definitions must have one executable source of
truth.

The file must remain primarily an execution shell, not a second implementation
of all the algorithms — not causal estimators, PyTorch architectures, or
geospatial algorithms.

Keep only the operational arguments actually needed to select a stage, dataset,
output location, or required array variant. Do not expose dozens of speculative
scientific hyperparameters.

## 5. Completely rework the existing scripts

Audit and remove the old public script layer. The following should not survive
as separate entry points unless there is a demonstrated, necessary reason:

- `setup.sh`
- `fetch_data.sh`
- `fetch_speeds_history.sh`
- `fetch_window.py`
- `convert_to_parquet.py`
- `download_street_graph.py`
- `prepare_roads.py`
- `build_graph.py`
- `check_dataset.py`
- `experiment.py`
- `event_study.py`
- `benchmark.py`
- `aggregate.py`

Preserve necessary behavior inside the canonical modules and `run.py`.

Do not create shims, deprecated aliases, or compatibility wrappers.

There should be one obvious command for each required research operation.

**Delete behavior, not just files.** Merging `setup.sh`, `fetch_data.sh`,
`fetch_window.py`, and `fetch_speeds_history.sh` into a 500-line `data.py` that
preserves every old option accomplishes very little. Retain only the one
data-acquisition workflow the study needs.

## 6. Minimal HPC implementation

Aim for only `hpc/submit.sh` and `hpc/job.sh`.

`submit.sh` submits the required CPU/GPU stages, dependencies, arrays, resource
requests, and resumable runs. `job.sh` executes the prescribed stage and records
essential provenance.

Move scientific experiment settings out of Bash. Bash schedules experiments; it
does not independently define their methodology.

Eliminate separate `experiments.sh`, `prepare.sh`, `smoke.sh`, and `status.sh`
where their functions can be replaced more simply by `run.py`, the submitter,
and ordinary Slurm commands.

Preserve:

- One-command complete Run A submission
- Independent CPU/GPU resource allocation
- Correct stage dependencies
- GPU-array concurrency control
- Resuming failed/incomplete stages
- Job exit statuses and useful logs
- Exact code revision and effective settings
- Dataset and output provenance
- Detection of missing experiment outputs

Do not create a custom workflow engine. Do not repeatedly validate the same
already-validated internal data. Do not replace safe failure behavior with
silent fallbacks.

## 7. Strict Python code style

Conventional, minimal research Python.

**Prefer:** plain functions; direct NumPy, pandas, SciPy, PyTorch, and PyG
calls; ordinary loops and conditionals; small, readable model classes; direct
function arguments; local variables with obvious names; pure scientific
transformations where practical; minimal `argparse`; minimal `uv` and dependency
configuration.

**Avoid:** mandatory type annotations; type-checking frameworks; protocols and
abstract base classes; generic factories, adapters, or registries; dependency
injection; configuration classes; excessive dataclasses; utility files
containing unrelated helpers; one-line forwarding functions; broad exception
handling and speculative recovery; redundant input checking deep inside trusted
code; helpers used once that make direct code harder to follow; boilerplate
docstrings and explanatory comments; clever compression to reduce LOC.

Type hints are optional and should appear only when they clarify an important or
otherwise confusing interface. Do not enforce typing completeness.

Comments should explain genuinely non-obvious scientific decisions, not narrate
ordinary Python statements.

Use classes only when they are genuinely useful for PyTorch models, datasets, or
related state.

A little local duplication is preferable to unnecessary indirection. Do not
introduce abstractions merely to eliminate repeated lines.

## 8. Configuration must be minimal and authoritative

Keep `uv`, `pyproject.toml`, and `uv.lock`.

Do not add Hydra, OmegaConf, Click, Typer, or a custom configuration framework.

The fixed experiment matrix lives in one straightforward Python location. Avoid
duplicating scientific settings in Bash and argparse defaults.

Every experimental output must record the actual effective settings used.

Preserve all frozen Run A values, including the data window, 10-minute bins,
12-bin history, 6-bin horizon, chronological splits, 2-hop exposure radius,
20-epoch training budget, patience 4, matched-control settings, model
configurations, and required sensitivity variants.

**No scientific setting may change silently during simplification.**

## 9. Reduce tests radically, but preserve scientific guarantees

Aim for one `test_research.py` of approximately 350–550 lines. A second test
file is allowed if one file becomes difficult to navigate.

Delete obsolete tests, duplicated scenarios, excessive mocking, parser trivia,
trivial assertions, and tests for removed behavior.

Retain independent tests capable of detecting:

- Temporal leakage
- Incorrect split embargo
- Crash-contaminated training inputs or targets
- Incorrect 0/1/2-hop exposure
- Wrong road-node or edge identity
- Invalid matching or event weighting
- Incorrect DiD and placebo calculations
- Synthetic intervention leakage
- Synthetic ground-truth errors
- Graph-ablation unfairness
- Incorrect treatment-head selection
- Frozen configuration drift
- Broken end-to-end reproduction or Slurm dependencies

Do not optimize for test count or coverage percentage. A test earns its place by
protecting a meaningful scientific or operational guarantee.

## 10. Remove documentation and historical clutter

The README is the main reproducibility document. It should explain:

- What the study measures
- Required data and where it comes from
- Installation using `uv`
- How to prepare the dataset
- How to run the experiments
- How to submit and resume on Rivanna
- Where the paper tables and figures come from
- The fixed settings and relevant research assumptions

The manuscript contains the full methodology and scientific limitations.

Delete the separate documentation directory only after all unique scientific and
reproduction information has been preserved in the README or paper.

Do not retain an architecture document merely to explain the directory tree.

Remove the roadmap and obsolete manuscript versions from the final research tree
once they are no longer needed, preserving them in a separate archive or
historical tag.

Do not delete required course deliverables or unique source data.

## 11. Aggressive dependency cleanup

Audit every dependency. Remove dependencies whose only remaining callers are
deleted features. In particular, examine whether mapping, interactive
visualization, and obsolete graph-processing packages remain necessary.

Do not replace mature scientific-library operations with custom implementations
merely to reduce dependencies.

Keep dependencies that demonstrably simplify correct numerical, geospatial,
graph, or ML operations.

## 12. Do not interfere with active Rivanna experiments

The existing Run A submission may still be queued or executing.

- Work on a separate Git branch/worktree.
- Tag or otherwise preserve the existing baseline and its dependency lockfile.
- Do not edit the working checkout used by active Slurm jobs.
- Do not modify existing inputs, processed datasets, output manifests, or
  completed results.
- Do not deploy the rewrite to Rivanna until scientific equivalence and job
  execution have been verified.

## 13. Require scientific equivalence

Do not accept the rewrite solely because its own tests pass. Compare it against
the original frozen implementation on identical fixtures and representative real
data.

Require exact agreement, where deterministic, for:

- Road identifiers and graph edges
- Crash matching and event eligibility
- Chronological split membership
- Exposure rings and labels
- Input/target contamination masks
- Usable target cells
- Matched controls under the same seed
- Synthetic treatment placement and realized truth
- Required experiment configurations
- Output schemas and result availability

Compare numerical estimators and metrics within justified tolerances.

For stochastic training, document reproducibility limits and compare model
configurations, objectives, evaluation cohorts, and appropriate numerical
results rather than assuming bitwise-identical GPU training.

Any intentional scientific difference must be explicitly reported and approved
before treating the new results as equivalent.

Equivalence is not required at the expense of clarity: the rewrite may change
implementation details, imports, and intermediate representations freely. It
must preserve the mathematical estimands, selected cohorts, frozen comparisons,
and reproducible results. When a change affects those, it is a research change,
not merely a refactor.

## 14. Deliverables and completion criteria

Provide:

1. The final repository tree.
2. Exact physical LOC per file.
3. Original versus new maintained LOC.
4. Every file and public command removed.
5. The functionality preserved from each removed script.
6. Dependencies removed.
7. Actual implementation simplifications, not just moves.
8. Scientific equivalence results.
9. Exact experiment settings and stage-dependency comparison.
10. The simplest complete reproduction instructions.

Before finishing, inspect the entire diff again. For every remaining file,
function, option, check, class, and abstraction, ask:

**Is this essential to the study, its correct implementation, or its
reproducibility?** If not, delete it.

**Do not invent complexity to satisfy conventional engineering practices. Do not
delete necessary science to satisfy LOC targets.**

The project must end up with fewer concepts, fewer execution paths, less code,
less documentation duplication, and a much smaller surface area for bugs.

The ideal result is approximately 3,000–4,000 maintained lines, but the actual
success criterion is the simplest correct, independently reproducible
implementation. Stop when there is no remaining simplification that improves
human comprehension without compromising the research.

## 15. Final standard

| Question | Passing standard |
| --- | --- |
| Can I understand the entire project quickly? | Read `README.md`, then `run.py`, then only the scientific module relevant to a question |
| Can I reproduce the entire paper? | Install with `uv`, obtain the frozen inputs, run one local command per experiment or one HPC submission command |
| Is every remaining piece necessary? | Every file, class, abstraction, dependency, and test has a demonstrated role in the actual study |

**Be ruthless about incidental software complexity, but exacting about the
scientific experiment.**
