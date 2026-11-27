"""Every figure in the paper, from the verified Run A exports in raw_results.

`raw_results` is the bundle `scripts/aggregate.py` wrote, with the sha256 of
each file in `result_index.csv`. The frozen road graph is read from the repo
only to draw Figure 1a; every number quoted in any figure comes from the
bundle.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

here = Path(__file__).resolve().parent
RAW = here.parents[1] / "raw_results"
REPO = here.parents[2]
NAVY, TEAL, BLUE, AMBER, RED, GRAY, GRID = (
    "#24374A",
    "#176F68",
    "#2C668E",
    "#B87825",
    "#B34A4A",
    "#73808C",
    "#DDE3E7",
)
# Muted on purpose: these fail a screen-chroma check and pass the ones that
# matter in print, colour-vision separation (worst adjacent pair dE 13.2
# protan) and contrast against white.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(here / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.055, dpi=300)
    plt.close(fig)


# ---- real frozen network, for the ring panel only; counts come from raw_results
seg = pd.read_parquet(REPO / "data/processed/segments.parquet")
geo = pd.DataFrame(json.loads((REPO / "data/chicago-traffic-segments.json").read_text()))
geo["segment_id"] = geo.segmentid.astype(int)
geo = geo.rename(columns={"start_lon": "x0", "_lif_lat": "y0", "_lit_lon": "x1", "_lit_lat": "y1"})
for c in ("x0", "y0", "x1", "y1"):
    geo[c] = geo[c].astype(float)
seg = seg.merge(geo[["segment_id", "x0", "y0", "x1", "y1"]], on="segment_id", how="left")
xy = {r.segment_id: ((r.x0, r.y0), (r.x1, r.y1)) for r in seg.itertuples()}
edges = pd.read_parquet(REPO / "data/processed/edges.parquet")
adj: dict[int, set[int]] = {}
for s, d in edges.itertuples(index=False):
    adj.setdefault(s, set()).add(d)
    adj.setdefault(d, set()).add(s)


def rings(root: int, hops: int = 2) -> dict[int, list[int]]:
    seen, out, frontier = {root}, {0: [root]}, [root]
    for hop in range(1, hops + 1):
        nxt = [n for f in frontier for n in adj.get(f, ()) if n not in seen]
        nxt = sorted(set(nxt))
        seen.update(nxt)
        out[hop] = nxt
        frontier = nxt
    return out


# A dense, well-connected crash segment: the ring structure is legible and
# typical rather than cherry-picked for size.
cands = [
    (s, len(rings(s, 2)[1]), len(rings(s, 2)[2]))
    for s in sorted(adj)
    if s in xy and 5 <= len(adj.get(s, ())) <= 9
]
root = max(cands, key=lambda c: c[2] + c[1])[0]
ring = rings(root, 2)

fig = plt.figure(figsize=(7.1, 4.55))
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.0, 0.95],
    width_ratios=[0.95, 1.05],
    hspace=0.44,
    wspace=0.26,
    left=0.055,
    right=0.985,
    top=0.905,
    bottom=0.105,
)

# ---------------- A: exposure rings on the frozen network ----------------
ax = fig.add_subplot(gs[0, 0])
ax.set_title("a   Exposure is defined on the road graph", loc="left", pad=6)
for (x0, y0), (x1, y1) in xy.values():
    ax.plot([x0, x1], [y0, y1], color="#C9D2DA", lw=0.6, solid_capstyle="round", zorder=1)
for hop, color, lw in ((2, BLUE, 1.7), (1, AMBER, 2.1), (0, RED, 3.4)):
    for sid in ring[hop]:
        if sid not in xy:
            continue
        (x0, y0), (x1, y1) = xy[sid]
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="round", zorder=3 + hop)
pts = np.array([p for sid in ring[0] + ring[1] + ring[2] if sid in xy for p in xy[sid]])
pad = 0.013
ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
ax.set_ylim(pts[:, 1].min() - pad * 0.72, pts[:, 1].max() + pad * 0.72)
ax.set_xticks([])
ax.set_yticks([])
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)
for i, (color, text) in enumerate(((RED, "crash segment"), (AMBER, "1 hop"), (BLUE, "2 hops"))):
    ax.plot(
        [0.035, 0.085],
        [0.085 - i * 0.072] * 2,
        color=color,
        lw=2.2,
        transform=ax.transAxes,
        solid_capstyle="round",
        clip_on=False,
    )
    ax.text(
        0.098,
        0.085 - i * 0.072,
        text,
        transform=ax.transAxes,
        fontsize=7.0,
        color=NAVY,
        va="center",
    )
# whole-city locator
loc = ax.inset_axes([0.665, 0.565, 0.335, 0.435])
for (x0, y0), (x1, y1) in xy.values():
    loc.plot([x0, x1], [y0, y1], color="#CAD3DA", lw=0.4, zorder=1)
loc.add_patch(
    Rectangle(
        (ax.get_xlim()[0], ax.get_ylim()[0]),
        np.ptp(ax.get_xlim()),
        np.ptp(ax.get_ylim()),
        fill=False,
        edgecolor=RED,
        lw=1.1,
        zorder=4,
    )
)
loc.set_xticks([])
loc.set_yticks([])
for spine in loc.spines.values():
    spine.set_color(GRID)
loc.set_facecolor("white")
loc.set_zorder(10)

# ---------------- B: the funnel, as a ribbon ----------------
ax = fig.add_subplot(gs[0, 1])
ax.set_title("b   Few crashes can be used at all", loc="left", pad=6)
funnel = pd.read_csv(RAW / "audit__coverage_funnel.csv")
n = funnel.crashes.to_numpy(dtype=float)
x = np.arange(len(n))
half = n / n[0] / 2
ax.fill_between(x, -half, half, color=BLUE, alpha=0.17, lw=0, zorder=2)
ax.plot(x, half, color=BLUE, lw=1.1, zorder=3)
ax.plot(x, -half, color=BLUE, lw=1.1, zorder=3)
shown = {
    0: "reported\ncrashes",
    2: "on a monitored\nsegment",
    4: "analysis\nsample",
    5: "fully\nobserved",
}
for i, text in shown.items():
    ax.plot([i, i], [-half[i], half[i]], color=BLUE, lw=1.0, zorder=4)
    ax.annotate(
        f"{int(n[i]):,}\n{n[i] / n[0] * 100:.1f}%",
        xy=(i, half[i]),
        xytext=(i, 0.60),
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=NAVY,
        zorder=5,
        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7, shrinkA=1, shrinkB=2),
    )
    ax.text(i, -0.60 if i != 5 else -0.80, text, ha="center", va="top", fontsize=6.7, color=GRAY)
ax.set_xlim(-0.45, len(n) - 0.55)
ax.set_ylim(-1.18, 1.02)
ax.axis("off")

# ---------------- C: signal against noise, one axis ----------------
ax = fig.add_subplot(gs[1, :])
ax.set_title(
    "c   The effect to be recovered is smaller than the error of the forecasts that recover it",
    loc="left",
    pad=8,
)
recovery = pd.read_csv(RAW / "benchmark-seed42__effect_recovery.csv")
hop = recovery[
    (recovery.model == "gcn") & (recovery.estimator == "corrected") & (recovery.group == "hop")
].set_index("value")
placebo = pd.read_csv(RAW / "event__effect_against_placebo.csv")
observed = pd.read_csv(RAW / "event__effect_by_hop.csv")
forecast = pd.read_csv(RAW / "main-seed42__results.csv")

marks = [
    ("2 hops", abs(hop.loc["2", "true_mph"]), TEAL, "o"),
    ("1 hop", abs(hop.loc["1", "true_mph"]), TEAL, "o"),
    ("crash segment", abs(hop.loc["0", "true_mph"]), TEAL, "o"),
    ("best of seven models", forecast.mae.min(), RED, "D"),
    (
        "pre-crash placebo, 1 hop",
        abs(placebo.query("hop == 1 and minutes == 30").placebo.iloc[0]),
        GRAY,
        "s",
    ),
    (
        "matched contrast, 1 hop",
        abs(observed.query("hop == 1 and minutes == 30").effect.iloc[0]),
        BLUE,
        "s",
    ),
]
for y, (_, value, color, marker) in enumerate(marks):
    ax.plot([0.034, value], [y, y], color=GRID, lw=0.8, zorder=1)
    ax.plot([value], [y], marker=marker, ms=6.2, color=color, mec="white", mew=0.9, zorder=4)
    ax.annotate(
        f"{value:.2f}",
        xy=(value, y),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=7.0,
        color=NAVY,
    )
ax.set_yticks(range(len(marks)), [m[0] for m in marks])
ax.tick_params(axis="y", length=0, pad=4)
for y, text, color in (
    (2.22, "Injected effect, known truth", TEAL),
    (3.22, "Error of the forecasts that estimate it", RED),
    (5.46, "Real crashes, 30 minutes after", BLUE),
):
    ax.text(0.036, y, text, ha="left", va="center", fontsize=7.0, color=color)
ax.axhline(2.5, color=GRID, lw=0.7)
ax.axhline(3.5, color=GRID, lw=0.7)
ax.set_xscale("log")
ax.set_xlim(0.034, 14)
ax.set_ylim(-0.6, len(marks) - 0.08)
ax.set_xlabel("mph, absolute (log scale)", labelpad=3)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)

save(fig, "figure1_study_design")


# ============================ Figure 2: topology ============================
# Paired within-seed penalties, then the test that decides the paper's
# question: is the effect-error gain anything other than the forecast gain?
g = pd.read_csv(RAW / "graph_ablation.csv")
real = g[g.topology == "real"].set_index(["model", "seed"])
paired = g[g.topology != "real"].copy()
for column in ("forecast_mae", "effect_mae"):
    paired["d_" + column] = [
        row[column] - real.loc[(row.model, row.seed), column] for _, row in paired.iterrows()
    ]

fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.30), width_ratios=[1, 1, 1.02])
rows = [
    ("gcn", "none", "GCN   self-only", 3),
    ("gcn", "permuted", "GCN   relabelled", 2),
    ("stgcn", "none", "GRU-GCN   self-only", 0.8),
    ("stgcn", "permuted", "GRU-GCN   relabelled", -0.2),
]
for ax, metric, title in zip(
    axes[:2],
    ["d_forecast_mae", "d_effect_mae"],
    ["a   Forecasting", "b   Corrected effect error"],
    strict=False,
):
    ax.axvline(0, color="#75828B", lw=1.0, zorder=1)
    for model, topology, _, y in rows:
        d = paired[(paired.model == model) & (paired.topology == topology)][metric].to_numpy()
        color = BLUE if model == "gcn" else AMBER
        if len(d) > 1:
            ax.scatter(
                d,
                y + np.linspace(-0.12, 0.12, len(d)),
                s=18,
                color=color,
                alpha=0.44,
                edgecolor="white",
                lw=0.45,
                zorder=4,
            )
            ax.plot([d.min(), d.max()], [y, y], color=color, lw=1.8, zorder=3)
            ax.scatter(
                [d.mean()],
                [y],
                s=70,
                color=color,
                marker="D",
                edgecolor="white",
                linewidth=0.95,
                zorder=5,
            )
        else:
            ax.scatter(
                d, [y], s=75, color=color, marker="s", edgecolor="white", linewidth=0.9, zorder=5
            )
    ax.set_xlim(0, 0.055)
    ax.set_xticks([0, 0.02, 0.04], ["0", "0.02", "0.04"])
    ax.set_ylim(-0.65, 3.53)
    ax.set_yticks([r[3] for r in rows], [r[2] for r in rows])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_title(title, loc="left", pad=9)
    ax.grid(axis="x", color=GRID, lw=0.65, zorder=0)
    ax.set_xlabel("added MAE vs. real graph (mph)", labelpad=6)
axes[1].set_yticklabels([])

ax = axes[2]
ax.set_title("c   The two gains are the same gain", loc="left", pad=9)
limit = 0.058
ax.plot([0, limit], [0, limit], color=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.text(
    0.0525,
    0.0545,
    "equal gain",
    fontsize=6.8,
    color=GRAY,
    ha="right",
    va="top",
    rotation=38,
    rotation_mode="anchor",
)
for model, marker in (("gcn", "o"), ("stgcn", "s")):
    sub = paired[paired.model == model]
    ax.scatter(
        sub.d_forecast_mae,
        sub.d_effect_mae,
        s=26,
        marker=marker,
        color=BLUE if model == "gcn" else AMBER,
        edgecolor="white",
        lw=0.5,
        zorder=4,
        label="GCN" if model == "gcn" else "GRU-GCN",
    )
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_xticks([0, 0.02, 0.04], ["0", "0.02", "0.04"])
ax.set_yticks([0, 0.02, 0.04], ["0", "0.02", "0.04"])
ax.set_xlabel("added forecast MAE (mph)", labelpad=6)
ax.set_ylabel("added effect MAE (mph)", labelpad=3)
ax.grid(color=GRID, lw=0.65, zorder=0)
ax.legend(fontsize=6.8, frameon=False, loc="upper left", handletextpad=0.3)
fig.subplots_adjust(left=0.155, right=0.985, bottom=0.235, top=0.86, wspace=0.34)
save(fig, "figure2_topology_results")


# ========================= Figure 3: observational =========================
main = pd.read_csv(RAW / "event__effect_by_hop.csv")
fixed = pd.read_csv(RAW / "event__effect_fixed_cohort.csv")
placebo = pd.read_csv(RAW / "event__effect_against_placebo.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.42), sharey=True)
for ax, d, title in zip(
    axes[:2],
    [main, fixed],
    ["a   Available observed cells", "b   Common cohort (9,275 events)"],
    strict=False,
):
    ax.axhline(0, color=GRAY, ls=":", lw=1.05)
    for k, c in ((0, BLUE), (1, AMBER), (2, TEAL)):
        s = d[d.hop == k].sort_values("minutes")
        ax.fill_between(s.minutes, s.low, s.high, alpha=0.12, color=c, lw=0)
        ax.plot(s.minutes, s.effect, "-o", color=c, ms=3.0, lw=1.5)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Minutes after crash")
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    ax.grid(axis="y", color=GRID, lw=0.65)

# The check the neighbour claim turns on: how much of each estimate is already
# there before the crash.
ax = axes[2]
ax.set_title("c   Estimate vs. its pre-crash placebo", loc="left")
ax.axhline(0, color=GRAY, ls=":", lw=1.05)
for k, c in ((0, BLUE), (1, AMBER), (2, TEAL)):
    d = placebo[placebo.hop == k].sort_values("minutes")
    ax.plot(
        d.minutes,
        d.placebo,
        ls=(0, (3, 2)),
        color=c,
        lw=1.3,
        marker="o",
        ms=3.4,
        mfc="white",
        mew=1.1,
        zorder=3,
    )
    ax.plot(d.minutes, d.effect, "-o", color=c, ms=3.0, lw=1.5, zorder=4)
    for _, r in d.iterrows():
        ax.plot([r.minutes, r.minutes], [r.effect, r.placebo], color=c, lw=0.7, alpha=0.5, zorder=2)
ax.set_xlabel("Minutes after crash")
ax.set_xticks([10, 20, 30])
ax.grid(axis="y", color=GRID, lw=0.65)
ax.legend(
    handles=[
        Line2D([0], [0], color=GRAY, lw=1.5, marker="o", ms=3.2, label="after the crash"),
        Line2D(
            [0],
            [0],
            color=GRAY,
            lw=1.3,
            ls=(0, (3, 2)),
            marker="o",
            ms=3.4,
            mfc="white",
            mew=1.1,
            label="before it (placebo)",
        ),
    ],
    fontsize=6.6,
    frameon=False,
    loc="lower left",
    handlelength=1.8,
)
axes[0].set_ylabel("Matched speed contrast (mph)")
axes[0].set_ylim(-0.83, 0.13)
fig.legend(
    handles=[
        Line2D([0], [0], color=c, lw=1.7, marker="o", ms=3.5, label=f"{k}-hop")
        for k, c in ((0, BLUE), (1, AMBER), (2, TEAL))
    ],
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.54, 1.03),
    fontsize=7.5,
)
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.235, top=0.80, wspace=0.16)
save(fig, "figure3_observational")


# ==================== Figure 4: recovery against the null ====================
# Every injected effect is non-positive, so the MAE of estimating no effect at
# all is |true_mph| under the same weighting. Without it a reader cannot tell a
# working estimator from one that has shrunk its predictions to zero.
b = pd.read_csv(RAW / "benchmark-seed42__effect_recovery.csv")
hops = b[b.group == "hop"].copy()
hops["value"] = hops.value.astype(int)
series = [
    ("gcn", "corrected", "GCN matched residual", BLUE, "^"),
    ("treatment_mlp", "exposure-head", "Treatment MLP", AMBER, "s"),
    ("treatment_gcn", "exposure-head", "Treatment GCN", TEAL, "o"),
]
fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.62), width_ratios=[1.02, 1, 1])

ax = axes[0]
ax.set_title("a   Error against the null", loc="left", pad=8)
x = np.arange(3)
null = [
    abs(
        hops[
            (hops.model == "gcn") & (hops.estimator == "corrected") & (hops.value == k)
        ].true_mph.iloc[0]
    )
    for k in x
]
ax.bar(x, null, width=0.62, color="#E7ECF0", edgecolor=GRAY, lw=0.8, zorder=2)
for model, estimator, _, color, marker in series:
    s = hops[(hops.model == model) & (hops.estimator == estimator)].set_index("value")
    ax.plot(
        x,
        [s.loc[k, "effect_mae"] for k in x],
        color=color,
        lw=1.5,
        marker=marker,
        ms=4.6,
        mec="white",
        mew=0.7,
        zorder=4,
    )
ax.set_xticks(x, ["0", "1", "2"])
ax.set_xlabel("Hops from the crash road")
ax.set_ylabel("Effect MAE (mph), lower is better")
ax.set_ylim(0, 4.3)
ax.grid(axis="y", color=GRID, lw=0.65, zorder=0)

ax = axes[1]
ax.set_title("b   Magnitude recovered", loc="left", pad=8)
truth = hops[(hops.model == "treatment_gcn") & (hops.estimator == "exposure-head")].set_index(
    "value"
)
ax.plot(x, [truth.loc[k, "true_mph"] for k in x], "-D", color=NAVY, lw=1.9, ms=4.2, zorder=5)
for model, estimator, _, color, marker in series:
    s = hops[(hops.model == model) & (hops.estimator == estimator)].set_index("value")
    ax.plot(
        x,
        [s.loc[k, "estimated_mph"] for k in x],
        color=color,
        lw=1.45,
        marker=marker,
        ms=4.0,
        mec="white",
        mew=0.6,
        zorder=4,
    )
ax.axhline(0, color=GRAY, lw=0.8, ls=":")
ax.set_xticks(x, ["0", "1", "2"])
ax.set_xlabel("Hops from the crash road")
ax.set_ylabel("Mean effect (mph)")
ax.set_ylim(-4.6, 0.35)
ax.grid(axis="y", color=GRID, lw=0.65)
handles = [
    Rectangle(
        (0, 0), 1, 1, facecolor="#E7ECF0", edgecolor=GRAY, lw=0.8, label="Estimate no effect"
    ),
    Line2D([0], [0], color=NAVY, lw=1.9, marker="D", ms=4.2, label="Realized effect"),
]
handles += [
    Line2D([0], [0], color=c, lw=1.45, marker=m, ms=4.0, label=n) for _, _, n, c, m in series
]
fig.legend(
    handles=handles,
    frameon=False,
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.54, 1.045),
    fontsize=7.0,
    columnspacing=1.3,
    handletextpad=0.45,
)

# B1 injects an exact zero on both outer rings, so any neighbour estimate there
# is invented. This is the panel the observational result has to be read beside.
ax = axes[2]
ax.set_title("c   Invented spillover, B1", loc="left", pad=8)
scen = pd.read_csv(RAW / "scenarios.csv")
b1 = scen[(scen.scenario == "B1 direct-only") & (scen.group == "hop")].copy()
b1["value"] = b1.value.astype(int)
ax.axhline(0, color=NAVY, lw=1.1, zorder=3)
for model, estimator, _, color, marker in series:
    g = b1[(b1.model == model) & (b1.estimator == estimator) & (b1.value > 0)]
    stat = g.groupby("value").estimated_mph.agg(["mean", "std"])
    ax.errorbar(
        stat.index,
        stat["mean"],
        yerr=stat["std"],
        color=color,
        lw=1.45,
        marker=marker,
        ms=4.0,
        mec="white",
        mew=0.6,
        capsize=2.5,
        zorder=4,
    )
seen = pd.read_csv(RAW / "event__effect_by_hop.csv")
real = seen.query("hop in (1, 2) and minutes == 30").set_index("hop").effect
ax.plot(
    real.index, real.to_numpy(), color=RED, lw=1.4, ls=(0, (4, 2)), marker="*", ms=8.0, zorder=5
)
ax.annotate(
    "real crashes",
    xy=(1, real.loc[1]),
    xytext=(7, 10),
    textcoords="offset points",
    fontsize=6.9,
    color=RED,
    ha="left",
)
ax.set_xticks([1, 2])
ax.set_xlim(0.72, 2.28)
ax.set_xlabel("Hops from the crash road")
ax.set_ylabel("Mean estimated effect (mph)")
ax.grid(axis="y", color=GRID, lw=0.65)
fig.subplots_adjust(left=0.062, right=0.99, bottom=0.19, top=0.845, wspace=0.36)
save(fig, "figure4_recovery")

print("wrote figures 1-4")
