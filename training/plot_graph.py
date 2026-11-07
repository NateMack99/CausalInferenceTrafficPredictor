"""Plot a window graph: green=fast, red=slow, purple=crash, gray=missing."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
import numpy as np
import osmnx as ox


def plot_graph(graph, path=None, show=False):
    """Save an OSMnx congestion map beside the GraphML; return (figure, axes)."""
    colors, widths = [], []
    for _, _, edge in graph.edges(data=True):
        value = edge["congestion"]
        colors.append(to_rgba("#9c27b0" if edge["crash_count"] else
                              plt.colormaps["RdYlGn_r"](value) if np.isfinite(value) else "#bdbdbd"))
        widths.append(3 if edge["crash_count"] else 1)
    fig, ax = ox.plot_graph(graph, node_size=0, edge_color=colors, edge_linewidth=widths,
                            bgcolor="white", figsize=(12, 10), show=False, close=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="RdYlGn_r"), ax=ax,
                 shrink=.6, label="Congestion proxy: 1 − mean speed / observed maximum")
    ax.legend(handles=[Line2D([0], [0], color="#9c27b0", lw=3, label="Crash in selected window"),
                       Line2D([0], [0], color="#bdbdbd", label="No speed in selected window")], loc="lower left")
    ax.set_title(f"Speeds: {graph.graph['speed_start']} to {graph.graph['speed_end']}\n"
                 f"Crashes: {graph.graph['crash_start']} to {graph.graph['crash_end']}")
    path = Path(path) if path else Path(graph.graph["output_path"]).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    from build_graph import build_graph

    graph = build_graph(
        crash_window=("2025-01-02 11:30", "2025-01-03 12:00"),
        speed_window=("2025-01-02 12:00", "2025-01-03 13:30"),
    )
    plot_graph(graph, show=True)
