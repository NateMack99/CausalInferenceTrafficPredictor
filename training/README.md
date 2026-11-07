# Window graphs

```python
from training.build_graph import build_graph
from training.plot_graph import plot_graph

G = build_graph(
    crash_window=("2025-01-01 12:00", "2025-01-01 13:00"),
    speed_window=("2025-01-01 12:00", "2025-01-01 13:00"),
    only_80_percent=True,  # False uses all speed-observed roads
)
plot_graph(G)
```

Windows are `[start, end)` in portal wall-clock time. GraphML, crash assignments,
and the PNG go in `training/output/` with timestamps in their filenames.

Fetching prepares the speed-road reference in `data/`; rebuild it with
`python data/prepare_roads.py`.

`only_80_percent=True` keeps edges whose traffic segment has a valid speed in
at least 80% of the reference period's expected 10-minute intervals. Coverage is
`distinct intervals with valid speeds / (calendar days × 144)`; negative/missing
speeds do not count, zero does, and duplicates within an interval count once.
For all of 2025, that means at least 42,048 of 52,560 nominal intervals.
This uses the full reference period, not your requested window, so an included
road can still have missing speeds in that window. `False` includes every road
with any reported valid speed. The denominator is nominal and does not adjust
for daylight-saving changes.

Congestion is `1 - mean speed / historical maximum`, a visualization proxy.
Purple marks crashes; gray means missing speed. Crash direction remains unresolved.
Coverage and maxima use the full reference period; use training-only data when
preparing references for prediction experiments.
