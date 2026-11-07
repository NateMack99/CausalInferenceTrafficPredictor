"""Train one no-accident speed forecaster and report MAE and RMSE.

    uv run python scripts/run_experiment.py --data data/mock --model gcn --seed 42

See `--help` for the window, model, and optimisation flags.
"""

from __future__ import annotations

from traffic_gml.train import main

if __name__ == "__main__":
    main()
