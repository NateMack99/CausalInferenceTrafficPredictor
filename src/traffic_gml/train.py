"""Optimisation, checkpointing, and the experiment entry point.

Trains the no-accident forecaster on crash-free windows with a masked speed
loss, so a missing or crash-contaminated target contributes nothing to the
gradient instead of pulling the prediction towards zero. Early stopping on
validation MAE; the best epoch's weights are restored before the test pass and
written to the run directory.

"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .evaluate import LOSSES, evaluate, masked_loss
from .events import LOCAL_TZ, Windows, build_windows, load_tables, local_time
from .models import BASELINES, CONVS, build_model, predict


def data_version(tables: dict, path: Path) -> dict:
    times = tables["speeds"]["time"]
    return {
        "path": str(path),
        "rows": {name: int(len(table)) for name, table in tables.items()},
        "span": [str(times.min()), str(times.max())],
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- legacy global seed, on purpose
    torch.manual_seed(seed)


def fit(
    model,
    sets: dict[str, Windows],
    edge_index: Tensor,
    loss_name: str = "mae",
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """Train a forecaster, or fit a baseline, on the training split."""
    if not isinstance(model, torch.nn.Module):
        model.fit(sets["train"])
        return []

    power = LOSSES[loss_name]
    # Losses are computed on normalised speeds; rescale to mph (mph^2 for mse).
    scale = sets["train"].std ** (2 if loss_name == "mse" else 1)
    unit = "mph^2" if loss_name == "mse" else "mph"
    loader = DataLoader(sets["train"], batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, state, history = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            out = predict(model, batch, edge_index)
            loss = masked_loss(out, batch["y"], batch["y_mask"], power)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        val = evaluate(model, sets["val"], edge_index, batch_size)
        train_loss = float(np.mean(losses)) * scale
        history.append(
            {"epoch": epoch, f"train_{loss_name}": round(train_loss, 4)}
            | {f"val_{k}": v for k, v in val.items()}
        )
        if verbose:
            print(
                f"epoch {epoch:3d}  train {loss_name.upper()} {train_loss:7.3f} {unit}"
                f"  val MAE {val['mae']:6.3f} mph"
            )
        if val["mae"] < best - 1e-5:
            best, best_epoch = val["mae"], epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"early stop at epoch {epoch}; best was epoch {best_epoch}")
            break
    if state is not None:
        model.load_state_dict(state)
    return history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one no-accident speed forecaster.")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default="gcn", choices=[*BASELINES, *CONVS])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("outputs"))

    windows = parser.add_argument_group("windows")
    windows.add_argument("--history", type=int, default=12, help="input bins of 10 minutes")
    windows.add_argument("--horizon", type=int, default=6, help="predicted bins ahead")
    windows.add_argument("--exposure-hops", type=int, default=1, help="crash masking radius")
    windows.add_argument("--stride", type=int, default=1, help="bins between window origins")
    windows.add_argument("--static-features", nargs="*", default=[])

    model = parser.add_argument_group("model")
    model.add_argument("--hidden", type=int, default=64)
    model.add_argument("--layers", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.1)

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--loss", default="mae", choices=sorted(LOSSES))
    optimisation.add_argument("--epochs", type=int, default=30)
    optimisation.add_argument("--lr", type=float, default=1e-3)
    optimisation.add_argument("--weight-decay", type=float, default=1e-4)
    optimisation.add_argument("--batch-size", type=int, default=64)
    optimisation.add_argument("--patience", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)

    tables = load_tables(args.data)
    sets, edge_index = build_windows(
        tables,
        history=args.history,
        horizon=args.horizon,
        exposure_hops=args.exposure_hops,
        stride=args.stride,
        static_features=args.static_features,
    )
    seen = torch.isfinite(sets["train"].values)
    local = local_time(sets["train"].times)
    print(
        f"{sets['train'].num_roads} roads ({int((~seen.any(dim=0)).sum())} with no speed data), "
        f"{edge_index.size(1)} edges, coverage {seen.float().mean():.1%}, bins "
        f"{local[0]:%Y-%m-%d %H:%M} to {local[-1]:%Y-%m-%d %H:%M} {LOCAL_TZ}"
    )
    print("  ".join(f"{name}={len(data):,} windows" for name, data in sets.items()))

    if args.model in CONVS and args.model != "mlp" and args.exposure_hops < args.layers:
        print(
            f"warning: {args.layers} layers read {args.layers} hops but crash masking covers "
            f"{args.exposure_hops}; raise --exposure-hops or expect contaminated neighbour inputs"
        )
    model = build_model(args.model, sets["train"], args.hidden, args.layers, args.dropout)
    history = fit(
        model,
        sets,
        edge_index,
        loss_name=args.loss,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    metrics = {
        name: evaluate(model, data, edge_index, args.batch_size) for name, data in sets.items()
    }
    for name, values in metrics.items():
        print(f"{name:>5s}  MAE {values['mae']:6.3f} mph  RMSE {values['rmse']:6.3f} mph")

    run = args.out / f"{args.model}-seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    record = {
        "data": data_version(tables, args.data),
        "graph": {"edges": int(edge_index.size(1)), "exposure_hops": args.exposure_hops},
        "normalizer": {"mean": sets["train"].mean, "std": sets["train"].std},
        "splits": metrics,
        "history": history,
    }
    (run / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), run / "model.pt")
    print(f"wrote {run}")
    return metrics


if __name__ == "__main__":
    main()
