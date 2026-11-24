#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a GPU allocation}"
smoke_out=$1
mkdir "$smoke_out"
uv run --no-sync python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable; refusing a CPU fallback"
x = torch.ones((64, 64), device="cuda", requires_grad=True)
(x @ x).sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print("CUDA verified:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
uv run --no-sync python -m traffic_gml.mock --out "$smoke_out/data" \
    --rows 3 --cols 3 --days 90 --crashes 60 --seed 42
uv run --no-sync python scripts/midpoint.py --data "$smoke_out/data" \
    --data-only --static-features length_m degree --controls 5 --out "$smoke_out/audit"
uv run --no-sync python scripts/midpoint.py --data "$smoke_out/data" \
    --static-features length_m degree --epochs 1 --patience 1 --controls 5 \
    --device cuda --out "$smoke_out/main"
uv run --no-sync python scripts/event_study.py --data "$smoke_out/data" \
    --hops 2 --controls 5 --out "$smoke_out/event-study"
uv run --no-sync python scripts/benchmark.py --data "$smoke_out/data" \
    --models mlp gcn tgru stgcn --static-features length_m degree \
    --events 8 --controls 5 --epochs 1 --patience 1 --device cuda \
    --out "$smoke_out/synthetic"
