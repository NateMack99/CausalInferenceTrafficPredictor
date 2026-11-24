#!/bin/bash
# Runs every stage of the matrix once on mock data. SMOKE_DEVICE=cpu drops
# the GPU check so the sequence can be rehearsed off the cluster.
set -euo pipefail
smoke_out=$1
device=${SMOKE_DEVICE:-cuda}
mkdir -p "$smoke_out"
if [[ "$device" == cuda ]]; then
: "${SLURM_JOB_ID:?Run inside a GPU allocation}"
uv run --no-sync python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable; refusing a CPU fallback"
x = torch.ones((64, 64), device="cuda", requires_grad=True)
(x @ x).sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print("CUDA verified:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
fi
uv run --no-sync python -m traffic_gml.mock --out "$smoke_out/data" \
    --rows 3 --cols 3 --days 90 --crashes 60 --seed 42
uv run --no-sync python scripts/experiment.py --data "$smoke_out/data" \
    --data-only --static-features length_m degree --controls 5 --out "$smoke_out/audit"
uv run --no-sync python scripts/experiment.py --data "$smoke_out/data" \
    --static-features length_m degree --epochs 1 --patience 1 --controls 5 \
    --device "$device" --out "$smoke_out/main-seed42"
uv run --no-sync python scripts/event_study.py --data "$smoke_out/data" \
    --hops 2 --controls 5 --out "$smoke_out/event"
uv run --no-sync python scripts/benchmark.py --data "$smoke_out/data" \
    --models mlp gcn tgru stgcn --static-features length_m degree \
    --events 8 --controls 5 --epochs 1 --patience 1 --device "$device" \
    --out "$smoke_out/benchmark-seed42"
# The stages the full matrix adds: an ablation arm, both treatment arms, the
# target-road matching rule, and the aggregation that builds the paper tables.
uv run --no-sync python scripts/benchmark.py --data "$smoke_out/data" \
    --models gcn --static-features length_m --events 8 --controls 5 \
    --epochs 1 --patience 1 --device "$device" --graph-ablation permuted \
    --out "$smoke_out/graph-ablation/gcn-permuted-seed42"
for model in treatment_mlp treatment_gcn; do
    uv run --no-sync python -m traffic_gml.train --data "$smoke_out/data" \
        --static-features length_m degree --neighbour-hops 1 --model "$model" \
        --epochs 1 --patience 1 --device "$device" --out "$smoke_out/treatment"
done
uv run --no-sync python scripts/event_study.py --data "$smoke_out/data" \
    --hops 2 --controls 5 --match-target --out "$smoke_out/matching/matched-target"
uv run --no-sync python scripts/aggregate.py --run "$smoke_out" \
    --work "$smoke_out" --out "$smoke_out/paper"
echo "smoke test complete: $smoke_out/paper/result_index.csv"
