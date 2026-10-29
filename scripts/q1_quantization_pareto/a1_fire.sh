#!/bin/bash
# a1_fire.sh — A1 (M08 재접근) sequential runner. Run INSIDE the danns container:
#   docker exec danns bash /workspace/.claude/worktrees/single-engine/scripts/q1_quantization_pareto/a1_fire.sh
# deep100M then spacev100M; each = 2 graphs (B iter150 / SP lits_k4_iter150) x
# 6 modes x 8 L, multi-label counters (balanced/centroid/nc/lits) in one traversal.
set -u
cd "$(dirname "$0")"
OUT_ROOT=/experiments/results/m08_quantization_pareto
mkdir -p "$OUT_ROOT"
for cfg in a1_deep100M.yaml a1_spacev100M.yaml; do
    name="${cfg%.yaml}"
    echo "=== A1 fire: $cfg -> $OUT_ROOT/results_$name ==="
    python run_q1.py --config "$cfg" --out "$OUT_ROOT/results_$name" \
        > "${name}_dpu3.log" 2>&1
    echo "A1_EXIT($cfg)=$?" | tee -a "${name}_dpu3.log"
done
echo A1_ALL_DONE
