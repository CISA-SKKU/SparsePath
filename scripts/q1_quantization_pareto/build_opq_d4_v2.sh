#!/bin/bash
# MR#8 OPQ D/4 REBUILD v2 (2026-08-21) — fixes the spacev/sift regression.
#
# Diagnosis (diag_opq_vs_pq.py + diag_opq_init.py, held-out 50K of the active
# window, mean ||x - x_hat||^2 relative to plain PQ on the same sample):
#
#     dataset      random-init OPQ (shipped M*_opq)   identity-init OPQ
#     deep100M                0.8598  (helps)               0.9194
#     spacev100M              1.2533  (HURTS)               1.0020
#     sift100M                1.5608  (HURTS)               1.0174
#
# There is NO harness bug: the LUT identity  LUT(R q) == ||q - x_hat||^2  holds
# to 1e-6 on all three (incl. int8/uint8), stored codes re-encode 100%
# identically, and 100% of active gids are inside the encode window. The codes
# are simply worse, because faiss's OPQMatrix starts from a RANDOM rotation and
# on sift/spacev the NATURAL axis order is already near-optimal (SIFT = 16
# spatial cells x 8 orientation bins; a consecutive-4 chunk stays inside one
# cell). A random start lands in a worse local optimum that 50 iterations never
# escape. deep100M is PCA-like with strongly decaying per-dim variance, so the
# natural chunking is badly unbalanced and a random restart genuinely wins.
#
# v2 therefore changes TWO things versus M*_opq:
#   1. --opq-init best     : train BOTH initialisations, keep whichever wins on
#                            a held-out slice. OPQ is a refinement of PQ and
#                            must never lose to it.
#   2. --train-sample 1000000 : match the shipped pipeline codebook protocol
#                            (M*_active used 1M / niter 15; the 08-06 OPQ used
#                            200K). Worth 1.01x deep / 1.01x spacev / 1.12x sift.
# The rotation itself is still learned on 200K rows (--opq-train-sample): a
# dim x dim matrix does not need 1M samples, and it keeps the runtime bounded.
#
# Outputs go to a SEPARATE directory M{M}_opq_v2 — M*_opq is never touched.
#
# Usage (inside the danns container):
#   bash build_opq_d4_v2.sh deep spacev sift
set -u

Q1=/workspace/.claude/worktrees/mr8-quant-harness/scripts/q1_quantization_pareto
export OMP_PROC_BIND=false     # faiss collapses CPU affinity to 2 cores without this
export OMP_NUM_THREADS=2

GID_LO=150000000
GID_N=100000000
TRAIN_SAMPLE=1000000           # codebook sample (matches the pipeline protocol)
OPQ_TRAIN_SAMPLE=200000        # rotation sample
WORKERS=32

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

prewarm() {   # $1=base $2=elem_bytes $3=dim
  local off=$(( 8 + GID_LO * $3 * $2 ))
  local len=$(( GID_N * $3 * $2 ))
  echo "[prewarm $(stamp)] dd $1 off=$off len=$len"
  dd if="$1" of=/dev/null bs=64M iflag=skip_bytes,count_bytes \
     skip="$off" count="$len" 2>&1 | tail -1
}

build_one() {   # $1=ds $2=dim $3=dtype $4=M $5=base $6=pool_n $7=elem_bytes
  local ds=$1 dim=$2 dt=$3 M=$4 base=$5 pool=$6 eb=$7
  local out=/experiments/artifacts/${ds}/pq/M${M}_opq_v2
  echo "==================================================================="
  echo "[$ds $(stamp)] v2 M=${M} dim=${dim} dtype=${dt} -> ${out}"
  df -h /experiments | tail -1
  if [ -f "${out}/pq_meta.json" ]; then
    echo "[$ds] pq_meta.json already present — SKIP (nothing overwritten)"
    return 0
  fi
  mkdir -p "$out"
  prewarm "$base" "$eb" "$dim"
  python "$Q1/build_pq_m.py" --base "$base" --out "$out" --M "$M" --dim "$dim" \
      --gid-lo $GID_LO --gid-n $GID_N --pool-n "$pool" \
      --train-sample $TRAIN_SAMPLE --opq-train-sample $OPQ_TRAIN_SAMPLE \
      --workers $WORKERS --dtype "$dt" --opq --opq-init best
  local rc=$?
  echo "[$ds $(stamp)] rc=${rc}"
  ls -la "$out"
  return $rc
}

for ds in "$@"; do
  case "$ds" in
    deep)   build_one deep100M   96  float32 24 \
              /workspace/datasets_nfs/DEEP1B/deep300M_base.fbin   300000000 4 ;;
    spacev) build_one spacev100M 100 int8    25 \
              /workspace/datasets_nfs/SPACEV1B/spacev300M_base.i8bin 300000000 1 ;;
    sift)   build_one sift100M   128 uint8   32 \
              /workspace/datasets_nfs/SIFT1B/sift300M_base.bin    300000000 1 ;;
    laion)  build_one laion100M  512 float32 128 \
              /workspace/datasets_nfs/LAION400M/laion300M_base.fbin 250000000 4 ;;
    *) echo "unknown dataset: $ds" ;;
  esac
done
echo "[all done $(stamp)]"
