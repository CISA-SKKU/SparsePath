#!/bin/bash
# MR#8 auxiliary quantizer campaign — OPQ D/4 artifact builds (2026-08-21).
#
# One byte per four vector dimensions (M = dim/4), same code budget as the plain
# D/4 PQ campaign, so OPQ vs PQ is an apples-to-apples swap on the SAME graph
# (lits_k4_iter150), the SAME queries/GT and the SAME L grid.
#
# The rotation is learned by build_pq_m.py --opq (faiss OPQMatrix, 08-06 protocol:
# niter=50, max_train_points=max(train_sample,65536), strided sample of the active
# window). The base is rotated CHUNK-WISE IN MEMORY during encoding — NO rotated
# base file is written (the 08-06 run materialised a 115 GB rotated deep base; we
# must not repeat that with only ~529 GB free on the NFS home).
#
# Infra pitfalls (2026-08-06, project_pq_build_infra_pitfalls):
#   * `import faiss` inside the danns container collapses CPU affinity to 2 cores
#     -> OMP_PROC_BIND=false is MANDATORY.
#   * NFS mmap page-fault reads run at ~18 MB/s; dd-prewarming the active window
#     into page cache turns a 94 s scan into 3 s.
#
# Usage (inside the danns container):
#   bash build_opq_d4.sh deep spacev sift        # laion is HELD pending NFS cleanup
set -u

Q1=/workspace/.claude/worktrees/mr8-quant-harness/scripts/q1_quantization_pareto
export OMP_PROC_BIND=false     # faiss affinity collapse guard (see header)
export OMP_NUM_THREADS=2       # 32 encode workers x 2 BLAS threads

GID_LO=150000000
GID_N=100000000
TRAIN_SAMPLE=200000
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
  local out=/experiments/artifacts/${ds}/pq/M${M}_opq
  echo "==================================================================="
  echo "[$ds $(stamp)] M=${M} dim=${dim} dtype=${dt} -> ${out}"
  df -h /experiments | tail -1
  if [ -f "${out}/pq_meta.json" ]; then
    echo "[$ds] pq_meta.json already present — SKIP (nothing overwritten)"
    return 0
  fi
  mkdir -p "$out"
  prewarm "$base" "$eb" "$dim"
  python "$Q1/build_pq_m.py" --base "$base" --out "$out" --M "$M" --dim "$dim" \
      --gid-lo $GID_LO --gid-n $GID_N --pool-n "$pool" \
      --train-sample $TRAIN_SAMPLE --workers $WORKERS --dtype "$dt" --opq
  local rc=$?
  echo "[$ds $(stamp)] rc=${rc}"
  ls -la "$out"
  df -h /experiments | tail -1
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
    # HELD 2026-08-21: 32 GB of codes; needs explicit go-ahead after NFS cleanup.
    laion)  build_one laion100M  512 float32 128 \
              /workspace/datasets_nfs/LAION400M/laion300M_base.fbin 250000000 4 ;;
    *) echo "unknown dataset: $ds" ;;
  esac
done
echo "[all done $(stamp)]"
