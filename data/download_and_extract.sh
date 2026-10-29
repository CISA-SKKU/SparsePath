#!/bin/bash
# Download + extract a 300M subset for one of the artifact's datasets.
#
# Usage:
#   DATASETS_ROOT=/path/to/datasets RAW_DIR=/path/to/raw bash download_and_extract.sh <dataset>
#
# <dataset> is one of: sift1B | deep1B | spacev1B | laion
#
# Layout produced:
#   $RAW_DIR/<src>/...                  raw downloaded files (not used at runtime)
#   $DATASETS_ROOT/vldb/<ds>300M/<ds>300M_{base,query}.{bin,fbin,i8bin}
#
# Notes:
#   - SIFT1B: ~ 92 GB compressed download + ~250 GB raw bvecs.  Subset = ~38 GB bin.
#   - DEEP1B: ~ 38 GB shards (4 of 37) + queries ~15 MB.        Subset = ~115 GB fbin.
#   - SPACEV1B: streamed via Hugging Face (BAAI/SPACEV).         Subset = ~30 GB i8bin.
#   - LAION (400M): ~390 GB float16 npy.                        Subset = ~570 GB fbin.
#
# Each downloader is idempotent — re-running only fetches missing files.

set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "Usage: $0 <sift1B|deep1B|spacev1B|laion>" >&2
    exit 1
fi

: "${RAW_DIR:?RAW_DIR env var must be set}"
: "${DATASETS_ROOT:?DATASETS_ROOT env var must be set}"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXTRACT="$HERE/extract_subset.py"
DL="$HERE/downloaders"

case "$DATASET" in
    sift1B)
        bash "$DL/download_sift1B.sh"
        OUT="$DATASETS_ROOT/vldb/sift300M"
        mkdir -p "$OUT"
        python3 "$EXTRACT" \
            --input  "$RAW_DIR/sift1B/bigann_base.bvecs" \
            --output "$OUT/sift300M_base.bin" \
            --count  300_000_000 \
            --dtype  uint8 --format bvecs \
            --query-input  "$RAW_DIR/sift1B/bigann_query.bvecs" \
            --query-output "$OUT/sift300M_query.bin" \
            --query-format bvecs
        ;;

    deep1B)
        # 4 shards is enough for 300M (each shard ≈ 26M vectors of 96-dim float)
        # so 4 × 26M = 104M ... wait that's not enough. Each shard is 10GB at
        # 388 bytes/vec ≈ 27M vectors. 300M / 27M ≈ 12 shards. Override to 12.
        python3 "$DL/download_deep1B.py" --base-shards 12 --merge
        OUT="$DATASETS_ROOT/vldb/deep300M"
        mkdir -p "$OUT"
        python3 "$EXTRACT" \
            --input  "$RAW_DIR/deep1B/base.fvecs" \
            --output "$OUT/deep300M_base.fbin" \
            --count  300_000_000 \
            --dtype  float32 --format fvecs \
            --query-input  "$RAW_DIR/DEEP1B/deep1B_queries.fvecs" \
            --query-output "$OUT/deep300M_query.fbin" \
            --query-format fvecs
        ;;

    spacev1B)
        # Direct streaming download produces a DiskANN-bin already; we do NOT
        # need extract_subset.py if --num matches our target. Use it anyway
        # so that the workflow stays uniform and a 1B raw download can be
        # reused for other subset sizes if needed.
        python3 "$DL/download_spacev1B.py" --num 300M
        OUT="$DATASETS_ROOT/vldb/spacev300M"
        mkdir -p "$OUT"
        python3 "$EXTRACT" \
            --input  "$RAW_DIR/spacev1B/spacev1B_base.bin" \
            --output "$OUT/spacev300M_base.i8bin" \
            --count  300_000_000 \
            --dtype  int8 --format bin
        if [[ -f "$RAW_DIR/spacev1B/spacev1B_query.bin" ]]; then
            python3 "$EXTRACT" \
                --input  "$RAW_DIR/spacev1B/spacev1B_query.bin" \
                --output "$OUT/spacev300M_query.i8bin" \
                --count  10_000_000 \
                --dtype  int8 --format bin
        else
            echo "[warn] spacev1B_query.bin not found at $RAW_DIR/spacev1B/."
            echo "       Place the MS-Turing query file there manually."
            echo "       (See data/downloaders/download_spacev1B.py header.)"
        fi
        ;;

    laion)
        # LAION downloader writes fbin directly; we only need to copy/rename
        # into the standard layout. 300M base + 10K queries split off the tail.
        python3 "$DL/download_laion.py" --num 300M --query-count 10000
        OUT="$DATASETS_ROOT/vldb/laion300M"
        mkdir -p "$OUT"
        ln -sf "$RAW_DIR/laion/laion300M_base.fbin"  "$OUT/laion300M_base.fbin"
        ln -sf "$RAW_DIR/laion/laion300M_query.fbin" "$OUT/laion300M_query.fbin"
        ;;

    *)
        echo "Unknown dataset: $DATASET" >&2
        echo "Valid: sift1B deep1B spacev1B laion" >&2
        exit 1
        ;;
esac

echo
echo "=== Output ==="
ls -lh "$OUT"
