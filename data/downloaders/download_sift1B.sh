#!/bin/bash
# Download SIFT1B (BIGANN) raw bvecs from IRISA/TEXMEX.
#
# Source: ftp://ftp.irisa.fr/local/texmex/corpus/
# Files:
#   bigann_base.bvecs.gz       ~92 GB compressed
#   bigann_query.bvecs.gz      ~5  MB compressed
#   bigann_gnd.tar.gz          ~1.5 GB compressed (groundtruth — NOT used by artifact)
#
# Output (decompressed):
#   $RAW_DIR/sift1B/bigann_base.bvecs
#   $RAW_DIR/sift1B/bigann_query.bvecs
#
# Usage:
#   RAW_DIR=/path/to/raw_datasets bash download_sift1B.sh
#
# Notes:
#   - Disk requirement: ~125 GB (compressed download + decompressed bvecs).
#   - If the IRISA FTP is slow / down, mirrors exist; check the BIGANN webpage.
#   - Decompression is single-threaded; expect 30+ minutes for the base file.

set -euo pipefail

: "${RAW_DIR:?RAW_DIR env var must be set (target download directory)}"

DST="$RAW_DIR/sift1B"
mkdir -p "$DST"
cd "$DST"

BASE_URL="ftp://ftp.irisa.fr/local/texmex/corpus"

download() {
    local fname=$1
    if [[ -f "$fname" || -f "${fname%.gz}" ]]; then
        echo "[skip] $fname already exists (or already decompressed)"
        return
    fi
    echo "[get]  $fname"
    curl -fL -o "$fname" "$BASE_URL/$fname"
}

decompress() {
    local fname=$1
    local out="${fname%.gz}"
    if [[ -f "$out" ]]; then
        echo "[skip] $out already decompressed"
        return
    fi
    echo "[gunzip] $fname"
    gunzip -k "$fname"
}

download bigann_base.bvecs.gz
download bigann_query.bvecs.gz

decompress bigann_base.bvecs.gz
decompress bigann_query.bvecs.gz

echo "=== SIFT1B raw files ready ==="
ls -lh bigann_base.bvecs bigann_query.bvecs
echo "Next: extract 300M subset via data/download_and_extract.sh sift1B"
