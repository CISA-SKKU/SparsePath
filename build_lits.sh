#!/usr/bin/env bash
# Build the LITS-aware graph-build tree (third_party/lits_build_tree) — needed
# ONLY for Phase B-full (rebuilding the SparsePath graph from scratch).
# Produces third_party/lits_build_tree/lib/diskann_bindings.so, consumed by
#   tools/graph_build/build_updatecycle_lits.py --bindings-lib third_party/lits_build_tree/lib
# The search-side engine (./build_all.sh, lib/*.so) is independent of this tree.
#
# Run inside the container:  docker compose exec danns ./build_lits.sh [--clean]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$ROOT/third_party/lits_build_tree"
[[ "${1:-}" == "--clean" ]] && rm -rf "$TREE/build" "$TREE/lib"
echo "=== LITS build tree: $TREE ==="
# The branch predates DANNS_ARCH; its build_all.sh uses the portable -mavx2 -mfma baseline.
( cd "$TREE" && ./build_all.sh )
ls -lh "$TREE"/lib/diskann_bindings*.so
