#!/bin/bash
# Build all DANNS components from project root
# Usage: ./build_all.sh [--clean]
#
# Must be run inside the Docker container (vecdb-dev)

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

CLEAN=false
if [ "$1" = "--clean" ]; then
    CLEAN=true
fi

echo "=== DANNS Build ==="
echo "Root: ${ROOT_DIR}"
echo ""

if [ "$CLEAN" = true ]; then
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

echo "--- CMake Configure ---"
# DANNS_ARCH / DANNS_FAST_MATH raise the ISA baseline for an A/B arm, e.g.
#   DANNS_ARCH=icelake-server ./build_all.sh --clean
# Never pass "native": orchestrator-host (build host, Sapphire Rapids) has AMX/fp16/bf16
# that worker-host (Ice Lake) lacks, and both load the same .so.
# Always pass both explicitly: they are CACHE variables, so omitting the flag
# silently reuses the previous configure's value and an unset DANNS_ARCH would
# keep building the last -march arm (hit 2026-07-27 — an ISA A/B leaked into
# the next build). Empty string restores the legacy -mavx2 baseline.
# Defaults (2026-08-30): the production arm icelake-server + FAST_MATH=ON is now
# the DEFAULT — a bare ./build_all.sh reproduces the deployed engine exactly.
# Override by exporting DANNS_ARCH / DANNS_FAST_MATH; a set-but-EMPTY var
# (DANNS_ARCH= ./build_all.sh) restores the legacy -mavx2 baseline — the
# expansions below use ${VAR-default}, not ${VAR:-default}, on purpose.
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
      -DDANNS_ARCH="${DANNS_ARCH-icelake-server}" \
      -DDANNS_FAST_MATH="${DANNS_FAST_MATH-ON}"

echo ""
echo "--- CMake Build ---"
cmake --build "${BUILD_DIR}" -j$(nproc)

echo ""
echo "=== Build Complete ==="
if ls "${ROOT_DIR}/lib/"*.so 2>/dev/null; then
    echo "Python modules:"
    ls -la "${ROOT_DIR}/lib/"*.so
else
    echo "No .so files generated yet (bindings have TODOs)"
fi
