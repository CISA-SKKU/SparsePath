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
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

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
