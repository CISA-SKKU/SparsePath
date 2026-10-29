#!/bin/bash
# Rename run directories that lack a non-empty engine_report.txt with a
# `FAILED_` prefix. Idempotent — already-prefixed dirs are skipped.
#
# Usage:
#   bash scripts/mark_failed_runs.sh [--dry-run] [--root <experiments_root>]
#
# Default root: experiments/runs
# Scans every <root>/vldb_*/distributed_engine/<run_dir>.

set -euo pipefail

DRY_RUN=0
ROOT="${DANNS_EXPERIMENT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/experiments}/runs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --root)    ROOT="$2"; shift 2 ;;
        --help|-h)
            grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -d "$ROOT" ]] || { echo "Root not found: $ROOT"; exit 1; }

n_total=0
n_failed=0
n_skipped=0

while IFS= read -r d; do
    [[ -d "$d" ]] || continue
    n_total=$((n_total + 1))
    bn="$(basename "$d")"
    parent="$(dirname "$d")"
    rep="$d/engine_report.txt"

    # Already marked
    if [[ "$bn" == FAILED_* ]]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Skip non-run dirs: only consider names that start with a timestamp
    # `YYYYMMDD_HHMMSS_*`. Auxiliary dirs (gt_cache, _old, _before_*, _test)
    # don't have engine_report.txt by design and must NOT be marked FAILED.
    if [[ ! "$bn" =~ ^[0-9]{8}_[0-9]{6}_ ]]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Has a non-empty engine_report → keep
    if [[ -s "$rep" ]]; then
        continue
    fi

    n_failed=$((n_failed + 1))
    new="$parent/FAILED_$bn"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] mv $d -> $new"
    else
        mv "$d" "$new"
        echo "marked: $new"
    fi
done < <(find "$ROOT" -mindepth 3 -maxdepth 3 -type d -path '*/vldb_*/distributed_engine/*')

echo
echo "Scanned: $n_total"
echo "Failed marked: $n_failed"
echo "Already marked (skipped): $n_skipped"
[[ $DRY_RUN -eq 1 ]] && echo "(dry-run — no changes made)"
