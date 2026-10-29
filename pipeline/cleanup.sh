#!/bin/bash
# ============================================================================
# cleanup.sh — Kill stale pipeline processes across all nodes
# ============================================================================
#
# Kills leftover orchestrator/worker/monitor processes from previous runs.
# Safe to run anytime — only targets pipeline-specific DANNS scripts.
#
# Usage:
#   ./cleanup.sh                          # cleanup all (local + all DPU nodes)
#   ./cleanup.sh --nodes 1,2              # cleanup specific nodes only
#   ./cleanup.sh --local-only             # cleanup local container only
#   ./cleanup.sh --dry-run                # show what would be killed
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python interpreter
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -f "${SCRIPT_DIR}/.env" ]]; then
        source "${SCRIPT_DIR}/.env"
    fi
fi
PYTHON="${PYTHON:-python3}"

# Defaults
NODES=""
LOCAL_ONLY=false
DRY_RUN=false
NODES_YAML=""

# Pipeline scripts and monitors to kill. Keep these patterns specific to DANNS:
# monitors can survive when a worker aborts before its normal stop hook runs.
TARGETS=(
    "orchestrator.py"
    "node_worker.py"
    "run_pipeline.py"
    "nic_bw_monitor.py"
    "rdma_monitor.sh"
    "pmu_monitor.sh"
)

# Host-side helpers spawned by launchers. These are not always visible from
# inside the danns container, so remote cleanup checks the host as well.
HOST_EXTRA_TARGETS=(
    "tail -F .*SparsePath_experiments.*/logs/.*\\.log"
)

while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes)        NODES="$2"; shift 2 ;;
        --local-only)   LOCAL_ONLY=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --nodes-yaml)   NODES_YAML="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--nodes <id,...>] [--local-only] [--dry-run] [--nodes-yaml <path>]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Resolve nodes.yaml
if [[ -z "$NODES_YAML" ]]; then
    # Try common locations
    EXPERIMENT_ROOT_HOST="${DANNS_EXPERIMENT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/experiments}"
    for candidate in "${EXPERIMENT_ROOT_HOST}/configs/nodes.yaml" "${SCRIPT_DIR}/../configs/nodes.yaml" "${SCRIPT_DIR}/../../scripts/configs/nodes.yaml"; do
        if [[ -f "$candidate" ]]; then
            NODES_YAML="$candidate"
            break
        fi
    done
    if [[ -z "$NODES_YAML" ]]; then
        NODES_YAML="${EXPERIMENT_ROOT_HOST}/configs/nodes.yaml"  # fallback for error message
    fi
fi

if [[ ! -f "$NODES_YAML" ]]; then
    echo "ERROR: nodes.yaml not found: $NODES_YAML"
    exit 1
fi

# Build grep pattern for targets
GREP_PATTERN=$(printf "|%s" "${TARGETS[@]}")
GREP_PATTERN="${GREP_PATTERN:1}"  # remove leading |
HOST_GREP_PATTERN=$(printf "|%s" "${TARGETS[@]}" "${HOST_EXTRA_TARGETS[@]}")
HOST_GREP_PATTERN="${HOST_GREP_PATTERN:1}"  # remove leading |

killed=0

kill_container_pattern() {
    local container="$1"
    local pattern="$2"
    docker exec "$container" pkill -TERM -f "$pattern" 2>/dev/null || true
}

kill_container_pattern_force() {
    local container="$1"
    local pattern="$2"
    docker exec "$container" pkill -KILL -f "$pattern" 2>/dev/null || true
}

kill_remote_container_pattern() {
    local host="$1"
    local container="$2"
    local pattern="$3"
    ssh "$host" "docker exec $container pkill -TERM -f '$pattern' 2>/dev/null || true" 2>/dev/null || true
}

kill_remote_container_pattern_force() {
    local host="$1"
    local container="$2"
    local pattern="$3"
    ssh "$host" "docker exec $container pkill -KILL -f '$pattern' 2>/dev/null || true" 2>/dev/null || true
}

kill_remote_host_pattern() {
    local host="$1"
    local pattern="$2"
    ssh "$host" "pkill -TERM -f '$pattern' 2>/dev/null || true" 2>/dev/null || true
}

kill_remote_host_pattern_force() {
    local host="$1"
    local pattern="$2"
    ssh "$host" "pkill -KILL -f '$pattern' 2>/dev/null || true" 2>/dev/null || true
}

# --- Local container (orchestrator-host orchestrator) ---
echo "=== Local container (orchestrator-host) ==="

LOCAL_CONTAINER=$($PYTHON -c "
import yaml, sys
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
# Check common local container names
for name in ['vecdb-build', 'danns']:
    print(name)
" "$NODES_YAML")

for CONTAINER in $LOCAL_CONTAINER; do
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
        continue
    fi

    PROCS=$(docker exec "$CONTAINER" bash -c "ps aux | grep -E '${GREP_PATTERN}' | grep -v grep" 2>/dev/null || true)
    if [[ -z "$PROCS" ]]; then
        echo "  [$CONTAINER] No stale processes"
        continue
    fi

    echo "  [$CONTAINER] Found:"
    echo "$PROCS" | while read -r line; do echo "    $line"; done

    if ! $DRY_RUN; then
        for target in "${TARGETS[@]}"; do
            kill_container_pattern "$CONTAINER" "$target"
        done
        sleep 1
        for target in "${TARGETS[@]}"; do
            kill_container_pattern_force "$CONTAINER" "$target"
        done
        echo "  [$CONTAINER] Killed"
        killed=$((killed + 1))
    else
        echo "  [$CONTAINER] (dry-run, skipping)"
    fi
done

if $LOCAL_ONLY; then
    echo ""
    echo "Done (local-only mode)."
    exit 0
fi

# --- DPU nodes ---
echo ""
echo "=== DPU nodes ==="

# Parse all node IDs if not specified
if [[ -z "$NODES" ]]; then
    NODES=$($PYTHON -c "
import yaml, sys
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
ids = [str(n['id']) for n in data['nodes']]
print(','.join(ids))
" "$NODES_YAML")
fi

IFS=',' read -ra NODE_IDS <<< "$NODES"

for NODE_ID in "${NODE_IDS[@]}"; do
    INFO=$($PYTHON -c "
import yaml, sys
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
for n in data['nodes']:
    if n['id'] == int(sys.argv[2]):
        print(f\"{n['ssh_host']}|{n.get('container', 'danns')}\")
        break
" "$NODES_YAML" "$NODE_ID")

    SSH_HOST=$(echo "$INFO" | cut -d'|' -f1)
    CONTAINER=$(echo "$INFO" | cut -d'|' -f2)

    if [[ -z "$SSH_HOST" ]]; then
        echo "  [Node $NODE_ID] not found in $NODES_YAML, skipping"
        continue
    fi

    echo "  [Node $NODE_ID] ${SSH_HOST}/${CONTAINER}"

    # Check SSH connectivity first
    if ! ssh -o ConnectTimeout=3 -o BatchMode=yes "$SSH_HOST" true 2>/dev/null; then
        echo "    SSH unreachable, skipping"
        continue
    fi

    PROCS=$(ssh "$SSH_HOST" "docker exec $CONTAINER bash -c \"ps aux | grep -E '${GREP_PATTERN}' | grep -v grep\"" 2>/dev/null || true)
    HOST_PROCS=$(ssh "$SSH_HOST" "ps aux | grep -E '${HOST_GREP_PATTERN}' | grep -v grep" 2>/dev/null || true)

    if [[ -z "$PROCS" && -z "$HOST_PROCS" ]]; then
        echo "    No stale processes"
        continue
    fi

    if [[ -n "$PROCS" ]]; then
        echo "    Found in container:"
        echo "$PROCS" | while read -r line; do echo "      $line"; done
    fi

    if [[ -n "$HOST_PROCS" ]]; then
        echo "    Found on host:"
        echo "$HOST_PROCS" | while read -r line; do echo "      $line"; done
    fi

    if ! $DRY_RUN; then
        for target in "${TARGETS[@]}"; do
            kill_remote_container_pattern "$SSH_HOST" "$CONTAINER" "$target"
            kill_remote_host_pattern "$SSH_HOST" "$target"
        done
        for target in "${HOST_EXTRA_TARGETS[@]}"; do
            kill_remote_host_pattern "$SSH_HOST" "$target"
        done
        sleep 1
        for target in "${TARGETS[@]}"; do
            kill_remote_container_pattern_force "$SSH_HOST" "$CONTAINER" "$target"
            kill_remote_host_pattern_force "$SSH_HOST" "$target"
        done
        for target in "${HOST_EXTRA_TARGETS[@]}"; do
            kill_remote_host_pattern_force "$SSH_HOST" "$target"
        done
        echo "    Killed"
        killed=$((killed + 1))
    else
        echo "    (dry-run, skipping)"
    fi
done

# --- Check freed ports ---
if ! $DRY_RUN; then
    echo ""
    echo "=== Port check ==="
    SYNC_PORT=$($PYTHON -c "
import yaml, sys
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
print(data.get('orchestrator', {}).get('sync_port', 19700))
" "$NODES_YAML")

    if ss -tlnp 2>/dev/null | grep -q ":${SYNC_PORT} "; then
        echo "  WARNING: Port $SYNC_PORT still in use (may take a moment to release)"
    else
        echo "  Port $SYNC_PORT is free"
    fi
fi

echo ""
echo "Done."
