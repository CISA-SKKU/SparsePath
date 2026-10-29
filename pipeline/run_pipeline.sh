#!/bin/bash
# ============================================================================
# run_pipeline.sh - Distributed DiskANN Pipeline Orchestrator
# ============================================================================
#
# Orchestrates the full pipeline:
#   LOCAL stages (build, partition, colocation, pq_build, gt) → build host execution
#   DISTRIBUTED stages (distributed_engine, search) → SSH + docker exec on DPU nodes
#
# Usage:
#   ./run_pipeline.sh --config configs/deep1M_e2e.yaml --nodes 1,2,3,4
#   ./run_pipeline.sh --config configs/deep1M_e2e.yaml --nodes 1,2 --stages search
#   ./run_pipeline.sh --config configs/deep1M_e2e.yaml --dry-run
#
# Requirements:
#   - SSH access to DPU nodes (via ~/.ssh/config aliases)
#   - Docker containers already running on each node
#   - python3 + yaml on this machine
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NODES_YAML=""  # Resolved from config
WORKER_SCRIPT="pipeline/node_worker.py"
ORCHESTRATOR_SCRIPT="pipeline/run_pipeline.py"
CONTAINER_WORKDIR="/workspace"
BUILD_CONTAINER="${BUILD_CONTAINER:-danns}"

# Python interpreter: override with PYTHON env var or .env file
if [[ -z "${PYTHON:-}" ]]; then
    # Try .env file in script directory
    if [[ -f "${SCRIPT_DIR}/.env" ]]; then
        source "${SCRIPT_DIR}/.env"
    fi
fi
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="python3"
fi
if [[ ! -x "$(command -v "$PYTHON")" ]]; then
    echo "ERROR: Python not found: $PYTHON"
    echo "  Set PYTHON env var or create ${SCRIPT_DIR}/.env with: PYTHON=/path/to/python"
    exit 1
fi

# --- Defaults ---
CONFIG=""
NODES=""
COORDINATOR=""
STAGES=""
POSTFIX=""
DRY_RUN=false
COMPILE_ONLY=false
NUMA_BIND=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --compile)      COMPILE_ONLY=true; shift ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --nodes)        NODES="$2"; shift 2 ;;
        --coordinator)  COORDINATOR="$2"; shift 2 ;;
        --stages)       STAGES="$2"; shift 2 ;;
        --postfix)      POSTFIX="$2"; shift 2 ;;
        --numa)         NUMA_BIND="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 --compile"
            echo "       $0 --config <yaml> [--nodes <id,id,...>] [--stages <stage,...>] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --compile      Build all C++ components in ${BUILD_CONTAINER} container"
            echo "  --config       Path to YAML config file (required for pipeline stages)"
            echo "  --nodes        Comma-separated node IDs for distributed stages (e.g., 1,2,3,4)"
            echo "  --coordinator  Coordinator node ID (default: from config or lowest node)"
            echo "  --stages       Override stages to run (comma-separated, e.g., build,search)"
            echo "  --postfix      Postfix appended to timestamp dir (e.g., --postfix trace_L2 → 20260314_211913_trace_L2)"
            echo "  --numa         NUMA node to bind (e.g., --numa 0 → numactl --cpunodebind=0 --membind=0)"
            echo "  --dry-run      Print commands without executing"
            echo ""
            echo "Stage classification:"
            echo "  LOCAL:                build, partition, colocation, pq_build, gt"
            echo "  DISTRIBUTED (DPU):    search, distributed_engine"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Compile-only mode ---
if $COMPILE_ONLY; then
    echo "============================================"
    echo " C++ Build (${BUILD_CONTAINER})"
    echo "============================================"
    echo "  CMD: cd /workspace && ./build_all.sh"
    docker exec "${BUILD_CONTAINER}" bash -c "cd /workspace && ./build_all.sh"
    echo "============================================"
    echo " Build complete."
    echo "============================================"
    exit 0
fi

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required"
    exit 1
fi

# --- Resolve paths ---
# If config path is relative, make it relative to SCRIPT_DIR
if [[ ! "$CONFIG" = /* ]]; then
    CONFIG_ABS="${PROJECT_ROOT}/${CONFIG}"
else
    CONFIG_ABS="$CONFIG"
fi

if [[ ! -f "$CONFIG_ABS" ]]; then
    echo "ERROR: Config file not found: $CONFIG_ABS"
    exit 1
fi

CONFIG_DIR="$(dirname "$CONFIG_ABS")"

# --- Parse stages from config if not overridden ---
get_stages() {
    PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${SCRIPT_DIR}" $PYTHON -c "
import yaml, sys
from pipeline.config import normalize_canonical_config
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
normalize_canonical_config(cfg)
stages = cfg.get('pipeline', {}).get('stages', ['build', 'partition', 'colocation', 'gt', 'distributed_engine'])
print(','.join(stages))
" "$1"
}

if [[ -z "$STAGES" ]]; then
    STAGES=$(get_stages "$CONFIG_ABS")
fi

IFS=',' read -ra STAGE_LIST <<< "$STAGES"

# --- Resolve nodes.yaml path from config ---
resolve_nodes_yaml() {
    $PYTHON -c "
import yaml, sys, os
config_path = sys.argv[1]
with open(config_path) as f:
    cfg = yaml.safe_load(f)
nodes_file = cfg.get('cluster', {}).get('nodes_file', 'nodes.yaml')
config_dir = os.path.dirname(config_path)
nodes_path = os.path.join(config_dir, nodes_file)
print(nodes_path)
" "$1"
}

NODES_YAML=$(resolve_nodes_yaml "$CONFIG_ABS")

# --- Parse node info from nodes.yaml ---
read_node_info() {
    $PYTHON -c "
import yaml, sys
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
for n in data['nodes']:
    if n['id'] == int(sys.argv[2]):
        print(f\"{n['ssh_host']}|{n.get('container', 'vecdb-node' + str(n['id']))}\")
        break
" "$NODES_YAML" "$1"
}

# --- Stage classification ---
is_distributed_stage() {
    case "$1" in
        search|distributed_engine) return 0 ;;
        *)                        return 1 ;;
    esac
}


# --- Setup ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$POSTFIX" ]]; then
    TIMESTAMP="${TIMESTAMP}_${POSTFIX}"
fi

echo "============================================"
echo " Distributed DiskANN Pipeline"
echo "============================================"
echo "  Config:       $CONFIG"
echo "  Nodes YAML:   $NODES_YAML"
echo "  Stages:       ${STAGE_LIST[*]}"
echo "  Nodes:        ${NODES:-"(local only)"}"
echo "  Coordinator:  ${COORDINATOR:-"(from config)"}"
echo "  Timestamp:    $TIMESTAMP"
echo "  Postfix:      ${POSTFIX:-"(none)"}"
echo "  Dry run:      $DRY_RUN"
echo "============================================"
echo ""

# --- Build coordinator arg ---
COORD_ARG=""
if [[ -n "$COORDINATOR" ]]; then
    COORD_ARG="--coordinator $COORDINATOR"
fi

# --- Cleanup on Ctrl+C ---
declare -a PIDS=()
declare -a LAUNCHED_SSH_HOSTS=()
declare -a LAUNCHED_CONTAINERS=()

cleanup() {
    echo ""
    echo "============================================"
    echo " Interrupted! Cleaning up..."
    echo "============================================"

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "  Killed local PID $pid"
        fi
    done

    for i in "${!LAUNCHED_SSH_HOSTS[@]}"; do
        local host="${LAUNCHED_SSH_HOSTS[$i]}"
        local container="${LAUNCHED_CONTAINERS[$i]}"
        echo "  Killing python3 in ${host}/${container}..."
        ssh "$host" "docker exec $container pkill -f '${WORKER_SCRIPT}'" 2>/dev/null || true
    done

    echo "  Done."
    exit 130
}
trap cleanup SIGINT SIGTERM

# --- Execute stages ---
PIPELINE_T0=$(date +%s)

for STAGE in "${STAGE_LIST[@]}"; do
    echo ""
    echo "============================================"
    echo "  Stage: $STAGE"
    echo "============================================"

    STAGE_T0=$(date +%s)

    if is_distributed_stage "$STAGE"; then
        # --- DISTRIBUTED stage ---
        if [[ -z "$NODES" ]]; then
            echo "ERROR: --nodes is required for distributed stage '$STAGE'"
            exit 1
        fi

        IFS=',' read -ra NODE_IDS <<< "$NODES"

        # Get output dir from config for log storage
        # Translate container path (/workspace/...) to host path if needed
        BASELINE_DIR="${PROJECT_ROOT}"
        # Stage-specific log subdirectory
        if [[ "$STAGE" == "distributed_engine" ]]; then
            LOG_SUBDIR="distributed_engine"
        else
            LOG_SUBDIR="search_rdma"
        fi
        LOG_DIR=$($PYTHON -c "
import yaml, sys, os
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
base_dir = cfg.get('output', {}).get('base_dir', '.')
exp_name = cfg.get('experiment_name', 'unnamed')
log_path = os.path.join(base_dir, exp_name, sys.argv[4], sys.argv[2])
# Translate /workspace/ to host baseline dir
if log_path.startswith('/workspace/'):
    log_path = os.path.join(sys.argv[3], log_path[len('/workspace/'):])
print(log_path)
" "$CONFIG_ABS" "$TIMESTAMP" "$BASELINE_DIR" "$LOG_SUBDIR")
        mkdir -p "$LOG_DIR"

        # Pre-compute GT for search ops (runs on orchestrator-host before DPU dispatch)
        if [[ "$STAGE" == "distributed_engine" ]]; then
            echo "  [Pre-dispatch] Computing GT for search ops in workload..."
            PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${SCRIPT_DIR}" $PYTHON "${PROJECT_ROOT}/${ORCHESTRATOR_SCRIPT}" --config "$CONFIG_ABS" --stages distributed_engine 2>&1 \
                | tee "${LOG_DIR}/gt_precompute.log" || true
        fi

        PIDS=()
        LAUNCHED_SSH_HOSTS=()
        LAUNCHED_CONTAINERS=()

        for NODE_ID in "${NODE_IDS[@]}"; do
            INFO=$(read_node_info "$NODE_ID")
            SSH_HOST=$(echo "$INFO" | cut -d'|' -f1)
            CONTAINER=$(echo "$INFO" | cut -d'|' -f2)

            if [[ -z "$SSH_HOST" ]]; then
                echo "ERROR: No ssh_host found for node $NODE_ID in $NODES_YAML"
                exit 1
            fi

            NUMA_ARG=""
            if [[ -n "$NUMA_BIND" ]]; then
                NUMA_ARG=" --numa-node ${NUMA_BIND}"
            fi
            # Forward optional pipeline env vars into the container.
            PMU_FWD=""
            [[ -n "${PIPELINE_PMU:-}" ]] && PMU_FWD="PIPELINE_PMU=${PIPELINE_PMU} "
            CMD="cd ${CONTAINER_WORKDIR} && ${PMU_FWD}python3 -u ${WORKER_SCRIPT} --config ${CONFIG} --node-id ${NODE_ID} --nodes ${NODES} ${COORD_ARG} --stage ${STAGE} --timestamp ${TIMESTAMP}${NUMA_ARG}"
            LOG_FILE="${LOG_DIR}/node${NODE_ID}.log"

            echo "[Node $NODE_ID] ${SSH_HOST} / ${CONTAINER}"
            echo "  CMD: $CMD"
            echo "  LOG: $LOG_FILE"

            if $DRY_RUN; then
                echo "  (dry-run, skipping)"
                continue
            fi

            ssh "${SSH_HOST}" "docker exec ${CONTAINER} bash -c '${CMD}'" \
                > "$LOG_FILE" 2>&1 &
            PIDS+=($!)
            LAUNCHED_SSH_HOSTS+=("$SSH_HOST")
            LAUNCHED_CONTAINERS+=("$CONTAINER")
            echo "  PID: ${PIDS[-1]}"
        done

        if ! $DRY_RUN && [[ ${#PIDS[@]} -gt 0 ]]; then
            echo ""
            echo "  Waiting for all nodes..."
            echo "  Logs: tail -f ${LOG_DIR}/node*.log"
            echo ""

            FAILED=0
            FAILED_NODES=()
            for i in "${!PIDS[@]}"; do
                PID=${PIDS[$i]}
                NID=${NODE_IDS[$i]}
                if wait "$PID"; then
                    echo "  [Node $NID] Done (exit 0)"
                else
                    EXIT_CODE=$?
                    echo "  [Node $NID] FAILED (exit $EXIT_CODE)"
                    FAILED=$((FAILED + 1))
                    FAILED_NODES+=("$NID")
                fi
            done

            if [[ ${#FAILED_NODES[@]} -gt 0 ]]; then
                echo ""
                echo "  Failed node log tails:"
                for NID in "${FAILED_NODES[@]}"; do
                    echo "  --- [Node $NID] last 20 lines ---"
                    tail -20 "${LOG_DIR}/node${NID}.log" 2>/dev/null || echo "  (log not found)"
                done

                # Rename output dir to FAILED_
                FAILED_DIR="$(dirname "$LOG_DIR")/FAILED_$(basename "$LOG_DIR")"
                if [[ -d "$LOG_DIR" && ! -d "$FAILED_DIR" ]]; then
                    mv "$LOG_DIR" "$FAILED_DIR"
                    echo ""
                    echo "  Output dir renamed to: $FAILED_DIR"
                fi

                echo ""
                echo "ERROR: $FAILED node(s) failed in stage '$STAGE'"
                exit 1
            fi

            # --- Generate report (search stage only) ---
            if [[ "$STAGE" == "search" ]]; then
                echo ""
                echo "  Generating report..."
                FIRST_NODE=${NODE_IDS[0]}
                FIRST_INFO=$(read_node_info "$FIRST_NODE")
                FIRST_SSH=$(echo "$FIRST_INFO" | cut -d'|' -f1)
                FIRST_CONTAINER=$(echo "$FIRST_INFO" | cut -d'|' -f2)

                # Compute container-side result path from config
                RESULT_SUBDIR="search_rdma"
                REPORT_TYPE_ARG=""
                RESULT_PATH=$($PYTHON -c "
import yaml, sys, os
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
base_dir = cfg.get('output', {}).get('base_dir', '.')
exp_name = cfg.get('experiment_name', 'unnamed')
print(os.path.join(base_dir, exp_name, sys.argv[2], sys.argv[3]))
" "$CONFIG_ABS" "$RESULT_SUBDIR" "$TIMESTAMP")

                REPORT_CMD="cd ${CONTAINER_WORKDIR} && python3 -u pipeline/report.py ${REPORT_TYPE_ARG} ${RESULT_PATH}"
                ssh "${FIRST_SSH}" "docker exec ${FIRST_CONTAINER} bash -c '${REPORT_CMD}'" 2>&1 || \
                    echo "  (report generation failed, results still available)"
            fi
        fi

    else
        # --- LOCAL stage ---
        if [[ "$STAGE" == "build" || "$STAGE" == "pq_build" || "$STAGE" == "gt" || -n "${BUILD_CONTAINER_ALL:-}" ]]; then
            # build/pq_build require native libs only available inside the container
            # BUILD_CONTAINER_ALL=1: run ALL local stages in the container (for DPU nodes)
            # Translate host path to container /workspace/ path
            BASELINE_DIR_LOCAL="${PROJECT_ROOT}"
            CONTAINER_CONFIG="${CONFIG_ABS/#${BASELINE_DIR_LOCAL}\//\/workspace\/}"
            CONTAINER_CMD="cd ${CONTAINER_WORKDIR} && python3 -u ${ORCHESTRATOR_SCRIPT} --config ${CONTAINER_CONFIG} --stages ${STAGE}"

            echo "  CMD (container): docker exec ${BUILD_CONTAINER} bash -c '${CONTAINER_CMD}'"

            if $DRY_RUN; then
                echo "  (dry-run, skipping)"
            else
                docker exec "${BUILD_CONTAINER}" bash -c "${CONTAINER_CMD}"
            fi
        else
            CMD="PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${SCRIPT_DIR} ${PYTHON} ${PROJECT_ROOT}/${ORCHESTRATOR_SCRIPT} --config ${CONFIG_ABS} --stages ${STAGE}"

            echo "  CMD: $CMD"

            if $DRY_RUN; then
                echo "  (dry-run, skipping)"
            else
                eval "$CMD"
            fi
        fi
    fi

    STAGE_T1=$(date +%s)
    STAGE_ELAPSED=$((STAGE_T1 - STAGE_T0))

    if ! $DRY_RUN; then
        echo ""
        echo "  Stage '$STAGE' completed in ${STAGE_ELAPSED}s"
    fi
done

PIPELINE_T1=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_T1 - PIPELINE_T0))

echo ""
echo "============================================"
if $DRY_RUN; then
    echo " Dry run complete. No processes launched."
else
    echo " Pipeline completed in ${TOTAL_ELAPSED}s"
fi
echo "============================================"
