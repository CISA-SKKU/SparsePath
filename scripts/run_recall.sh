#!/bin/bash
# ============================================================================
# Post-hoc Recall Computation (runs GT + recall via container)
# ============================================================================
# Computes recall for the most recent experiment run, or a specific one.
#
# Usage (from scripts/):
#   ./run_recall.sh configs/deep100K_4node.yaml              # latest run
#   ./run_recall.sh configs/deep100K_4node.yaml --run 20260413_135726
#   ./run_recall.sh configs/deep100K_4node.yaml --force      # recompute GT
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
EXPERIMENT_ROOT_HOST="${DANNS_EXPERIMENT_ROOT:-$PROJECT_ROOT/experiments}"
EXPERIMENT_ROOT_CONTAINER="${DANNS_EXPERIMENT_CONTAINER_ROOT:-/experiments}"
CORE_CONFIGS_ROOT="${SCRIPT_DIR}/configs"
EXPERIMENT_CONFIGS_ROOT="${EXPERIMENT_ROOT_HOST}/configs"
RUN_ROOT_HOST="${EXPERIMENT_ROOT_HOST}/runs"
RUN_ROOT_CONTAINER="${EXPERIMENT_ROOT_CONTAINER}/runs"

STAGE="distributed_engine"
RUN_TIMESTAMP=""
FORCE=""

# --- Parse args -------------------------------------------------------------
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)    RUN_TIMESTAMP="$2"; shift 2 ;;
        --stage)  STAGE="$2"; shift 2 ;;
        --force)  FORCE="--force-recompute-gt"; shift ;;
        --help|-h)
            echo "Usage: $0 <config.yaml> [--run TIMESTAMP] [--stage STAGE] [--force]"
            echo ""
            echo "  config.yaml    Experiment config (e.g. configs/deep100K_4node.yaml)"
            echo "  --run TS       Specific run timestamp (default: latest)"
            echo "  --stage STG    Stage name (default: distributed_engine)"
            echo "  --force        Recompute GT even if cached"
            exit 0 ;;
        *)
            if [[ -z "$CONFIG" ]]; then
                CONFIG="$1"; shift
            else
                echo "Unknown arg: $1"; exit 1
            fi ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: config file required"
    echo "Usage: $0 <config.yaml> [--run TIMESTAMP] [--force]"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    if [[ -f "$EXPERIMENT_ROOT_HOST/$CONFIG" ]]; then
        CONFIG="$EXPERIMENT_ROOT_HOST/$CONFIG"
    elif [[ -f "$EXPERIMENT_CONFIGS_ROOT/$CONFIG" ]]; then
        CONFIG="$EXPERIMENT_CONFIGS_ROOT/$CONFIG"
    elif [[ -f "$CORE_CONFIGS_ROOT/$CONFIG" ]]; then
        CONFIG="$CORE_CONFIGS_ROOT/$CONFIG"
    else
        echo "ERROR: config file not found: $CONFIG"
        exit 1
    fi
fi
CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
if [[ "$CONFIG_ABS" == "$EXPERIMENT_CONFIGS_ROOT"/* ]]; then
    CONFIG_REL="${CONFIG_ABS#$EXPERIMENT_CONFIGS_ROOT/}"
    CONFIG_CONTAINER="$EXPERIMENT_ROOT_CONTAINER/configs/$CONFIG_REL"
elif [[ "$CONFIG_ABS" == "$CORE_CONFIGS_ROOT"/* ]]; then
    CONFIG_REL="${CONFIG_ABS#$CORE_CONFIGS_ROOT/}"
    CONFIG_CONTAINER="/workspace/scripts/configs/$CONFIG_REL"
else
    echo "ERROR: config must live under $EXPERIMENT_CONFIGS_ROOT or $CORE_CONFIGS_ROOT"
    exit 1
fi
CONFIG_BASENAME="$CONFIG_REL"

# --- Resolve experiment name from config ------------------------------------
EXPERIMENT_NAME=$(docker compose -f "$COMPOSE_FILE" exec -T danns \
    python3 -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['experiment_name'])" "$CONFIG_CONTAINER")
EXPERIMENT_NAME=$(echo "$EXPERIMENT_NAME" | tr -d '\r')

RUNS_DIR="${RUN_ROOT_HOST}/${EXPERIMENT_NAME}/${STAGE}"

if [[ ! -d "$RUNS_DIR" ]]; then
    echo "ERROR: no runs found at $RUNS_DIR"
    exit 1
fi

# --- Resolve run directory --------------------------------------------------
if [[ -n "$RUN_TIMESTAMP" ]]; then
    RUN_DIR="${RUNS_DIR}/${RUN_TIMESTAMP}"
else
    # Latest run
    RUN_TIMESTAMP=$(ls -1t "$RUNS_DIR" | head -1)
    RUN_DIR="${RUNS_DIR}/${RUN_TIMESTAMP}"
fi

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: run directory not found: $RUN_DIR"
    exit 1
fi

# Container-internal path
CONTAINER_RUN_DIR="${RUN_ROOT_CONTAINER}/${EXPERIMENT_NAME}/${STAGE}/${RUN_TIMESTAMP}"

echo "============================================================"
echo "Post-hoc Recall Computation"
echo "============================================================"
echo "  Config:    $CONFIG_BASENAME"
echo "  Run:       $RUN_TIMESTAMP"
echo "  Run dir:   $RUN_DIR"
echo "============================================================"

# --- Run recall computation in container ------------------------------------
docker compose -f "$COMPOSE_FILE" exec -T -e PYTHONUNBUFFERED=1 danns \
    python3 pipeline/compute_recall_posthoc.py \
        "$CONTAINER_RUN_DIR" $FORCE

# --- Fix ownership ----------------------------------------------------------
HOST_UID=$(id -u)
HOST_GID=$(id -g)
docker compose -f "$COMPOSE_FILE" exec -T danns \
    chown -R "${HOST_UID}:${HOST_GID}" "$CONTAINER_RUN_DIR" 2>/dev/null || true

echo ""
echo "============================================================"
echo "Done. Results: $RUN_DIR/recall_posthoc.json"
echo "============================================================"
