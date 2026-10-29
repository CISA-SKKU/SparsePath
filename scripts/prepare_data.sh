#!/bin/bash
# ============================================================================
# DANNS Data Preparation Pipeline (runs on host, outside container)
# ============================================================================
# Prepares data for distributed experiments:
#   1. build        — Build DiskANN graph index
#   2. partition     — K-means partition vectors
#   3. colocation    — Build RDMA colocation layout
#   4. deploy        — Copy colocation files to DPU local SSDs
#
# Steps 1-3 execute inside the orchestrator-host container automatically.
# Step 4 uses SSH + cp (DPU nodes can see NFS).
#
# Usage (from scripts/):
#   ./prepare_data.sh configs/deep1M_test.yaml
#   ./prepare_data.sh configs/deep1M_test.yaml --from colocation
#   ./prepare_data.sh configs/deep1M_test.yaml --deploy-only
#   ./prepare_data.sh configs/deep1M_test.yaml --skip-deploy
# ============================================================================

set -euo pipefail

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
EXPERIMENT_ROOT_HOST="${DANNS_EXPERIMENT_ROOT:-$PROJECT_ROOT/experiments}"
EXPERIMENT_ROOT_CONTAINER="${DANNS_EXPERIMENT_CONTAINER_ROOT:-/experiments}"
CORE_CONFIGS_ROOT="${SCRIPT_DIR}/configs"
EXPERIMENT_CONFIGS_ROOT="${EXPERIMENT_ROOT_HOST}/configs"
RUN_ROOT_HOST="${EXPERIMENT_ROOT_HOST}/runs"
RUN_ROOT_CONTAINER="${EXPERIMENT_ROOT_CONTAINER}/runs"

# --- Defaults ---------------------------------------------------------------
START_STAGE="build"
DEPLOY_ONLY=false
SKIP_DEPLOY=false

# --- Parse args -------------------------------------------------------------
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)        START_STAGE="$2"; shift 2 ;;
        --deploy-only) DEPLOY_ONLY=true; shift ;;
        --skip-deploy) SKIP_DEPLOY=true; shift ;;
        --help|-h)
            echo "Usage: $0 <config.yaml> [--from build|partition|colocation] [--deploy-only] [--skip-deploy]"
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
    echo "Usage: $0 <config.yaml> [--from build|partition|colocation] [--deploy-only] [--skip-deploy]"
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

# --- Verify container is running --------------------------------------------
if ! docker compose -f "$COMPOSE_FILE" exec -T danns true 2>/dev/null; then
    echo "ERROR: danns container not running. Run: docker compose up -d"
    exit 1
fi

# --- Parse config via container python (single call) ------------------------
LAUNCH_VARS=$(docker compose -f "$COMPOSE_FILE" exec -T danns \
    python3 pipeline/config.py "$CONFIG_CONTAINER")
eval "$LAUNCH_VARS"

echo "============================================================"
echo "DANNS Data Preparation Pipeline"
echo "============================================================"
echo "  Config:       $CONFIG_BASENAME"
echo "  Experiment:   $EXPERIMENT_NAME"
echo "  Partitions:   $NUM_PARTITIONS"
echo "  Nodes:        $NODES_CSV"
if $DEPLOY_ONLY; then
    echo "  Mode:         deploy only"
elif $SKIP_DEPLOY; then
    echo "  Mode:         build only (skip deploy)"
else
    echo "  Start from:   $START_STAGE"
fi
echo "============================================================"

# --- Build + Partition + Colocation (inside container) ----------------------
if ! $DEPLOY_ONLY; then
    case "$START_STAGE" in
        build)      STAGES="build,partition,colocation" ;;
        partition)  STAGES="partition,colocation" ;;
        colocation) STAGES="colocation" ;;
        *) echo "ERROR: unknown stage: $START_STAGE"; exit 1 ;;
    esac

    echo ""
    echo "[Prepare] Running stages: $STAGES"
    echo "------------------------------------------------------------"

    # --stages expects space-separated args
    STAGES_SPACE="${STAGES//,/ }"

    docker compose -f "$COMPOSE_FILE" exec -T danns \
        python3 pipeline/run_pipeline.py \
            --config "$CONFIG_CONTAINER" \
            --stages $STAGES_SPACE

    # Fix ownership: container runs as root, host user needs write access
    docker compose -f "$COMPOSE_FILE" exec -T danns \
        chown -R "${HOST_UID:-$(id -u)}:${HOST_GID:-$(id -g)}" \
        "${RUN_ROOT_CONTAINER}/${EXPERIMENT_NAME}" 2>/dev/null || true

    echo "------------------------------------------------------------"
    echo "[Prepare] Build stages complete."
fi

# --- Deploy colocation files to DPU local SSDs ------------------------------
if ! $SKIP_DEPLOY; then
    echo ""
    echo "[Prepare] Deploying colocation files to DPU local SSDs..."

    # Find colocation output directory (relative to project root).
    # Colocation output: experiments/<exp>/colocation/
    # The actual .coloc files may be in a subdirectory (e.g., 4_partitions/<variant>/).
    # We find the directory containing .coloc files.
    COLOC_BASE="$RUN_ROOT_HOST/$EXPERIMENT_NAME/colocation"

    if [[ ! -d "$COLOC_BASE" ]]; then
        echo "ERROR: colocation directory not found: $COLOC_BASE"
        echo "  Run build+partition+colocation stages first."
        exit 1
    fi

    # Find the directory containing .coloc files (may be nested)
    COLOC_DIR=$(find "$COLOC_BASE" -name "*.coloc" -print -quit 2>/dev/null | xargs dirname 2>/dev/null || true)
    if [[ -z "$COLOC_DIR" ]]; then
        echo "ERROR: no .coloc files found under $COLOC_BASE"
        exit 1
    fi

    # Relative path from project root (for container path mapping)
    COLOC_REL="${COLOC_DIR#$RUN_ROOT_HOST/}"

    echo "  Source (NFS): $COLOC_DIR"
    echo "  Relative:     $COLOC_REL"

    IFS=',' read -ra NODE_LIST <<< "$NODES_CSV"
    for NID in "${NODE_LIST[@]}"; do
        SSH_VAR="NODE_SSH_${NID}"
        SSH_HOST="${!SSH_VAR:-}"
        SSD_VAR="NODE_LOCAL_SSD_${NID}"
        LOCAL_SSD="${!SSD_VAR:-}"

        if [[ -z "$SSH_HOST" ]]; then
            echo "  [WARN] No SSH host for node $NID, skipping"
            continue
        fi
        if [[ -z "$LOCAL_SSD" ]]; then
            echo "  [WARN] No local_ssd for node $NID, skipping deploy"
            continue
        fi

        # Deploy to local_ssd preserving relative path structure.
        # Inside container: local_ssd is mounted at /workspace/datasets_local
        # So files at local_ssd/<rel_path> are visible at /workspace/datasets_local/<rel_path>
        DEST_DIR="$LOCAL_SSD/$COLOC_REL"

        echo "  Node $NID ($SSH_HOST): → $DEST_DIR"
        ssh "$SSH_HOST" "mkdir -p '$DEST_DIR'"

        # Copy colocation files (.coloc, metadata, tag mappings)
        ssh "$SSH_HOST" "\
            cp '$COLOC_DIR'/*.coloc '$DEST_DIR/' 2>/dev/null || true; \
            cp '$COLOC_DIR'/colocation_meta.json '$DEST_DIR/' 2>/dev/null || true; \
            cp '$COLOC_DIR'/tag_to_local_*.bin '$DEST_DIR/' 2>/dev/null || true; \
            cp '$COLOC_DIR'/centroids.fbin '$DEST_DIR/' 2>/dev/null || true; \
            echo \"    $(ls '$DEST_DIR'/*.coloc 2>/dev/null | wc -l) .coloc files deployed\""
    done

    echo ""
    echo "[Prepare] Deploy complete."
    echo "  Inside DPU containers, files are at:"
    echo "    /workspace/datasets_local/$COLOC_REL/"
fi

echo ""
echo "============================================================"
echo "[Prepare] All done."
echo "============================================================"
