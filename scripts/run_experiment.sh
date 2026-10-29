#!/bin/bash
# ============================================================================
# DANNS Experiment Launcher (runs on host, outside container)
# ============================================================================
# Launches orchestrator (orchestrator-host container) + workers (DPU containers).
# No host python dependencies — all parsing done via container python.
#
# Usage (from scripts/):
#   ./run_experiment.sh configs/deep1M_test.yaml
#   ./run_experiment.sh configs/deep1M_test.yaml --stage search
#   ./run_experiment.sh configs/deep1M_test.yaml --nodes 1,2
# ============================================================================

set -euo pipefail

# Detach stdin from any TTY so background descendants (ssh, docker) don't
# SIGTTIN when invoked via `nohup ... &` from an interactive shell.
exec < /dev/null

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# DANNS_COMPOSE_FILE: 코디네이터(danns) compose 파일 오버라이드 — worktree에서
# 라이브 컨테이너를 쓰기 위함 (compose는 -f 디렉토리로 프로젝트를 해석하므로
# worktree 파일로는 기존 danns를 못 찾음). 오케스트레이터는 라이브 코드로 무방.
COMPOSE_FILE="${DANNS_COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
# DANNS_PLAIN_DOCKER=1 (2026-08-19): compose 프로젝트 밖 호스트(worker-host 우회 등)에서
# 컨테이너 이름으로 직접 exec — 기본(미설정) 동작은 기존과 동일.
if [ -n "${DANNS_PLAIN_DOCKER:-}" ]; then
    DEXEC="docker exec"
else
    DEXEC="docker compose -f $COMPOSE_FILE exec -T"
fi
EXPERIMENT_ROOT_HOST="${DANNS_EXPERIMENT_ROOT:-$PROJECT_ROOT/experiments}"
EXPERIMENT_ROOT_CONTAINER="${DANNS_EXPERIMENT_CONTAINER_ROOT:-/experiments}"
CORE_CONFIGS_ROOT="${SCRIPT_DIR}/configs"
EXPERIMENT_CONFIGS_ROOT="${EXPERIMENT_ROOT_HOST}/configs"
ANALYSIS_DIR="${EXPERIMENT_ROOT_HOST}/analysis"

# --- Defaults ---------------------------------------------------------------
STAGE="distributed_engine"
NODES_OVERRIDE=""
POSTFIX=""
CONFIG_OVERRIDES=()

# --- Parse args -------------------------------------------------------------
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)    STAGE="$2"; shift 2 ;;
        --nodes)    NODES_OVERRIDE="$2"; shift 2 ;;
        --postfix)  POSTFIX="$2"; shift 2 ;;
        --set)      CONFIG_OVERRIDES+=("$2"); shift 2 ;;
        --help|-h)
            echo "Usage: $0 <config.yaml> [--stage distributed_engine|search] [--nodes 1,2,3,4] [--postfix LABEL] [--set KEY=VALUE]"
            echo "  Config lookup order: path as given, ${DANNS_EXPERIMENT_ROOT:-experiments}/configs, scripts/configs"
            echo "  --postfix LABEL   Appended to timestamp dir: <timestamp>_<LABEL>"
            echo "  --set KEY=VALUE   Override config via dotted path; repeatable"
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
    echo "Usage: $0 <config.yaml> [--stage ...] [--nodes ...]"
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
        echo "       Searched as given, $EXPERIMENT_ROOT_HOST/$CONFIG, $EXPERIMENT_CONFIGS_ROOT/$CONFIG, $CORE_CONFIGS_ROOT/$CONFIG"
        exit 1
    fi
fi

CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
if [[ "$CONFIG_ABS" == "$EXPERIMENT_CONFIGS_ROOT"/* ]]; then
    CONFIG_REL="${CONFIG_ABS#$EXPERIMENT_CONFIGS_ROOT/}"
    CONFIG_CONTAINER="$EXPERIMENT_ROOT_CONTAINER/configs/$CONFIG_REL"
    CONFIG_LABEL="configs/$CONFIG_REL"
    RUN_ROOT_HOST="$EXPERIMENT_ROOT_HOST/runs"
    RUN_ROOT_CONTAINER="$EXPERIMENT_ROOT_CONTAINER/runs"
elif [[ "$CONFIG_ABS" == "$CORE_CONFIGS_ROOT"/* ]]; then
    CONFIG_REL="${CONFIG_ABS#$CORE_CONFIGS_ROOT/}"
    CONFIG_CONTAINER="/workspace/scripts/configs/$CONFIG_REL"
    CONFIG_LABEL="scripts/configs/$CONFIG_REL"
    RUN_ROOT_HOST="$EXPERIMENT_ROOT_HOST/runs"
    RUN_ROOT_CONTAINER="$EXPERIMENT_ROOT_CONTAINER/runs"
else
    echo "ERROR: config must live under $EXPERIMENT_CONFIGS_ROOT or $CORE_CONFIGS_ROOT"
    echo "       Got: $CONFIG_ABS"
    exit 1
fi
CONFIG_BASENAME="$CONFIG_REL"

# --- Verify container is running --------------------------------------------
if ! $DEXEC danns true 2>/dev/null; then
    echo "ERROR: danns container not running. Run: docker compose up -d"
    exit 1
fi

# --- Parse config via container python (single call) ------------------------
# config.py outputs bash-sourceable key=value lines
CONFIG_SET_ARGS=()
for kv in "${CONFIG_OVERRIDES[@]}"; do
    CONFIG_SET_ARGS+=(--set "$kv")
done

# Auto-inject results.base_dir so results always land in results/<dataset>/<label>_<ts>/
# (clean split-results layout) unless the caller has already set a results path.
_has_results_override=0
for _kv in "${CONFIG_OVERRIDES[@]}"; do
    case "$_kv" in results.*|output.results_base_dir*|output.run_base_dir*) _has_results_override=1; break ;; esac
done
if [[ "$_has_results_override" == "0" ]]; then
    CONFIG_OVERRIDES+=("results.base_dir=${EXPERIMENT_ROOT_CONTAINER}/results")
    CONFIG_SET_ARGS+=(--set "results.base_dir=${EXPERIMENT_ROOT_CONTAINER}/results")
fi

LAUNCH_VARS=$($DEXEC danns \
    python3 pipeline/config.py "$CONFIG_CONTAINER" "${CONFIG_SET_ARGS[@]}")

# Source the variables: ORCH_HOST, ORCH_PORT, EXPERIMENT_NAME, NODES_CSV, NODE_SSH_*
eval "$LAUNCH_VARS"

# The orchestrator process is launched in the local container below.
# Refuse to run from a different host than cluster.orchestrator.host; otherwise
# workers wait on ORCH_HOST while the listener is bound on this machine.
LOCAL_HOST="$(hostname -s | tr '[:upper:]' '[:lower:]')"
ORCH_HOST_NORM="$(echo "$ORCH_HOST" | tr '[:upper:]' '[:lower:]')"
if [[ "${DANNS_SKIP_ORCH_HOST_CHECK:-0}" != "1" && "$LOCAL_HOST" != "$ORCH_HOST_NORM" ]]; then
    echo "ERROR: config orchestrator.host is '$ORCH_HOST', but this launcher is running on '$LOCAL_HOST'."
    echo "       Run this command on '$ORCH_HOST', or set DANNS_SKIP_ORCH_HOST_CHECK=1 if this host is intentionally equivalent."
    exit 1
fi

# Apply node override if specified
if [[ -n "$NODES_OVERRIDE" ]]; then
    NODES_CSV="$NODES_OVERRIDE"
fi

# Build NODE_SSH associative array from NODE_SSH_* variables
declare -A NODE_SSH
IFS=',' read -ra NODE_LIST <<< "$NODES_CSV"
for NID in "${NODE_LIST[@]}"; do
    varname="NODE_SSH_${NID}"
    NODE_SSH[$NID]="${!varname:-}"
done

# --- Timestamp ---------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$POSTFIX" ]]; then
    TIMESTAMP="${TIMESTAMP}_${POSTFIX}"
fi

container_to_host_path() {
    local path="$1"
    if [[ "$path" == "$EXPERIMENT_ROOT_CONTAINER"/* ]]; then
        printf "%s/%s\n" "$EXPERIMENT_ROOT_HOST" "${path#$EXPERIMENT_ROOT_CONTAINER/}"
    elif [[ "$path" == "$RUN_ROOT_CONTAINER"/* ]]; then
        printf "%s/%s\n" "$RUN_ROOT_HOST" "${path#$RUN_ROOT_CONTAINER/}"
    else
        printf "%s\n" "$path"
    fi
}

# --- Log dir (host side) — matches runtime run_dir in config ----------------
if [[ "${SPLIT_RESULTS:-0}" == "1" ]]; then
    RUN_ROOT_CONTAINER="${RESULTS_ROOT_CONTAINER%/}"
    RUN_ROOT_HOST="$(container_to_host_path "$RUN_ROOT_CONTAINER")"
    RUN_DIR="${RUN_ROOT_HOST}/${DATASET_ID}/${EXPERIMENT_LABEL}/${TIMESTAMP}"
    CONTAINER_RUN_DIR="${RUN_ROOT_CONTAINER}/${DATASET_ID}/${EXPERIMENT_LABEL}/${TIMESTAMP}"
else
    RUN_DIR="${RUN_ROOT_HOST}/${EXPERIMENT_NAME}/${STAGE}/${TIMESTAMP}"
    CONTAINER_RUN_DIR="${RUN_ROOT_CONTAINER}/${EXPERIMENT_NAME}/${STAGE}/${TIMESTAMP}"
fi
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "DANNS Experiment Launcher"
echo "============================================================"
echo "  Config:       $CONFIG_LABEL"
echo "  Stage:        $STAGE"
echo "  Nodes:        $NODES_CSV"
echo "  Timestamp:    $TIMESTAMP"
echo "  Dataset:      ${DATASET_ID:-"(legacy)"}"
echo "  Run label:    ${EXPERIMENT_LABEL:-$EXPERIMENT_NAME}"
echo "  Postfix:      ${POSTFIX:-"(none)"}"
if [[ ${#CONFIG_OVERRIDES[@]} -gt 0 ]]; then
    printf "  Overrides:   "
    printf "%s " "${CONFIG_OVERRIDES[@]}"
    printf "\n"
else
    echo "  Overrides:   (none)"
fi
echo "  Logs:         $LOG_DIR"
echo "  Orchestrator: $ORCH_HOST (port $ORCH_PORT)"
echo "============================================================"

# --- Cleanup on exit ---------------------------------------------------------
PIDS=()
cleanup() {
    echo ""
    echo "[Launcher] Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    # Sweep stale DPU/worker container processes belonging to THIS run only.
    # Filters by $TIMESTAMP — guaranteed unique per run, present in every cmdline.
    if [[ -n "${TIMESTAMP:-}" ]]; then
        IFS=',' read -ra _NODE_LIST <<< "${NODES_CSV:-1,2,3,4}"
        for n in "${_NODE_LIST[@]}"; do
            # timeout + ConnectTimeout: don't let a stale ssh ControlSocket
            # or unresponsive host hang the launcher forever (we've already
            # written the result files at this point — cleanup is best-effort).
            timeout 15 ssh -n -T \
                -o BatchMode=yes \
                -o ConnectTimeout=5 \
                -o ServerAliveInterval=3 \
                -o ServerAliveCountMax=2 \
                "dpu$n" \
                "docker exec danns pkill -TERM -f \"$TIMESTAMP\" 2>/dev/null; \
                 sleep 1; \
                 docker exec danns pkill -KILL -f \"$TIMESTAMP\" 2>/dev/null" \
                </dev/null >/dev/null 2>&1 &
        done
        $DEXEC danns \
            bash -c "pkill -TERM -f \"$TIMESTAMP\" 2>/dev/null; sleep 1; pkill -KILL -f \"$TIMESTAMP\" 2>/dev/null" \
            </dev/null >/dev/null 2>&1 || true
        wait 2>/dev/null
    fi
    echo "[Launcher] Done."
}
trap cleanup EXIT INT TERM

# --- Start orchestrator (orchestrator-host container, background) -----------------------
echo "[Launcher] Starting orchestrator in orchestrator-host container..."

$DEXEC -e PYTHONUNBUFFERED=1 danns \
    python3 pipeline/orchestrator.py \
        --config "$CONFIG_CONTAINER" \
        --nodes "$NODES_CSV" \
        --stage "$STAGE" \
        --timestamp "$TIMESTAMP" \
        --project-root /workspace \
        "${CONFIG_SET_ARGS[@]}" \
    > "$LOG_DIR/orchestrator.log" 2>&1 &
ORCH_PID=$!
PIDS+=($ORCH_PID)
echo "[Launcher] Orchestrator PID=$ORCH_PID"

# Give orchestrator a moment to start listening
sleep 2

# --- Start workers (DPU containers via SSH, background) ----------------------
for NID in "${NODE_LIST[@]}"; do
    SSH_HOST="${NODE_SSH[$NID]:-}"
    if [[ -z "$SSH_HOST" ]]; then
        echo "[Launcher] WARNING: No SSH host for node $NID, skipping"
        continue
    fi

    echo "[Launcher] Starting worker on $SSH_HOST (node $NID)..."

    # Forward optional pipeline env vars into the container (docker exec
    # strips host env unless explicitly passed).
    PMU_FWD=""
    [[ -n "${PIPELINE_PMU:-}" ]] && PMU_FWD="PIPELINE_PMU=${PIPELINE_PMU} "
    [[ -n "${DANNS_PIN_NIC_NUMA:-}" ]] && PMU_FWD+="DANNS_PIN_NIC_NUMA=${DANNS_PIN_NIC_NUMA} "
    [[ -n "${DANNS_PERDI_CACHE:-}" ]] && PMU_FWD+="DANNS_PERDI_CACHE=${DANNS_PERDI_CACHE} "
    [[ -n "${DANNS_CONSOL_DI_ORDER:-}" ]] && PMU_FWD+="DANNS_CONSOL_DI_ORDER=${DANNS_CONSOL_DI_ORDER} "
    [[ -n "${DANNS_CACHE_DUMP:-}" ]] && PMU_FWD+="DANNS_CACHE_DUMP=${DANNS_CACHE_DUMP} "
    [[ -n "${DANNS_CACHE_LOAD:-}" ]] && PMU_FWD+="DANNS_CACHE_LOAD=${DANNS_CACHE_LOAD} "
    [[ -n "${DANNS_CONSOL_PACE_GBPS:-}" ]] && PMU_FWD+="DANNS_CONSOL_PACE_GBPS=${DANNS_CONSOL_PACE_GBPS} "
    [[ -n "${DANNS_SEARCH_WARMER_OFF:-}" ]] && PMU_FWD+="DANNS_SEARCH_WARMER_OFF=${DANNS_SEARCH_WARMER_OFF} "
    [[ -n "${DANNS_CACHE_NUMA_NODE:-}" ]] && PMU_FWD+="DANNS_CACHE_NUMA_NODE=${DANNS_CACHE_NUMA_NODE} "
    [[ -n "${DANNS_CONCURRENT_QUIET:-}" ]] && PMU_FWD+="DANNS_CONCURRENT_QUIET=${DANNS_CONCURRENT_QUIET} "
    [[ -n "${DANNS_CONCURRENT_QP_TOTAL:-}" ]] && PMU_FWD+="DANNS_CONCURRENT_QP_TOTAL=${DANNS_CONCURRENT_QP_TOTAL} "
    [[ -n "${DANNS_STORE_NUMA_NODE:-}" ]] && PMU_FWD+="DANNS_STORE_NUMA_NODE=${DANNS_STORE_NUMA_NODE} "
    # glibc allocator knobs (thread-count-cliff diagnosis: arena locks / >128KB
    # mmap-path churn under the process-wide mmap_lock)
    for _mv in MALLOC_ARENA_MAX MALLOC_MMAP_THRESHOLD_ MALLOC_TRIM_THRESHOLD_; do
        [[ -n "${!_mv:-}" ]] && PMU_FWD+="${_mv}=${!_mv} "
    done
    # OpenMP wait-policy knobs (P1 item 5: join-barrier / between-L spin —
    # e.g. OMP_WAIT_POLICY=passive or KMP_BLOCKTIME=0 for libomp,
    # GOMP_SPINCOUNT for libgomp). Forwarded only when set on the launcher.
    # Thread<->core affinity (P1b): the worker process is already confined to the
    # NIC socket via DANNS_PIN_NIC_NUMA, but its OpenMP threads still migrate
    # inside that set, which evicts the thread_local SearchScratch from L1/L2.
    # OMP_SCHEDULE only takes effect where the pragma says schedule(runtime).
    for _mv in OMP_PROC_BIND OMP_PLACES OMP_SCHEDULE; do
        [[ -n "${!_mv:-}" ]] && PMU_FWD+="${_mv}=${!_mv} "
    done
    for _mv in OMP_WAIT_POLICY KMP_BLOCKTIME GOMP_SPINCOUNT; do
        [[ -n "${!_mv:-}" ]] && PMU_FWD+="${_mv}=${!_mv} "
    done
    # DANNS_WORKSPACE: 워커가 사용할 in-container 코드 루트 (기본 /workspace = 라이브).
    # worktree A/B: DANNS_WORKSPACE=/workspace/.claude/worktrees/p1-hotpath 로 지정하면
    # 워커가 그 트리의 pipeline/ + lib/*.so 를 사용 (repo 마운트 안이라 추가 마운트 불필요).
    WORKSPACE_C="${DANNS_WORKSPACE:-/workspace}"
    WORKER_CMD="cd $WORKSPACE_C && ${PMU_FWD}DANNS_EXPERIMENT_CONTAINER_ROOT=${EXPERIMENT_ROOT_CONTAINER} PYTHONUNBUFFERED=1 python3 pipeline/node_worker.py \
        --config $CONFIG_CONTAINER \
        --node-id $NID \
        --nodes $NODES_CSV \
        --coordinator 0 \
        --orchestrator-host $ORCH_HOST \
        --orchestrator-port $ORCH_PORT \
        --stage $STAGE \
        --timestamp $TIMESTAMP \
        --project-root $WORKSPACE_C"

    for kv in "${CONFIG_OVERRIDES[@]}"; do
        printf -v _quoted_kv '%q' "$kv"
        WORKER_CMD+=" --set $_quoted_kv"
    done

    # -n: redirect stdin from /dev/null so bg ssh doesn't SIGTTIN
    # -T -o BatchMode=yes: no TTY, non-interactive (no password/key prompt hangs)
    ssh -n -T -o BatchMode=yes "$SSH_HOST" \
        "docker exec -i danns bash -c '$WORKER_CMD'" \
        < /dev/null > "$LOG_DIR/worker_node${NID}.log" 2>&1 &
    WORKER_PID=$!
    PIDS+=($WORKER_PID)
    echo "[Launcher] Worker node $NID PID=$WORKER_PID ($SSH_HOST)"
done

echo ""
echo "[Launcher] All processes started. Tailing logs..."
echo "           Press Ctrl-C to stop."
echo "============================================================"

# --- Tail all logs -----------------------------------------------------------
tail -F "$LOG_DIR"/*.log &
TAIL_PID=$!
PIDS+=($TAIL_PID)

# Wait for orchestrator to finish
wait $ORCH_PID 2>/dev/null
ORCH_EXIT=$?

sleep 3

echo ""

# --- Fix file ownership (container creates as root) -------------------------
echo "[Launcher] Fixing file ownership..."
$DEXEC danns \
    chown -R "${HOST_UID:-10011}:${HOST_GID:-10000}" \
    "${CONTAINER_RUN_DIR}" \
    2>/dev/null || true

# --- Optional recall and results.csv ----------------------------------------
# Recall is opt-in. Set RUN_RECALL_POSTHOC=1 to compute post-hoc recall after
# the engine finishes, or set search.in_run_recall: true for in-engine recall.
# Search results are still saved and can be evaluated later with run_recall.sh.
if [[ $ORCH_EXIT -eq 0 ]] && [[ -d "$RUN_DIR/node_stats" ]] && \
   [[ "${RUN_RECALL_POSTHOC:-0}" == "1" ]] && \
   [[ "${SKIP_RECALL_POSTHOC:-0}" != "1" ]]; then
    CONTAINER_RUN="${CONTAINER_RUN_DIR}"
    echo "[Launcher] Computing recall_posthoc (RUN_RECALL_POSTHOC=1)..."
    $DEXEC -e PYTHONUNBUFFERED=1 danns \
        python3 pipeline/compute_recall_posthoc.py \
        "$CONTAINER_RUN" 2>&1 | tail -3 || true
elif [[ "${RUN_RECALL_POSTHOC:-0}" != "1" ]]; then
    echo "[Launcher] Skipping recall_posthoc (not requested; set RUN_RECALL_POSTHOC=1 to enable)"
elif [[ "${SKIP_RECALL_POSTHOC:-0}" == "1" ]]; then
    echo "[Launcher] Skipping recall_posthoc (SKIP_RECALL_POSTHOC=1)"
fi

if [[ -f "$RUN_DIR/recall_posthoc.json" ]]; then
    echo "[Launcher] Appending RECALL SUMMARY to engine_report.txt..."
    python3 -c "
import json, sys
rd = '$RUN_DIR'
r = json.load(open(f'{rd}/recall_posthoc.json'))
lines = ['', '=' * 88, '  RECALL SUMMARY (all seqs, posthoc)', '=' * 88]
for op in r.get('search_ops', []):
    seq = op['seq']
    L_map = op.get('recall_by_L', {})
    if not L_map: continue
    meta = []
    for k in ('num_threads', 'pq_alpha', 'batch_size'):
        v = op.get(k)
        if v is not None and v != '':
            meta.append(f'{k}={v}')
    meta_s = ', ' + ', '.join(meta) if meta else ''
    lines.append(f\"  seq={seq} (N={op.get('num_active')}, K={op.get('K')}, Q={op.get('num_queries')}{meta_s}):\")
    for L, v in sorted(L_map.items(), key=lambda kv: int(kv[0])):
        lines.append(f'    L={L}: {float(v):.4f}')
with open(f'{rd}/engine_report.txt', 'a') as f:
    f.write('\n' + '\n'.join(lines) + '\n')
" 2>&1 | tail -3 || true
fi

echo "[Launcher] Generating results.csv..."
python3 "${ANALYSIS_DIR}/stats_to_csv.py" "$RUN_DIR" \
    -o "$RUN_DIR/results.csv" 2>&1 | tail -3 || true

# --- Append row to runs.tsv (central experiment log) ------------------------
RUNS_TSV="${RUN_ROOT_HOST}/runs.tsv"
if [[ ! -f "$RUNS_TSV" ]]; then
    printf "timestamp\tdataset\tcategory\tconfig_name\trun_dir\tstatus\tnum_seqs\tbest_qps\tbest_recall\tnotes\n" > "$RUNS_TSV"
fi

if [[ $ORCH_EXIT -eq 0 ]]; then
    RUN_STATUS="OK"
else
    RUN_STATUS="FAILED"
fi

# Parse dataset/category from config path (e.g. vldb/deep100M/solution/foo.yaml)
DATASET_TAG=""
CATEGORY_TAG=""
CONFIG_NAME="$(basename "$CONFIG_REL" .yaml)"
IFS='/' read -ra CFG_PARTS <<< "$CONFIG_REL"
if [[ ${#CFG_PARTS[@]} -ge 3 ]] && [[ "${CFG_PARTS[0]}" == "vldb" ]]; then
    DATASET_TAG="${CFG_PARTS[1]}"
    CATEGORY_TAG="${CFG_PARTS[2]}"
fi

# Extract best_qps / best_recall / num_seqs from results.csv and engine_report
BEST_QPS=""
BEST_RECALL=""
NUM_SEQS=""
if [[ -f "$RUN_DIR/results.csv" ]]; then
    BEST_QPS=$(python3 -c "
import csv, sys
try:
    with open('$RUN_DIR/results.csv') as f:
        rows = list(csv.DictReader(f))
    if not rows: sys.exit()
    # SUM-OF-PER-NODE convention (2026-07-28, user decision): cluster qps at
    # (seq,L) = sum of per-node qps. Wall-clock (total queries / slowest node)
    # is still computed below for reference.
    #
    # Note when comparing against baselines: CoTra and RED-ANNS barrier-sync
    # their ranks, so their sum IS their wall-clock -- switching conventions
    # moves only our number, by 1.11-1.21x at L=100. State that in any table.
    from collections import defaultdict
    agg = defaultdict(lambda: {'q':0.0,'t':0.0,'s':0.0})
    for r in rows:
        try:
            a = agg[(r['seq'], r['L'])]
            qp = float(r.get('qps') or 0); nq = float(r.get('num_queries') or 0)
            el = float(r.get('elapsed_s') or 0)
            nt = el if el > 0 else (nq/qp if qp > 0 else 0.0)
            a['q'] += nq; a['t'] = max(a['t'], nt); a['s'] += qp
        except Exception: pass
    vals = [a['s'] if a['s'] > 0 else (a['q']/a['t'] if a['q']>0 and a['t']>0 else 0) for a in agg.values()]
    if vals: print(f'{max(vals):.1f}')
except Exception: pass" 2>/dev/null)
    BEST_RECALL=$(python3 -c "
import csv, sys
try:
    with open('$RUN_DIR/results.csv') as f:
        rows = list(csv.DictReader(f))
    vals = [float(r['recall_10']) for r in rows if r.get('recall_10') not in (None,'')]
    if vals: print(f'{max(vals):.4f}')
except Exception: pass" 2>/dev/null)
    NUM_SEQS=$(python3 -c "
import csv
try:
    with open('$RUN_DIR/results.csv') as f:
        seqs = {r['seq'] for r in csv.DictReader(f)}
    print(len(seqs))
except Exception: pass" 2>/dev/null)
fi

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$TIMESTAMP" \
    "$DATASET_TAG" \
    "$CATEGORY_TAG" \
    "$CONFIG_NAME" \
    "${RUN_DIR#$EXPERIMENT_ROOT_HOST/}" \
    "$RUN_STATUS" \
    "${NUM_SEQS:-}" \
    "${BEST_QPS:-}" \
    "${BEST_RECALL:-}" \
    "${POSTFIX:-}" \
    >> "$RUNS_TSV"


exit $ORCH_EXIT
