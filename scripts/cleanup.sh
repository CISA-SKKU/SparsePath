#!/bin/bash
# cleanup.sh — Kill leftover orchestrator / worker / monitor processes.
#
# Note on monitors: rdma_monitor.sh / nic_bw_monitor.py / pmu_monitor.sh
# are spawned by engine.py / node_worker.py for the duration of an op.
# If their parent dies or is killed abruptly, the samplers get orphaned
# and keep reading /sys counters. Always cleanup monitors alongside workers.
set -e

echo "=== Cleaning up DANNS processes ==="

# Monitor script patterns — extend this list when adding new samplers.
MONITOR_PATTERNS=(
    "rdma_monitor.sh"
    "nic_bw_monitor.py"
    "pmu_monitor.sh"
)

# ---------------------------------------------------------------------------
# Local (orchestrator-host) container
# ---------------------------------------------------------------------------
echo "[orchestrator-host] Killing orchestrator/worker/monitor in danns container..."
docker exec danns pkill -f orchestrator 2>/dev/null || true
docker exec danns pkill -f node_worker 2>/dev/null || true
for pat in "${MONITOR_PATTERNS[@]}"; do
    docker exec danns pkill -f "$pat" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# DPU nodes — check BOTH inside container and on the host. Monitor scripts
# that read /sys counters are usually spawned from inside the danns
# container (which runs as root), so when the container's parent dies they
# get orphaned to init on the DPU host — STILL AS ROOT. Plain `pkill` by a
# non-root user can't kill those; we must use `sudo`. Try passwordless
# first, and if that fails tell the user to rerun the command interactively.
# ---------------------------------------------------------------------------
for node in ${DANNS_NODES:-}; do
    echo "[$node] Killing node_worker + monitors..."
    ssh "$node" "docker exec danns pkill -f node_worker 2>/dev/null" 2>/dev/null || true
    for pat in "${MONITOR_PATTERNS[@]}"; do
        # 1. In-container (rarely, but cheap)
        ssh "$node" "docker exec danns pkill -f '$pat' 2>/dev/null" 2>/dev/null || true
        # 2. Host-level same-user
        ssh "$node" "pkill -f '$pat' 2>/dev/null" 2>/dev/null || true
        # 3. Host-level root orphans (need sudo). Try passwordless first.
        if ! ssh "$node" "sudo -n pkill -9 -f '$pat' 2>/dev/null" 2>/dev/null; then
            # Passwordless sudo not configured — probe if anything is still there,
            # and if so emit the one-liner the user should run interactively.
            alive=$(ssh "$node" "ps -ef | grep -E '$pat' | grep -v grep | wc -l" 2>/dev/null || echo 0)
            if [ "${alive:-0}" -gt 0 ]; then
                echo "  [WARN] $node has $alive root-owned '$pat' orphan(s) — passwordless sudo not set."
                echo "         Run once interactively:   ssh -t $node 'sudo pkill -9 -f $pat'"
            fi
        fi
    done
done

# ---------------------------------------------------------------------------
# Host-level on orchestrator-host (processes started outside containers)
# ---------------------------------------------------------------------------
pkill -f "orchestrator.py.*distributed_engine" 2>/dev/null || true
pkill -f "node_worker.py" 2>/dev/null || true
for pat in "${MONITOR_PATTERNS[@]}"; do
    pkill -f "$pat" 2>/dev/null || true
done

sleep 1

# ---------------------------------------------------------------------------
# Verify — include monitors in the survey
# ---------------------------------------------------------------------------
survey_pat="node_worker|orchestrator.py|rdma_monitor|nic_bw_monitor|pmu_monitor"
remaining=$(ps aux | grep -E "$survey_pat" | grep -v grep | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "=== orchestrator-host clean ==="
else
    echo "=== WARNING: $remaining processes still running on orchestrator-host ==="
    ps aux | grep -E "$survey_pat" | grep -v grep
    echo "Try: kill -9 <pid>"
fi

# Also survey the DPUs so orphans don't silently pollute future probes.
echo ""
echo "=== DPU monitor survey ==="
for node in ${DANNS_NODES:-}; do
    count=$(ssh "$node" "ps aux | grep -E 'rdma_monitor|nic_bw_monitor|pmu_monitor' | grep -v grep | wc -l" 2>/dev/null || echo "?")
    echo "  $node: $count monitor process(es) still alive"
done
