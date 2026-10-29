#!/bin/bash
# ============================================================================
# pmu_monitor.sh - Intel PMU counter collection for DDIO / memory subsystem.
#
# Mirrors rdma_monitor.sh API (snapshot / diff / continuous) so pipeline
# integration stays consistent. The actual collection is a thin wrapper
# around `perf stat`.
#
# Usage:
#   pmu_monitor.sh discover                          # List available events on this host
#   pmu_monitor.sh snapshot <output_file> [secs]     # One-shot aggregate (default 1s window)
#   pmu_monitor.sh diff <before_file> <after_file>   # Human-readable delta summary
#   pmu_monitor.sh continuous <output_file> <ms>     # perf-stat -I sampling (ctrl-C to stop)
#   pmu_monitor.sh window <pid> <output_file>        # Attach to PID until it exits
#
# What we measure (see comments per block below):
#   - Basic:    cycles, instructions              → IPC
#   - Cache:    LLC loads/misses, l3_hit/l3_miss  → LLC hit rate for demand loads
#   - DRAM:     uncore_imc  cas_count_read/write  → DRAM bytes (64B line)
#   - DDIO:     uncore_iio  PCIItoM / ItoM        → NIC→LLC DMA write bytes (DDIO traffic)
#
# Uncore events require root + CAP_PERFMON (or /proc/sys/kernel/perf_event_paranoid <= 0).
# Event names differ across Xeon gens — `discover` shows what this host has.
#
# ============================================================================

set -uo pipefail

# Sudo is only required when perf_event_paranoid > 0. On hosts with
# paranoid = -1 or 0, uncore events are readable by regular users.
_perf_prefix() {
    local par=2
    [[ -r /proc/sys/kernel/perf_event_paranoid ]] && \
        par=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 2)
    if [[ "$par" -le 0 ]]; then
        echo ""
    else
        echo "sudo"
    fi
}

# ---------------------------------------------------------------------------
# Event sets (override via env vars PMU_EVENTS_EXTRA / PMU_EVENTS_OVERRIDE)
# ---------------------------------------------------------------------------
#
# Ice Lake-SP baseline (Xeon Gold 6338/6338N). These symbolic names usually
# resolve via `perf list`. If `discover` reports a different name on your
# host, set PMU_EVENTS_OVERRIDE to a comma-separated list and re-run.

EVENTS_BASIC=(
    cycles
    instructions
)

EVENTS_CACHE=(
    # All-demand LLC (sum over cores)
    LLC-loads
    LLC-load-misses
    # Per-load attribution (needs CAP_PERFMON)
    mem_load_retired.l3_hit
    mem_load_retired.l3_miss
)

# DRAM read/write bytes via integrated memory controller.
# 1 increment == 64-byte cacheline transfer. Event is per-IMC/per-channel
# on Ice Lake; the glob handles multiple units.
EVENTS_DRAM=(
    'uncore_imc/cas_count_read/'
    'uncore_imc/cas_count_write/'
)

# DDIO (NIC -> CPU DMA).
# On Ice Lake-SP the programmable event is `unc_iio_data_req_of_cpu.mem_write.part*`
# (NIC-originated writes to CPU memory; this is DDIO-allocating traffic when
# DDIO is on). `.part0..part7` are the 8 lanes of the IIO stack; for BlueField-2
# at PCIe x16 we sum across parts 0-3 on the stack that owns the NIC bus.
# Stack index is socket-dependent — use `pmu_monitor.sh nic_stack` to find
# which uncore_iio_N owns the NIC bus on this host.
#
# We leave events unqualified (`unc_iio_data_req_of_cpu.mem_write.partN`);
# perf expands each to all uncore_iio_* units, and the diff aggregator sums
# them. If you want to isolate a single socket's NIC, use PMU_EVENTS_OVERRIDE
# with a fully-qualified name like `uncore_iio_0/unc_iio_data_req_of_cpu.mem_write.part0/`.
#
# `bw_in_port*` on `uncore_iio_free_running_*` is an alternative signal —
# coarser (bytes across all traffic types per port) but always available and
# does not need programmable counter slots.
EVENTS_IIO=(
    'unc_iio_data_req_of_cpu.mem_write.part0'
    'unc_iio_data_req_of_cpu.mem_write.part1'
    'unc_iio_data_req_of_cpu.mem_write.part2'
    'unc_iio_data_req_of_cpu.mem_write.part3'
    'unc_iio_data_req_of_cpu.mem_read.part0'
    'unc_iio_data_req_of_cpu.mem_read.part1'
    'unc_iio_data_req_of_cpu.mem_read.part2'
    'unc_iio_data_req_of_cpu.mem_read.part3'
)

# ---------------------------------------------------------------------------
# Host discovery
# ---------------------------------------------------------------------------
do_discover() {
    command -v perf >/dev/null 2>&1 || {
        echo "perf not found. apt install linux-tools-$(uname -r) linux-tools-generic"
        exit 1
    }

    echo "=== CPU ==="
    grep "model name" /proc/cpuinfo | head -1
    echo ""
    echo "=== perf_event_paranoid ==="
    cat /proc/sys/kernel/perf_event_paranoid
    echo "  (uncore events typically need <= 0; set via"
    echo "   echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid)"
    echo ""
    echo "=== Uncore IMC (DRAM BW) ==="
    perf list 2>/dev/null | grep -iE 'uncore_imc.*(cas_count|mem_bw|DRAM)' | head -10
    echo ""
    echo "=== Uncore IIO (DDIO) — look for ItoM / PCIItoM / mem_write ==="
    perf list 2>/dev/null | grep -iE 'uncore_iio' | head -20
    echo ""
    echo "=== Memory-load PMU events (LLC hit/miss attribution) ==="
    perf list 2>/dev/null | grep -iE 'mem_load_retired' | head -10
    echo ""
    echo "=== Sanity test (1 second) ==="
    sudo perf stat -a -e cycles,instructions,LLC-loads,LLC-load-misses sleep 1 2>&1 | tail -10
}

# ---------------------------------------------------------------------------
# Event list assembly (skips events not available on this host)
# ---------------------------------------------------------------------------
_available_events() {
    # If user forced an override, trust it.
    if [[ -n "${PMU_EVENTS_OVERRIDE:-}" ]]; then
        echo "$PMU_EVENTS_OVERRIDE"
        return
    fi

    # Probe each event by actually trying to open it with a zero-length
    # workload. `perf list` is unreliable in some containers (output may
    # be empty or formatted differently than the canonical event name).
    # `perf stat --null` opens the event but skips counter output, so it's
    # fast (~30-50ms per event). With 16 events ≈ 0.5s one-time setup cost.
    local prefix
    prefix=$(_perf_prefix)
    _evt_ok() {
        local ev="$1"
        $prefix perf stat --null -a -e "$ev" -- true 2>/dev/null
    }

    local kept=()
    for ev in "${EVENTS_BASIC[@]}" "${EVENTS_CACHE[@]}" \
              "${EVENTS_DRAM[@]}" "${EVENTS_IIO[@]}"; do
        if _evt_ok "$ev"; then
            kept+=("$ev")
        fi
    done
    if [[ -n "${PMU_EVENTS_EXTRA:-}" ]]; then
        kept+=("$PMU_EVENTS_EXTRA")
    fi

    if [[ ${#kept[@]} -eq 0 ]]; then
        echo "cycles,instructions"   # ultimate fallback
        return
    fi
    local joined
    joined=$(printf "%s," "${kept[@]}")
    echo "${joined%,}"
}

# ---------------------------------------------------------------------------
# snapshot: short aggregate measurement (one line per event -> key=value file)
# ---------------------------------------------------------------------------
do_snapshot() {
    local outfile="$1" secs="${2:-1}"
    local events
    events=$(_available_events)

    local tmp
    tmp=$(mktemp)

    $(_perf_prefix) perf stat -a -x , -e "$events" -o "$tmp" sleep "$secs" 2>/dev/null

    {
        echo "timestamp=$(date +%s.%N)"
        echo "duration_sec=${secs}"
        # perf -x , format:  value,unit,event,run_time_ns,pct_enabled,...
        awk -F, '
            /^[[:space:]]*#/ { next }
            NF >= 3 && $3 != "" {
                # $1=value (possibly <not counted>), $3=event name
                val=$1; ev=$3;
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", val);
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", ev);
                if (val == "<not counted>" || val == "<not supported>") val = "0";
                # Sanitize event name for key (replace / and = with _)
                gsub(/[\/=,]/, "_", ev);
                print ev "=" val;
            }
        ' "$tmp"
    } > "$outfile"
    rm -f "$tmp"
}

# ---------------------------------------------------------------------------
# diff: compute derived metrics from two snapshots.
# Snapshots from do_snapshot already contain deltas (perf aggregated over the
# window). We treat them directly as per-window counts.
# ---------------------------------------------------------------------------
do_diff() {
    local before_file="$1" after_file="$2"

    # For PMU, snapshots are already window-aggregated rather than cumulative,
    # so we just present the 'after' values plus derived ratios.
    declare -A after
    while IFS='=' read -r key val; do
        after["$key"]="$val"
    done < "$after_file"

    local dur="${after[duration_sec]:-1}"
    # Values may be floats (perf auto-scales uncore_imc to MiB via .scale file).
    # Allow both integers and decimals; reject anything else.
    get() { local k="$1"; local v="${after[$k]:-0}"; [[ "$v" =~ ^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]] || v=0; echo "$v"; }
    # Float-safe accumulate: sum = awk "BEGIN{printf \"%.6f\", a+b}"
    fadd() { awk "BEGIN{printf \"%.6f\", ($1)+($2)}"; }

    local cycles instr llcl llcm l3h l3m imc_r imc_w iio_mw iio_mr
    cycles=$(get cycles)
    instr=$(get instructions)
    llcl=$(get LLC-loads)
    llcm=$(get LLC-load-misses)
    l3h=$(get mem_load_retired.l3_hit)
    l3m=$(get mem_load_retired.l3_miss)
    # IMC is per-unit; key names look like uncore_imc_0_cas_count_read_ -> sum them.
    # IIO events resolve per-stack; perf emits names like
    #   unc_iio_data_req_of_cpu.mem_write.part2   (aggregated across stacks)
    # or per-stack uncore_iio_0_unc_iio_data_req_of_cpu.mem_write.part2_
    # Count is in 4B units per the Ice Lake uncore spec — we convert to bytes.
    imc_r=0    # MiB (perf auto-scaled)
    imc_w=0    # MiB (perf auto-scaled)
    iio_mw=0   # NIC -> CPU writes (4B request count)
    iio_mr=0   # NIC <- CPU reads  (4B request count)
    for k in "${!after[@]}"; do
        v=$(get "$k")
        case "$k" in
            uncore_imc*cas_count_read*)  imc_r=$(fadd "$imc_r" "$v") ;;
            uncore_imc*cas_count_write*) imc_w=$(fadd "$imc_w" "$v") ;;
            *unc_iio_data_req_of_cpu.mem_write*|*nic_to_llc*|*PCIItoM*|*ItoM*)
                iio_mw=$(fadd "$iio_mw" "$v") ;;
            *unc_iio_data_req_of_cpu.mem_read*)
                iio_mr=$(fadd "$iio_mr" "$v") ;;
        esac
    done

    # Derived metrics
    local ipc=0 llc_hit_rate=0 l3_hit_rate=0
    local dram_read_gbps=0 dram_write_gbps=0
    local iio_write_gbps=0 iio_read_gbps=0
    local ddio_absorbed_pct=0
    if [[ $cycles -gt 0 ]]; then
        ipc=$(awk "BEGIN{printf \"%.3f\", $instr/$cycles}")
    fi
    if [[ $(( llcl + 1 )) -gt 1 ]]; then
        llc_hit_rate=$(awk "BEGIN{printf \"%.1f\", ($llcl - $llcm)/$llcl*100}")
    fi
    local l3tot=$(( l3h + l3m ))
    if [[ $l3tot -gt 0 ]]; then
        l3_hit_rate=$(awk "BEGIN{printf \"%.1f\", $l3h/$l3tot*100}")
    fi
    # imc_r/imc_w are perf-scaled MiB values -> bytes = MiB * 2^20
    dram_read_gbps=$(awk "BEGIN{printf \"%.2f\", ($imc_r)*1048576/($dur)/1e9}")
    dram_write_gbps=$(awk "BEGIN{printf \"%.2f\", ($imc_w)*1048576/($dur)/1e9}")
    # IIO data_req_of_cpu counters are 4-byte element counts.
    iio_write_gbps=$(awk "BEGIN{printf \"%.3f\", ($iio_mw)*4/($dur)/1e9}")
    iio_read_gbps=$(awk "BEGIN{printf \"%.3f\", ($iio_mr)*4/($dur)/1e9}")
    # DDIO absorption: fraction of NIC-inbound writes NOT reaching DRAM.
    # Caveat: IMC write includes ALL sources (NIC + CPU dirty eviction).
    # Best used as a DELTA across idle/under-load snapshots.
    ddio_absorbed_pct=$(awk "BEGIN{
        iio_b = ($iio_mw) * 4;
        imc_b = ($imc_w) * 1048576;
        if (iio_b <= 0) { print \"0.0\"; exit }
        a = (iio_b - imc_b) / iio_b * 100;
        if (a < 0) a = 0; if (a > 100) a = 100;
        printf \"%.1f\", a
    }")

    echo "============================================================"
    echo "PMU Summary  (duration=${dur}s)"
    echo "============================================================"
    printf "  IPC:                   %s\n" "$ipc"
    printf "  LLC hit rate (all):    %s%%  (loads=%s misses=%s)\n" \
        "$llc_hit_rate" "$llcl" "$llcm"
    printf "  L3 hit rate (demand):  %s%%  (hits=%s misses=%s)\n" \
        "$l3_hit_rate" "$l3h" "$l3m"
    printf "  DRAM read  BW:         %s GB/s  (%s MiB)\n" \
        "$dram_read_gbps" "$imc_r"
    printf "  DRAM write BW:         %s GB/s  (%s MiB)\n" \
        "$dram_write_gbps" "$imc_w"
    printf "  NIC→CPU  write BW:     %s GB/s  (%s x 4B requests)\n" \
        "$iio_write_gbps" "$iio_mw"
    printf "  NIC←CPU  read  BW:     %s GB/s  (%s x 4B requests)\n" \
        "$iio_read_gbps" "$iio_mr"
    printf "  DDIO absorption (est): %s%%  (NIC writes not reaching DRAM)\n" \
        "$ddio_absorbed_pct"
    echo ""
    if awk "BEGIN{exit !(($iio_mw) == 0)}"; then
        echo "  NOTE: IIO write counter is zero. Either events didn't resolve"
        echo "        on this host, or no NIC-inbound DMA during window. Try:"
        echo "           pmu_monitor.sh discover"
        echo "        and/or set PMU_EVENTS_OVERRIDE explicitly."
    fi
}

# ---------------------------------------------------------------------------
# continuous: long-running perf stat -I sampling, CSV output
# ---------------------------------------------------------------------------
do_continuous() {
    local outfile="$1" interval_ms="${2:-1000}"
    local events
    events=$(_available_events)

    echo "# pmu_monitor continuous: interval=${interval_ms}ms events=${events}" > "$outfile"
    echo "# Columns: time,count,unit,event,run_time_ns,pct_enabled,metric,metric_unit" >> "$outfile"

    exec $(_perf_prefix) perf stat -a -I "$interval_ms" -x , -e "$events" -o "$outfile" --append
}

# ---------------------------------------------------------------------------
# start: bracket-style — launch perf stat in background, write PID + tmp paths
#        to sidecar files so do_stop can finalize. Returns immediately.
#
# Usage in pipeline:
#   pmu_monitor.sh start /run/.../pmu_seq3.kv          # bracket open
#   ... search work ...
#   pmu_monitor.sh stop  /run/.../pmu_seq3.kv          # bracket close → kv file
# ---------------------------------------------------------------------------
do_start() {
    local outfile="$1"
    local events
    events=$(_available_events)

    local pidfile="${outfile}.pid"
    local tmpfile="${outfile}.tmp"
    if [[ -e "$pidfile" ]]; then
        # Stale bracket — clean up best-effort.
        local old_pid; old_pid=$(cat "$pidfile" 2>/dev/null || true)
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            kill -INT "$old_pid" 2>/dev/null || true
            sleep 0.2
            kill -KILL "$old_pid" 2>/dev/null || true
        fi
        rm -f "$pidfile" "$tmpfile"
    fi

    # `sleep infinity` is the workload; we'll kill the child in stop,
    # which makes perf finalize and dump counters to -o file. Use setsid
    # so perf survives the start subshell exit.
    setsid $(_perf_prefix) perf stat -a -x , -e "$events" -o "$tmpfile" -- \
        sleep infinity </dev/null >/dev/null 2>&1 &
    local perf_pid=$!
    echo "$perf_pid" > "$pidfile"
    echo "$tmpfile" > "${outfile}.tmppath"
    # Record the start timestamp externally. perf's own "X.XX seconds time
    # elapsed" line is unreliable when the workload is signaled rather
    # than exited — so we measure wall time ourselves and compute
    # duration in stop. Without this, all GB/s derivations are wrong
    # (divide by 1s instead of actual wall).
    date +%s.%N > "${outfile}.start_ts"
    # Brief wait so perf has time to install its counters before we return.
    sleep 0.05
}

# ---------------------------------------------------------------------------
# stop: SIGINT the perf started by do_start, parse its output to key=value,
#       compute derived metrics, write everything to <outfile>.
# ---------------------------------------------------------------------------
do_stop() {
    local outfile="$1"
    local pidfile="${outfile}.pid"
    local tmppathfile="${outfile}.tmppath"

    if [[ ! -f "$pidfile" || ! -f "$tmppathfile" ]]; then
        echo "pmu_stop: no bracket open at $outfile" >&2
        # Emit empty kv so consumer can still parse.
        : > "$outfile"
        return 0
    fi
    local perf_pid; perf_pid=$(cat "$pidfile")
    local tmpfile; tmpfile=$(cat "$tmppathfile")

    if kill -0 "$perf_pid" 2>/dev/null; then
        # IMPORTANT: when perf is monitoring an external workload
        # (`perf stat ... -- sleep infinity`), SIGINT-ing perf does NOT
        # flush counters to file. Instead, kill the WORKLOAD (the child
        # `sleep`); perf's wait() returns, it finalizes, dumps counters
        # to -o, and exits cleanly.
        local workload_pid
        workload_pid=$(pgrep -P "$perf_pid" 2>/dev/null | head -1 || true)
        if [[ -n "$workload_pid" ]]; then
            kill "$workload_pid" 2>/dev/null || true
        else
            # No child found — fall back to SIGINT on perf (e.g. for
            # `perf stat -a` without trailing workload, but our start uses one).
            kill -INT "$perf_pid" 2>/dev/null || true
        fi
        # Wait briefly for perf to flush.
        for i in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$perf_pid" 2>/dev/null || break
            sleep 0.1
        done
        # If still alive, force.
        if kill -0 "$perf_pid" 2>/dev/null; then
            kill -KILL "$perf_pid" 2>/dev/null || true
            wait "$perf_pid" 2>/dev/null || true
        fi
    fi

    # Compute duration from externally-recorded start timestamp.
    # perf's own "seconds time elapsed" line is unreliable when the
    # workload is signaled rather than exited — rely on this instead.
    local duration=1
    local start_ts_file="${outfile}.start_ts"
    if [[ -f "$start_ts_file" ]]; then
        local start_ts end_ts
        start_ts=$(cat "$start_ts_file")
        end_ts=$(date +%s.%N)
        duration=$(awk "BEGIN{d=($end_ts)-($start_ts); if (d<=0) d=1; printf \"%.6f\", d}")
    fi

    # Parse perf -x, output → key=value (same logic as do_snapshot).
    {
        echo "timestamp=$(date +%s.%N)"
        echo "duration_sec=$duration"
        if [[ -f "$tmpfile" ]]; then
            awk -F, '
                /^[[:space:]]*#/ { next }
                NF >= 3 && $3 != "" {
                    val=$1; ev=$3;
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", val);
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", ev);
                    if (val == "<not counted>" || val == "<not supported>") val = "0";
                    gsub(/[\/=,]/, "_", ev);
                    print ev "=" val;
                }
            ' "$tmpfile"
        fi
    } > "$outfile"

    # Compute derived metrics by reusing do_diff logic (it reads <after_file>
    # and treats it as window-aggregated, which matches our case exactly).
    # Append derived block as additional key=value pairs for easy consumption.
    {
        echo ""
        echo "# --- derived ---"
        # Use a here-doc-like inline subshell to extract values and compute.
        declare -A kv
        while IFS='=' read -r k v; do
            [[ -z "$k" || "$k" == \#* ]] && continue
            kv["$k"]="$v"
        done < "$outfile"
        get() { local k="$1"; local v="${kv[$k]:-0}"; [[ "$v" =~ ^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]] || v=0; echo "$v"; }
        fadd() { awk "BEGIN{printf \"%.6f\", ($1)+($2)}"; }

        local dur cycles instr llcl llcm l3h l3m
        dur=$(get duration_sec); [[ "$dur" == "0" ]] && dur=1
        cycles=$(get cycles)
        instr=$(get instructions)
        llcl=$(get LLC-loads)
        llcm=$(get LLC-load-misses)
        l3h=$(get mem_load_retired.l3_hit)
        l3m=$(get mem_load_retired.l3_miss)
        local imc_r=0 imc_w=0 iio_mw=0 iio_mr=0
        for k in "${!kv[@]}"; do
            local v; v=$(get "$k")
            case "$k" in
                uncore_imc*cas_count_read*)  imc_r=$(fadd "$imc_r" "$v") ;;
                uncore_imc*cas_count_write*) imc_w=$(fadd "$imc_w" "$v") ;;
                *unc_iio_data_req_of_cpu.mem_write*|*nic_to_llc*|*PCIItoM*|*ItoM*)
                    iio_mw=$(fadd "$iio_mw" "$v") ;;
                *unc_iio_data_req_of_cpu.mem_read*)
                    iio_mr=$(fadd "$iio_mr" "$v") ;;
            esac
        done
        local ipc=0 llc_hit=0 l3_hit=0 dr=0 dw=0 iiow=0 iior=0 ddio=0
        if awk "BEGIN{exit !($cycles > 0)}"; then
            ipc=$(awk "BEGIN{printf \"%.3f\", $instr/$cycles}")
        fi
        if awk "BEGIN{exit !($llcl > 0)}"; then
            llc_hit=$(awk "BEGIN{printf \"%.2f\", ($llcl - $llcm)/$llcl*100}")
        fi
        local l3tot; l3tot=$(fadd "$l3h" "$l3m")
        if awk "BEGIN{exit !($l3tot > 0)}"; then
            l3_hit=$(awk "BEGIN{printf \"%.2f\", $l3h/$l3tot*100}")
        fi
        dr=$(awk "BEGIN{printf \"%.3f\", ($imc_r)*1048576/($dur)/1e9}")
        dw=$(awk "BEGIN{printf \"%.3f\", ($imc_w)*1048576/($dur)/1e9}")
        iiow=$(awk "BEGIN{printf \"%.3f\", ($iio_mw)*4/($dur)/1e9}")
        iior=$(awk "BEGIN{printf \"%.3f\", ($iio_mr)*4/($dur)/1e9}")
        ddio=$(awk "BEGIN{
            iio_b = ($iio_mw) * 4; imc_b = ($imc_w) * 1048576;
            if (iio_b <= 0) { print \"0\"; exit }
            a = (iio_b - imc_b) / iio_b * 100;
            if (a < 0) a = 0; if (a > 100) a = 100;
            printf \"%.2f\", a
        }")
        echo "derived_ipc=$ipc"
        echo "derived_llc_hit_rate_pct=$llc_hit"
        echo "derived_l3_hit_rate_pct=$l3_hit"
        echo "derived_dram_read_gbps=$dr"
        echo "derived_dram_write_gbps=$dw"
        echo "derived_iio_write_gbps=$iiow"
        echo "derived_iio_read_gbps=$iior"
        echo "derived_ddio_absorbed_pct=$ddio"
    } >> "$outfile"

    rm -f "$pidfile" "$tmppathfile" "$tmpfile" "${outfile}.start_ts"
}

# ---------------------------------------------------------------------------
# window: attach to an existing PID until it exits
# ---------------------------------------------------------------------------
do_window() {
    local pid="$1" outfile="$2"
    local events
    events=$(_available_events)

    if ! [[ -d /proc/$pid ]]; then
        echo "PID $pid not running"; exit 1
    fi

    exec $(_perf_prefix) perf stat -p "$pid" -x , -e "$events" -o "$outfile"
}

# ---------------------------------------------------------------------------
# nic_stack: report which uncore_iio_N stack owns the NIC's PCI bus.
# Useful when you want to use PMU_EVENTS_OVERRIDE with per-stack qualifiers
# like `uncore_iio_0/unc_iio_data_req_of_cpu.mem_write.part2/`.
# ---------------------------------------------------------------------------
do_nic_stack() {
    local nic_bus_hex="${1:-17}"
    local target_bus="0000:${nic_bus_hex}"
    local found=""
    for d in /sys/bus/event_source/devices/uncore_iio_[0-9]*; do
        [[ -d "$d" ]] || continue
        # Some kernels expose "mapping"; others don't. Fall back to sibling
        # directory that exposes PCI domain:bus via "die" or sysfs link.
        local n=$(basename "$d")
        # Try the free-running counterpart for a sanity probe.
        echo "  $n"
    done
    echo ""
    echo "  PCI root-port to stack mapping (read from lspci + uncore_iio topology):"
    # Show which root ports host the NIC (lspci secondary-bus match).
    local rp
    rp=$(lspci -v 2>/dev/null | awk -v tgt="$nic_bus_hex" '
        /^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]+.*PCI bridge/ { rp=$1 }
        /secondary=/ {
            for (i=1;i<=NF;i++) if ($i ~ /^secondary=/) {
                split($i,a,"=");
                if (a[2]==tgt) print rp
            }
        }
        /Bus:.*secondary=/ {
            for (i=1;i<=NF;i++) if ($i ~ /^secondary=/) {
                split($i,a,"=");
                gsub(",","",a[2]);
                if (a[2]==tgt) print rp
            }
        }
    ')
    if [[ -n "$rp" ]]; then
        echo "  Root port for NIC bus 0x${nic_bus_hex}: ${rp}"
        echo "  NIC devices on bus 0x${nic_bus_hex}:"
        lspci -s "${nic_bus_hex}:" 2>/dev/null | sed 's/^/    /'
    else
        echo "  (could not locate root port — run with bus hex, e.g. nic_stack 17)"
    fi
    echo ""
    echo "  To target just the NIC's stack in PMU_EVENTS_OVERRIDE, pick the"
    echo "  uncore_iio_N whose sibling \"mapping\" or bus-range covers the"
    echo "  root port above (socket-0 stack is usually uncore_iio_0)."
}

# ---------------------------------------------------------------------------
# main dispatch
# ---------------------------------------------------------------------------
CMD="${1:-}"
shift || true

case "$CMD" in
    discover)
        do_discover
        ;;
    nic_stack)
        do_nic_stack "$@"
        ;;
    snapshot)
        [[ $# -lt 1 ]] && { echo "Usage: $0 snapshot <output_file> [secs=1]"; exit 1; }
        do_snapshot "$@"
        ;;
    start)
        [[ $# -lt 1 ]] && { echo "Usage: $0 start <output_file>"; exit 1; }
        do_start "$@"
        ;;
    stop)
        [[ $# -lt 1 ]] && { echo "Usage: $0 stop <output_file>"; exit 1; }
        do_stop "$@"
        ;;
    diff)
        [[ $# -lt 2 ]] && { echo "Usage: $0 diff <before_file> <after_file>"; exit 1; }
        do_diff "$@"
        ;;
    continuous)
        [[ $# -lt 2 ]] && { echo "Usage: $0 continuous <output_file> <interval_ms>"; exit 1; }
        do_continuous "$@"
        ;;
    window)
        [[ $# -lt 2 ]] && { echo "Usage: $0 window <pid> <output_file>"; exit 1; }
        do_window "$@"
        ;;
    *)
        cat <<EOF
Usage: $0 <command> [args]

  discover                                   List available PMU events on this host
  nic_stack  [bus_hex=17]                    Report uncore_iio stack for NIC bus
  snapshot   <output_file> [secs=1]          One-shot short aggregate
  start      <output_file>                   Bracket-open: launch perf in bg
  stop       <output_file>                   Bracket-close: SIGINT perf, dump kv + derived
  diff       <before_file> <after_file>      Human-readable derived metrics
  continuous <output_file> <interval_ms>     Periodic CSV (perf stat -I)
  window     <pid> <output_file>             Attach to a running process

Environment:
  PMU_EVENTS_OVERRIDE   Comma-separated event list (replaces defaults)
  PMU_EVENTS_EXTRA      Extra events appended to defaults

Recommended workflow for DDIO verification:
  1. pmu_monitor.sh discover                # on one DPU first
  2. pmu_monitor.sh snapshot baseline.pmu 5 # idle baseline
  3. # start search experiment in another shell
  4. pmu_monitor.sh snapshot under_load.pmu 5
  5. pmu_monitor.sh diff baseline.pmu under_load.pmu
EOF
        exit 1
        ;;
esac
