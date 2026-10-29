#!/bin/bash
# ============================================================================
# rdma_monitor.sh - RDMA counter collection for benchmarking
# ============================================================================
#
# Reads sysfs RDMA counters (zero overhead on data path) and computes deltas.
#
# Usage:
#   rdma_monitor.sh snapshot <output_file>                       # Save counters
#   rdma_monitor.sh diff <before_file> <after_file>              # Show delta
#   rdma_monitor.sh continuous <output_file> <interval_sec>      # Periodic CSV
#
# ============================================================================

set -euo pipefail

# Auto-detect: use RDMA_DEVICE env var if set and device exists, else fall back
# to mlx5_0. Handles hosts A/B (only mlx5_0) vs hosts C/D (mlx5_1 set
# in container env but mlx5_0 is actually the active NIC on some nodes).
_preferred="${RDMA_DEVICE:-mlx5_1}"
if [[ -d "/sys/class/infiniband/${_preferred}" ]]; then
    DEVICE="${_preferred}"
elif [[ -d "/sys/class/infiniband/mlx5_0" ]]; then
    DEVICE="mlx5_0"
else
    DEVICE="${_preferred}"
fi
PORT="${RDMA_PORT:-1}"
COUNTERS_DIR="/sys/class/infiniband/${DEVICE}/ports/${PORT}/counters"
HW_COUNTERS_DIR="/sys/class/infiniband/${DEVICE}/ports/${PORT}/hw_counters"

# Counters to collect
TRAFFIC_COUNTERS=(
    port_xmit_data
    port_rcv_data
    port_xmit_packets
    port_rcv_packets
)

CONGESTION_COUNTERS=(
    port_xmit_wait
)

HW_CONGESTION_COUNTERS=(
    np_cnp_sent
    np_ecn_marked_roce_packets
)

HW_ERROR_COUNTERS=(
    out_of_buffer
    roce_adp_retrans
    req_rnr_retries_exceeded
    req_transport_retries_exceeded
    packet_seq_err
    local_ack_timeout_err
)

HW_RDMA_COUNTERS=(
    rx_read_requests
    rx_write_requests
)

# mlx5 debugfs diagnostics (QPC cache, etc.)
# These may not exist on all systems; collected best-effort
MLX5_DIAG_DIR=""
for d in /sys/kernel/debug/mlx5/*/diagnostics; do
    if [[ -d "$d" ]]; then
        MLX5_DIAG_DIR="$d"
        break
    fi
done

MLX5_DIAG_COUNTERS=(
    # Common mlx5 diagnostic counters (availability varies by FW/driver)
    rx_write_requests
    rx_read_requests
    rx_atomic_requests
    sq_num_bre
    rq_num_bre
    sq_num_wrfe
    rq_num_wrfe
    sq_num_mce
    rq_num_mce
    sq_num_rae
    rq_num_rae
)

# PCIe device discovery via sysfs symlink
PCIE_DEVICE_DIR=""
if [[ -L "/sys/class/infiniband/${DEVICE}/device" ]]; then
    PCIE_DEVICE_DIR=$(readlink -f "/sys/class/infiniband/${DEVICE}/device")
fi
PCIE_BDF=""
if [[ -n "$PCIE_DEVICE_DIR" ]]; then
    PCIE_BDF=$(basename "$PCIE_DEVICE_DIR")
fi

# PCIe link info files (static, from config registers)
PCIE_STATIC_FILES=(
    current_link_speed
    current_link_width
    max_link_speed
    max_link_width
)

# Physical layer byte counters (includes ALL headers — true wire-level traffic)
PHY_ETHTOOL_COUNTERS=(
    rx_bytes_phy
    tx_bytes_phy
    rx_packets_phy
    tx_packets_phy
)

# RDMA-specific vport counters (unicast RDMA traffic, payload level)
RDMA_VPORT_COUNTERS=(
    rx_vport_rdma_unicast_bytes
    tx_vport_rdma_unicast_bytes
    rx_vport_rdma_unicast_packets
    tx_vport_rdma_unicast_packets
)

# PCIe stall counters (from ethtool -S on mlx5 net device)
PCIE_ETHTOOL_COUNTERS=(
    tx_pci_signal_integrity
    rx_pci_signal_integrity
    outbound_pci_stalled_rd
    outbound_pci_stalled_wr
    outbound_pci_stalled_rd_events
    outbound_pci_stalled_wr_events
)

# Discover the network device name for ethtool
NETDEV=""
if [[ -d "/sys/class/infiniband/${DEVICE}/device/net" ]]; then
    NETDEV=$(ls "/sys/class/infiniband/${DEVICE}/device/net" 2>/dev/null | head -1)
fi

# ============================================================================
# Helper functions
# ============================================================================

read_counter() {
    local dir="$1" name="$2"
    local path="${dir}/${name}"
    if [[ -r "$path" ]]; then
        cat "$path"
    else
        echo "0"
    fi
}

do_snapshot() {
    local outfile="$1"
    local ts
    ts=$(date +%s.%N)

    {
        echo "timestamp=${ts}"

        # Standard counters
        for c in "${TRAFFIC_COUNTERS[@]}" "${CONGESTION_COUNTERS[@]}"; do
            echo "${c}=$(read_counter "$COUNTERS_DIR" "$c")"
        done

        # HW counters
        for c in "${HW_CONGESTION_COUNTERS[@]}" "${HW_ERROR_COUNTERS[@]}" "${HW_RDMA_COUNTERS[@]}"; do
            echo "${c}=$(read_counter "$HW_COUNTERS_DIR" "$c")"
        done

        # mlx5 debugfs diagnostics (best-effort)
        if [[ -n "$MLX5_DIAG_DIR" ]]; then
            for c in "${MLX5_DIAG_COUNTERS[@]}"; do
                local path="${MLX5_DIAG_DIR}/${c}"
                if [[ -r "$path" ]]; then
                    echo "mlx5_diag_${c}=$(cat "$path")"
                fi
            done
        fi

        # PCIe static info
        if [[ -n "$PCIE_DEVICE_DIR" ]]; then
            echo "pcie_bdf=${PCIE_BDF}"
            for f in "${PCIE_STATIC_FILES[@]}"; do
                local path="${PCIE_DEVICE_DIR}/${f}"
                if [[ -r "$path" ]]; then
                    echo "pcie_${f}=$(cat "$path" 2>/dev/null || echo unknown)"
                fi
            done
        fi

        # Ethtool counters (best-effort)
        if [[ -n "$NETDEV" ]] && command -v ethtool &>/dev/null; then
            local ethtool_out
            ethtool_out=$(ethtool -S "$NETDEV" 2>/dev/null || true)
            if [[ -n "$ethtool_out" ]]; then
                # Physical layer byte counters (true wire-level traffic, all headers included)
                for counter in "${PHY_ETHTOOL_COUNTERS[@]}"; do
                    local val
                    val=$(echo "$ethtool_out" | grep -w "$counter" | awk '{print $2}')
                    if [[ -n "$val" ]]; then
                        echo "phy_${counter}=${val}"
                    fi
                done
                # RDMA vport counters (RDMA-specific unicast traffic)
                for counter in "${RDMA_VPORT_COUNTERS[@]}"; do
                    local val
                    val=$(echo "$ethtool_out" | grep -w "$counter" | awk '{print $2}')
                    if [[ -n "$val" ]]; then
                        echo "vport_${counter}=${val}"
                    fi
                done
                # PCIe stall counters
                for counter in "${PCIE_ETHTOOL_COUNTERS[@]}"; do
                    local val
                    val=$(echo "$ethtool_out" | grep -w "$counter" | awk '{print $2}')
                    if [[ -n "$val" ]]; then
                        echo "pcie_${counter}=${val}"
                    fi
                done
            fi
        fi
    } > "$outfile"
}

do_diff() {
    local before_file="$1" after_file="$2"

    # Load before/after into associative arrays
    declare -A before after
    while IFS='=' read -r key val; do
        before["$key"]="$val"
    done < "$before_file"
    while IFS='=' read -r key val; do
        after["$key"]="$val"
    done < "$after_file"

    local duration
    duration=$(awk "BEGIN {printf \"%.3f\", ${after[timestamp]} - ${before[timestamp]}}")

    # Compute deltas (guard against non-numeric values)
    delta() {
        local key="$1"
        local b="${before[$key]:-0}" a="${after[$key]:-0}"
        if [[ "$b" =~ ^[0-9]+$ && "$a" =~ ^[0-9]+$ ]]; then
            echo $((a - b))
        else
            echo "N/A"
        fi
    }

    # IB port_xmit_data/port_rcv_data are in 32-bit words (×4 for bytes)
    local tx_words rx_words
    tx_words=$(delta port_xmit_data)
    rx_words=$(delta port_rcv_data)
    local tx_bytes=$((tx_words * 4))
    local rx_bytes=$((rx_words * 4))
    local tx_packets rx_packets
    tx_packets=$(delta port_xmit_packets)
    rx_packets=$(delta port_rcv_packets)

    # Bandwidth
    local tx_mbps rx_mbps tx_mbs rx_mbs
    tx_mbs=$(awk "BEGIN {printf \"%.2f\", ${tx_bytes} / 1048576}")
    rx_mbs=$(awk "BEGIN {printf \"%.2f\", ${rx_bytes} / 1048576}")
    tx_mbps=$(awk "BEGIN {printf \"%.1f\", ${tx_bytes} / ${duration} / 1048576}")
    rx_mbps=$(awk "BEGIN {printf \"%.1f\", ${rx_bytes} / ${duration} / 1048576}")
    local tx_gbps rx_gbps
    tx_gbps=$(awk "BEGIN {printf \"%.2f\", ${tx_bytes} * 8 / ${duration} / 1000000000}")
    rx_gbps=$(awk "BEGIN {printf \"%.2f\", ${rx_bytes} * 8 / ${duration} / 1000000000}")

    # RDMA operations
    local read_reqs write_reqs
    read_reqs=$(delta rx_read_requests)
    write_reqs=$(delta rx_write_requests)

    # Congestion
    local xmit_wait cnp_sent ecn_marked
    xmit_wait=$(delta port_xmit_wait)
    cnp_sent=$(delta np_cnp_sent)
    ecn_marked=$(delta np_ecn_marked_roce_packets)

    # Errors
    local oob retrans rnr_exceeded transport_exceeded seq_err ack_timeout
    oob=$(delta out_of_buffer)
    retrans=$(delta roce_adp_retrans)
    rnr_exceeded=$(delta req_rnr_retries_exceeded)
    transport_exceeded=$(delta req_transport_retries_exceeded)
    seq_err=$(delta packet_seq_err)
    ack_timeout=$(delta local_ack_timeout_err)

    # Wire-level traffic (physical layer — includes ALL headers)
    local phy_tx_bytes_val="${after[phy_tx_bytes_phy]:-}"
    local phy_rx_bytes_val="${after[phy_rx_bytes_phy]:-}"
    local phy_tx_before="${before[phy_tx_bytes_phy]:-0}"
    local phy_rx_before="${before[phy_rx_bytes_phy]:-0}"
    local has_phy=false
    local wire_tx_gbps="" wire_rx_gbps="" wire_tx_bytes="" wire_rx_bytes=""
    if [[ -n "$phy_tx_bytes_val" && -n "$phy_rx_bytes_val" && \
          "$phy_tx_bytes_val" =~ ^[0-9]+$ && "$phy_rx_bytes_val" =~ ^[0-9]+$ && \
          "$phy_tx_before" =~ ^[0-9]+$ && "$phy_rx_before" =~ ^[0-9]+$ ]]; then
        has_phy=true
        wire_tx_bytes=$((phy_tx_bytes_val - phy_tx_before))
        wire_rx_bytes=$((phy_rx_bytes_val - phy_rx_before))
        wire_tx_gbps=$(awk "BEGIN {printf \"%.2f\", ${wire_tx_bytes} * 8 / ${duration} / 1000000000}")
        wire_rx_gbps=$(awk "BEGIN {printf \"%.2f\", ${wire_rx_bytes} * 8 / ${duration} / 1000000000}")
    fi

    # Output
    echo "======================================================================"
    echo "RDMA Resource Usage (device=${DEVICE}, duration=${duration}s)"
    echo "======================================================================"
    echo ""
    printf "Traffic (IB payload only — excludes Ethernet/IP/UDP/BTH headers):\n"
    printf "  TX: %10s MB  (%s MB/s = %s Gbps)\n" "$tx_mbs" "$tx_mbps" "$tx_gbps"
    printf "  RX: %10s MB  (%s MB/s = %s Gbps)\n" "$rx_mbs" "$rx_mbps" "$rx_gbps"
    printf "  TX packets: %s  RX packets: %s\n" "$tx_packets" "$rx_packets"
    if $has_phy; then
        echo ""
        printf "Wire-level traffic (physical layer — ALL headers included):\n"
        printf "  TX: %s Gbps   RX: %s Gbps   (line rate = 100 Gbps)\n" "$wire_tx_gbps" "$wire_rx_gbps"
        local util_tx util_rx
        util_tx=$(awk "BEGIN {printf \"%.1f\", ${wire_tx_gbps} / 100 * 100}")
        util_rx=$(awk "BEGIN {printf \"%.1f\", ${wire_rx_gbps} / 100 * 100}")
        printf "  Utilization: TX=%s%%  RX=%s%%\n" "$util_tx" "$util_rx"
    fi
    echo ""
    printf "RDMA Operations:\n"
    printf "  READ requests received:  %s\n" "$read_reqs"
    printf "  WRITE requests received: %s\n" "$write_reqs"
    echo ""
    printf "Congestion:\n"
    printf "  xmit_wait: %s  CNP sent: %s  ECN marked: %s\n" \
        "$xmit_wait" "$cnp_sent" "$ecn_marked"
    echo ""
    printf "Errors:\n"
    printf "  out_of_buffer: %s  retrans: %s\n" "$oob" "$retrans"
    printf "  rnr_exceeded: %s  transport_exceeded: %s\n" "$rnr_exceeded" "$transport_exceeded"
    printf "  seq_err: %s  ack_timeout: %s\n" "$seq_err" "$ack_timeout"
    echo ""

    # PCIe link info (from after snapshot)
    local pcie_speed="${after[pcie_current_link_speed]:-}"
    local pcie_width="${after[pcie_current_link_width]:-}"
    if [[ -n "$pcie_speed" || -n "$pcie_width" ]]; then
        printf "PCIe Link:\n"
        printf "  Current: %s x %s  (Max: %s x %s)\n" \
            "${pcie_speed:-unknown}" "${pcie_width:-unknown}" \
            "${after[pcie_max_link_speed]:-unknown}" "${after[pcie_max_link_width]:-unknown}"

        # PCIe stall counters (delta if available)
        local has_pcie_stall=false
        for counter in outbound_pci_stalled_rd outbound_pci_stalled_wr \
                       outbound_pci_stalled_rd_events outbound_pci_stalled_wr_events; do
            local pkey="pcie_${counter}"
            if [[ -n "${after[$pkey]:-}" ]]; then
                has_pcie_stall=true
                break
            fi
        done
        if $has_pcie_stall; then
            local stall_rd stall_wr stall_rd_ev stall_wr_ev
            stall_rd=$(delta pcie_outbound_pci_stalled_rd)
            stall_wr=$(delta pcie_outbound_pci_stalled_wr)
            stall_rd_ev=$(delta pcie_outbound_pci_stalled_rd_events)
            stall_wr_ev=$(delta pcie_outbound_pci_stalled_wr_events)
            printf "  Stalls: rd=%s  wr=%s  rd_events=%s  wr_events=%s\n" \
                "$stall_rd" "$stall_wr" "$stall_rd_ev" "$stall_wr_ev"
        fi

        # PCIe signal integrity
        local sig_tx sig_rx
        sig_tx=$(delta pcie_tx_pci_signal_integrity)
        sig_rx=$(delta pcie_rx_pci_signal_integrity)
        if [[ "$sig_tx" != "N/A" || "$sig_rx" != "N/A" ]]; then
            printf "  Signal integrity: tx=%s  rx=%s\n" "$sig_tx" "$sig_rx"
        fi
        echo ""
    fi
}

do_continuous() {
    local outfile="$1" interval="$2"

    # Discover available mlx5 diag counters for CSV header
    local -a available_diag=()
    if [[ -n "$MLX5_DIAG_DIR" ]]; then
        for c in "${MLX5_DIAG_COUNTERS[@]}"; do
            if [[ -r "${MLX5_DIAG_DIR}/${c}" ]]; then
                available_diag+=("$c")
            fi
        done
    fi

    # Discover available PHY and PCIe ethtool counters
    local -a available_phy=()
    local -a available_pcie=()
    if [[ -n "$NETDEV" ]] && command -v ethtool &>/dev/null; then
        local ethtool_check
        ethtool_check=$(ethtool -S "$NETDEV" 2>/dev/null || true)
        for counter in "${PHY_ETHTOOL_COUNTERS[@]}"; do
            if echo "$ethtool_check" | grep -qw "$counter"; then
                available_phy+=("$counter")
            fi
        done
        for counter in "${PCIE_ETHTOOL_COUNTERS[@]}"; do
            if echo "$ethtool_check" | grep -qw "$counter"; then
                available_pcie+=("$counter")
            fi
        done
    fi

    # CSV header
    {
        printf "timestamp"
        for c in "${TRAFFIC_COUNTERS[@]}" "${CONGESTION_COUNTERS[@]}"; do
            printf ",%s" "$c"
        done
        for c in "${HW_CONGESTION_COUNTERS[@]}" "${HW_ERROR_COUNTERS[@]}" "${HW_RDMA_COUNTERS[@]}"; do
            printf ",%s" "$c"
        done
        for c in "${available_diag[@]}"; do
            printf ",mlx5_diag_%s" "$c"
        done
        for c in "${available_phy[@]}"; do
            printf ",phy_%s" "$c"
        done
        for c in "${available_pcie[@]}"; do
            printf ",pcie_%s" "$c"
        done
        echo ""
    } > "$outfile"

    while true; do
        local ts
        ts=$(date +%s.%N)

        # Pre-fetch ethtool output once per iteration (avoid per-counter overhead)
        local ethtool_out=""
        if [[ ${#available_phy[@]} -gt 0 || ${#available_pcie[@]} -gt 0 ]]; then
            ethtool_out=$(ethtool -S "$NETDEV" 2>/dev/null || true)
        fi

        {
            printf "%s" "$ts"
            for c in "${TRAFFIC_COUNTERS[@]}" "${CONGESTION_COUNTERS[@]}"; do
                printf ",%s" "$(read_counter "$COUNTERS_DIR" "$c")"
            done
            for c in "${HW_CONGESTION_COUNTERS[@]}" "${HW_ERROR_COUNTERS[@]}" "${HW_RDMA_COUNTERS[@]}"; do
                printf ",%s" "$(read_counter "$HW_COUNTERS_DIR" "$c")"
            done
            for c in "${available_diag[@]}"; do
                printf ",%s" "$(cat "${MLX5_DIAG_DIR}/${c}" 2>/dev/null || echo 0)"
            done
            for c in "${available_phy[@]}" "${available_pcie[@]}"; do
                local val
                val=$(echo "$ethtool_out" | grep -w "$c" | awk '{print $2}')
                printf ",%s" "${val:-0}"
            done
            echo ""
        } >> "$outfile"
        sleep "$interval"
    done
}

# ============================================================================
# Main
# ============================================================================

CMD="${1:-}"
shift || true

case "$CMD" in
    snapshot)
        [[ $# -lt 1 ]] && { echo "Usage: $0 snapshot <output_file>"; exit 1; }
        do_snapshot "$1"
        ;;
    diff)
        [[ $# -lt 2 ]] && { echo "Usage: $0 diff <before_file> <after_file>"; exit 1; }
        do_diff "$1" "$2"
        ;;
    continuous)
        [[ $# -lt 2 ]] && { echo "Usage: $0 continuous <output_file> <interval_sec>"; exit 1; }
        do_continuous "$1" "$2"
        ;;
    *)
        echo "Usage: $0 {snapshot|diff|continuous} [args...]"
        echo ""
        echo "  snapshot <output_file>                  Save current RDMA counters"
        echo "  diff <before_file> <after_file>         Compute and display delta"
        echo "  continuous <output_file> <interval_sec>  Periodic CSV sampling"
        exit 1
        ;;
esac
