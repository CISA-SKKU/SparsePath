"""Experiment diagnostics helpers for the distributed engine."""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict


_rdma_monitor_warned = False

def rdma_snapshot() -> dict:
    global _rdma_monitor_warned
    script = Path(__file__).resolve().parent.parent / "scripts" / "common" / "rdma_monitor.sh"
    if not script.exists():
        if not _rdma_monitor_warned:
            print(f"[WARN] rdma_monitor.sh not found at {script}, NIC BW stats unavailable", flush=True)
            _rdma_monitor_warned = True
        return {}
    with tempfile.NamedTemporaryFile(mode='r', suffix='.snap', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run([str(script), "snapshot", tmp_path],
                        capture_output=True, timeout=10)
        snap = {}
        with open(tmp_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    k, v = line.split('=', 1)
                    snap[k] = v
        return snap
    except Exception:
        return {}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

def rdma_compute_bw(before: dict, after: dict, elapsed: float) -> dict:
    if not before or not after or elapsed <= 0:
        return {}
    def _delta(key):
        try:
            return int(after.get(key, '0')) - int(before.get(key, '0'))
        except (ValueError, TypeError):
            return 0
    # port_xmit/rcv_data (32-bit words × 4 = bytes) are WIRE-LEVEL on mlx5
    # RoCE: calibrated 2026-06-11 against ib_read_bw at 392B saturation —
    # payload 82.0G ↔ port_rcv_data 99.0G ↔ ethtool rx_bytes_phy 99.9G
    # (~473B counted per 478B wire frame; headers INCLUDED, preamble/IFG not).
    # Do NOT read these as payload Gbps; payload ≈ value × payload/wire ratio
    # of the message mix (392B msgs: ×0.83).
    ib_tx = _delta('port_xmit_data') * 4
    ib_rx = _delta('port_rcv_data') * 4
    # Physical-layer byte counters (ethtool): full wire frame incl. FCS;
    # essentially the same level as port_*_data (±1%).
    wire_tx = _delta('phy_tx_bytes_phy')
    wire_rx = _delta('phy_rx_bytes_phy')
    # RDMA-specific unicast vport counter — RDMA traffic only, at payload level.
    rdma_tx = _delta('vport_tx_vport_rdma_unicast_bytes')
    rdma_rx = _delta('vport_rx_vport_rdma_unicast_bytes')
    # Queueing / congestion counters (rdma_monitor.sh already collects these).
    # Delta = events during the measurement window. Summing across nodes is
    # meaningful for cluster-level pressure indicators.
    xmit_wait = _delta('port_xmit_wait')              # HCA transmit queue wait ticks
    cnp_sent = _delta('np_cnp_sent')                  # ECN CNPs sent by this NIC
    ecn_marked = _delta('np_ecn_marked_roce_packets') # ECN-marked RoCE packets received
    oob = _delta('out_of_buffer')                     # buffer-overflow drops
    rx_reads = _delta('rx_read_requests')             # NIC-level RDMA READ count
    out = {
        'nic_ib_tx_gbps': round(ib_tx * 8 / elapsed / 1e9, 3),
        'nic_ib_rx_gbps': round(ib_rx * 8 / elapsed / 1e9, 3),
    }
    if wire_tx > 0 or wire_rx > 0:
        out['nic_wire_tx_gbps'] = round(wire_tx * 8 / elapsed / 1e9, 3)
        out['nic_wire_rx_gbps'] = round(wire_rx * 8 / elapsed / 1e9, 3)
    if rdma_tx > 0 or rdma_rx > 0:
        out['nic_rdma_tx_gbps'] = round(rdma_tx * 8 / elapsed / 1e9, 3)
        out['nic_rdma_rx_gbps'] = round(rdma_rx * 8 / elapsed / 1e9, 3)
    # Congestion deltas: emit even when 0 so aggregation is stable across
    # nodes (missing keys would skew per-op sums).
    out['nic_xmit_wait'] = xmit_wait
    out['nic_cnp_sent'] = cnp_sent
    out['nic_ecn_marked'] = ecn_marked
    out['nic_out_of_buffer'] = oob
    out['nic_rx_read_requests'] = rx_reads
    out['nic_msg_rate_mps'] = round(rx_reads / elapsed / 1e6, 3)
    return out

# --- PMU monitor (perf-stat based: IPC, LLC, DRAM BW, DDIO) ---
# Mirrors the RDMA monitor pattern but uses bracket-style start/stop
# subcommands (perf needs to be running during the window — it can't be
# snapshotted like RDMA HW counters).
_pmu_monitor_warned = False

def pmu_start(out_path: str) -> bool:
    """Open a PMU bracket. Returns True if perf launched, False if PMU
    unavailable (script missing, perf missing, or perms). Caller must
    always pair with pmu_stop(out_path)."""
    global _pmu_monitor_warned
    script = Path(__file__).resolve().parent.parent / "scripts" / "common" / "pmu_monitor.sh"
    if not script.exists():
        if not _pmu_monitor_warned:
            print(f"[WARN] pmu_monitor.sh not found at {script}, PMU stats unavailable",
                  flush=True)
            _pmu_monitor_warned = True
        return False
    try:
        r = subprocess.run([str(script), "start", out_path],
                            capture_output=True, timeout=10)
        if r.returncode != 0:
            if not _pmu_monitor_warned:
                print(f"[WARN] pmu start failed: {r.stderr.decode(errors='ignore')[:200]}",
                      flush=True)
                _pmu_monitor_warned = True
            return False
        return True
    except Exception as e:
        if not _pmu_monitor_warned:
            print(f"[WARN] pmu start exception: {e}", flush=True)
            _pmu_monitor_warned = True
        return False

def pmu_stop(out_path: str) -> dict:
    """Close PMU bracket and parse derived metrics. Returns dict with
    pmu_* keys (empty if anything failed). The kv file at out_path is
    kept under run_dir/pmu/ for later inspection / re-derivation."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "common" / "pmu_monitor.sh"
    if not script.exists():
        return {}
    try:
        subprocess.run([str(script), "stop", out_path],
                        capture_output=True, timeout=15)
        return pmu_parse(out_path)
    except Exception:
        return {}

def pmu_parse(path: str) -> dict:
    """Parse a pmu_monitor stop-file (key=value lines) into a stats dict.
    Returns only the derived_* fields, renamed to pmu_*."""
    out: Dict[str, float] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if not k.startswith('derived_'):
                    continue
                try:
                    out['pmu_' + k[len('derived_'):]] = float(v)
                except ValueError:
                    pass
    except FileNotFoundError:
        return {}
    return out

def nic_burst_monitor_start(csv_path: str, interval_s: float = 0.001):
    """Spawn nic_bw_monitor.py in background for fine-grained sampling.

    Returns (proc, mono_start) — mono_start is used to align per-L
    timestamps with the monitor's elapsed_s column.
    """
    import os
    _exp_root = Path(os.environ.get('DANNS_EXPERIMENT_CONTAINER_ROOT', '/experiments'))
    script = _exp_root / "analysis" / "nic_bw_monitor.py"
    # Fallback: host path (when running outside container)
    if not script.exists():
        script = Path(__file__).resolve().parent.parent.parent / "02_SparsePath_experiments" / "analysis" / "nic_bw_monitor.py"
    if not script.exists():
        return None, None
    try:
        proc = subprocess.Popen(
            ["python3", str(script), "--csv", csv_path,
             "--interval", str(interval_s),
             # Orphan watchdog: the monitor exits by itself if this worker
             # dies without reaching nic_burst_monitor_stop (crash/kill -9).
             "--parent-pid", str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Give the monitor a moment to initialize (read counters, open CSV).
        # Capture mono_start AFTER init so it aligns with monitor's elapsed_s=0.
        time.sleep(0.1)
        mono_start = time.monotonic()
        return proc, mono_start
    except Exception as e:
        print(f"[WARN] nic_burst_monitor_start failed: {e}", flush=True)
        return None, None

def nic_burst_monitor_stop(proc):
    if proc is None:
        return
    try:
        proc.terminate()  # SIGTERM — handler writes final summary
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass
    except Exception:
        pass

def nic_burst_parse_csv(csv_path: str, l_windows: list,
                         link_gbps: float = 100.0) -> dict:
    """Slice CSV samples per L window and compute burst stats.

    Filters out samples > link_gbps * 1.1 (wraparound / scheduling jitter
    artifacts) and samples < 0 to keep the report free of bogus spikes.
    """
    if not os.path.exists(csv_path):
        return {}
    samples = []
    # Safety cap: a 32-bit word counter wraps at ~17GB; with scheduling
    # jitter in a CPU-saturated system, a single 1ms sample can report
    # arbitrary values. Anything above this threshold is a measurement
    # artifact, not real traffic.
    cap = link_gbps * 1.1
    dropped = 0
    try:
        with open(csv_path) as f:
            f.readline()  # header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                try:
                    e = float(parts[0])
                    tv = float(parts[1])
                    rv = float(parts[2])
                except ValueError:
                    continue
                if tv < 0 or rv < 0 or tv > cap or rv > cap:
                    dropped += 1
                    continue
                samples.append((e, tv, rv))
    except Exception:
        return {}
    if not samples:
        return {}

    result: Dict[object, Dict[str, float]] = {}
    for L, t_start, t_end in l_windows:
        txs, rxs = [], []
        for e, tv, rv in samples:
            if t_start <= e <= t_end:
                txs.append(tv)
                rxs.append(rv)
        if not txs:
            continue
        txs.sort()
        rxs.sort()
        n = len(txs)
        p99_idx = min(int(n * 0.99), n - 1)
        result[L] = {
            'nic_burst_tx_max_gbps': round(txs[-1], 2),
            'nic_burst_rx_max_gbps': round(rxs[-1], 2),
            'nic_burst_tx_p99_gbps': round(txs[p99_idx], 2),
            'nic_burst_rx_p99_gbps': round(rxs[p99_idx], 2),
            'nic_burst_tx_avg_gbps': round(sum(txs) / n, 2),
            'nic_burst_rx_avg_gbps': round(sum(rxs) / n, 2),
            'nic_burst_samples': n,
            'nic_burst_dropped': dropped,
        }
    return result
