#!/usr/bin/env python3
"""NIC bandwidth monitor using InfiniBand hardware counters.

Samples port_xmit_data / port_rcv_data at 1ms intervals and reports
summary statistics when stopped (Ctrl-C or SIGTERM).

Usage:
    python nic_bw_monitor.py                        # auto-detect device
    python nic_bw_monitor.py --dev mlx5_0 --port 1  # explicit device
    python nic_bw_monitor.py --csv bw_trace.csv      # also dump per-sample trace
"""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path


def find_ib_device():
    """Pick the RDMA device. Honors RDMA_DEVICE/RDMA_PORT env vars
    (same convention as scripts/common/rdma_monitor.sh — default mlx5_1 port 1).
    Falls back to the ACTIVE port with the highest link speed, to avoid
    picking a slow management NIC by accident.
    """
    ib_root = Path("/sys/class/infiniband")
    if not ib_root.exists():
        sys.exit("ERROR: /sys/class/infiniband not found. Is this a node with RDMA?")

    env_dev = os.environ.get("RDMA_DEVICE", "").strip()
    env_port = os.environ.get("RDMA_PORT", "").strip()
    if env_dev and (ib_root / env_dev).exists():
        port = int(env_port) if env_port.isdigit() else 1
        return env_dev, port

    devices = sorted(ib_root.iterdir())
    if not devices:
        sys.exit("ERROR: no InfiniBand devices found")

    # Default if env hints are absent: prefer mlx5_1 (matches rdma_monitor.sh)
    # but only if it is ACTIVE — otherwise fall through to link-speed pick.
    mlx5_1_state = ib_root / "mlx5_1" / "ports" / "1" / "state"
    if mlx5_1_state.exists() and "ACTIVE" in mlx5_1_state.read_text():
        return "mlx5_1", 1

    # Otherwise: pick the ACTIVE port with the highest link speed
    best = None  # (speed, dev, port)
    for dev_path in devices:
        ports_dir = dev_path / "ports"
        if not ports_dir.exists():
            continue
        for port_path in sorted(ports_dir.iterdir()):
            state_file = port_path / "state"
            if not (state_file.exists() and "ACTIVE" in state_file.read_text()):
                continue
            rate_path = port_path / "rate"
            try:
                speed = float(rate_path.read_text().strip().split()[0])
            except Exception:
                speed = 0.0
            if best is None or speed > best[0]:
                best = (speed, dev_path.name, int(port_path.name))
    if best:
        return best[1], best[2]
    return devices[0].name, 1


def read_counter(counter_dir, name):
    with open(os.path.join(counter_dir, name)) as f:
        return int(f.read().strip())


def get_link_speed_gbps(dev, port):
    """Try to read active link speed. Returns None if unavailable."""
    rate_path = f"/sys/class/infiniband/{dev}/ports/{port}/rate"
    try:
        text = Path(rate_path).read_text().strip()  # e.g. "100 Gb/sec"
        return float(text.split()[0])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="NIC BW monitor (IB counters)")
    parser.add_argument("--dev", default=None, help="IB device name (e.g. mlx5_0)")
    parser.add_argument("--port", type=int, default=None, help="IB port number")
    parser.add_argument("--interval", type=float, default=0.001,
                        help="Sampling interval in seconds (default: 1ms)")
    parser.add_argument("--csv", default=None, dest="csv_path",
                        help="Optional: dump per-sample trace to CSV")
    parser.add_argument("--summary", default=None, dest="summary_path",
                        help="Optional: periodically write running summary to this file "
                             "(survives SIGKILL). Default: <csv>.summary if --csv given.")
    parser.add_argument("--parent-pid", type=int, default=0, dest="parent_pid",
                        help="exit cleanly when this PID disappears "
                             "(orphan watchdog; 0 = disabled)")
    parser.add_argument("--summary-interval", type=float, default=10.0,
                        help="Seconds between summary dumps (default: 10s)")
    parser.add_argument("--summary-downsample", type=int, default=10,
                        help="Average N consecutive samples for summary stats "
                             "(default: 10 → 1ms*10 = 10ms effective, smooths "
                             "scheduling jitter without affecting raw csv).")
    args = parser.parse_args()

    if args.summary_path is None and args.csv_path:
        args.summary_path = args.csv_path + ".summary"

    if args.dev and args.port:
        dev, port = args.dev, args.port
    else:
        dev, port = find_ib_device()

    counter_dir = f"/sys/class/infiniband/{dev}/ports/{port}/counters"
    if not os.path.isdir(counter_dir):
        sys.exit(f"ERROR: counter dir not found: {counter_dir}")

    link_speed = get_link_speed_gbps(dev, port)
    link_str = f"{link_speed} Gb/s" if link_speed else "unknown"
    print(f"Monitoring {dev}/port {port}  link_speed={link_str}  interval={args.interval*1000:.1f}ms")
    print("Press Ctrl-C to stop and print summary.\n")

    # CSV trace (optional)
    csv_file = None
    csv_writer = None
    if args.csv_path:
        csv_file = open(args.csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["elapsed_s", "tx_gbps", "rx_gbps"])

    # Collect samples
    tx_samples = []
    rx_samples = []

    def downsample(samples, factor):
        """Average every `factor` consecutive samples. Smooths scheduling jitter
        on small intervals (1ms) → effective interval = factor * args.interval."""
        if factor <= 1 or len(samples) < factor:
            return samples
        n = (len(samples) // factor) * factor
        return [sum(samples[i:i + factor]) / factor
                for i in range(0, n, factor)]

    def stats(samples):
        if not samples:
            return None
        s = sorted(samples)
        n = len(s)
        return {
            "min": s[0],
            "max": s[-1],
            "avg": sum(s) / n,
            "p50": s[n // 2],
            "p99": s[int(n * 0.99)],
        }

    def format_summary(tx_samples, rx_samples, elapsed):
        ds = max(1, args.summary_downsample)
        eff_interval_ms = args.interval * 1000 * ds
        tx_ds = downsample(tx_samples, ds)
        rx_ds = downsample(rx_samples, ds)
        lines = []
        lines.append("=" * 55)
        lines.append(f" NIC BW Summary  ({dev}/port {port}, link {link_str})")
        lines.append(f" Duration: {elapsed:.3f}s   Samples: {len(tx_samples)} raw  ({args.interval*1000:.1f}ms interval)")
        if ds > 1:
            lines.append(f" Stats:    averaged x{ds} → {len(tx_ds)} samples ({eff_interval_ms:.1f}ms effective)")
        lines.append("=" * 55)
        tx_s = stats(tx_ds)
        rx_s = stats(rx_ds)
        if tx_s is None:
            lines.append(" (no samples)")
            return "\n".join(lines)
        lines.append(f"         {'min':>10}  {'avg':>10}  {'p50':>10}  {'p99':>10}  {'max':>10}")
        lines.append(f"  TX  {tx_s['min']:>9.2f}  {tx_s['avg']:>9.2f}  {tx_s['p50']:>9.2f}  {tx_s['p99']:>9.2f}  {tx_s['max']:>9.2f}  Gb/s")
        lines.append(f"  RX  {rx_s['min']:>9.2f}  {rx_s['avg']:>9.2f}  {rx_s['p50']:>9.2f}  {rx_s['p99']:>9.2f}  {rx_s['max']:>9.2f}  Gb/s")
        if link_speed:
            lines.append("")
            lines.append(f"  Peak utilization:  TX {tx_s['max']/link_speed*100:.1f}%   RX {rx_s['max']/link_speed*100:.1f}%")
        if dropped_bogus > 0:
            lines.append(f"  (dropped {dropped_bogus} raw samples above {bw_cap_gbps:.1f} Gbps)")
        lines.append("=" * 55)
        return "\n".join(lines)

    def dump_summary(tx_samples, rx_samples, elapsed):
        if not args.summary_path:
            return
        try:
            tmp = args.summary_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(format_summary(tx_samples, rx_samples, elapsed) + "\n")
            os.replace(tmp, args.summary_path)  # atomic rename
        except Exception as e:
            print(f"WARN: failed to write summary: {e}", file=sys.stderr)

    prev_tx = read_counter(counter_dir, "port_xmit_data")
    prev_rx = read_counter(counter_dir, "port_rcv_data")
    prev_t = time.monotonic()
    start_t = prev_t
    last_summary_t = prev_t

    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    # >1.1× wire rate is a wraparound or other artifact → DROP.
    # Smaller jitter overshoots (wire rate ~ 1.1×) are smoothed in summary stats
    # by averaging consecutive samples (see SUMMARY_DOWNSAMPLE below).
    bw_cap_gbps = (link_speed if link_speed else 200.0) * 1.1
    # Counter wraparound detection (32-bit word counter wraps at 2^32)
    WORD_WRAP = 1 << 32
    dropped_bogus = 0

    # Orphan watchdog: if the spawning worker dies without calling
    # nic_burst_monitor_stop (crash/kill -9), exit via the normal shutdown
    # path instead of running forever — two leaked monitors from a 06-30 run
    # burned ~0.5 core + 1 kHz NFS writes on hosts A/B for 8 days.
    parent_pid = int(args.parent_pid or 0)
    parent_check_t = time.monotonic()

    while running:
        time.sleep(args.interval)
        now = time.monotonic()
        if parent_pid and now - parent_check_t >= 1.0:
            parent_check_t = now
            if not os.path.exists(f"/proc/{parent_pid}"):
                running = False
                break
        tx = read_counter(counter_dir, "port_xmit_data")
        rx = read_counter(counter_dir, "port_rcv_data")

        dt = now - prev_t
        if dt <= 0:
            prev_tx, prev_rx, prev_t = tx, rx, now
            continue

        # Handle 32-bit word counter wraparound (17GB @ 4B/word)
        d_tx = tx - prev_tx
        d_rx = rx - prev_rx
        if d_tx < 0:
            d_tx += WORD_WRAP
        if d_rx < 0:
            d_rx += WORD_WRAP

        # counters are in 4-byte words
        tx_gbps = d_tx * 4 * 8 / dt / 1e9
        rx_gbps = d_rx * 4 * 8 / dt / 1e9

        # Drop obviously bogus samples (>1.1× wire rate — wraparound / OS hiccup)
        if tx_gbps > bw_cap_gbps or rx_gbps > bw_cap_gbps:
            dropped_bogus += 1
            prev_tx, prev_rx, prev_t = tx, rx, now
            continue

        tx_samples.append(tx_gbps)
        rx_samples.append(rx_gbps)

        if csv_writer:
            csv_writer.writerow([f"{now - start_t:.6f}", f"{tx_gbps:.2f}", f"{rx_gbps:.2f}"])

        prev_tx, prev_rx, prev_t = tx, rx, now

        # periodic summary dump (survives SIGKILL)
        if now - last_summary_t >= args.summary_interval:
            dump_summary(tx_samples, rx_samples, now - start_t)
            if csv_file:
                csv_file.flush()
            last_summary_t = now

    # Cleanup
    if csv_file:
        csv_file.close()
        print(f"Trace saved to {args.csv_path}")

    if not tx_samples:
        print("No samples collected.")
        return

    total_t = prev_t - start_t
    summary_text = format_summary(tx_samples, rx_samples, total_t)
    print("\n" + summary_text)
    dump_summary(tx_samples, rx_samples, total_t)
    if args.summary_path:
        print(f"Summary saved to {args.summary_path}")


if __name__ == "__main__":
    main()
