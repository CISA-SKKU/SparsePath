#!/usr/bin/env python3
"""Emit one CSV row per (run, seq, L, node) from node_stats/*.json.

Joins per-op workload params from engine_report WORKLOAD section (T, L_values,
pq_alpha, batch_size) and recall from recall_posthoc.json if present.

Usage:
    python stats_to_csv.py <run_dir_or_glob> [<run_dir_or_glob> ...]  -o out.csv
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path


SEARCH_COLS = [
    'num_queries', 'elapsed_s', 'qps',
    'n_hops', 'n_cmps',
    'local_vector_accesses', 'remote_vector_accesses',
    'local_neighbor_accesses', 'remote_neighbor_accesses',
    'fetch_entry_calls', 'fetch_entry_items', 'resolve_fail',
    'pq_filtered', 'pq_cache_hits', 'pq_cache_misses',
    'rdma_distance_us', 'rdma_fetch_us', 'rdma_total_us',
    'sum_query_total_us', 'cpu_total_us',
    'nic_rx_gbps', 'nic_tx_gbps',
    'nic_wire_rx_gbps', 'nic_wire_tx_gbps',
    'nic_rdma_rx_gbps', 'nic_rdma_tx_gbps',
    'nic_xmit_wait', 'nic_cnp_sent', 'nic_ecn_marked', 'nic_out_of_buffer',
    'nic_rx_read_requests', 'nic_msg_rate_mps',
    'pmu_ipc', 'pmu_llc_hit_rate_pct', 'pmu_l3_hit_rate_pct',
    'pmu_dram_read_gbps', 'pmu_dram_write_gbps',
    'pmu_iio_write_gbps', 'pmu_iio_read_gbps', 'pmu_ddio_absorbed_pct',
    'nic_burst_tx_max_gbps', 'nic_burst_rx_max_gbps',
    'nic_burst_tx_p99_gbps', 'nic_burst_rx_p99_gbps',
    'nic_burst_tx_avg_gbps', 'nic_burst_rx_avg_gbps',
    'nic_burst_samples', 'nic_burst_dropped',
]

CPU_COLS = [
    'visited_us', 'resolve_us', 'distance_us', 'queue_us',
    'nbr_loop_us', 'map_us', 'pq_us',
    'overlap_phase_us',
    'pass1_us', 'pass2_us', 'pass3_us', 'pass3_poll_us', 'pass3_compute_us',
    'batch_nbr_poll_us', 'batch_vec_poll_us',
]


def parse_workload(run_dir: Path):
    """Map seq -> {T, L_values, pq_alpha, batch_size, overlap_mode}."""
    rpt = (run_dir / 'engine_report.txt').read_text() if (run_dir / 'engine_report.txt').exists() else ''
    out = {}
    # Pattern: "    0. search         num_queries=10000, K=10, num_threads=32, L_values=[100, 150, 200], pq_alpha=0.2"
    for m in re.finditer(r'^\s*(\d+)\.\s+search\s+(.*)$', rpt, re.MULTILINE):
        seq = int(m.group(1))
        params = m.group(2)
        d = {}
        for kv in re.findall(r'(\w+)=([^,]+?)(?=,|\s*$)', params):
            k, v = kv
            v = v.strip()
            if v.startswith('['): d[k] = v
            elif v.lstrip('-').replace('.', '').isdigit():
                d[k] = float(v) if '.' in v else int(v)
            else:
                d[k] = v.strip('"')
        out[seq] = d
    return out


def parse_recall(run_dir: Path):
    """Map (seq, L) -> recall@10."""
    rf = run_dir / 'recall_posthoc.json'
    if not rf.exists():
        return {}
    r = json.loads(rf.read_text())
    out = {}
    for op in r.get('search_ops', []):
        seq = op.get('seq')
        for L, rv in (op.get('recall_by_L') or {}).items():
            out[(seq, int(L))] = float(rv)
    return out


def process_run(run_dir: Path, writer):
    stats_dir = run_dir / 'node_stats'
    if not stats_dir.is_dir():
        print(f"[skip] {run_dir.name}: no node_stats/", file=sys.stderr)
        return 0
    workload = parse_workload(run_dir)
    recalls = parse_recall(run_dir)
    count = 0
    for f in sorted(stats_dir.glob('node*_seq*.json')):
        m = re.match(r'node(\d+)_seq(\d+)\.json', f.name)
        if not m: continue
        node_id = int(m.group(1))
        seq = int(m.group(2))
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"[warn] {f}: {e}", file=sys.stderr)
            continue
        if data.get('op') != 'search': continue
        op_params = workload.get(seq, {})
        T = op_params.get('num_threads', '')
        pq_alpha = op_params.get('pq_alpha', '')
        batch_size = op_params.get('batch_size', '')
        overlap_mode = op_params.get('overlap_mode', '')
        for L_str, L_stats in data.get('search_stats_by_L', {}).items():
            L = int(L_str)
            row = {
                'run': run_dir.name,
                'seq': seq,
                'node_id': node_id,
                'T': T,
                'L': L,
                'pq_alpha': pq_alpha,
                'batch_size': batch_size,
                'overlap_mode': overlap_mode,
                'recall_10': recalls.get((seq, L), ''),
            }
            for k in SEARCH_COLS:
                row[k] = L_stats.get(k, '')
            cpu = L_stats.get('cpu_breakdown', {}) or {}
            for k in CPU_COLS:
                row[k] = cpu.get(k, '')
            writer.writerow(row)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='Run dirs or glob patterns')
    ap.add_argument('-o', '--out', default='stats.csv')
    args = ap.parse_args()

    run_dirs = []
    for pat in args.runs:
        matches = [Path(p) for p in glob.glob(pat)] or [Path(pat)]
        run_dirs.extend(m for m in matches if m.is_dir())

    fields = (['run', 'seq', 'node_id', 'T', 'L', 'pq_alpha', 'batch_size',
               'overlap_mode', 'recall_10']
              + SEARCH_COLS + CPU_COLS)

    with open(args.out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        total = 0
        for rd in run_dirs:
            n = process_run(rd, writer)
            total += n
            print(f"[{rd.name}] {n} rows")
    print(f"Wrote {total} rows → {args.out}")


if __name__ == '__main__':
    main()
