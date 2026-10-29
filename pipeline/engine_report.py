"""Human-readable report generation for OperationEngine runs."""

import json
import os as _os
from pathlib import Path
from typing import Dict, List

import numpy as np


# ------------------------------------------------------------------
# Human-readable report generation
# ------------------------------------------------------------------

def _load_pmu_csv(engine, seq) -> Dict[str, float]:
    """Load per-op PMU counters from <run_dir>/pmu_<seq>.csv (if present).

    Format accepted (either):
      long:  two-column CSV with header 'metric,value', one row per metric
      wide:  header line of metric names + a single row of values
    Missing/invalid → returns {} so the caller just skips the section.
    """
    run_dir = getattr(engine, 'run_dir', None)
    if run_dir is None:
        return {}
    # Accept a few plausible per-seq filenames
    for fname in (f'pmu_{seq}.csv', f'pmu_seq{seq}.csv'):
        p = Path(run_dir) / fname
        if p.exists():
            try:
                rows = [ln.strip() for ln in p.read_text().splitlines()
                        if ln.strip() and not ln.lstrip().startswith('#')]
                if not rows:
                    return {}
                header = [h.strip() for h in rows[0].split(',')]
                # long format (metric,value)
                if [h.lower() for h in header] == ['metric', 'value']:
                    out: Dict[str, float] = {}
                    for ln in rows[1:]:
                        parts = [x.strip() for x in ln.split(',')]
                        if len(parts) >= 2:
                            try:
                                out[parts[0].lower()] = float(parts[1])
                            except ValueError:
                                pass
                    return out
                # wide format: header row + one data row
                if len(rows) >= 2:
                    vals = [x.strip() for x in rows[1].split(',')]
                    out = {}
                    for h, v in zip(header, vals):
                        try:
                            out[h.lower()] = float(v)
                        except ValueError:
                            pass
                    return out
            except Exception:
                return {}
    return {}

def generate_engine_report(engine) -> str:
    """Generate a human-readable text report from op_history."""
    lines = []
    W = 88  # report width

    def hr(ch='='):
        lines.append(ch * W)

    def section(title):
        lines.append('')
        hr()
        lines.append(f'  {title}')
        hr()

    def _total_items(op, key='items_processed'):
        """Sum items across all nodes from per_node data."""
        pn = op.get('per_node', {})
        if pn:
            return sum(ns.get(key, 0) for ns in pn.values())
        return op.get(key, 0)

    def _max_time(op):
        """Get max wall_time across nodes (= actual elapsed)."""
        pn = op.get('per_node', {})
        if pn:
            return max(ns.get('wall_time_s', 0) for ns in pn.values())
        return op.get('wall_time_s', 0)

    def _get_node(pn, nid):
        """Lookup per_node entry by nid (handles int/str key mismatch)."""
        return pn.get(nid, pn.get(str(nid), pn.get(int(nid), {})))

    node_ids = set()
    for op in engine.op_history:
        pn = op.get('per_node', {})
        node_ids.update(pn.keys())
    # Normalize to str for consistent lookups (keys may be int or str)
    node_ids = sorted([str(k) for k in node_ids], key=lambda x: int(x))

    total_wall = sum(op.get('wall_time_s', 0) for op in engine.op_history)

    # --- Top 1-liner summary (glance-level) ---
    update_cfg_top = engine.ctx.config.get('distributed_engine', {})
    workload_cfg_top = update_cfg_top.get('workload', [])
    best_qps = 0.0
    best_L = best_T = None
    for _idx, _op in enumerate(engine.op_history):
        if _op.get('op') != 'search':
            continue
        _pn = _op.get('per_node', {})
        _sources = list(_pn.values()) if _pn else [_op]
        _qps_by_L: Dict[int, Dict[str, float]] = {}
        for _src in _sources:
            _sbl = _src.get('search_stats_by_L', {})
            for _Ls, _s in _sbl.items():
                _L = int(_Ls)
                _cur = _qps_by_L.setdefault(_L, {'q': 0.0, 'el_max': 0.0})
                _cur['q'] += float(_s.get('num_queries', 0))
                _cur['el_max'] = max(_cur['el_max'], float(_s.get('elapsed_s', 0)))
        _T = 1
        if _idx < len(workload_cfg_top):
            _T = workload_cfg_top[_idx].get('num_threads', 1)
        for _L, _v in _qps_by_L.items():
            if _v['el_max'] > 0:
                _q = _v['q'] / _v['el_max']
                if _q > best_qps:
                    best_qps = _q
                    best_L, best_T = _L, _T

    _exp_name = engine.ctx.config.get('experiment_name', 'unnamed')
    _bits = [f'Run: {_exp_name}',
             f'{len(engine.op_history)} ops',
             f'{total_wall:.1f}s']
    if best_qps > 0:
        _bits.append(f'best QPS: {best_qps:,.1f} @ L={best_L} T={best_T}')
    lines.append('  ' + ' | '.join(_bits))

    # --- Header ---
    section('DISTRIBUTED ENGINE REPORT')
    lines.append(f'  Experiment : {engine.ctx.config.get("experiment_name", "unnamed")}')
    lines.append(f'  Nodes      : {len(node_ids)} ({", ".join(str(n) for n in node_ids)})')
    lines.append(f'  Operations : {len(engine.op_history)}')
    lines.append(f'  Total time : {total_wall:.2f}s')

    # --- Configuration snapshot ---
    cfg = engine.ctx.config
    section('CONFIG')
    # Dataset
    ds = cfg.get('dataset', {})
    lines.append(f'  Dataset     : {ds.get("name", "?")} '
                  f'(dim={ds.get("dimension", "?")}, type={ds.get("data_type", "?")})')
    # Build
    build = cfg.get('build', {})
    lines.append(f'  Build       : R={build.get("R", "?")}, L={build.get("L", "?")}, '
                  f'alpha={build.get("alpha", "?")}, '
                  f'initial={build.get("initial_size", "?"):,}')
    lines.append(f'  Graph var.  : {build.get("graph_variant", "(latest)")}')
    # Partition
    part = cfg.get('partition', {})
    lines.append(f'  Partitions  : {part.get("num_partitions", "?")}, '
                  f'insert_strategy={part.get("insert_strategy", "nearest_centroid")}')
    # Colocation
    coloc = cfg.get('colocation', {})
    lines.append(f'  Colocation  : fetch_mode={coloc.get("fetch_mode", "?")}, '
                  f'max_cap_factor={coloc.get("max_capacity_factor", "?")}')
    # Search
    search_cfg = cfg.get('search', {})
    lines.append(f'  Search      : K={search_cfg.get("K", "?")}, '
                  f'L_values={search_cfg.get("L_values", "?")}, '
                  f'num_threads={search_cfg.get("num_threads", 1)}')
    # Convergence-Aware Prefetching
    cap = cfg.get('convergence_aware_prefetch', {})
    bf = cap.get('batched_fetching', {}) if cap else {}
    pqf = cap.get('pq_filtering', {}) if cap else cfg.get('pq_filter', {})
    if not bf:
        bp = cfg.get('batch_prefetch', {})
        bf = {
            'batch_size': bp.get('batch_size', 0),
            'convergence_window': bp.get('conv_window', 3),
            'convergence_threshold_pct': bp.get('conv_threshold_pct', 1.0),
        }
    lines.append(f'  Conv-aware : enabled={cap.get("enabled", False)}, '
                  f'batch={bf.get("batch_size", 0)}, '
                  f'window={bf.get("convergence_window", bf.get("conv_window", 3))}, '
                  f'pq={pqf.get("enabled", False)}')
    # Consolidation
    consol = cfg.get('consolidation', {})
    cache = cfg.get('consolidation_cache', {})
    lines.append(f'  Consolidate : cache={cache.get("enabled", consol.get("enable_vec_cache", False))}, '
                  f'num_threads={consol.get("num_threads", 1)}, '
                  f'multi_qp={consol.get("use_multi_qp", False)}')
    # Distributed search latency model
    ds_cfg = cfg.get('distributed_search', {}).get('latency', {})
    if ds_cfg:
        lines.append(f'  Latency sim : local={ds_cfg.get("local_us", 0)}us, '
                      f'remote={ds_cfg.get("remote_us", 0)}us, '
                      f'jitter={ds_cfg.get("enable_jitter", False)}')

    # --- 0. Workload Config ---
    update_cfg = cfg.get('distributed_engine', {})
    workload_cfg = update_cfg.get('workload', [])
    if workload_cfg:
        section('0. WORKLOAD')
        for i, spec in enumerate(workload_cfg):
            op_name = spec.get('op', '?')
            params = {k: v for k, v in spec.items() if k != 'op'}
            param_str = ', '.join(f'{k}={v}' for k, v in params.items())
            lines.append(f'  {i:3d}. {op_name:<14} {param_str}')

    # --- 1. Operation Timeline (all-node totals) ---
    # Pull max loop_only across nodes for this op (only consolidate sets it).
    def _op_loop_only(op):
        pn = op.get('per_node', {})
        if pn:
            vals = [ns.get('loop_only_sec', 0.0) for ns in pn.values()]
            return max(vals) if vals else 0.0
        return op.get('loop_only_sec', 0.0)

    section('1. OPERATION TIMELINE')
    has_active = any('active_points' in op for op in engine.op_history)
    if has_active:
        lines.append(f'  {"Seq":>3}  {"Op":<14} {"Status":<10} {"Wall(s)":>8}  '
                      f'{"Loop(s)":>8}  '
                      f'{"Active":>10}  {"Total":>10}  {"Failed":>7}  {"Per-Node Items":}')
        lines.append(f'  {"---":>3}  {"--":<14} {"------":<10} {"-------":>8}  '
                      f'{"-------":>8}  '
                      f'{"------":>10}  {"-----":>10}  {"------":>7}  {"-" * 30}')
    else:
        lines.append(f'  {"Seq":>3}  {"Op":<14} {"Status":<10} {"Wall(s)":>8}  '
                      f'{"Loop(s)":>8}  '
                      f'{"Total":>10}  {"Failed":>7}  {"Per-Node Items":}')
        lines.append(f'  {"---":>3}  {"--":<14} {"------":<10} {"-------":>8}  '
                      f'{"-------":>8}  '
                      f'{"-----":>10}  {"------":>7}  {"-" * 30}')
    for op in engine.op_history:
        total_proc = _total_items(op)
        total_fail = _total_items(op, 'items_failed')
        active = op.get('active_points', '')
        loop_only = _op_loop_only(op)
        loop_str = f'{loop_only:8.2f}' if loop_only > 0 else f'{"":>8}'
        # Per-node item breakdown
        pn = op.get('per_node', {})
        if pn and len(pn) > 1:
            node_items_str = '  '.join(
                f'N{nid}:{ns.get("items_processed", 0)}'
                for nid, ns in sorted(pn.items(), key=lambda x: int(x[0])))
        else:
            node_items_str = ''
        # Per-node active points
        active_per_node = op.get('active_per_node', {})
        if active_per_node:
            active_pn_str = '  '.join(
                f'N{nid}:{cnt:,}'
                for nid, cnt in sorted(active_per_node.items(), key=lambda x: int(x[0])))
        else:
            active_pn_str = ''
        if has_active:
            active_str = f'{active:10,}' if isinstance(active, int) else f'{"":>10}'
            lines.append(f'  {op["seq"]:3d}  {op["op"]:<14} {op["status"]:<10} '
                          f'{op["wall_time_s"]:8.2f}  '
                          f'{loop_str}  '
                          f'{active_str}  '
                          f'{total_proc:10,}  '
                          f'{total_fail:7,}  {node_items_str}')
            if active_pn_str:
                lines.append(f'  {"":>3}  {"":14} {"":10} {"":>8}  '
                              f'{"":>8}  '
                              f'{"(per-node)":>10}  {active_pn_str}')
        else:
            lines.append(f'  {op["seq"]:3d}  {op["op"]:<14} {op["status"]:<10} '
                          f'{op["wall_time_s"]:8.2f}  '
                          f'{loop_str}  '
                          f'{total_proc:10,}  '
                          f'{total_fail:7,}  {node_items_str}')

    # --- 1.5: Consolidation timing breakdown ---
    # Surfaces qp_setup / body / loop_only metrics so the table can be
    # used directly in figures. loop_only is the "pure prune work" — the
    # fair algorithm-cost comparison vs DiskANN baseline, since setup
    # (vec_storage populate, vid_to_idx build) is amortizable.
    _consol_ops = [op for op in engine.op_history if op['op'] == 'consolidate']
    if _consol_ops:
        section('1.5 CONSOLIDATION TIMING BREAKDOWN (per-node max)')
        lines.append(f'  {"Seq":>3}  {"Wall(s)":>8}  {"qp_setup(s)":>12}  '
                      f'{"body(s)":>8}  {"loop_only(s)":>12}  '
                      f'{"vec_buf(GB)":>11}  {"N_di":>10}')
        lines.append(f'  {"---":>3}  {"-------":>8}  {"-----------":>12}  '
                      f'{"-------":>8}  {"------------":>12}  '
                      f'{"-----------":>11}  {"----------":>10}')
        for _cop in _consol_ops:
            _seq = _cop.get('seq', 0)
            _wall = _cop.get('wall_time_s', 0)
            # Custom fields surface from C++ stats / engine.py wrappers.
            # Look at per-node values and report max (slowest node bounds wall).
            def _max_field(field, default=0.0):
                if 'per_node' in _cop and _cop['per_node']:
                    return max(ns.get(field, default)
                               for ns in _cop['per_node'].values())
                return _cop.get(field, default)
            _qp = _max_field('qp_setup_sec')
            _body = _max_field('consol_body_sec')
            _loop = _max_field('loop_only_sec')
            # Per-Di / per-vid cache memory footprint. vec_buffer_peak_bytes
            # is the size of the static buffer (mmap allocation). For the
            # per-Di path it scales with N_di × slack_R × vec_bytes_per
            # and is the dominant memory cost; track it here so sweep
            # cells that approach DPU RAM are visible at a glance.
            _peak_b = _max_field('vec_buffer_peak_bytes')
            _peak_gb = _peak_b / (1024 ** 3) if _peak_b else 0.0
            # unique_vec_fetches is N_di in the per-Di cache path
            # (set to len(di_list) ≈ unique deleted vids referenced).
            _ndi = int(_max_field('unique_vec_fetches'))
            _peak_str = f'{_peak_gb:11.2f}' if _peak_b else f'{"":>11}'
            _ndi_str = f'{_ndi:10,}' if _ndi else f'{"":>10}'
            lines.append(f'  {_seq:3d}  {_wall:8.2f}  {_qp:12.2f}  '
                          f'{_body:8.2f}  {_loop:12.2f}  '
                          f'{_peak_str}  {_ndi_str}')

        # 1.6: Per-node NIC BW for each consolidate op. Surfaces both
        # delta-based avg over the wall window and 1ms-burst peak/p99
        # so cache ON vs OFF can be compared on actual link utilization,
        # not just C++ bytes_fetched counters.
        _has_bw = any(
            any('nic_ib_rx_gbps' in ns for ns in _cop.get('per_node', {}).values())
            or 'nic_ib_rx_gbps' in _cop
            for _cop in _consol_ops)
        if _has_bw:
            section('1.6 CONSOLIDATION NIC BANDWIDTH (per-node)')
            lines.append(f'  {"Seq":>3}  {"Node":>4}  {"avg_rx":>8}  {"avg_tx":>8}  '
                          f'{"peak_rx":>8}  {"peak_tx":>8}  {"p99_rx":>8}  {"p99_tx":>8}  '
                          f'{"xmit_wait":>10}')
            lines.append(f'  {"---":>3}  {"----":>4}  {"------":>8}  {"------":>8}  '
                          f'{"-------":>8}  {"-------":>8}  {"------":>8}  {"------":>8}  '
                          f'{"---------":>10}')
            for _cop in _consol_ops:
                _seq = _cop.get('seq', 0)
                _pn = _cop.get('per_node', {}) or {0: _cop}
                for _nid in sorted(_pn.keys(), key=lambda x: int(x)):
                    _ns = _pn[_nid]
                    _rx = _ns.get('nic_ib_rx_gbps', 0.0) or 0.0
                    _tx = _ns.get('nic_ib_tx_gbps', 0.0) or 0.0
                    _prx = _ns.get('nic_burst_rx_max_gbps', 0.0) or 0.0
                    _ptx = _ns.get('nic_burst_tx_max_gbps', 0.0) or 0.0
                    _99rx = _ns.get('nic_burst_rx_p99_gbps', 0.0) or 0.0
                    _99tx = _ns.get('nic_burst_tx_p99_gbps', 0.0) or 0.0
                    _xw = int(_ns.get('nic_xmit_wait', 0) or 0)
                    lines.append(f'  {_seq:3d}  {int(_nid):4d}  {_rx:8.2f}  {_tx:8.2f}  '
                                  f'{_prx:8.2f}  {_ptx:8.2f}  {_99rx:8.2f}  {_99tx:8.2f}  '
                                  f'{_xw:10d}')

        # 1.7: Per-phase NIC BW slicing. Lets the loop-only window be
        # compared apples-to-apples between cache ON (where loop is just
        # the prune for-loop) and OFF (where loop covers the full body).
        phase_names = ('scan', 'setup', 'populate', 'loop')
        _has_phase = any(
            any(f'nic_phase_{p}_nic_burst_rx_max_gbps' in ns
                for p in phase_names
                for ns in _cop.get('per_node', {}).values())
            or any(f'nic_phase_{p}_nic_burst_rx_max_gbps' in _cop
                   for p in phase_names)
            for _cop in _consol_ops)
        if _has_phase:
            section('1.7 CONSOLIDATION NIC BW BY PHASE (per-node)')
            lines.append(f'  {"Seq":>3}  {"Node":>4}  {"Phase":<9}  '
                          f'{"avg_rx":>8}  {"avg_tx":>8}  '
                          f'{"peak_rx":>8}  {"peak_tx":>8}  '
                          f'{"p99_rx":>8}  {"p99_tx":>8}')
            lines.append(f'  {"---":>3}  {"----":>4}  {"---------":<9}  '
                          f'{"------":>8}  {"------":>8}  '
                          f'{"-------":>8}  {"-------":>8}  '
                          f'{"------":>8}  {"------":>8}')
            for _cop in _consol_ops:
                _seq = _cop.get('seq', 0)
                _pn = _cop.get('per_node', {}) or {0: _cop}
                for _nid in sorted(_pn.keys(), key=lambda x: int(x)):
                    _ns = _pn[_nid]
                    for _ph in phase_names:
                        _key_root = f'nic_phase_{_ph}_'
                        if (_key_root + 'nic_burst_rx_max_gbps') not in _ns:
                            continue
                        _arx = _ns.get(_key_root + 'nic_burst_rx_avg_gbps', 0.0) or 0.0
                        _atx = _ns.get(_key_root + 'nic_burst_tx_avg_gbps', 0.0) or 0.0
                        _prx = _ns.get(_key_root + 'nic_burst_rx_max_gbps', 0.0) or 0.0
                        _ptx = _ns.get(_key_root + 'nic_burst_tx_max_gbps', 0.0) or 0.0
                        _9rx = _ns.get(_key_root + 'nic_burst_rx_p99_gbps', 0.0) or 0.0
                        _9tx = _ns.get(_key_root + 'nic_burst_tx_p99_gbps', 0.0) or 0.0
                        lines.append(f'  {_seq:3d}  {int(_nid):4d}  {_ph:<9}  '
                                      f'{_arx:8.2f}  {_atx:8.2f}  '
                                      f'{_prx:8.2f}  {_ptx:8.2f}  '
                                      f'{_9rx:8.2f}  {_9tx:8.2f}')

    # (Section 2 removed — "Time by operation type" wasn't informative
    # once the timeline at Section 1 already shows per-op wall times.)
    # We still need op_types for downstream per-op-type table in Sec 4.
    op_types: Dict[str, Dict[str, Any]] = {}
    for op in engine.op_history:
        t = op['op']
        if t not in op_types:
            op_types[t] = {'count': 0, 'total_s': 0.0, 'items': 0}
        op_types[t]['count'] += 1
        op_types[t]['total_s'] += op.get('wall_time_s', 0)
        op_types[t]['items'] += _total_items(op)

    # --- 2. Search Summary (QPS × L × T, compact comparison table) ---
    _search_ops_for_summary = [op for op in engine.op_history
                                if op['op'] == 'search']
    if _search_ops_for_summary:
        section('2. SEARCH SUMMARY (QPS across L × T)')
        _workload_top = engine.ctx.config.get(
            'distributed_engine', {}).get('workload', [])
        _rows = []
        for _idx, _op in enumerate(_search_ops_for_summary):
            _seq = _op.get('seq', _idx)
            _T = 1
            if _seq < len(_workload_top):
                _T = _workload_top[_seq].get('num_threads', 1)
            _pn = _op.get('per_node', {})
            _srcs = list(_pn.values()) if _pn else [_op]
            _per_L: Dict[int, Dict[str, float]] = {}
            _nic_max: Dict[int, float] = {}
            for _src in _srcs:
                _sbl = _src.get('search_stats_by_L', {})
                for _Ls, _s in _sbl.items():
                    _L = int(_Ls)
                    _c = _per_L.setdefault(_L, {'q': 0.0, 'el_max': 0.0,
                                                 'fec': 0, 'rdma_us': 0.0,
                                                 'fetches': 0})
                    _c['q'] += float(_s.get('num_queries', 0))
                    _c['el_max'] = max(_c['el_max'], float(_s.get('elapsed_s', 0)))
                    _c['fec'] += int(_s.get('fetch_entry_calls', 0))
                    _c['rdma_us'] += float(_s.get('rdma_total_us', 0))
                    _c['fetches'] += int(_s.get('fetch_entry_items', 0))
                    _nic_max[_L] = max(_nic_max.get(_L, 0),
                                       float(_s.get('nic_rx_gbps', 0)))
            for _L, _c in sorted(_per_L.items()):
                _qps = _c['q'] / _c['el_max'] if _c['el_max'] > 0 else 0
                _us_call = (_c['rdma_us'] / _c['fec']) if _c['fec'] > 0 else 0
                _rows.append({
                    'seq': _seq, 'L': _L, 'T': _T,
                    'qps': _qps,
                    'wall': _c['el_max'],
                    'nic': _nic_max.get(_L, 0),
                    'us_call': _us_call,
                    'fetches': _c['fetches'],
                })
        if _rows:
            lines.append(f'  {"Seq":>3}  {"L":>4}  {"T":>3}  {"QPS":>10}  '
                         f'{"Wall(s)":>8}  {"IB_RX":>7}  {"us/call":>8}  '
                         f'{"Fetches":>12}')
            lines.append(f'  {"---":>3}  {"--":>4}  {"--":>3}  {"-"*8:>10}  '
                         f'{"-"*7:>8}  {"-"*6:>7}  {"-"*7:>8}  {"-"*11:>12}')
            # Sort rows: by L then by T for easy vertical scan of scaling
            for r in sorted(_rows, key=lambda x: (x['L'], x['T'])):
                _nic_s = f'{r["nic"]:>5.1f}G' if r['nic'] > 0 else f'{"N/A":>7}'
                _usc_s = f'{r["us_call"]:>5.2f}us' if r['us_call'] > 0 else f'{"N/A":>8}'
                lines.append(f'  {r["seq"]:>3}  {r["L"]:>4}  {r["T"]:>3}  '
                             f'{r["qps"]:>10,.1f}  {r["wall"]:>8.2f}  '
                             f'{_nic_s}  {_usc_s}  {r["fetches"]:>12,}')
            # Peak-per-L one-liner
            _by_L: Dict[int, dict] = {}
            for r in _rows:
                cur = _by_L.get(r['L'])
                if cur is None or r['qps'] > cur['qps']:
                    _by_L[r['L']] = r
            if _by_L:
                lines.append('')
                lines.append('  Peak QPS per L:')
                for _L in sorted(_by_L.keys()):
                    r = _by_L[_L]
                    lines.append(f'    L={_L:<4} → {r["qps"]:>10,.1f} QPS  @ T={r["T"]}')

    # --- 3. Insert Latency Breakdown (all nodes merged) ---
    insert_ops = [op for op in engine.op_history if op['op'] == 'insert'
                  and ('latency_breakdown' in op or 'per_node' in op)]
    if insert_ops:
        section('3. INSERT LATENCY BREAKDOWN (ms) — all nodes')
        phases = ['search_ms', 'prune_ms', 'write_ms', 'reverse_edge_ms']

        # Map flat C++ insert_stats (total sec) to report phase names (ms)
        _sec_to_phase = {
            'search_sec': 'search_ms',
            'prune_sec': 'prune_ms',
            'write_sec': 'write_ms',
            'reverse_sec': 'reverse_edge_ms',
        }

        def _flat_lb_to_stats(lb, items):
            """Convert flat {phase_sec: total_float} to {phase_ms: {mean,...}}."""
            result = {}
            for sec_key, ms_key in _sec_to_phase.items():
                val = lb.get(sec_key, 0)
                if not val or not isinstance(val, (int, float)):
                    continue
                mean_ms = val * 1000 / items if items > 0 else 0.0
                result[ms_key] = {
                    'mean': round(mean_ms, 3),
                    'p50': round(mean_ms, 3),
                    'p99': round(mean_ms, 3),
                    'min': round(mean_ms, 3),
                    'max': round(mean_ms, 3),
                    'count': items,
                }
            return result

        def _merge_latency_breakdowns(op):
            """Merge latency_breakdown from all nodes via per_node data."""
            pn = op.get('per_node', {})
            if not pn:
                # Fallback: coordinator-only breakdown
                lb = op.get('latency_breakdown', {})
                # Handle flat format (search_sec, prune_sec, ...)
                if lb and any(k.endswith('_sec') for k in lb):
                    return _flat_lb_to_stats(lb, op.get('items_processed', 0))
                return lb
            # First pass: convert flat format per node if needed
            for nid, ns in pn.items():
                lb = ns.get('latency_breakdown', {})
                if lb and any(k.endswith('_sec') for k in lb):
                    ns['latency_breakdown'] = _flat_lb_to_stats(
                        lb, ns.get('items_processed', 0))
            # Collect per-phase raw stats from all nodes, compute weighted merge
            merged = {}
            for phase in phases:
                total_count = 0
                weighted_mean_sum = 0.0
                all_min = []
                all_max = []
                # Collect all per-node p50/p99 values for re-aggregation
                all_p50 = []
                all_p99 = []
                node_counts = []
                for nid, ns in pn.items():
                    lb = ns.get('latency_breakdown', {})
                    s = lb.get(phase, {})
                    if not s:
                        continue
                    c = s.get('count', 0)
                    if c == 0:
                        continue
                    total_count += c
                    weighted_mean_sum += s['mean'] * c
                    all_min.append(s['min'])
                    all_max.append(s['max'])
                    all_p50.append((c, s['p50']))
                    all_p99.append((c, s['p99']))
                    node_counts.append(c)
                if total_count == 0:
                    continue
                # Weighted average for p50/p99 (approximation — exact
                # requires raw data, but count-weighted mean of percentiles
                # is close enough for report purposes)
                w_p50 = sum(c * v for c, v in all_p50) / total_count
                w_p99 = sum(c * v for c, v in all_p99) / total_count
                merged[phase] = {
                    'mean': round(weighted_mean_sum / total_count, 3),
                    'p50': round(w_p50, 3),
                    'p99': round(w_p99, 3),
                    'min': round(min(all_min), 3),
                    'max': round(max(all_max), 3),
                    'count': total_count,
                }
            return merged

        def _sum_lits(op):
            """Sum LITS forward-pool {checked, filtered} across all nodes."""
            pn = op.get('per_node', {})
            if pn:
                checked = sum(ns.get('lits', {}).get('checked', 0)
                              for ns in pn.values())
                filtered = sum(ns.get('lits', {}).get('filtered', 0)
                               for ns in pn.values())
            else:
                l = op.get('lits', {})
                checked, filtered = l.get('checked', 0), l.get('filtered', 0)
            return checked, filtered

        def _sum_lb(op):
            """Aggregate LoadBalancer (Stage B) stats across all nodes.

            Returns (considered, would_redirect, inserted, counters) where
            counters merges each node's batch-start snapshot {node: count}.
            """
            pn = op.get('per_node', {})
            sources = list(pn.values()) if pn else [op]
            considered = would = inserted = 0
            r_out = r_in = r_fb = 0
            counters = {}
            for ns in sources:
                lb = ns.get('load_balancer', {})
                considered += lb.get('considered', 0)
                would += lb.get('would_redirect', 0)
                inserted += lb.get('inserted', 0)
                r_out += lb.get('redirected_out', 0)
                r_in += lb.get('redirected_in', 0)
                r_fb += lb.get('redirect_fallback', 0)
                for nid, cnt in lb.get('counters', {}).items():
                    counters[int(nid)] = max(counters.get(int(nid), 0), int(cnt))
            return considered, would, inserted, counters, (r_out, r_in, r_fb)

        lines.append(f'  {"Seq":>3}  {"Phase":<16} {"Mean":>8} {"P50":>8} '
                      f'{"P99":>8} {"Min":>8} {"Max":>8} {"Count":>7}')
        lines.append(f'  {"---":>3}  {"-----":<16} {"----":>8} {"---":>8} '
                      f'{"---":>8} {"---":>8} {"---":>8} {"-----":>7}')
        all_merged = []
        for op in insert_ops:
            merged_lb = _merge_latency_breakdowns(op)
            all_merged.append(merged_lb)
            for phase in phases:
                s = merged_lb.get(phase, {})
                if not s:
                    continue
                lines.append(
                    f'  {op["seq"]:3d}  {phase:<16} {s["mean"]:8.3f} '
                    f'{s["p50"]:8.3f} {s["p99"]:8.3f} '
                    f'{s["min"]:8.3f} {s["max"]:8.3f} {s["count"]:7d}')
            lc, lf = _sum_lits(op)
            if lc > 0:
                lines.append(
                    f'  {op["seq"]:3d}  {"LITS filter":<16} '
                    f'checked={lc:,}  filtered={lf:,}  '
                    f'ratio={100*lf/lc:.1f}%')
            # LoadBalancer (Stage B, measure-only): the Stage-C go/no-go
            # numbers — would-redirect rate + per-node counter imbalance.
            lb_cons, lb_would, lb_ins, lb_counters, lb_redir = _sum_lb(op)
            if lb_ins > 0:
                r_out, r_in, r_fb = lb_redir
                redir_str = (f'  moved={r_in:,}/{r_out:,} fallback={r_fb:,}'
                             if r_out else '')
                lines.append(
                    f'  {op["seq"]:3d}  {"LoadBalancer":<16} '
                    f'considered={lb_cons:,} ({100*lb_cons/lb_ins:.1f}%)  '
                    f'would_redirect={lb_would:,}  '
                    f'redirect_rate={100*lb_would/lb_ins:.2f}%{redir_str}')
                if lb_counters:
                    cmin, cmax = min(lb_counters.values()), max(lb_counters.values())
                    imb = (cmax / cmin) if cmin > 0 else float('inf')
                    per_node = '  '.join(f'n{n}={c:,}'
                                         for n, c in sorted(lb_counters.items()))
                    lines.append(
                        f'  {"":>3}  {"":<16} counters: {per_node}  '
                        f'imbalance(max/min)={imb:.3f}')
            lines.append('')

        # Aggregate across all inserts
        lines.append('  --- Aggregate (all inserts) ---')
        for phase in phases:
            all_means = [lb.get(phase, {}).get('mean', 0)
                         for lb in all_merged if lb.get(phase)]
            if all_means:
                lines.append(f'  {"":>3}  {phase:<16} '
                              f'avg_mean={sum(all_means)/len(all_means):.3f}ms  '
                              f'range=[{min(all_means):.3f}, {max(all_means):.3f}]')

        # Per-node breakdown
        if any(op.get('per_node') for op in insert_ops):
            lines.append('')
            lines.append('  --- Per-node mean (ms) ---')
            lines.append(f'  {"Node":>6}  {"search":>8} {"prune":>8} '
                          f'{"write":>8} {"reverse":>8} {"inserts":>8}')
            lines.append(f'  {"----":>6}  {"------":>8} {"-----":>8} '
                          f'{"-----":>8} {"-------":>8} {"-------":>8}')
            for nid in node_ids:
                phase_totals = {p: (0.0, 0) for p in phases}
                for op in insert_ops:
                    ns = _get_node(op.get('per_node', {}), nid)
                    lb = ns.get('latency_breakdown', {})
                    for p in phases:
                        s = lb.get(p, {})
                        if s and s.get('count', 0) > 0:
                            wsum, cnt = phase_totals[p]
                            phase_totals[p] = (wsum + s['mean'] * s['count'],
                                               cnt + s['count'])
                vals = {}
                total_inserts = 0
                for p in phases:
                    wsum, cnt = phase_totals[p]
                    vals[p] = wsum / cnt if cnt > 0 else 0.0
                    if p == phases[0]:
                        total_inserts = cnt
                lines.append(
                    f'  {nid:>6}  {vals["search_ms"]:8.3f} '
                    f'{vals["prune_ms"]:8.3f} {vals["write_ms"]:8.3f} '
                    f'{vals["reverse_edge_ms"]:8.3f} {total_inserts:8d}')

    # --- 3b. Consolidation Profiling Breakdown ---
    consol_ops = [op for op in engine.op_history
                  if op['op'] == 'consolidate' and op.get('custom')]
    if consol_ops:
        section('3b. CONSOLIDATION PROFILING BREAKDOWN')
        for op in consol_ops:
            c = op['custom']
            elapsed_s = c.get('elapsed_seconds', 0)
            elapsed_ns = elapsed_s * 1e9
            threads = c.get('num_threads', 1)

            def _ns_pct(key):
                v = c.get(key, 0)
                pct = v / elapsed_ns * 100 if elapsed_ns > 0 else 0
                return v / 1e9, pct

            lock_s, lock_p = _ns_pct('twohop_lock_wait_ns')
            vlock_s, vlock_p = _ns_pct('vecfetch_lock_wait_ns')
            clock_s, clock_p = _ns_pct('vecfetch_cache_lock_ns')
            rdma2_s, rdma2_p = _ns_pct('twohop_rdma_ns')
            rdmav_s, rdmav_p = _ns_pct('vecfetch_rdma_ns')
            loc2_s, loc2_p = _ns_pct('twohop_local_ns')
            locv_s, locv_p = _ns_pct('vecfetch_local_ns')
            dist_s, dist_p = _ns_pct('dist_compute_ns')
            prn_s, prn_p = _ns_pct('prune_ns')

            total_lock = lock_s + vlock_s + clock_s
            total_rdma = rdma2_s + rdmav_s
            total_local = loc2_s + locv_s

            lines.append(f'  Seq {op["seq"]}  threads={threads}  elapsed={elapsed_s:.2f}s  '
                          f'modified={c.get("modified", 0)}')
            lines.append(f'    {"Phase":<24} {"Time(s)":>8} {"%":>6}')
            lines.append(f'    {"-----":<24} {"------":>8} {"--":>6}')
            lines.append(f'    {"Lock Wait (total)":<24} {total_lock:8.3f} '
                          f'{(lock_p+vlock_p+clock_p):5.1f}%')
            lines.append(f'      2hop fetcher lock     {lock_s:8.3f} {lock_p:5.1f}%')
            lines.append(f'      vec fetcher lock      {vlock_s:8.3f} {vlock_p:5.1f}%')
            lines.append(f'      cache lock            {clock_s:8.3f} {clock_p:5.1f}%')
            lines.append(f'    {"RDMA (total)":<24} {total_rdma:8.3f} '
                          f'{(rdma2_p+rdmav_p):5.1f}%')
            lines.append(f'      2hop neighbor fetch   {rdma2_s:8.3f} {rdma2_p:5.1f}%')
            lines.append(f'      vec batch fetch       {rdmav_s:8.3f} {rdmav_p:5.1f}%')
            lines.append(f'    {"Local Access (total)":<24} {total_local:8.3f} '
                          f'{(loc2_p+locv_p):5.1f}%')
            lines.append(f'    {"Distance Compute":<24} {dist_s:8.3f} {dist_p:5.1f}%')
            lines.append(f'    {"RobustPrune":<24} {prn_s:8.3f} {prn_p:5.1f}%')
            lines.append(f'    Contentions: 2hop={c.get("twohop_lock_contentions", 0):,}  '
                          f'vecfetch={c.get("vecfetch_lock_contentions", 0):,}')
            # LITS filter stats
            lits_checked = c.get('lits_checked', 0)
            lits_filtered = c.get('lits_filtered', 0)
            if lits_checked > 0:
                lines.append(f'    LITS filter: checked={lits_checked:,}  '
                              f'filtered={lits_filtered:,}  '
                              f'ratio={100*lits_filtered/lits_checked:.1f}%')
            pt = c.get('per_thread_entries', [])
            if pt:
                lines.append(f'    Per-thread entries: {list(pt)}')

    # --- 4. RDMA Communication (all nodes) ---
    section('4. RDMA COMMUNICATION (all nodes)')

    def _accumulate_rdma(ops_list):
        """Sum RDMA counters across all nodes from per_node data."""
        totals = {
            'fetch_entry': {'calls': 0, 'items': 0},
            'vector_accesses': {'local': 0, 'remote': 0},
            'neighbor_accesses': {'local': 0, 'remote': 0},
            'write_neighbors': {'local': 0, 'remote': 0},
            'search_colocation_calls': 0,
        }
        for op in ops_list:
            pn = op.get('per_node', {})
            sources = list(pn.values()) if pn else [op]
            for src in sources:
                rc = src.get('rdma_counts', {})
                fe = rc.get('fetch_entry', {})
                totals['fetch_entry']['calls'] += fe.get('calls', 0)
                totals['fetch_entry']['items'] += fe.get('items', 0)
                for key in ['vector_accesses', 'neighbor_accesses', 'write_neighbors']:
                    d = rc.get(key, {})
                    totals[key]['local'] += d.get('local', 0)
                    totals[key]['remote'] += d.get('remote', 0)
                totals['search_colocation_calls'] += rc.get(
                    'search_colocation_calls', 0)
        return totals

    total_rdma = _accumulate_rdma(engine.op_history)

    # 4a. Actual RDMA fetch_entry (colocation entry = vec + nbr together)
    lines.append('  --- Actual RDMA round-trips ---')
    fe = total_rdma['fetch_entry']
    lines.append(f'  {"fetch_entry_calls":<22} {fe["calls"]:>12,}')
    lines.append(f'  {"fetch_entry_items":<22} {fe["items"]:>12,}')
    avg_batch = fe['items'] / fe['calls'] if fe['calls'] else 0
    lines.append(f'  {"avg_items_per_call":<22} {avg_batch:>12.1f}')

    # Per-op type breakdown (includes local vs remote in a compact form).
    # The old standalone "Logical accesses" table was a sum of this one,
    # so we drop it — this table gives the same data per-op-type.
    def _lr(local, remote):
        """Format local/remote pair as fixed-width string."""
        return f'{local:>11,} / {remote:<11,}'

    # Precompute all rows so we can hide columns whose total is zero
    # (e.g. write_remote is 0 in search-only runs; entry_* are 0 for
    # update-only runs).
    _col_defs = [
        ('entry_calls',  lambda r: r['fetch_entry']['calls']),
        ('entry_items',  lambda r: r['fetch_entry']['items']),
        ('vec_remote',   lambda r: r['vector_accesses']['remote']),
        ('nbr_remote',   lambda r: r['neighbor_accesses']['remote']),
        ('write_remote', lambda r: r['write_neighbors']['remote']),
        ('search',       lambda r: r['search_colocation_calls']),
    ]
    _type_rows = []
    for op_type in sorted(op_types.keys()):
        type_ops = [op for op in engine.op_history if op['op'] == op_type]
        _type_rows.append((op_type, _accumulate_rdma(type_ops)))
    _active_cols = [c for c in _col_defs
                    if any(c[1](r) for _, r in _type_rows)]

    lines.append('')
    lines.append('  --- Per operation type ---')
    _hdr_cells = [f'{"Type":<14}']
    _sep_cells = [f'{"----":<14}']
    for name, _ in _active_cols:
        w = max(12, len(name))
        _hdr_cells.append(f'{name:>{w}}')
        _sep_cells.append(f'{"-"*len(name):>{w}}')
    lines.append('  ' + '  '.join(_hdr_cells))
    lines.append('  ' + '  '.join(_sep_cells))
    for op_type, r in _type_rows:
        cells = [f'{op_type:<14}']
        for name, getter in _active_cols:
            w = max(12, len(name))
            cells.append(f'{getter(r):>{w},}')
        lines.append('  ' + '  '.join(cells))

    # Per-node totals
    if node_ids:
        lines.append('')
        lines.append('  --- Per node ---')
        lines.append(f'  {"Node":>6}  {"entry(calls/items)":>22}  '
                      f'{"vec_access":>25}  {"nbr_access":>25}  '
                      f'{"write_nbr":>25}  {"search":>8}')
        lines.append(f'  {"----":>6}  {"":>22}  '
                      f'{"(local / remote)":>25}  {"(local / remote)":>25}  '
                      f'{"(local / remote)":>25}  {"------":>8}')
        for nid in node_ids:
            fe_c = fe_i = va_l = va_r = na_l = na_r = wn_l = wn_r = sc = 0
            for op in engine.op_history:
                ns = _get_node(op.get('per_node', {}), nid)
                rc = ns.get('rdma_counts', {})
                fe = rc.get('fetch_entry', {})
                fe_c += fe.get('calls', 0); fe_i += fe.get('items', 0)
                va = rc.get('vector_accesses', {})
                va_l += va.get('local', 0); va_r += va.get('remote', 0)
                na = rc.get('neighbor_accesses', {})
                na_l += na.get('local', 0); na_r += na.get('remote', 0)
                wn = rc.get('write_neighbors', {})
                wn_l += wn.get('local', 0); wn_r += wn.get('remote', 0)
                sc += rc.get('search_colocation_calls', 0)
            lines.append(f'  {nid:>6}  {fe_c:>10,} / {fe_i:<10,}  '
                          f'{_lr(va_l, va_r)}  {_lr(na_l, na_r)}  '
                          f'{_lr(wn_l, wn_r)}  {sc:>8,}')

    # Search pure-colocation stats
    search_ops_with_stats = [op for op in engine.op_history
                             if op['op'] == 'search']
    has_search_stats = any(
        any(_get_node(op.get('per_node', {}), nid).get('search_pure_stats')
            for nid in node_ids)
        for op in search_ops_with_stats
    ) if node_ids else False
    if has_search_stats:
        lines.append('')
        lines.append('  --- Search pure-colocation stats (all search ops, all nodes) ---')
        agg = {}
        for op in search_ops_with_stats:
            pn = op.get('per_node', {})
            sources = list(pn.values()) if pn else [op]
            for src in sources:
                sr = src.get('search_pure_stats', {})
                for k, v in sr.items():
                    if isinstance(v, (int, float)):
                        agg[k] = agg.get(k, 0) + v
        # Essentials: hops, comparisons, vector local/remote ratio, fails.
        # Neighbor local/remote + entry calls/items are already shown in
        # sections above (Per-op-type / Per-node); don't repeat them.
        n_hops = int(agg.get('n_hops', 0))
        n_cmps = int(agg.get('n_cmps', 0))
        vec_l = int(agg.get('local_vector_accesses', 0))
        vec_r = int(agg.get('remote_vector_accesses', 0))
        vec_tot = vec_l + vec_r
        vec_rem_pct = (vec_r * 100.0 / vec_tot) if vec_tot else 0
        lines.append(f'    hops           : {n_hops:>14,}')
        lines.append(f'    comparisons    : {n_cmps:>14,}')
        lines.append(f'    vec local/rem  : {vec_l:>14,} / {vec_r:<14,}  ({vec_rem_pct:.1f}% remote)')
        resolve_fail = int(agg.get('resolve_fail', 0))
        if resolve_fail > 0:
            lines.append(f'    resolve_fail   : {resolve_fail:>14,}')

    # Per-search-op breakdown (one section per search op)
    for op_idx, op in enumerate(search_ops_with_stats):
        # Extract num_threads from op params for section header
        op_threads = 1
        # Load per-node stats from node_stats/ files
        pn = op.get('per_node', {})
        if not pn:
            seq = op.get('seq', op_idx)
            stats_dir = engine.run_dir / "node_stats" if hasattr(engine, 'run_dir') else None
            if stats_dir and stats_dir.exists():
                import glob as _glob
                for sf in sorted(stats_dir.glob(f"node*_seq{seq}.json")):
                    try:
                        with open(sf) as _f:
                            nd_stats = json.load(_f)
                        nid = nd_stats.get('node_id', sf.stem)
                        pn[nid] = nd_stats
                    except Exception:
                        pass
        for nd in pn.values():
            sbl = nd.get('search_stats_by_L', {})
            if sbl:
                break
        # Try to get num_threads from workload config
        wl = op.get('workload_params', {})
        if not wl:
            # Fallback: check if wall_time suggests multi-thread
            seq = op.get('seq', op_idx)
            wl_cfg = engine.ctx.config.get('distributed_engine', {}).get('workload', []) if hasattr(engine, 'ctx') else []
            if seq < len(wl_cfg):
                op_threads = wl_cfg[seq].get('num_threads', 1)

        lines.append('')
        lines.append(f'  {"="*80}')
        lines.append(f'  SEARCH #{op.get("seq", op_idx)} (num_threads={op_threads}, '
                     f'wall={op.get("wall_time_s", 0):.1f}s)')
        lines.append(f'  {"="*80}')

        # --- Per-op logical stats (this op only, summed across nodes) ---
        # Hops, comparisons, vec/nbr local-remote, PQ filter.
        op_agg: Dict[str, float] = {}
        _op_sources = list(pn.values()) if pn else [op]
        for _src in _op_sources:
            _sr = _src.get('search_pure_stats', {})
            for _k, _v in _sr.items():
                if isinstance(_v, (int, float)):
                    op_agg[_k] = op_agg.get(_k, 0) + _v
        if op_agg:
            _hops = int(op_agg.get('n_hops', 0))
            _cmps = int(op_agg.get('n_cmps', 0))
            _vec_l = int(op_agg.get('local_vector_accesses', 0))
            _vec_r = int(op_agg.get('remote_vector_accesses', 0))
            _nbr_l = int(op_agg.get('local_neighbor_accesses', 0))
            _nbr_r = int(op_agg.get('remote_neighbor_accesses', 0))
            _vec_tot = _vec_l + _vec_r
            _nbr_tot = _nbr_l + _nbr_r
            _vec_rpct = (_vec_r * 100.0 / _vec_tot) if _vec_tot else 0
            _nbr_rpct = (_nbr_r * 100.0 / _nbr_tot) if _nbr_tot else 0
            _pq_f = int(op_agg.get('pq_filtered', 0))
            _pq_pct = (_pq_f * 100.0 / _vec_tot) if _vec_tot else 0
            _resf = int(op_agg.get('resolve_fail', 0))
            lines.append('')
            lines.append('  Per-op stats (summed across nodes):')
            lines.append(f'    hops          : {_hops:>14,}')
            lines.append(f'    comparisons   : {_cmps:>14,}')
            lines.append(f'    vec local/rem : {_vec_l:>14,} / {_vec_r:<14,}  ({_vec_rpct:.1f}% remote)')
            lines.append(f'    nbr local/rem : {_nbr_l:>14,} / {_nbr_r:<14,}  ({_nbr_rpct:.1f}% remote)')
            if _pq_f > 0:
                lines.append(f'    pq_filtered   : {_pq_f:>14,}   ({_pq_pct:.1f}% of vec accesses)')
            if _resf > 0:
                lines.append(f'    resolve_fail={_resf:,}')
            lines.append('')

        # Aggregate per-L stats across nodes for THIS op only
        agg_by_L: Dict[int, Dict[str, float]] = {}
        nic_max_by_L: Dict[int, float] = {}  # NIC IB RX: max across nodes (payload)
        wire_max_by_L: Dict[int, float] = {}  # NIC wire-level RX: max across nodes
        msg_rate_max_by_L: Dict[int, float] = {}  # per-node Mmsg/s: max across nodes
        burst_rx_max_by_L: Dict[int, float] = {}   # fine-grained 1ms peak RX
        burst_rx_p99_by_L: Dict[int, float] = {}
        burst_rx_avg_by_L: Dict[int, float] = {}
        burst_tx_max_by_L: Dict[int, float] = {}
        elapsed_by_L: Dict[int, List[float]] = {}  # per-node elapsed_s (for avg/min/max)
        sources = list(pn.values()) if pn else [op]
        _nic_keys = ('nic_rx_gbps', 'nic_tx_gbps',
                     'nic_wire_rx_gbps', 'nic_wire_tx_gbps',
                     'nic_rdma_rx_gbps', 'nic_rdma_tx_gbps',
                     'nic_msg_rate_mps')
        _burst_keys = ('nic_burst_rx_max_gbps', 'nic_burst_rx_p99_gbps',
                       'nic_burst_rx_avg_gbps', 'nic_burst_tx_max_gbps',
                       'nic_burst_tx_p99_gbps', 'nic_burst_tx_avg_gbps',
                       'nic_burst_samples')
        for src in sources:
            sbl = src.get('search_stats_by_L', {})
            for L_str, stats in sbl.items():
                L = int(L_str)
                if L not in agg_by_L:
                    agg_by_L[L] = {}
                    elapsed_by_L[L] = []
                # Track per-node elapsed_s separately (don't sum)
                elapsed_by_L[L].append(stats.get('elapsed_s', 0))
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        if k == 'elapsed_s':
                            continue  # tracked separately above
                        if k == 'nic_rx_gbps':
                            nic_max_by_L[L] = max(nic_max_by_L.get(L, 0), v)
                        elif k == 'nic_wire_rx_gbps':
                            wire_max_by_L[L] = max(wire_max_by_L.get(L, 0), v)
                        elif k == 'nic_msg_rate_mps':
                            # max across nodes (per-node spec comparison)
                            msg_rate_max_by_L[L] = max(msg_rate_max_by_L.get(L, 0), v)
                        elif k == 'nic_burst_rx_max_gbps':
                            burst_rx_max_by_L[L] = max(burst_rx_max_by_L.get(L, 0), v)
                        elif k == 'nic_burst_rx_p99_gbps':
                            burst_rx_p99_by_L[L] = max(burst_rx_p99_by_L.get(L, 0), v)
                        elif k == 'nic_burst_rx_avg_gbps':
                            burst_rx_avg_by_L[L] = max(burst_rx_avg_by_L.get(L, 0), v)
                        elif k == 'nic_burst_tx_max_gbps':
                            burst_tx_max_by_L[L] = max(burst_tx_max_by_L.get(L, 0), v)
                        elif k in _nic_keys or k in _burst_keys:
                            pass  # track max separately if needed later
                        else:
                            # All other numeric fields (incl. congestion counter
                            # deltas like nic_xmit_wait/nic_cnp_sent/nic_ecn_marked/
                            # nic_out_of_buffer) are summed across nodes to give a
                            # cluster-wide event count for the operation.
                            agg_by_L[L][k] = agg_by_L[L].get(k, 0) + v

        if not agg_by_L:
            continue

        # --- Table A: Performance (throughput / latency / BW) ---
        # Wall is per-node (NOT summed). avg/min/max shown separately.
        # QPS = total_queries / max_per_node_wall (slowest node = system throughput).
        # Mmsg/s = per-node NIC RDMA read rate (max across nodes). ConnectX-5 @
        # 100G caps around ~20 Mmsg/s regardless of payload.
        lines.append(
            f'  {"L":>4}  {"QPS":>8}  {"Wall avg":>9}  {"min":>6}  {"max":>6}  '
            f'{"Avg/Q":>7}  {"IB_RX":>7}  {"WIRE_RX":>8}  {"%NIC":>5}  '
            f'{"Mmsg/s":>7}  {"us/call":>8}  {"Fetches":>11}')
        lines.append(
            f'  {"--":>4}  {"------":>8}  {"--------":>9}  {"-----":>6}  {"-----":>6}  '
            f'{"-----":>7}  {"-----":>7}  {"-------":>8}  {"----":>5}  '
            f'{"------":>7}  {"--------":>8}  {"-----------":>11}')

        L_iter = sorted(agg_by_L.keys())
        cpu_bd_by_L: Dict[int, Dict[str, float]] = {}
        row_ctx: Dict[int, Dict[str, float]] = {}
        for L in L_iter:
            s = agg_by_L[L]
            nq = int(s.get('num_queries', 0)) or 1
            # Per-node elapsed (not summed): avg/min/max across nodes
            el_list = elapsed_by_L.get(L, [])
            el_avg = sum(el_list) / len(el_list) if el_list else 0
            el_min = min(el_list) if el_list else 0
            el_max = max(el_list) if el_list else 0
            e2e_us = s.get('sum_query_total_us', 0)
            rdma_us = s.get('rdma_total_us', 0)
            # QPS = total queries / slowest node wall (system-level throughput)
            qps = round(nq / el_max, 1) if el_max > 0 else 0

            # Aggregate CPU breakdown across nodes
            cpu_bd: Dict[str, float] = {}
            for src in sources:
                sbl = src.get('search_stats_by_L', {})
                for lk, sv in sbl.items():
                    if int(lk) == L and isinstance(sv, dict):
                        for k, v in sv.get('cpu_breakdown', {}).items():
                            if isinstance(v, (int, float)):
                                cpu_bd[k] = cpu_bd.get(k, 0) + v
            cpu_bd_by_L[L] = cpu_bd

            nic_rx = nic_max_by_L.get(L, 0)
            wire_rx = wire_max_by_L.get(L, 0)
            msg_rate = msg_rate_max_by_L.get(L, 0)
            fetches = int(s.get('fetch_entry_items', 0))
            avg_q_ms = (e2e_us / nq / 1000) if nq > 0 else 0
            nic_str = f'{nic_rx:>6.1f}G' if nic_rx > 0 else f'{"N/A":>7}'
            wire_str = f'{wire_rx:>7.1f}G' if wire_rx > 0 else f'{"N/A":>8}'
            nic_pct = wire_rx if wire_rx > 0 else 0
            nic_pct_str = f'{nic_pct:>4.1f}%' if wire_rx > 0 else f'{"N/A":>5}'
            msg_rate_str = f'{msg_rate:>6.2f}' if msg_rate > 0 else f'{"N/A":>7}'

            row_ctx[L] = {'nq': nq, 'e2e_us': e2e_us, 'rdma_us': rdma_us,
                          'el_avg': el_avg, 'el_max': el_max}

            # Per-RDMA-call avg latency (thread-time per fetch_entry call).
            # Rises with queueing pressure — useful with IB_RX to see if NIC
            # is just busy vs actively queueing WRs.
            fec = int(s.get('fetch_entry_calls', 0))
            per_call_us = (rdma_us / fec) if fec > 0 else 0
            per_call_str = f'{per_call_us:>6.2f}us' if fec > 0 else f'{"N/A":>8}'

            lines.append(
                f'  {L:>4}  {qps:>8.1f}  {el_avg*1000:>7.0f}ms  '
                f'{el_min*1000:>4.0f}ms  {el_max*1000:>4.0f}ms  '
                f'{avg_q_ms:>5.1f}ms  {nic_str}  {wire_str}  {nic_pct_str}  '
                f'{msg_rate_str}  {per_call_str}  {fetches:>11,}')

        # --- RDMA call breakdown (where does us/call go?) ---
        # The search path now uses one canonical bulk-sync schedule:
        #   pass1_cpu/call = pass1_us / fec
        #   nic_drain/call = (rdma_total_us - pass1_us - pass3_poll_us) / fec
        #   total/call     = rdma_total_us / fec
        any_bd_call = any(
            cpu_bd_by_L.get(L, {}).get('pass1_us', 0) > 0 and
            int(agg_by_L[L].get('fetch_entry_calls', 0)) > 0
            for L in L_iter)
        if any_bd_call:
            lines.append('')
            lines.append(
                '  RDMA call breakdown (us/call, summed across nodes / fetch_entry_calls):')
            lines.append('    pass1_cpu = pass1_us/fec  (PQ + cache-hit + state inserts + async post)')
            lines.append('    nic_drain = NIC round-trip wait from bulk-sync drain after Pass1')
            lines.append(f'  {"L":>4}  {"pass1_cpu":>11}  {"nic_drain":>13}  {"total":>9}')
            lines.append(f'  {"--":>4}  {"-"*9:>11}  {"-"*11:>13}  {"-"*7:>9}')
            for L in L_iter:
                fec = int(agg_by_L[L].get('fetch_entry_calls', 0))
                if fec <= 0:
                    continue
                p1_us = float(cpu_bd_by_L.get(L, {}).get('pass1_us', 0))
                p3p_us = float(cpu_bd_by_L.get(L, {}).get('pass3_poll_us', 0))
                rdma_tot = float(agg_by_L[L].get('rdma_total_us', 0))
                pc_p1 = p1_us / fec
                pc_tot = rdma_tot / fec
                pc_drain = max(pc_tot - pc_p1 - (p3p_us / fec), 0)
                lines.append(
                    f'  {L:>4}  {pc_p1:>9.2f}us  '
                    f'{pc_drain:>11.2f}us  {pc_tot:>7.2f}us')

        # --- Queueing / congestion counters (summed across nodes per L) ---
        # Direct queue-pressure evidence from mlx5 hw_counters. Rising
        # port_xmit_wait with T indicates HCA-level SQ queueing. np_cnp_sent
        # / ecn_marked rise when fabric marks packets for congestion.
        # out_of_buffer > 0 means saturation drops.
        any_cong = any(
            (agg_by_L[L].get('nic_xmit_wait', 0) > 0 or
             agg_by_L[L].get('nic_cnp_sent', 0) > 0 or
             agg_by_L[L].get('nic_ecn_marked', 0) > 0 or
             agg_by_L[L].get('nic_out_of_buffer', 0) > 0)
            for L in L_iter)
        if any_cong:
            lines.append('')
            lines.append('  Queueing / congestion counters (delta over L window, summed across nodes):')
            lines.append(
                f'  {"L":>4}  {"xmit_wait":>12}  {"cnp_sent":>10}  '
                f'{"ecn_marked":>11}  {"out_of_buf":>11}  {"rx_reads":>14}')
            lines.append(
                f'  {"--":>4}  {"---------":>12}  {"--------":>10}  '
                f'{"----------":>11}  {"----------":>11}  {"--------":>14}')
            for L in L_iter:
                xw = int(agg_by_L[L].get('nic_xmit_wait', 0))
                cn = int(agg_by_L[L].get('nic_cnp_sent', 0))
                em = int(agg_by_L[L].get('nic_ecn_marked', 0))
                ob = int(agg_by_L[L].get('nic_out_of_buffer', 0))
                rr = int(agg_by_L[L].get('nic_rx_read_requests', 0))
                lines.append(
                    f'  {L:>4}  {xw:>12,}  {cn:>10,}  '
                    f'{em:>11,}  {ob:>11,}  {rr:>14,}')

        # --- Burst BW (1ms sampled, max across nodes) ---
        # IB_RX above is wall-clock avg; this shows what the NIC actually
        # burst to. Hidden by default (rarely read); set
        # PIPELINE_REPORT_VERBOSE=1 to show.
        _show_burst = bool(_os.environ.get('PIPELINE_REPORT_VERBOSE', ''))
        if _show_burst and any(L in burst_rx_max_by_L for L in L_iter):
            lines.append('')
            lines.append('  Fine-grained NIC RX (1ms sampling, max across nodes):')
            lines.append(
                f'  {"L":>4}  {"burst_max":>10}  {"burst_p99":>10}  {"burst_avg":>10}  {"%burst":>7}')
            lines.append(
                f'  {"--":>4}  {"---------":>10}  {"---------":>10}  {"---------":>10}  {"------":>7}')
            for L in L_iter:
                bmax = burst_rx_max_by_L.get(L, 0)
                bp99 = burst_rx_p99_by_L.get(L, 0)
                bavg = burst_rx_avg_by_L.get(L, 0)
                if bmax <= 0:
                    continue
                # %burst = burst max as percentage of 100 Gbps line rate
                lines.append(
                    f'  {L:>4}  {bmax:>8.1f}G   {bp99:>8.1f}G   {bavg:>8.1f}G   {bmax:>6.1f}%')

        # --- Table B: Per-query thread-time breakdown ---
        # dist  : SIMD distance compute (cpu_distance_us) — useful work
        # graph : visited + resolve + nbr_loop_pure (graph traversal)
        # state : map + queue (search state bookkeeping)
        # rdma  : rdma_total_us (post + poll + barrier wait)
        # pq    : cpu_pq_us (PQ filter scoring; only shown when > 0)
        # other : e2e - sum(above); residual = batch-path work (when
        #         batch_size > 0; not timed by the C++ inner timers) +
        #         instrumentation gaps (PQ LUT build, result extraction,
        #         OpenMP reduction). When `other` is dominant in PQ-batch
        #         runs, the bulk is the post-convergence batch path.
        any_bd = any(cpu_bd_by_L[L] and row_ctx[L]['e2e_us'] > 0 for L in L_iter)
        if any_bd:
            # Detect whether any L has non-zero pq_us — if not, hide the
            # column entirely.
            show_pq = any(cpu_bd_by_L[L].get('pq_us', 0) > 0 for L in L_iter)
            lines.append('')
            lines.append('  Per-query thread-time breakdown (us/q, % of e2e):')
            CW = 13
            hdr_cells = [f'{"dist":>{CW}}', f'{"graph":>{CW}}',
                         f'{"state":>{CW}}', f'{"rdma":>{CW}}']
            if show_pq:
                hdr_cells.append(f'{"pq":>{CW}}')
            hdr_cells.append(f'{"other":>{CW}}')
            lines.append(f'  {"L":>4}  ' + '  '.join(hdr_cells))
            sep_cells = [f'{"-"*11:>{CW}}'] * len(hdr_cells)
            lines.append(f'  {"--":>4}  ' + '  '.join(sep_cells))
            for L in L_iter:
                cpu_bd = cpu_bd_by_L[L]
                ctx_row = row_ctx[L]
                e2e_us = ctx_row['e2e_us']
                rdma_us = ctx_row['rdma_us']
                nq = ctx_row['nq']
                if e2e_us <= 0 or nq <= 0:
                    continue
                vis_us = cpu_bd.get('visited_us', 0)
                res_us = cpu_bd.get('resolve_us', 0)
                nbr_raw = cpu_bd.get('nbr_loop_us', 0)
                nbr_pure = max(nbr_raw - vis_us - res_us, 0)

                dist_us = cpu_bd.get('distance_us', 0)
                graph_us = vis_us + res_us + nbr_pure
                state_us = cpu_bd.get('map_us', 0) + cpu_bd.get('queue_us', 0)
                pq_us = cpu_bd.get('pq_us', 0)
                if show_pq:
                    measured = dist_us + graph_us + state_us + rdma_us + pq_us
                else:
                    measured = dist_us + graph_us + state_us + rdma_us
                other_us = max(e2e_us - measured, 0)

                def _c(v):
                    return f'{v/nq:>6.0f}us({v/e2e_us*100:>3.0f}%)'
                cells = [_c(dist_us), _c(graph_us), _c(state_us), _c(rdma_us)]
                if show_pq:
                    cells.append(_c(pq_us))
                cells.append(_c(other_us))
                lines.append(f'  {L:>4}  ' + '  '.join(cells))
            note = ('  note: graph = visited+resolve+nbr_loop_pure; '
                    'state = map+queue;')
            if show_pq:
                note += ' pq = pq_filter scoring;'
            note += (' other = residual + batch-path work (untimed)'
                     if any(cpu_bd_by_L[L].get('batch_nbr_poll_us', 0) > 0
                            or cpu_bd_by_L[L].get('batch_vec_poll_us', 0) > 0
                            for L in L_iter)
                     else ' other = residual.')
            lines.append(note)

            # --- Table C: Pass 1/2/3 breakdown ---
            # Shown whenever pass timers are populated. The canonical search
            # path drains remote reads before local/remote distance compute.
            any_pass = any(cpu_bd_by_L[L].get('pass1_us', 0) > 0 for L in L_iter)
            if any_pass:
                # Detect whether converge-batch poll columns have any data
                has_batch_poll = any(
                    cpu_bd_by_L[L].get('batch_nbr_poll_us', 0) > 0
                    or cpu_bd_by_L[L].get('batch_vec_poll_us', 0) > 0
                    for L in L_iter)

                lines.append('')
                lines.append('  Pass-wise wall within hop (us/q):')
                lines.append(
                    '    P1=remote post/PQ  P2=local compute  P3=remote compute after bulk-sync drain')
                lines.append(
                    '    P3_poll should be small because remote reads are pre-drained')
                PW = 13
                header = (f'  {"L":>4}  {"P1":>{PW}}  {"P2":>{PW}}  {"P3":>{PW}}  '
                          f'{"P3_poll":>{PW}}  {"P3_comp":>{PW}}')
                sep = (f'  {"--":>4}  {"-"*11:>{PW}}  {"-"*11:>{PW}}  {"-"*11:>{PW}}  '
                       f'{"-"*11:>{PW}}  {"-"*11:>{PW}}')
                if has_batch_poll:
                    header += f'  {"nbr_poll":>{PW}}  {"vec_poll":>{PW}}'
                    sep += f'  {"-"*11:>{PW}}  {"-"*11:>{PW}}'
                lines.append(header)
                lines.append(sep)
                for L in L_iter:
                    cpu_bd = cpu_bd_by_L[L]
                    ctx_row = row_ctx[L]
                    nq = ctx_row['nq']
                    if nq <= 0:
                        continue
                    phase = cpu_bd.get('candidate_fetch_phase_us', 0)
                    p1 = cpu_bd.get('pass1_us', 0)
                    p2 = cpu_bd.get('pass2_us', 0)
                    p3 = cpu_bd.get('pass3_us', 0)
                    p3_poll = cpu_bd.get('pass3_poll_us', 0)
                    p3_comp = cpu_bd.get('pass3_compute_us', 0)

                    def _phase(v):
                        pct = (v / phase * 100) if phase > 0 else 0
                        return f'{v/nq:>5.0f}us({pct:>3.0f}%)'
                    def _p3(v):
                        pct = (v / p3 * 100) if p3 > 0 else 0
                        return f'{v/nq:>5.0f}us({pct:>3.0f}%)'
                    row = (f'  {L:>4}  {_phase(p1)}  {_phase(p2)}  {_phase(p3)}  '
                           f'{_p3(p3_poll)}  {_p3(p3_comp)}')
                    if has_batch_poll:
                        nbr_poll = cpu_bd.get('batch_nbr_poll_us', 0)
                        vec_poll = cpu_bd.get('batch_vec_poll_us', 0)
                        row += f'  {nbr_poll/nq:>9.0f}us  {vec_poll/nq:>9.0f}us'
                    lines.append(row)
                lines.append('  note: % shown for P1/P2/P3 = fraction of candidate-fetch phase; for P3_poll/P3_comp = fraction of P3.')

        # --- PMU counters (per-node, from pmu_monitor.sh start/stop) ---
        # Populated by SearchOp when env PIPELINE_PMU=1. Per-node values
        # are aggregated from search_stats_by_L[L]['pmu_*']; we report
        # avg across nodes plus [min..max] range when meaningful.
        _pmu_keys_pretty = [
            ('pmu_ipc',                 'IPC',                 ''),
            ('pmu_llc_hit_rate_pct',    'LLC hit rate',        ' %'),
            ('pmu_l3_hit_rate_pct',     'L3 hit rate (demand)',' %'),
            ('pmu_dram_read_gbps',      'DRAM read BW',        ' GB/s'),
            ('pmu_dram_write_gbps',     'DRAM write BW',       ' GB/s'),
            ('pmu_iio_write_gbps',      'NIC->host (IIO write)',' GB/s'),
            ('pmu_iio_read_gbps',       'NIC<-host (IIO read)', ' GB/s'),
            ('pmu_ddio_absorbed_pct',   'DDIO absorbed',       ' %'),
        ]
        # Per-L per-node aggregation: collect each pmu_* across nodes,
        # avg / min / max. We pick the FIRST non-empty L to summarize
        # (most search ops have single L; report as a single block).
        _pmu_pernode_by_L: Dict[int, Dict[str, list]] = {}
        for src in (list(pn.values()) if pn else [op]):
            _sbl = src.get('search_stats_by_L', {})
            for _Ls, _s in _sbl.items():
                _L = int(_Ls)
                bucket = _pmu_pernode_by_L.setdefault(_L, {})
                for _pk, _, _ in _pmu_keys_pretty:
                    if _pk in _s:
                        bucket.setdefault(_pk, []).append(float(_s[_pk]))
        _pmu_have = any(any(b) for b in _pmu_pernode_by_L.values())
        if _pmu_have:
            lines.append('')
            lines.append('  PMU (per-node avg [min..max] across nodes):')
            _pmu_first_L = sorted(_pmu_pernode_by_L.keys())[0]
            _bucket = _pmu_pernode_by_L[_pmu_first_L]
            _n_nodes = max((len(v) for v in _bucket.values()), default=0)
            if len(_pmu_pernode_by_L) > 1:
                lines.append(f'    (showing L={_pmu_first_L}; {_n_nodes} node(s))')
            for _pk, _label, _suffix in _pmu_keys_pretty:
                vals = _bucket.get(_pk, [])
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                lo, hi = min(vals), max(vals)
                if len(vals) > 1 and (hi - lo) > 0.01:
                    rng = f'  [{lo:.2f} .. {hi:.2f}]'
                else:
                    rng = ''
                lines.append(f'    {_label:<22}: {avg:>8.2f}{_suffix}{rng}')

        # (legacy fallback: load from CSV if present — kept for
        # back-compat with externally-generated pmu_*.csv files)
        _pmu_rows = _load_pmu_csv(engine, op.get('seq', op_idx))
        if _pmu_rows and not _pmu_have:
            lines.append('')
            lines.append('  PMU (from external CSV):')
            for k in ['ipc', 'l3_hit_rate', 'dram_read_gbps',
                       'dram_write_gbps', 'ddio_gbps']:
                if k in _pmu_rows:
                    v = _pmu_rows[k]
                    suffix = ''
                    if k.endswith('_gbps'):
                        suffix = ' GB/s'
                    elif k == 'l3_hit_rate':
                        suffix = ' %'
                    lines.append(f'  {k:<22} {v:>12.2f}{suffix}')

    # Per-L latency breakdown
    has_lat_by_L = any(
        any(_get_node(op.get('per_node', {}), nid).get('latency_by_L')
            for nid in node_ids)
        for op in search_ops_with_stats
    ) if node_ids and search_ops_with_stats else False
    if has_lat_by_L:
        lines.append('')
        lines.append('  --- Latency per L (all nodes, per-query) ---')
        lines.append(f'  {"L":>5}  {"mean(us)":>10}  {"p50(us)":>10}  '
                      f'{"p95(us)":>10}  {"p99(us)":>10}  {"max(us)":>10}')
        lines.append(f'  {"--":>5}  {"--------":>10}  {"-------":>10}  '
                      f'{"-------":>10}  {"-------":>10}  {"------":>10}')
        lat_agg: Dict[int, Dict[str, list]] = {}
        for op in search_ops_with_stats:
            pn = op.get('per_node', {})
            sources = list(pn.values()) if pn else [op]
            for src in sources:
                lbl = src.get('latency_by_L', {})
                for L_str, lat in lbl.items():
                    L = int(L_str)
                    if L not in lat_agg:
                        lat_agg[L] = {k: [] for k in ['mean_us', 'p50_us', 'p95_us', 'p99_us', 'max_us']}
                    for k in lat_agg[L]:
                        lat_agg[L][k].append(lat.get(k, 0))
        for L in sorted(lat_agg.keys()):
            la = lat_agg[L]
            lines.append(
                f'  {L:>5}  '
                f'{np.mean(la["mean_us"]):>10.1f}  '
                f'{np.mean(la["p50_us"]):>10.1f}  '
                f'{np.mean(la["p95_us"]):>10.1f}  '
                f'{np.mean(la["p99_us"]):>10.1f}  '
                f'{max(la["max_us"]):>10.1f}')

    # --- 5. Recall Tracking ---
    search_ops = [op for op in engine.op_history
                  if op['op'] == 'search' and 'recall' in op]
    if search_ops:
        section('5. RECALL TRACKING')
        # Determine all L values across search ops
        all_L_values = set()
        for op in search_ops:
            r = op['recall']
            if 'recall_by_L' in r:
                all_L_values.update(int(l) for l in r['recall_by_L'].keys())
            elif 'recall_at_K' in r:
                all_L_values.add(r.get('L', 0))
        all_L_values = sorted(all_L_values)

        if all_L_values and any('recall_by_L' in op['recall'] for op in search_ops):
            # L sweep table
            l_headers = ''.join(f'  {"L="+str(l):>8}' for l in all_L_values)
            lines.append(f'  {"Seq":>3}  {"K":>4} {"Queries":>8} '
                          f'{"Active Pts":>12}{l_headers} {"GT(s)":>8}')
            l_dashes = ''.join(f'  {"--------":>8}' for _ in all_L_values)
            lines.append(f'  {"---":>3}  {"--":>4} {"-------":>8} '
                          f'{"----------":>12}{l_dashes} {"-----":>8}')
            for op in search_ops:
                r = op['recall']
                recall_by_L = r.get('recall_by_L', {})
                l_vals = ''.join(
                    f'  {recall_by_L.get(l, recall_by_L.get(str(l), 0)):8.4f}'
                    for l in all_L_values)
                lines.append(f'  {op["seq"]:3d}  {r["K"]:4d} {r["num_queries"]:8d} '
                              f'{r["num_active_points"]:12,}{l_vals} '
                              f'{r["gt_time_s"]:8.3f}')
        else:
            # Legacy single-L format
            lines.append(f'  {"Seq":>3}  {"Recall@K":>10} {"K":>4} '
                          f'{"Queries":>8} {"Active Pts":>12} {"GT Time(s)":>10}')
            lines.append(f'  {"---":>3}  {"--------":>10} {"--":>4} '
                          f'{"-------":>8} {"----------":>12} {"---------":>10}')
            for op in search_ops:
                r = op['recall']
                lines.append(f'  {op["seq"]:3d}  {r["recall_at_K"]:10.4f} '
                              f'{r["K"]:4d} {r["num_queries"]:8d} '
                              f'{r["num_active_points"]:12,} '
                              f'{r["gt_time_s"]:10.3f}')
    else:
        section('5. RECALL TRACKING')
        # Check for post-hoc recall results
        posthoc_file = engine.run_dir / "recall_posthoc.json" if engine.run_dir else None
        if posthoc_file and posthoc_file.exists():
            try:
                with open(posthoc_file) as f:
                    posthoc = json.load(f)
                lines.append('  (recall computed post-hoc)')
                posthoc_ops = [s for s in posthoc.get('search_ops', [])
                               if 'recall_by_L' in s]
                if posthoc_ops:
                    all_L = sorted({int(l) for s in posthoc_ops
                                    for l in s['recall_by_L']})
                    l_headers = ''.join(f'  {"L="+str(l):>8}' for l in all_L)
                    lines.append(f'  {"Seq":>3}  {"K":>4} {"Queries":>8} '
                                  f'{"Active Pts":>12}{l_headers}')
                    l_dashes = ''.join(f'  {"--------":>8}' for _ in all_L)
                    lines.append(f'  {"---":>3}  {"--":>4} {"-------":>8} '
                                  f'{"----------":>12}{l_dashes}')
                    for s in posthoc_ops:
                        rbl = s['recall_by_L']
                        l_vals = ''.join(
                            f'  {rbl.get(str(l), 0):8.4f}' for l in all_L)
                        lines.append(
                            f'  {s["seq"]:3d}  {s["K"]:4d} '
                            f'{s["num_queries"]:8d} '
                            f'{s["num_active"]:12,}{l_vals}')
            except Exception:
                lines.append('  (failed to load recall_posthoc.json)')
        else:
            lines.append('  (no search operations with recall data)')
            lines.append('  Run: python compute_recall_posthoc.py <run_dir>')

    # --- 6. Cache Statistics ---
    cache_ops = [op for op in engine.op_history if op['op'] == 'search']
    has_cache = any(
        ndata.get('cache_stats', {}).get('enabled', False)
        for op in cache_ops
        for ndata in op.get('per_node', {}).values()
    )
    if has_cache:
        section('6. SEARCH CACHE STATISTICS')
        for op in cache_ops:
            pn = op.get('per_node', {})
            # Aggregate across nodes
            tv_h = tv_m = tn_h = tn_m = 0
            tn_pins = tn_pin_skips = tn_puts = tn_evict = 0
            for nid, ndata in pn.items():
                cs = ndata.get('cache_stats', {})
                tv_h += cs.get('vec_hits', 0)
                tv_m += cs.get('vec_misses', 0)
                tn_h += cs.get('nbr_hits', 0)
                tn_m += cs.get('nbr_misses', 0)
                tn_pins += cs.get('nbr_pins', 0)
                tn_pin_skips += cs.get('nbr_pin_evict_skips', 0)
                tn_puts += cs.get('nbr_puts', 0)
                tn_evict += cs.get('nbr_evictions', 0)

            vt = tv_h + tv_m
            nt = tn_h + tn_m
            vr = tv_h / vt * 100 if vt else 0
            nr = tn_h / nt * 100 if nt else 0

            lines.append(f'  Search S{op["seq"]}:')
            lines.append(f'    vec  hit rate: {vr:6.1f}%  ({tv_h:>12,} / {vt:>12,})')
            lines.append(f'    nbr  hit rate: {nr:6.1f}%  ({tn_h:>12,} / {nt:>12,})')
            lines.append(f'    nbr  puts:     {tn_puts:>12,}   evictions: {tn_evict:>12,}')
            lines.append(f'    nbr  pins:     {tn_pins:>12,}   pin_evict_skips: {tn_pin_skips:>12,}')
            lines.append('')

            # Per-node breakdown
            lines.append(f'    {"Node":>6} {"vec_hit%":>8} {"nbr_hit%":>8} '
                         f'{"nbr_pins":>10} {"pin_skips":>12} {"nbr_puts":>10}')
            lines.append(f'    {"------":>6} {"--------":>8} {"--------":>8} '
                         f'{"----------":>10} {"------------":>12} {"----------":>10}')
            for nid in sorted(pn.keys(), key=int):
                cs = pn[nid].get('cache_stats', {})
                vh = cs.get('vec_hits', 0)
                vm = cs.get('vec_misses', 0)
                nh = cs.get('nbr_hits', 0)
                nm = cs.get('nbr_misses', 0)
                vr2 = vh / (vh + vm) * 100 if (vh + vm) else 0
                nr2 = nh / (nh + nm) * 100 if (nh + nm) else 0
                lines.append(f'    {nid:>6} {vr2:>7.1f}% {nr2:>7.1f}% '
                             f'{cs.get("nbr_pins", 0):>10,} '
                             f'{cs.get("nbr_pin_evict_skips", 0):>12,} '
                             f'{cs.get("nbr_puts", 0):>10,}')
            lines.append('')

    # --- 7. Per-Node Comparison ---
    section('7. PER-NODE COMPARISON')

    if not node_ids:
        lines.append('  (no per-node data available)')
    else:
        # Per-node total time and items per op type
        for op_type in sorted(op_types.keys()):
            type_ops = [op for op in engine.op_history if op['op'] == op_type
                        and 'per_node' in op]
            if not type_ops:
                continue
            lines.append(f'')
            lines.append(f'  [{op_type.upper()}] ({len(type_ops)} ops)')
            lines.append(f'  {"Node":>6} {"Total Time(s)":>14} '
                          f'{"Avg Time(s)":>12} {"Items":>10} {"Avg Items":>10}')
            lines.append(f'  {"----":>6} {"-------------":>14} '
                          f'{"----------":>12} {"-----":>10} {"---------":>10}')

            for nid in node_ids:
                t_total = 0.0
                items_total = 0
                count = 0
                for op in type_ops:
                    ns = _get_node(op.get('per_node', {}), nid)
                    t_total += ns.get('wall_time_s', 0)
                    items_total += ns.get('items_processed', 0)
                    count += 1
                avg_t = t_total / count if count else 0
                avg_items = items_total / count if count else 0
                lines.append(f'  {nid:>6} {t_total:14.2f} {avg_t:12.2f} '
                              f'{items_total:10,} {avg_items:10.1f}')

        # Load balance analysis
        lines.append('')
        lines.append('  --- Load Balance (insert) ---')
        insert_type_ops = [op for op in engine.op_history
                           if op['op'] == 'insert' and 'per_node' in op]
        if insert_type_ops and node_ids:
            node_items = {}
            node_times = {}
            for nid in node_ids:
                node_items[nid] = sum(
                    _get_node(op.get('per_node', {}), nid).get('items_processed', 0)
                    for op in insert_type_ops)
                node_times[nid] = sum(
                    _get_node(op.get('per_node', {}), nid).get('wall_time_s', 0)
                    for op in insert_type_ops)
            total_items = sum(node_items.values())
            max_items = max(node_items.values()) if node_items else 0
            min_items = min(node_items.values()) if node_items else 0
            max_time = max(node_times.values()) if node_times else 0
            min_time = min(node_times.values()) if node_times else 0
            imbalance = ((max_items - min_items) / max_items * 100) if max_items else 0
            lines.append(f'  Total inserts: {total_items:,}')
            lines.append(f'  Item range   : [{min_items:,}, {max_items:,}] '
                          f'(imbalance: {imbalance:.1f}%)')
            lines.append(f'  Time range   : [{min_time:.2f}s, {max_time:.2f}s]')

    # --- Footer ---
    lines.append('')
    hr()
    lines.append(f'  End of report')
    hr()
    lines.append('')

    return '\n'.join(lines)
