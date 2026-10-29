"""Consolidation operation for the distributed engine."""

from __future__ import annotations

import json as _json
import time
from pathlib import Path
from typing import Any

from pipeline import diagnostics
from pipeline.operation_base import Operation


class ConsolidateOp(Operation):
    """Consolidate deleted neighbors via 2-hop expansion + re-prune.

    No broadcast needed — each node consolidates its own partition
    using the globally-known deleted VID set from prior DeleteOp.
    """
    name = "consolidate"
    needs_broadcast = False

    def execute(self, ctx: UpdateContext, payload: Any):
        ctx.log("ConsolidateOp: starting...")

        # Run-condition marker: makes the de-overlapped baseline identifiable
        # from the run log alone (the wall-clock difference is large enough that
        # a mislabeled run would be easy to misread as a regression).
        _consol_cfg = ctx.config.get('consolidation', {}) or {}
        # Fetch in-flight window depth K, resolved the same way C++ resolves it:
        # explicit fetch_window wins; sync_fetch alone means K=1; neither means
        # K=0 (unlimited/batched). Logged unconditionally so every consolidation
        # run states its pipelining depth, which is the x-axis of the
        # fetch-depth sensitivity curve.
        _fw_cfg = max(0, int(_consol_cfg.get('fetch_window', 0) or 0))
        _sync_cfg = bool(_consol_cfg.get('sync_fetch', False))
        _fw = _fw_cfg if _fw_cfg > 0 else (1 if _sync_cfg else 0)
        ctx.log(f"[CONSOLIDATE] fetch_window={_fw}")
        if _fw_cfg > 0 and _sync_cfg:
            ctx.log("[CONSOLIDATE] fetch_window overrides sync_fetch "
                    "(sync_fetch is its K=1 special case)")
        if _fw == 1:
            if bool(_consol_cfg.get('enable_vec_cache', False)):
                # Cache arm: hits never reach a fetch site, so only the
                # interceptor's misses are serialized.
                ctx.log("[CONSOLIDATE] sync_fetch=ON (applies to cache-miss fetches)")
            else:
                ctx.log("[CONSOLIDATE] sync_fetch=ON (de-overlapped baseline)")
        elif _fw >= 2:
            if bool(_consol_cfg.get('enable_vec_cache', False)):
                ctx.log(f"[CONSOLIDATE] fetch_window={_fw} "
                        "(applies to cache-miss fetches)")
            else:
                ctx.log(f"[CONSOLIDATE] fetch_window={_fw} (throttled pipeline)")
        if bool(_consol_cfg.get('reuse_fetch_buffers', False)):
            ctx.log("[CONSOLIDATE] reuse_fetch_buffers=ON")

        # NIC BW instrumentation around the C++ consolidate call. Mirrors
        # SearchOp's approach so the same rdma_monitor.sh / nic_bw_monitor.py
        # infrastructure produces comparable per-op BW numbers.
        # Burst CSV for fine-grained timeline (1 ms sampling).
        _burst_seq = self.stats.seq
        _burst_dir = (ctx.run_dir / "nic_bw") if getattr(ctx, 'run_dir', None) else Path("/tmp")
        try:
            _burst_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            _burst_dir = Path("/tmp")
        _burst_csv = str(_burst_dir / f"nic_bw_consol_N{ctx.node_id}_S{_burst_seq}.csv")
        _burst_proc, _burst_mono_start = diagnostics.nic_burst_monitor_start(_burst_csv)

        # Delta-based snapshot covers the full op window.
        _nic_snap_before = diagnostics.rdma_snapshot()
        _burst_window_start = (time.monotonic() - _burst_mono_start) \
            if _burst_mono_start is not None else None

        t0 = time.time()
        stats = ctx._consolidate_local()
        elapsed = time.time() - t0

        _burst_window_end = (time.monotonic() - _burst_mono_start) \
            if _burst_mono_start is not None else None
        _nic_snap_after = diagnostics.rdma_snapshot()

        # Compute delta-based BW (avg over the whole window).
        _nic_bw = diagnostics.rdma_compute_bw(_nic_snap_before, _nic_snap_after, elapsed)
        if _nic_bw:
            stats.update(_nic_bw)

        # Stop burst monitor and merge per-phase peak/p99 into stats.
        # Phases come from C++ offsets so the ON/OFF comparison can be sliced
        # to e.g. the prune-loop window only (the fair algorithm-cost view).
        if _burst_proc is not None:
            diagnostics.nic_burst_monitor_stop(_burst_proc)
            try:
                if _burst_window_start is not None and _burst_window_end is not None:
                    # Build phase list from C++ offsets (relative to C++
                    # t_start). The Python wrapper's qp_setup runs before
                    # C++ t_start, so shift the base by qp_setup_sec so the
                    # phase shading aligns with where C++ actually started.
                    qp_setup = stats.get('qp_setup_sec', 0.0) or 0.0
                    base = _burst_window_start + qp_setup
                    scan_end = stats.get('scan_end_offset_sec', 0.0) or 0.0
                    pop_start = stats.get('pop_start_offset_sec', 0.0) or 0.0
                    pop_end = stats.get('pop_end_offset_sec', 0.0) or 0.0
                    loop_start = stats.get('pass2_loop_start_offset_sec', 0.0) or 0.0
                    loop_end = stats.get('pass2_loop_end_offset_sec', 0.0) or 0.0
                    phases = []
                    # Only emit phases whose duration is meaningful (>1ms).
                    def _maybe(name, s_off, e_off):
                        if e_off - s_off > 0.001:
                            phases.append({
                                'name': name,
                                'start_s': base + s_off,
                                'end_s': base + e_off,
                            })
                    _maybe('scan',     0.0,         scan_end)
                    _maybe('setup',    scan_end,    pop_start)
                    _maybe('populate', pop_start,   pop_end)
                    _maybe('loop',     loop_start,  loop_end)

                    # Outer window kept for backward compat with older parsers.
                    _windows = [(_burst_seq, _burst_window_start, _burst_window_end)]
                    try:
                        win_meta = {
                            'node_id': ctx.node_id,
                            'op_seq': _burst_seq,
                            'mono_start': _burst_mono_start,
                            'windows': [{'seq': _burst_seq,
                                         'start_s': _burst_window_start,
                                         'end_s': _burst_window_end}],
                            'phases': phases,
                        }
                        with open(_burst_csv + ".windows.json", "w") as _wf:
                            _json.dump(win_meta, _wf, indent=2)
                    except Exception:
                        pass
                    _burst_stats = diagnostics.nic_burst_parse_csv(_burst_csv, _windows)
                    if _burst_seq in _burst_stats:
                        stats.update(_burst_stats[_burst_seq])
                    # Per-phase BW slicing — uses the same parse helper but
                    # with phase-named windows. Keys land as
                    # nic_phase_<name>_burst_rx_max_gbps etc.
                    if phases:
                        phase_windows = [(p['name'], p['start_s'], p['end_s'])
                                         for p in phases]
                        phase_stats = diagnostics.nic_burst_parse_csv(
                            _burst_csv, phase_windows)
                        for pname, pstats in phase_stats.items():
                            for k, v in pstats.items():
                                stats[f'nic_phase_{pname}_{k}'] = v
            except Exception as e:
                ctx.log(f"[WARN] consolidate NIC burst parse failed: {e}")

        modified = stats.get('entries_modified', stats.get('modified', 0))
        self.stats.items_processed = modified
        self.stats.custom = stats
        # One-line BW summary for quick log inspection.
        _ib_rx = stats.get('nic_ib_rx_gbps', 0)
        _ib_tx = stats.get('nic_ib_tx_gbps', 0)
        _peak = stats.get('nic_burst_rx_max_gbps', 0)
        ctx.log(f"ConsolidateOp done: {modified} nodes modified in {elapsed:.2f}s "
                f"NIC RX={_ib_rx:.2f} Gbps  TX={_ib_tx:.2f} Gbps  peak_rx={_peak:.2f} Gbps")

