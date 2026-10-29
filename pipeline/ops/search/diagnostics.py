"""Search experiment diagnostics sessions and metric helpers."""

from __future__ import annotations

import os as _os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline import diagnostics


@dataclass
class SearchTraceConfig:
    level: int
    output_dir: str
    seq: int

    @classmethod
    def from_context(cls, ctx, seq: int) -> "SearchTraceConfig":
        trace_cfg = ctx.config.get('trace', {}) or {}
        enabled = bool(trace_cfg.get('enabled', False))
        level = int(trace_cfg.get('level', 1)) if enabled else 0
        output_dir = trace_cfg.get('output_dir', '') or str(ctx.run_dir / "traces")
        if level > 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        return cls(level=level, output_dir=output_dir, seq=seq)

    def start_l(self, ctx, L: int) -> None:
        if self.level <= 0 or not self.output_dir:
            return
        trace_path = str(Path(self.output_dir) /
                         f"search_trace_N{ctx.node_id}_S{self.seq}_L{L}.bin")
        ctx.rdma_store.start_search_trace(trace_path, self.level)
        ctx.log(f"  Search trace: level={self.level} path={trace_path}")

    def stop_l(self, ctx) -> None:
        if self.level > 0 and self.output_dir:
            ctx.rdma_store.stop_search_trace()


@dataclass
class SearchVfctSession:
    enabled: bool
    label: Optional[str]
    output_dir: str
    mode: str

    @classmethod
    def start(cls, ctx, params: dict) -> "SearchVfctSession":
        cfg = ctx.config.get('vec_fetch_count', {}) or {}
        enabled = bool(cfg.get('enabled', False))
        label = params.get('vfct_label', None)
        output_dir = cfg.get('output_dir', '') or str(ctx.run_dir / "search_vfct")
        mode = str(cfg.get('mode', 'deleted')).lower()
        mode_int = 1 if mode == 'neighbors' else 0

        if not enabled:
            return cls(False, label, output_dir, mode)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if not hasattr(ctx.rdma_store, 'start_search_vec_fetch_count'):
            ctx.log("[WARN] vec_fetch_count.enabled but rdma_store has no "
                    "start_search_vec_fetch_count — bindings need rebuild?")
            return cls(False, label, output_dir, mode)

        try:
            ctx.rdma_store.start_search_vec_fetch_count(mode_int)
        except TypeError:
            ctx.rdma_store.start_search_vec_fetch_count()
            if mode_int != 0:
                ctx.log("[WARN] vec_fetch_count.mode=neighbors but binding "
                        "lacks mode arg — forced to 'deleted'. Rebuild required.")
                mode = 'deleted'
        ctx.log(f"  Search VFCT: enabled (mode={mode}, label={label!r}, "
                f"dir={output_dir})")
        return cls(True, label, output_dir, mode)

    def dump(self, ctx) -> None:
        if not self.enabled or not self.label:
            return
        try:
            vfct_path = str(Path(self.output_dir) /
                            f"search_vfct_N{ctx.node_id}_{self.label}.bin")
            info = ctx.rdma_store.dump_search_vec_fetch_count(vfct_path)
            ctx.log(f"  Search VFCT snapshot {self.label!r}: "
                    f"unique={int(info.get('unique_fetched', 0)):,} "
                    f"total={int(info.get('total_fetches', 0)):,} "
                    f"max={int(info.get('max_fetches', 0)):,} "
                    f"-> {vfct_path}")
        except Exception as e:
            ctx.log(f"[WARN] Search VFCT dump failed: {e}")


class NicBurstSession:
    def __init__(self, proc, mono_start, csv_path: str, op_seq: int):
        self.proc = proc
        self.mono_start = mono_start
        self.csv_path = csv_path
        self.op_seq = op_seq
        self.windows = []

    @classmethod
    def start(cls, ctx) -> "NicBurstSession":
        op_seq = getattr(ctx, '_current_op_seq', 0)
        out_dir = (ctx.run_dir / "nic_bw") if getattr(ctx, 'run_dir', None) else Path("/tmp")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            out_dir = Path("/tmp")
        csv_path = str(out_dir / f"nic_bw_N{ctx.node_id}_S{op_seq}.csv")
        proc, mono_start = diagnostics.nic_burst_monitor_start(csv_path)
        return cls(proc, mono_start, csv_path, op_seq)

    def mark_l_start(self) -> Optional[float]:
        if self.mono_start is None:
            return None
        return time.monotonic() - self.mono_start

    def mark_l_end(self, L: int, start_s: Optional[float]) -> None:
        if self.mono_start is None or start_s is None:
            return
        self.windows.append((L, start_s, time.monotonic() - self.mono_start))

    def stop_and_merge(self, ctx, per_l_stats: dict) -> None:
        if self.proc is None:
            return
        diagnostics.nic_burst_monitor_stop(self.proc)
        try:
            self._write_windows(ctx)
            burst_stats = diagnostics.nic_burst_parse_csv(self.csv_path, self.windows)
            for L, stats in burst_stats.items():
                if L in per_l_stats:
                    per_l_stats[L].update(stats)
        except Exception as e:
            ctx.log(f"[WARN] NIC burst parse failed: {e}")

    def _write_windows(self, ctx) -> None:
        try:
            import json as _json
            win_meta = {
                'node_id': ctx.node_id,
                'op_seq': self.op_seq,
                'mono_start': self.mono_start,
                'windows': [
                    {'L': L, 'start_s': s, 'end_s': e}
                    for L, s, e in self.windows
                ],
            }
            with open(self.csv_path + ".windows.json", "w") as wf:
                _json.dump(win_meta, wf, indent=2)
        except Exception:
            pass


def rdma_snapshot() -> dict:
    return diagnostics.rdma_snapshot()


def merge_rdmabw_stats(per_l_stats: dict, L: int, before: dict, after: dict,
                       elapsed: float) -> None:
    nic_bw = diagnostics.rdma_compute_bw(before, after, elapsed)
    if not nic_bw:
        return
    per_l_stats[L]['nic_rx_gbps'] = nic_bw.get('nic_ib_rx_gbps', 0)
    per_l_stats[L]['nic_tx_gbps'] = nic_bw.get('nic_ib_tx_gbps', 0)
    for key in ('nic_wire_rx_gbps', 'nic_wire_tx_gbps',
                'nic_rdma_rx_gbps', 'nic_rdma_tx_gbps',
                'nic_xmit_wait', 'nic_cnp_sent', 'nic_ecn_marked',
                'nic_out_of_buffer', 'nic_rx_read_requests',
                'nic_msg_rate_mps'):
        if key in nic_bw:
            per_l_stats[L][key] = nic_bw[key]


def pmu_start_for_l(ctx, stats_seq: int, L: int) -> Optional[str]:
    _env = _os.environ.get('PIPELINE_PMU', '').strip().lower()
    enabled = _env in ('1', 'true', 'yes', 'on') or \
              bool(ctx.config.get('pipeline', {}).get('pmu', False))
    if not enabled:
        return None
    run_dir = getattr(ctx, 'run_dir', None)
    if run_dir is None:
        return None
    pmu_dir = Path(run_dir) / 'pmu'
    pmu_dir.mkdir(parents=True, exist_ok=True)
    pmu_path = str(pmu_dir / f'node{ctx.node_id}_seq{stats_seq}_L{L}.kv')
    return pmu_path if diagnostics.pmu_start(pmu_path) else None


def pmu_stop_and_merge(per_l_stats: dict, L: int, pmu_path: Optional[str]) -> None:
    if not pmu_path:
        return
    pmu_stats = diagnostics.pmu_stop(pmu_path)
    if pmu_stats:
        per_l_stats[L].update(pmu_stats)
