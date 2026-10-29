#!/usr/bin/env python3
"""
Experiment report generator for distributed pipeline.

Aggregates per-node results (results_node*.json) into a single
experiment report. Designed for extensibility: add new sections
by implementing a method and registering it in SECTION_REGISTRY.

Usage:
    # CLI
    python3 report.py /path/to/search_rdma/20260304_182729/

    # Programmatic
    from report import generate_report
    report = generate_report("/path/to/result_dir")
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Section Registry — add new sections here
# ============================================================================

SECTION_REGISTRY: List[str] = [
    "summary",
    "per_node",
    "latency_breakdown",
    "recall",
    "cache",
]


# ============================================================================
# ExperimentReport
# ============================================================================

class ExperimentReport:
    """Aggregates per-node results into a single experiment report."""

    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.node_results: Dict[int, dict] = {}
        self._load_node_results()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_node_results(self):
        pattern = "results_node*.json"
        files = sorted(self.result_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No {pattern} files found in {self.result_dir}")

        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            node_id = data.get("node_id", -1)
            self.node_results[node_id] = data

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self) -> dict:
        first = next(iter(self.node_results.values()))
        report = {
            "timestamp": first.get("timestamp", ""),
            "experiment_name": first.get("config", {}).get("experiment_name", ""),
            "result_dir": str(self.result_dir),
            "status": "completed",
            "num_nodes": len(self.node_results),
            "node_ids": sorted(self.node_results.keys()),
            "search_params": first.get("search_params", {}),
            "cluster": first.get("cluster", {}),
        }

        for section_name in SECTION_REGISTRY:
            method = getattr(self, f"_section_{section_name}", None)
            if method:
                report[section_name] = method()

        return report

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _section_summary(self) -> dict:
        """L-value sweep summary — weighted average across all nodes."""
        all_L_keys = set()
        for data in self.node_results.values():
            all_L_keys.update(data.get("results", {}).keys())

        L_keys = sorted(all_L_keys, key=lambda x: int(x.replace("L=", "")))
        summary = {}

        for L_key in L_keys:
            total_queries = 0
            weighted = {
                "total_us": 0.0, "p50_us": 0.0, "p99_us": 0.0,
                "rdma_total_us": 0.0, "n_hops": 0.0, "n_cmps": 0.0,
                "remote_vector_ratio": 0.0,
            }
            search_time_sum = 0.0

            for data in self.node_results.values():
                entry = data.get("results", {}).get(L_key)
                if not entry:
                    continue
                nq = entry.get("routing", {}).get("routed_queries", 0)
                if nq == 0:
                    continue
                total_queries += nq
                weighted["total_us"] += _get_mean(entry, "total_us") * nq
                weighted["p50_us"] += _get_p(entry, "total_us", "p50") * nq
                weighted["p99_us"] += _get_p(entry, "total_us", "p99") * nq
                weighted["rdma_total_us"] += _get_mean(entry, "rdma_total_us") * nq
                weighted["n_hops"] += _get_mean(entry, "n_hops") * nq
                weighted["n_cmps"] += _get_mean(entry, "n_cmps") * nq
                weighted["remote_vector_ratio"] += _get_mean(entry, "remote_vector_ratio") * nq
                search_time_sum += entry.get("search_time_s", 0)

            if total_queries > 0:
                for k in weighted:
                    weighted[k] = round(weighted[k] / total_queries, 2)

            # Compute weighted recall
            recall_weighted = 0.0
            recall_queries = 0
            for data in self.node_results.values():
                entry = data.get("results", {}).get(L_key)
                if not entry or "recall" not in entry:
                    continue
                nq = entry.get("routing", {}).get("routed_queries", 0)
                if nq > 0:
                    recall_weighted += entry["recall"] * nq
                    recall_queries += nq

            L_val = int(L_key.replace("L=", ""))
            entry_dict = {
                "L": L_val,
                "total_queries": total_queries,
                "mean_us": weighted["total_us"],
                "p50_us": weighted["p50_us"],
                "p99_us": weighted["p99_us"],
                "rdma_us": weighted["rdma_total_us"],
                "hops": weighted["n_hops"],
                "cmps": weighted["n_cmps"],
                "remote_ratio": weighted["remote_vector_ratio"],
                "search_time_s": round(search_time_sum, 2),
            }
            if recall_queries > 0:
                entry_dict["recall"] = round(recall_weighted / recall_queries, 6)

            # Cache neighbor hit rate (aggregated across nodes)
            total_nbr_hits, total_nbr_misses = 0, 0
            for data in self.node_results.values():
                entry = data.get("results", {}).get(L_key)
                if not entry:
                    continue
                total_nbr_hits += _get_sum(entry, "cache_nbr_hits")
                total_nbr_misses += _get_sum(entry, "cache_nbr_misses")
            nbr_total = total_nbr_hits + total_nbr_misses
            if nbr_total > 0:
                entry_dict["nbr_hit_rate"] = round(total_nbr_hits / nbr_total, 4)

            summary[L_key] = entry_dict

        return summary

    def _section_per_node(self) -> dict:
        """Per-node key metrics for each L value."""
        per_node = {}
        for node_id, data in sorted(self.node_results.items()):
            pi = data.get("partition", {})
            node_entry = {
                "partition": pi.get("my_partition"),
                "num_local_vectors": pi.get("num_local_vectors"),
                "total_vectors": pi.get("total_vectors"),
                "L_results": {},
            }
            for L_key, entry in sorted(data.get("results", {}).items(),
                                        key=lambda x: int(x[0].replace("L=", ""))):
                routing = entry.get("routing", {})
                if node_entry["partition"] is None:
                    node_entry["partition"] = routing.get("my_partition")

                l_result = {
                    "routed_queries": routing.get("routed_queries", 0),
                    "mean_us": _get_mean(entry, "total_us"),
                    "p50_us": _get_p(entry, "total_us", "p50"),
                    "p99_us": _get_p(entry, "total_us", "p99"),
                    "rdma_us": _get_mean(entry, "rdma_total_us"),
                    "hops": _get_mean(entry, "n_hops"),
                    "remote_ratio": _get_mean(entry, "remote_vector_ratio"),
                    "search_time_s": entry.get("search_time_s", 0),
                }
                nbr_h = _get_sum(entry, "cache_nbr_hits")
                nbr_m = _get_sum(entry, "cache_nbr_misses")
                if nbr_h + nbr_m > 0:
                    l_result["nbr_hit_rate"] = round(nbr_h / (nbr_h + nbr_m), 4)
                if "recall" in entry:
                    l_result["recall"] = entry["recall"]
                node_entry["L_results"][L_key] = l_result
            per_node[f"node_{node_id}"] = node_entry
        return per_node

    def _section_latency_breakdown(self) -> dict:
        """Pipeline stage breakdown — weighted average across nodes."""
        all_L_keys = set()
        for data in self.node_results.values():
            all_L_keys.update(data.get("results", {}).keys())

        L_keys = sorted(all_L_keys, key=lambda x: int(x.replace("L=", "")))
        breakdown = {}

        # Pipeline stages that sum to total (mutually exclusive)
        pipeline_stages = [
            ("neighbor_fetch_us", "neighbor_fetch"),
            ("distance_calc_us", "distance_calc"),
            ("queue_update_us", "queue_update"),
        ]
        # Sub-breakdowns and reference metrics (not included in % total)
        detail_stages = [
            ("neighbor_fetch_local_us", "neighbor_fetch_local"),
            ("neighbor_fetch_remote_us", "neighbor_fetch_remote"),
            ("distance_calc_local_us", "distance_calc_local"),
            ("distance_calc_remote_us", "distance_calc_remote"),
            ("rdma_total_us", "rdma_total"),
            # Fine-grained sub-breakdown
            ("dist_cache_check_us", "dist_cache_check"),
            ("dist_cache_put_us", "dist_cache_put"),
            ("dist_vid_resolution_us", "dist_vid_resolution"),
            ("dist_pure_distance_us", "dist_pure_distance"),
            ("fetch_cache_check_us", "fetch_cache_check"),
            ("fetch_cache_put_us", "fetch_cache_put"),
            ("fetch_vid_resolution_us", "fetch_vid_resolution"),
        ]
        all_stages = pipeline_stages + detail_stages

        for L_key in L_keys:
            total_queries = 0
            weighted = {name: 0.0 for _, name in all_stages}
            total_weighted = 0.0

            for data in self.node_results.values():
                entry = data.get("results", {}).get(L_key)
                if not entry:
                    continue
                nq = entry.get("routing", {}).get("routed_queries", 0)
                if nq == 0:
                    continue
                total_queries += nq
                total_weighted += _get_mean(entry, "total_us") * nq
                for json_key, name in all_stages:
                    weighted[name] += _get_mean(entry, json_key) * nq

            if total_queries > 0:
                total_mean = total_weighted / total_queries
                bd = {"total_us": round(total_mean, 2)}
                # Pipeline stages with % of total
                for _, name in pipeline_stages:
                    val = weighted[name] / total_queries
                    bd[name + "_us"] = round(val, 2)
                    bd[name + "_pct"] = round(100 * val / total_mean, 1) if total_mean > 0 else 0
                # Detail/reference metrics (no % — they overlap with pipeline stages)
                for _, name in detail_stages:
                    val = weighted[name] / total_queries
                    bd[name + "_us"] = round(val, 2)
                breakdown[L_key] = bd

        return breakdown

    def _section_recall(self) -> Optional[dict]:
        """Recall@K — weighted average across all nodes per L value."""
        all_L_keys = set()
        has_recall = False
        for data in self.node_results.values():
            for L_key, entry in data.get("results", {}).items():
                all_L_keys.add(L_key)
                if "recall" in entry:
                    has_recall = True

        if not has_recall:
            return None

        L_keys = sorted(all_L_keys, key=lambda x: int(x.replace("L=", "")))
        recall_data = {}

        for L_key in L_keys:
            total_queries = 0
            weighted_recall = 0.0

            for data in self.node_results.values():
                entry = data.get("results", {}).get(L_key)
                if not entry or "recall" not in entry:
                    continue
                nq = entry.get("routing", {}).get("routed_queries", 0)
                if nq == 0:
                    continue
                total_queries += nq
                weighted_recall += entry["recall"] * nq

            if total_queries > 0:
                L_val = int(L_key.replace("L=", ""))
                recall_data[L_key] = {
                    "L": L_val,
                    "recall": round(weighted_recall / total_queries, 6),
                    "total_queries": total_queries,
                }

                # Per-node recall
                per_node = {}
                for node_id, data in sorted(self.node_results.items()):
                    entry = data.get("results", {}).get(L_key)
                    if entry and "recall" in entry:
                        nq = entry.get("routing", {}).get("routed_queries", 0)
                        per_node[f"node_{node_id}"] = {
                            "recall": entry["recall"],
                            "queries": nq,
                        }
                recall_data[L_key]["per_node"] = per_node

        return recall_data

    def _section_cache(self) -> Optional[dict]:
        """Cache hit/miss stats — weighted average across nodes per L value."""
        # Check if any node has cache data
        has_cache = False
        for data in self.node_results.values():
            for entry in data.get("results", {}).values():
                if _get_sum(entry, "cache_vec_hits") > 0 or _get_sum(entry, "cache_vec_misses") > 0:
                    has_cache = True
                    break
            if has_cache:
                break

        if not has_cache:
            return None

        all_L_keys = set()
        for data in self.node_results.values():
            all_L_keys.update(data.get("results", {}).keys())

        L_keys = sorted(all_L_keys, key=lambda x: int(x.replace("L=", "")))
        cache_data = {}

        for L_key in L_keys:
            total_vh, total_vm = 0, 0
            total_nh, total_nm = 0, 0
            total_ve, total_ne = 0, 0

            for data in self.node_results.values():
                entry = data.get("results", {}).get(L_key)
                if not entry:
                    continue
                total_vh += _get_sum(entry, "cache_vec_hits")
                total_vm += _get_sum(entry, "cache_vec_misses")
                total_nh += _get_sum(entry, "cache_nbr_hits")
                total_nm += _get_sum(entry, "cache_nbr_misses")
                total_ve += _get_sum(entry, "cache_vec_evictions")
                total_ne += _get_sum(entry, "cache_nbr_evictions")

            vec_total = total_vh + total_vm
            nbr_total = total_nh + total_nm

            cache_data[L_key] = {
                "L": int(L_key.replace("L=", "")),
                "vec_hits": total_vh,
                "vec_misses": total_vm,
                "vec_hit_rate": round(total_vh / vec_total, 4) if vec_total > 0 else 0,
                "nbr_hits": total_nh,
                "nbr_misses": total_nm,
                "nbr_hit_rate": round(total_nh / nbr_total, 4) if nbr_total > 0 else 0,
                "vec_evictions": total_ve,
                "nbr_evictions": total_ne,
            }

        return cache_data

    # ------------------------------------------------------------------
    # Save & Print
    # ------------------------------------------------------------------

    def save(self, output_path: Path = None) -> dict:
        report = self.generate()

        if output_path is None:
            output_path = self.result_dir / "report.json"

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate human-readable text report
        txt_path = output_path.with_suffix('.txt')
        txt_content = self._format_text_report(report)
        with open(txt_path, 'w') as f:
            f.write(txt_content)

        # Print to stdout as well
        print(txt_content)
        print(f"Report saved to: {output_path}")
        print(f"Text report:     {txt_path}")
        return report

    # ------------------------------------------------------------------
    # Text Report Formatting
    # ------------------------------------------------------------------

    def _format_text_report(self, report: dict) -> str:
        """Generate a human-readable text report."""
        lines = []
        W = 88  # line width

        # ---- Header ----
        lines.append("=" * W)
        lines.append(f"  Experiment: {report.get('experiment_name', 'unknown')}")
        lines.append(f"  Timestamp:  {report.get('timestamp', '')}")
        lines.append(f"  Nodes:      {report.get('node_ids', [])}")

        sp = report.get("search_params", {})
        if sp:
            lines.append(f"  K={sp.get('K', '?')}, Queries={sp.get('num_queries', '?')}")

        # Graph / partition info from first node's config
        first = next(iter(self.node_results.values()), {})
        cfg = first.get("config", {})
        build = cfg.get("build", {})
        update = cfg.get("update", {})
        part = cfg.get("partition", {})
        if build:
            lines.append(f"  Graph:      R={build.get('R','?')}, L={build.get('L','?')}, "
                         f"alpha={build.get('alpha','?')}, "
                         f"initial={build.get('initial_size','?')}")
        if update and update.get("enabled"):
            lines.append(f"  Update:     {update.get('max_iterations','?')} iters, "
                         f"+{update.get('points_to_insert','?')}/-{update.get('points_to_delete','?')} per iter, "
                         f"consolidate every {update.get('consolidate_every','?')}")
        if part:
            lines.append(f"  Partitions: {part.get('num_partitions', '?')}")

        # Partition sizes from node results
        partition_sizes = {}
        for data in self.node_results.values():
            pi = data.get("partition", {})
            ps = pi.get("partition_sizes", {})
            if ps:
                partition_sizes = ps
                break
        if partition_sizes:
            parts_str = ", ".join(
                f"P{p}={c}" for p, c in sorted(partition_sizes.items(), key=lambda x: int(x[0])))
            lines.append(f"  Vectors:    {parts_str}")

        lines.append("=" * W)

        # ---- Summary Table ----
        summary = report.get("summary", {})
        if summary:
            lines.append("")
            lines.append("  SEARCH PERFORMANCE (weighted avg across nodes)")
            lines.append("")

            has_recall = any("recall" in s for s in summary.values())
            has_cache = any("nbr_hit_rate" in s for s in summary.values())

            # Fixed-width columns
            base_cols = (f"{'Mean(us)':>9}  {'P50(us)':>9}  "
                         f"{'P99(us)':>9}  {'RDMA(us)':>9}  {'Hops':>6}  {'Remote(%)':>9}")
            cache_col = f"  {'NbrHit(%)':>9}" if has_cache else ""
            if has_recall:
                hdr = f"  {'L':>5}  {'Recall':>8}  {base_cols}{cache_col}  {'QPS':>7}"
            else:
                hdr = f"  {'L':>5}  {base_cols}{cache_col}  {'QPS':>7}"
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))

            for L_key in sorted(summary.keys(), key=lambda x: int(x.replace("L=", ""))):
                s = summary[L_key]
                L_val = s.get("L", L_key)
                search_s = s.get("search_time_s", 0)
                nq = s.get("total_queries", 0)
                qps = int(nq / search_s) if search_s > 0 else 0
                remote_pct = s.get("remote_ratio", 0) * 100
                cache_str = f"  {s['nbr_hit_rate'] * 100:>8.1f}%" if has_cache else ""

                if has_recall:
                    recall_str = f"{s['recall']:.4f}" if "recall" in s else "   N/A  "
                    lines.append(
                        f"  {L_val:>5}  {recall_str:>8}  {s['mean_us']:>9.1f}  {s['p50_us']:>9.1f}  "
                        f"{s['p99_us']:>9.1f}  {s['rdma_us']:>9.1f}  {s['hops']:>6.1f}  "
                        f"{remote_pct:>8.1f}%{cache_str}  {qps:>7}")
                else:
                    lines.append(
                        f"  {L_val:>5}  {s['mean_us']:>9.1f}  {s['p50_us']:>9.1f}  "
                        f"{s['p99_us']:>9.1f}  {s['rdma_us']:>9.1f}  {s['hops']:>6.1f}  "
                        f"{remote_pct:>8.1f}%{cache_str}  {qps:>7}")

        # ---- Latency Breakdown ----
        breakdown = report.get("latency_breakdown", {})
        if breakdown:
            lines.append("")
            lines.append("  LATENCY BREAKDOWN (us)")
            lines.append("")

            # Header
            L_keys = sorted(breakdown.keys(), key=lambda x: int(x.replace("L=", "")))
            L_labels = [f"L={int(k.replace('L=', ''))}" for k in L_keys]
            col_w = 12
            hdr = f"  {'Stage':<24}" + "".join(f"{l:>{col_w}}" for l in L_labels)
            lines.append(hdr)
            lines.append("  " + "-" * (24 + col_w * len(L_keys)))

            # Total
            row = f"  {'total':<24}"
            for L_key in L_keys:
                bd = breakdown[L_key]
                row += f"{bd['total_us']:>{col_w}.1f}"
            lines.append(row)
            lines.append("")

            # Pipeline stages
            for stage in ["neighbor_fetch", "distance_calc", "queue_update"]:
                row_us = f"  {'  ' + stage:<24}"
                row_pct = f"  {'':24}"
                for L_key in L_keys:
                    bd = breakdown[L_key]
                    us = bd.get(stage + "_us", 0)
                    pct = bd.get(stage + "_pct", 0)
                    row_us += f"{us:>{col_w}.1f}"
                    row_pct += f"{'(' + f'{pct:.0f}%' + ')':>{col_w}}"
                lines.append(row_us)
                lines.append(row_pct)

            lines.append("")

            # Local/Remote sub-breakdown with hierarchical detail
            lines.append(f"  {'--- Local vs Remote ---'}")

            # Helper to render a row and optional sub-items
            def _render_row(label, key):
                row = f"  {label:<24}"
                for L_key in L_keys:
                    val = breakdown[L_key].get(key + "_us", 0)
                    row += f"{val:>{col_w}.1f}"
                lines.append(row)

            def _render_sub_row(label, key):
                """Render sub-item only if any L value > 0."""
                any_nonzero = any(
                    breakdown[L_key].get(key + "_us", 0) > 0 for L_key in L_keys
                )
                if not any_nonzero:
                    return
                row = f"  {label:<24}"
                for L_key in L_keys:
                    val = breakdown[L_key].get(key + "_us", 0)
                    row += f"{val:>{col_w}.1f}"
                lines.append(row)

            _render_row("  fetch local", "neighbor_fetch_local")
            _render_row("  fetch remote (RDMA)", "neighbor_fetch_remote")
            _render_sub_row("    cache check", "fetch_cache_check")
            _render_sub_row("    cache put", "fetch_cache_put")
            _render_sub_row("    VID resolution", "fetch_vid_resolution")

            _render_row("  dist  local", "distance_calc_local")
            _render_row("  dist  remote (RDMA)", "distance_calc_remote")
            _render_sub_row("    cache check", "dist_cache_check")
            _render_sub_row("    pure distance", "dist_pure_distance")
            _render_sub_row("    cache put", "dist_cache_put")
            _render_sub_row("    VID resolution", "dist_vid_resolution")

            _render_row("  RDMA total", "rdma_total")

        # ---- Cache Stats ----
        cache = report.get("cache")
        if cache:
            lines.append("")
            lines.append("  REMOTE CACHE STATS (aggregated across nodes)")
            lines.append("")
            hdr = (f"  {'L':>5}  {'Vec Hits':>10}  {'Vec Miss':>10}  {'Vec HR':>8}  "
                   f"{'Nbr Hits':>10}  {'Nbr Miss':>10}  {'Nbr HR':>8}  "
                   f"{'Vec Evic':>10}  {'Nbr Evic':>10}")
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))
            for L_key in sorted(cache.keys(), key=lambda x: int(x.replace("L=", ""))):
                c = cache[L_key]
                lines.append(
                    f"  {c['L']:>5}  {c['vec_hits']:>10}  {c['vec_misses']:>10}  "
                    f"{c['vec_hit_rate']*100:>7.1f}%  "
                    f"{c['nbr_hits']:>10}  {c['nbr_misses']:>10}  "
                    f"{c['nbr_hit_rate']*100:>7.1f}%  "
                    f"{c['vec_evictions']:>10}  {c['nbr_evictions']:>10}")

        # ---- Per-Node ----
        per_node = report.get("per_node", {})
        if per_node:
            lines.append("")
            lines.append("  PER-NODE DETAIL")

            for node_key in sorted(per_node.keys()):
                nd = per_node[node_key]
                partition = nd.get("partition", "?")
                num_vecs = nd.get("num_local_vectors")
                lines.append(f"")
                vec_str = f", {num_vecs:,} vectors" if num_vecs else ""
                lines.append(f"  {node_key} (partition {partition}{vec_str})")

                has_recall = any("recall" in v for v in nd.get("L_results", {}).values())
                nd_has_cache = any("nbr_hit_rate" in v for v in nd.get("L_results", {}).values())

                base_cols = (f"{'Mean(us)':>9}  {'P50(us)':>9}  "
                             f"{'P99(us)':>9}  {'RDMA(us)':>9}  {'Hops':>6}  {'Remote(%)':>9}")
                nd_cache_col = f"  {'NbrHit(%)':>9}" if nd_has_cache else ""
                if has_recall:
                    lines.append(f"    {'L':>5}  {'Recall':>8}  {base_cols}{nd_cache_col}  {'Queries':>8}")
                else:
                    lines.append(f"    {'L':>5}  {base_cols}{nd_cache_col}  {'Queries':>8}")
                lines.append(f"    {'-' * (len(lines[-1]) - 4)}")

                for L_key in sorted(nd.get("L_results", {}).keys(),
                                     key=lambda x: int(x.replace("L=", ""))):
                    lr = nd["L_results"][L_key]
                    L_val = int(L_key.replace("L=", ""))
                    remote_pct = lr.get("remote_ratio", 0) * 100
                    nq = lr.get("routed_queries", 0)
                    nd_cache_str = f"  {lr['nbr_hit_rate'] * 100:>8.1f}%" if nd_has_cache else ""

                    if has_recall:
                        recall_str = f"{lr['recall']:.4f}" if "recall" in lr else "   N/A  "
                        lines.append(
                            f"    {L_val:>5}  {recall_str:>8}  {lr['mean_us']:>9.1f}  "
                            f"{lr['p50_us']:>9.1f}  {lr['p99_us']:>9.1f}  "
                            f"{lr['rdma_us']:>9.1f}  {lr['hops']:>6.1f}  "
                            f"{remote_pct:>8.1f}%{nd_cache_str}  {nq:>8}")
                    else:
                        lines.append(
                            f"    {L_val:>5}  {lr['mean_us']:>9.1f}  "
                            f"{lr['p50_us']:>9.1f}  {lr['p99_us']:>9.1f}  "
                            f"{lr['rdma_us']:>9.1f}  {lr['hops']:>6.1f}  "
                            f"{remote_pct:>8.1f}%{nd_cache_str}  {nq:>8}")

        lines.append("")
        lines.append("=" * W)
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# Helpers
# ============================================================================

def _get_mean(entry: dict, key: str) -> float:
    v = entry.get(key, {})
    if isinstance(v, dict):
        return v.get("mean", 0.0)
    return float(v) if v else 0.0


def _get_p(entry: dict, key: str, percentile: str) -> float:
    v = entry.get(key, {})
    if isinstance(v, dict):
        return v.get(percentile, 0.0)
    return 0.0


def _get_sum(entry: dict, key: str) -> int:
    v = entry.get(key, {})
    if isinstance(v, dict):
        return int(v.get("sum", 0))
    return int(v) if v else 0


# ============================================================================
# Public API
# ============================================================================

def generate_report(result_dir: str, output_path: str = None) -> dict:
    """Generate and save experiment report.

    Args:
        result_dir: Directory containing results_node*.json files.
        output_path: Optional path for report.json (default: result_dir/report.json).

    Returns:
        The report dict.
    """
    rpt = ExperimentReport(result_dir)
    out = Path(output_path) if output_path else None
    return rpt.save(out)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment report from per-node results")
    parser.add_argument("result_dir", type=str,
                        help="Directory containing results_node*.json files")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for report.json (default: <result_dir>/report.json)")
    args = parser.parse_args()

    generate_report(args.result_dir, args.output)


if __name__ == "__main__":
    main()
