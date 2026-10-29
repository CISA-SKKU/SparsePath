#!/usr/bin/env python
"""LITS stats — single-node measurement of already-built graphs (no RDMA, no rebuild).

Loads a graph checkpoint's ADJACENCY + tags (via load_topology; no base vectors)
and its partition labels, then classifies every active out-edge as intra- or
inter-partition to reproduce the paper's remote-edge headline single-node:

    deep100M   lits graph + lits labels   ~7.56%   inter (paper)
               vamana graph + ncb 4p      ~24.8%
               vamana graph + rr  4p      ~75.0%

The LITS graphs are ALREADY BUILT (by diskann-lits, the single-node LITS graph
builder). This harness only MEASURES them — nothing is rebuilt or re-run. The
inter-partition edge fraction is the exact "remote edge" metric: an out-edge is
remote iff its endpoints sit in different partitions (= different machines in the
M-node system). It is a pure graph-structure property, so single-node is exact.

Partition indexing (auto per source):
  * lits `.labels`  : [n u32][K u32] then n u32, TAG-indexed (tags are
                      non-sequential after update cycles) -> by_tag=True.
  * `partition_ids.npy` (vamana rr/ncb): GID-indexed (gid = tag-1) -> by_tag=False.

Usage:
    python run_lits_stats.py --config datasets.yaml [--dataset deep100M] [--out results/]
"""
import argparse
import json
import os
import sys
import time
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "lib"))

try:
    import pq_search_bindings as pqsb
except ImportError as e:
    sys.exit(f"cannot import pq_search_bindings (build it first: docker exec danns ./build_all.sh): {e}")

try:
    import yaml
except ImportError:
    yaml = None

_HARNESS = {
    "float32": pqsb.PQSearchHarnessFloat, "float": pqsb.PQSearchHarnessFloat,
    "uint8": pqsb.PQSearchHarnessUInt8, "int8": pqsb.PQSearchHarnessInt8,
}


def load_part(spec):
    """Return (partition_array uint32, by_tag bool)."""
    kind = spec.get("kind", "labels")
    path = spec["file"]
    if kind == "labels":
        # lits .labels: skip [n u32][K u32] header (8 bytes); rest is tag-indexed.
        arr = np.fromfile(path, dtype=np.uint32, offset=8)
        return np.ascontiguousarray(arr, dtype=np.uint32), True
    if kind == "partition_ids":
        arr = np.load(path).astype(np.uint32)
        return np.ascontiguousarray(arr, dtype=np.uint32), False
    raise ValueError(f"unknown partition kind: {kind!r} (use 'labels' or 'partition_ids')")


def run_dataset(name, cfg, out_dir):
    Harness = _HARNESS[cfg["dtype"]]
    dim = int(cfg["dim"])
    max_points = int(cfg.get("max_points", 110000000))
    rows = []
    print(f"\n=== {name} (dim={dim}) ===", flush=True)
    for g in cfg["graphs"]:
        h = Harness(dim=dim, max_points=max_points)
        t0 = time.time()
        h.load_topology(g["graph_file"], g["tags_file"])
        print(f"[{name}/{g['name']}] topology loaded  ({time.time()-t0:.1f}s)", flush=True)
        for ps in g["partitions"]:
            part, by_tag = load_part(ps)
            t1 = time.time()
            s = h.graph_stats(part, by_tag)
            counts = np.asarray(s["partition_node_counts"], dtype=np.int64)
            active_counts = counts[counts > 0]
            balance = float(active_counts.max() / active_counts.min()) if active_counts.size else 0.0
            row = dict(
                dataset=name, graph=g["name"], partition=ps["name"],
                n_active=int(s["n_active"]), n_skipped=int(s["n_skipped"]),
                total_edges=int(s["total_edges"]), inter_edges=int(s["inter_edges"]),
                inter_pct=round(float(s["inter_pct"]), 4),
                avg_degree=round(float(s["avg_degree"]), 3),
                zero_degree_nodes=int(s["zero_degree_nodes"]),
                num_partitions=int(s["num_partitions"]),
                balance=round(balance, 3),
                partition_node_counts=counts.tolist(),
                stats_secs=round(time.time() - t1, 1),
            )
            rows.append(row)
            skip_note = f"  ⚠ skipped={row['n_skipped']}" if row["n_skipped"] else ""
            print(f"  [{ps['name']:<16s}] inter={row['inter_pct']:6.2f}%  "
                  f"avg_deg={row['avg_degree']:5.1f}  zero_deg={row['zero_degree_nodes']}  "
                  f"bal={row['balance']:.2f}  nodes/part={counts.tolist()}{skip_note}", flush=True)
        del h  # free the graph store before loading the next graph
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_lits_stats.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results_lits_stats")
    args = ap.parse_args()
    if yaml is None:
        sys.exit("pyyaml not available")
    with open(args.config) as f:
        conf = yaml.safe_load(f)
    datasets = conf["datasets"]
    keys = [args.dataset] if args.dataset else list(datasets.keys())
    all_rows = []
    for k in keys:
        all_rows += run_dataset(k, datasets[k], args.out)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "LITS_stats_table.md"), "w") as f:
        f.write("| dataset | graph | partition | inter-edge% | avg_deg | zero-deg | balance | n_active | skipped |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in all_rows:
            f.write(f"| {r['dataset']} | {r['graph']} | {r['partition']} | {r['inter_pct']:.2f} | "
                    f"{r['avg_degree']:.1f} | {r['zero_degree_nodes']} | {r['balance']:.2f} | "
                    f"{r['n_active']} | {r['n_skipped']} |\n")
    print(f"\nwrote {args.out}/LITS_stats_table.md", flush=True)


if __name__ == "__main__":
    main()
