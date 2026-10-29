#!/usr/bin/env python3
"""Post-hoc correction for PMU runs before the duration-capture fix.

Pre-fix: `pmu_monitor.sh stop` divided counters by 1s instead of the actual
wall time, so all *_gbps derived values are inflated by exactly wall_sec.
Ratios (IPC, hit rates, DDIO %) are ALSO wrong because the DDIO formula
mixes rate-derived values.

This script:
 1. Walks run_dir/pmu/*.kv
 2. Reads raw counters + looks up actual wall from op_history.json
    (search_stats_by_L[L]['elapsed_s'] per node)
 3. Re-derives *_gbps / hit-rate / DDIO using the correct duration
 4. Writes corrected pmu_* back into op_history.json
 5. Regenerates engine_report.txt via OperationEngine._generate_report

Usage: python pmu_posthoc_fix.py <run_dir>
"""
import json
import re
import sys
import types
import yaml
from pathlib import Path


def parse_kv(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def rederive(raw: dict, duration_sec: float) -> dict:
    """Recompute derived_* metrics from raw counters + correct duration."""
    cycles = raw.get('cycles', 0)
    instr = raw.get('instructions', 0)
    llcl = raw.get('LLC-loads', 0)
    llcm = raw.get('LLC-load-misses', 0)
    l3h = raw.get('mem_load_retired.l3_hit', 0)
    l3m = raw.get('mem_load_retired.l3_miss', 0)

    imc_r = 0.0
    imc_w = 0.0
    iio_mw = 0.0
    iio_mr = 0.0
    for k, v in raw.items():
        kl = k.lower()
        if 'uncore_imc' in kl and 'cas_count_read' in kl:
            imc_r += v
        elif 'uncore_imc' in kl and 'cas_count_write' in kl:
            imc_w += v
        elif 'iio' in kl and 'mem_write' in kl:
            iio_mw += v
        elif 'iio' in kl and 'mem_read' in kl:
            iio_mr += v

    out = {}
    out['pmu_ipc'] = round(instr / cycles, 3) if cycles > 0 else 0.0
    out['pmu_llc_hit_rate_pct'] = (
        round((llcl - llcm) / llcl * 100, 2) if llcl > 0 else 0.0)
    l3tot = l3h + l3m
    out['pmu_l3_hit_rate_pct'] = (
        round(l3h / l3tot * 100, 2) if l3tot > 0 else 0.0)
    # IMC values are MiB (perf auto-scaled); × 2^20 → bytes
    out['pmu_dram_read_gbps'] = round(imc_r * 1048576 / duration_sec / 1e9, 3)
    out['pmu_dram_write_gbps'] = round(imc_w * 1048576 / duration_sec / 1e9, 3)
    # IIO counts are 4-byte request counts
    out['pmu_iio_write_gbps'] = round(iio_mw * 4 / duration_sec / 1e9, 3)
    out['pmu_iio_read_gbps'] = round(iio_mr * 4 / duration_sec / 1e9, 3)
    iio_b = iio_mw * 4
    imc_b = imc_w * 1048576
    if iio_b > 0:
        a = (iio_b - imc_b) / iio_b * 100
        out['pmu_ddio_absorbed_pct'] = round(max(0, min(100, a)), 2)
    else:
        out['pmu_ddio_absorbed_pct'] = 0.0
    return out


_KV_RE = re.compile(r'node(\d+)_seq(\d+)_L(\d+)\.kv$')


def main(run_dir: Path):
    oph_path = run_dir / 'op_history.json'
    with open(oph_path) as f:
        oph = json.load(f)

    pmu_dir = run_dir / 'pmu'
    fixed = 0
    for kv_file in sorted(pmu_dir.glob('*.kv')):
        m = _KV_RE.search(kv_file.name)
        if not m:
            continue
        node, seq, L = int(m.group(1)), int(m.group(2)), int(m.group(3))
        raw = parse_kv(kv_file)

        # Find the op + per_node entry to fetch the correct wall.
        op = next((o for o in oph['operations'] if o.get('seq') == seq), None)
        if op is None:
            continue
        pn = op.get('per_node', {})
        ns = pn.get(str(node)) or pn.get(node)
        if ns is None:
            continue
        sbl = ns.get('search_stats_by_L', {})
        sL = sbl.get(str(L)) or sbl.get(L)
        if sL is None:
            continue
        wall = float(sL.get('elapsed_s', 0))
        if wall <= 0:
            continue

        new_derived = rederive(raw, wall)
        for k, v in new_derived.items():
            sL[k] = v
        fixed += 1

    # Backup original op_history once before overwriting
    bak = run_dir / 'op_history.json.prefix.bak'
    if not bak.exists():
        bak.write_text(oph_path.read_text())
    oph_path.write_text(json.dumps(oph, indent=2))
    print(f"Corrected {fixed} (node, seq, L) entries in {oph_path}")
    print(f"Original backed up to {bak}")

    # Regenerate engine_report.txt via OperationEngine._generate_report
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.engine import OperationEngine
    # Find config yaml in run_dir
    yamls = list(run_dir.glob('*.yaml'))
    if not yamls:
        print("(no yaml in run_dir, skipping report regen)")
        return
    with open(yamls[0]) as f:
        cfg = yaml.safe_load(f)
    eng = OperationEngine.__new__(OperationEngine)
    eng.op_history = oph['operations']
    eng.run_dir = run_dir
    ctx = types.SimpleNamespace()
    ctx.config = cfg
    eng.ctx = ctx
    out = eng._generate_report()
    (run_dir / 'engine_report.txt').write_text(out)
    print(f"Report regenerated: {run_dir}/engine_report.txt ({len(out.splitlines())} lines)")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: pmu_posthoc_fix.py <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).resolve())
