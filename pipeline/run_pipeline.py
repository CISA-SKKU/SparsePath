#!/usr/bin/env python3
"""
Distributed Pipeline — E2E Orchestrator.

Usage:
  python run_pipeline.py --config configs/deep1M_e2e.yaml
  python run_pipeline.py --config configs/deep1M_e2e.yaml --stages build partition
  python run_pipeline.py --config configs/deep1M_e2e.yaml --resume-from update
  python run_pipeline.py --config configs/deep1M_e2e.yaml --dry-run
"""

import argparse
import json
import os
import shutil
import sys
import time
import yaml
from pathlib import Path

# Path setup — ensure the pipeline package and scripts/common are importable
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
_scripts_dir = _project_root / 'scripts'
for _p in [str(_this_dir), str(_project_root), str(_scripts_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stages import (
    PipelineContext, STAGE_ORDER, STAGE_MAP,
    validate_stage_order, _resolve_workspace_path,
)
from pipeline.config import (
    apply_config_overrides, normalize_canonical_config, resolve_config_paths,
)
from pipeline.experiment_paths import artifact_experiment_dir


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)



def build_context(config: dict) -> PipelineContext:
    """Build PipelineContext from config."""
    experiment_name = config.get('experiment_name', 'unnamed')

    ctx = PipelineContext(
        config=config,
        base_dir=artifact_experiment_dir(config),
        experiment_name=experiment_name,
    )

    # Try to restore from checkpoint
    if ctx.load_checkpoint():
        print(f"Restored pipeline state from checkpoint")

    return ctx


def save_final_report(ctx: PipelineContext):
    """Save final experiment report."""
    report = {
        'experiment_name': ctx.experiment_name,
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'config': ctx.config,
        'search_results': ctx.search_results,
    }
    report_file = ctx.base_dir / "final_report.json"
    os.makedirs(ctx.base_dir, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFinal report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed DiskANN Pipeline — E2E Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config configs/deep1M_e2e.yaml
  %(prog)s --config configs/deep1M_e2e.yaml --stages build partition
  %(prog)s --config configs/deep1M_e2e.yaml --resume-from update
  %(prog)s --config configs/deep1M_e2e.yaml --dry-run
        """,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--stages", nargs="*", help="Override: run only these stages")
    parser.add_argument("--resume-from", help="Resume from this stage (skip earlier)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stages without executing")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config key via dotted path (e.g. "
             "--set partition.insert_strategy=round_robin). "
             "Values are parsed as YAML so ints/floats/bools/lists work.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    apply_config_overrides(config, args.overrides, log=True)
    normalize_canonical_config(config)
    resolve_config_paths(config, _project_root)
    experiment_name = config.get('experiment_name', 'unnamed')

    print("=" * 60)
    print(f"  Pipeline: {experiment_name}")
    print(f"  Config:   {args.config}")
    print("=" * 60)

    # Determine which stages to run
    configured_stages = config.get('pipeline', {}).get('stages', STAGE_ORDER)
    stages = args.stages or configured_stages

    # Validate stage names
    for s in stages:
        if s not in STAGE_MAP:
            parser.error(f"Unknown stage: '{s}'. Valid: {STAGE_ORDER}")

    # Handle --resume-from
    if args.resume_from:
        if args.resume_from not in STAGE_ORDER:
            parser.error(f"Unknown stage: '{args.resume_from}'. Valid: {STAGE_ORDER}")
        resume_idx = STAGE_ORDER.index(args.resume_from)
        stages = [s for s in stages if STAGE_ORDER.index(s) >= resume_idx]

    # Validate stage dependencies
    warnings = validate_stage_order(stages)
    for w in warnings:
        print(f"  [WARN] {w}")

    print(f"  Stages:   {' → '.join(stages)}")
    if args.dry_run:
        print(f"  Mode:     DRY RUN")
    print()

    # Dry run
    if args.dry_run:
        for name in stages:
            print(f"  [DRY RUN] Would execute: {name}")
        print("\nDone (dry run, nothing executed).")
        return

    # Build context
    ctx = build_context(config)

    # Save config snapshot for reproducibility
    os.makedirs(ctx.base_dir, exist_ok=True)
    config_copy = ctx.base_dir / "config.yaml"
    shutil.copy2(args.config, config_copy)
    print(f"  Config saved to {config_copy}")

    # Execute stages
    pipeline_t0 = time.time()
    for name in stages:
        print(f"\n{'=' * 60}")
        print(f"  Stage: {name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            ctx = STAGE_MAP[name](ctx)
        except Exception as e:
            print(f"\n  [ERROR] Stage '{name}' failed: {e}")
            ctx.save_checkpoint()
            raise

        elapsed = time.time() - t0
        print(f"\n  Stage '{name}' completed in {elapsed:.1f}s")
        ctx.save_checkpoint()

    total = time.time() - pipeline_t0
    print(f"\n{'=' * 60}")
    print(f"  Pipeline completed in {total:.1f}s")
    print(f"{'=' * 60}")

    save_final_report(ctx)


if __name__ == "__main__":
    main()
