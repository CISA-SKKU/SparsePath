#!/usr/bin/env python3
"""Download LAION-400M CLIP ViT-B/32 image embeddings.

Source: https://deploy.laion.ai/.../embeddings/img_emb/img_emb_<NNN>.npy
        (410 numpy files, ~1M float16 vectors per file, dim=512)

LAION's distribution is float16 numpy. We download the npy files as-is;
extract_subset.py is for the bin/fvecs/bvecs path, so for LAION we
convert directly here (npy → DiskANN fbin float32).

Disk requirement (300M subset):
    npy:  ~290 GB (kept after conversion unless you delete manually)
    fbin: ~570 GB (300M × 512 × 4)

Output: $RAW_DIR/laion/laion<num>_base.fbin

Queries: LAION does not ship a separate query set in the same distribution.
Common practice is to take the last K embeddings from the base set as queries
or use a held-out file. For the artifact, set --query-count K to slice off
the last K vectors of the conversion as the query set.

Usage:
    RAW_DIR=/path/to/raw_datasets python download_laion.py --num 300M
    RAW_DIR=/path/to/raw_datasets python download_laion.py --num 300M --query-count 10000
    RAW_DIR=/path/to/raw_datasets python download_laion.py --download-only
"""

import argparse
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


BASE_URL = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb"
TOTAL_FILES = 410       # ~1M vec/file ⇒ ~410M total
DIM = 512


def parse_count(s: str) -> int:
    s = s.strip().upper().replace("_", "")
    if s in ("0", "ALL"):
        return 0
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def download_one(idx: int, total: int, npy_dir: Path) -> str:
    url = f"{BASE_URL}/img_emb_{idx}.npy"
    dst = npy_dir / f"img_emb_{idx}.npy"
    if dst.exists() and dst.stat().st_size > 0:
        return f"[{idx:3d}/{total}] skip"
    tmp = dst.with_suffix(".npy.tmp")
    try:
        urlretrieve(url, tmp)
        tmp.rename(dst)
        return f"[{idx:3d}/{total}] done ({dst.stat().st_size/(1024**2):.0f} MB)"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return f"[{idx:3d}/{total}] FAIL {e}"


def download_all(num_files: int, workers: int, npy_dir: Path) -> bool:
    npy_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {num_files} files → {npy_dir}/  (workers={workers})")
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_one, i, num_files, npy_dir): i
                for i in range(num_files)}
        for fut in as_completed(futs):
            msg = fut.result()
            print(msg)
            if "FAIL" in msg:
                failed.append(futs[fut])
    if failed:
        print(f"\n{len(failed)} files failed: {failed}")
        print("Re-run to retry — script is idempotent.")
        return False
    return True


def convert_to_fbin(target: int, num_files: int, npy_dir: Path,
                    base_path: Path, query_count: int, query_path: Path):
    """Concatenate float16 npys → float32 fbin; optionally split off queries."""
    label = f"{target // 1_000_000}M" if target else "400M"
    print(f"Converting → {base_path} ({label})")

    missing = [i for i in range(num_files)
               if not (npy_dir / f"img_emb_{i}.npy").exists()]
    if missing:
        print(f"ERROR: missing {len(missing)} npy files (e.g. {missing[:5]})")
        sys.exit(1)

    base_path.parent.mkdir(parents=True, exist_ok=True)

    # Two-pass: write all into base, then move tail vectors to query.
    with open(base_path, "wb") as f:
        f.write(struct.pack("II", 0, DIM))  # placeholder

        count = 0
        for i in range(num_files):
            v = np.load(npy_dir / f"img_emb_{i}.npy").astype(np.float32)
            if target > 0:
                remaining = target - count
                if remaining <= 0:
                    break
                if len(v) > remaining:
                    v = v[:remaining]
            v.tofile(f)
            count += len(v)
            print(f"  [{i+1:3d}/{num_files}] {count:>12,} vec  "
                  f"({count*DIM*4/(1024**3):.1f} GB)")

    if query_count > 0:
        if query_count >= count:
            sys.exit(f"--query-count {query_count} >= total {count}")
        # Split: trim last query_count vectors out of base into query file.
        trim = query_count
        new_base_count = count - trim
        # Read tail
        with open(base_path, "rb") as f:
            f.seek(8 + new_base_count * DIM * 4)
            tail = f.read(trim * DIM * 4)
        # Write query
        query_path.parent.mkdir(parents=True, exist_ok=True)
        with open(query_path, "wb") as f:
            f.write(struct.pack("II", trim, DIM))
            f.write(tail)
        # Truncate base
        with open(base_path, "r+b") as f:
            f.seek(0)
            f.write(struct.pack("II", new_base_count, DIM))
            f.truncate(8 + new_base_count * DIM * 4)
        print(f"  Split off {trim:,} tail vectors → {query_path}")
        count = new_base_count
    else:
        with open(base_path, "r+b") as f:
            f.seek(0)
            f.write(struct.pack("II", count, DIM))

    print(f"\nDone — base: {count:,} vec, "
          f"{base_path.stat().st_size/(1024**3):.1f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", default="400M",
                   help="Total target vectors (default 400M). Accepts e.g. 300M.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--convert-only", action="store_true")
    p.add_argument("--query-count", type=int, default=10_000,
                   help="If >0, split off this many tail vectors into a query "
                        "file and remove them from base. 0 = no query split.")
    args = p.parse_args()

    raw_dir = os.environ.get("RAW_DIR")
    if not raw_dir:
        sys.exit("RAW_DIR env var must be set (target download directory)")

    target = parse_count(args.num)
    if target > 0:
        num_files = min(TOTAL_FILES, (target + 999_999) // 1_000_000)
    else:
        num_files = TOTAL_FILES

    root = Path(raw_dir) / "laion"
    npy_dir = root / "npy"
    label = f"{target // 1_000_000}M" if target else "400M"
    base_path = root / f"laion{label}_base.fbin"
    query_path = root / f"laion{label}_query.fbin"

    print(f"Plan: {num_files} files, "
          f"~{num_files*0.95:.0f} GB download, "
          f"~{(target or num_files*1_000_000) * DIM * 4 / (1024**3):.0f} GB fbin")

    if not args.convert_only:
        ok = download_all(num_files, args.workers, npy_dir)
        if not ok and not args.download_only:
            sys.exit("Stopping: download failures.")

    if not args.download_only:
        convert_to_fbin(target, num_files, npy_dir,
                        base_path, args.query_count, query_path)


if __name__ == "__main__":
    main()
