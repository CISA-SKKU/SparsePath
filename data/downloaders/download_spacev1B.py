#!/usr/bin/env python3
"""Download SPACEV1B from Hugging Face and stream into DiskANN bin format.

Source:  https://huggingface.co/datasets/BAAI/SPACEV  (split=base, streaming)
Output:  $RAW_DIR/spacev1B/spacev1B_base.bin    (DiskANN bin, int8)
         (queries are not on the same dataset card; see below)

SPACEV vectors are 100-dim int8. The Hugging Face stream yields one vector per
sample, so we stream-write rather than holding the full 1B in memory.

Disk requirement: ~93 GB for full 1B base. Use --num to limit (e.g. --num 300M).

Usage:
    RAW_DIR=/path/to/raw_datasets python download_spacev1B.py
    RAW_DIR=/path/to/raw_datasets python download_spacev1B.py --num 300M

Queries: SPACEV1B queries are NOT included in the BAAI/SPACEV HF dataset.
The original release ships them as `query.bin` from MS-Turing's SPACEV repo.
Place the raw query file at $RAW_DIR/spacev1B/spacev1B_query.bin manually
(or extend this script if a public mirror becomes available).

Requires: pip install datasets numpy
"""

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


def parse_count(s: str) -> int:
    s = s.strip().upper().replace("_", "")
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", default="0",
                   help="Vectors to extract; 0/all = full 1B (default 0). "
                        "Accepts e.g. 300M.")
    p.add_argument("--output-name", default="spacev1B_base.bin",
                   help="Output file name within $RAW_DIR/spacev1B/")
    args = p.parse_args()

    raw_dir = os.environ.get("RAW_DIR")
    if not raw_dir:
        sys.exit("RAW_DIR env var must be set (target download directory)")

    target = parse_count(args.num) or 1_000_000_000
    root = Path(raw_dir) / "spacev1B"
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / args.output_name

    # Lazy import — datasets is heavy
    from datasets import load_dataset

    print("Probing dim from first sample ...")
    ds = load_dataset("BAAI/SPACEV", split="base", streaming=True)
    first = next(iter(ds))
    dim = len(first["vector"])
    print(f"  dim = {dim}")

    print(f"Streaming {target:,} vectors → {out_path}")
    print(f"  estimated size: {target * dim / 1e9:.1f} GB (int8)")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    t0 = time.time()
    written = 0

    with open(tmp, "wb") as f:
        # Reserve header; rewrite count at the end if --num was 0/all.
        f.write(struct.pack("II", 0, dim))

        ds = load_dataset("BAAI/SPACEV", split="base", streaming=True)
        for sample in ds:
            if written >= target:
                break
            vec = np.asarray(sample["vector"], dtype=np.int8)
            f.write(vec.tobytes())
            written += 1
            if written % 1_000_000 == 0:
                elapsed = time.time() - t0
                rate = written / elapsed
                eta = (target - written) / rate
                print(f"  {written:,}/{target:,} "
                      f"({100*written/target:.1f}%) — "
                      f"{rate/1000:.0f}K vec/s  ETA {eta/3600:.1f}h")

    # Patch header with the actual count.
    with open(tmp, "r+b") as f:
        f.seek(0)
        f.write(struct.pack("II", written, dim))

    tmp.rename(out_path)
    print(f"\nDone in {(time.time()-t0)/3600:.2f}h — {out_path} "
          f"({out_path.stat().st_size/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
