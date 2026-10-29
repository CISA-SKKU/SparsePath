#!/usr/bin/env python3
"""Download DEEP1B (Yandex) base shards + query file.

Source:  Yandex Disk public link (https://yadi.sk/d/11eDCm7Dsn9GA), resolved via
         https://cloud-api.yandex.net/v1/disk/public/resources/download
Files:   base/base_00 ... base/base_NN (10 GB each, raw fvecs byte stream)
         deep1B_queries.fvecs

After download we expect callers to either (a) cat the base shards into a single
fvecs stream, or (b) feed the shards directly to extract_subset.py via a process
substitution. Strategy (a) is simpler and is what data/download_and_extract.sh
uses; the merged file lives at $RAW_DIR/deep1B/base.fvecs.

Default: 12 shards (~120 GB). Each shard ≈ 27M vectors of 96-dim float32, so
12 shards ≈ 324M vectors — enough to slice the 300M backing file used by the
deep100M experiment. Override --base-shards if you need a different size.

Usage:
    RAW_DIR=/path/to/raw_datasets python download_deep1B.py
    RAW_DIR=/path/to/raw_datasets python download_deep1B.py --base-shards 4
    RAW_DIR=/path/to/raw_datasets python download_deep1B.py --merge

The Yandex API is occasionally rate-limited; the script is idempotent
(re-running skips already-downloaded files), so just retry on failure.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

YADISK = "https://yadi.sk/d/11eDCm7Dsn9GA"
API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def resolve_url(filename: str) -> str:
    cmd = ["curl", "-sS", f'{API}?public_key={YADISK}&path=/{filename}']
    out = subprocess.check_output(cmd)
    try:
        return json.loads(out)["href"]
    except (json.JSONDecodeError, KeyError) as e:
        sys.stderr.write(f"Yandex API response: {out!r}\n")
        raise RuntimeError(f"Could not resolve URL for {filename}: {e}")


def wget(url: str, dst: Path):
    if dst.exists():
        print(f"[skip] {dst} already exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"[get]  {dst}")
    subprocess.run(["wget", "--quiet", "--show-progress", url, "-O", str(tmp)],
                   check=True, stdin=subprocess.DEVNULL)
    tmp.rename(dst)


def merge_shards(root: Path, num_shards: int):
    out = root / "base.fvecs"
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    shards = [str(root / "base" / f"base_{i:02d}") for i in range(num_shards)]
    print(f"[cat ] {' '.join(shards)} > {out}")
    with open(out, "wb") as fout:
        subprocess.run(["cat", *shards], stdout=fout, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-shards", type=int, default=12,
                   help="Number of base_NN shards to download (default 12 ≈ "
                        "324M vec — enough for the deep100M / 300M backing file).")
    p.add_argument("--merge", action="store_true",
                   help="After download, cat shards into a single base.fvecs.")
    args = p.parse_args()

    raw_dir = os.environ.get("RAW_DIR")
    if not raw_dir:
        sys.exit("RAW_DIR env var must be set (target download directory)")
    root = Path(raw_dir) / "deep1B"
    (root / "base").mkdir(parents=True, exist_ok=True)

    # Query file (small, ~15 MB)
    wget(resolve_url("deep1B_queries.fvecs"), root / "deep1B_queries.fvecs")

    # Base shards
    for i in range(args.base_shards):
        fname = f"base/base_{i:02d}"
        wget(resolve_url(fname), root / fname)

    if args.merge:
        merge_shards(root, args.base_shards)

    print(f"\nDone — {args.base_shards} base shards + queries in {root}")


if __name__ == "__main__":
    main()
