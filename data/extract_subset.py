#!/usr/bin/env python3
"""Extract a subset of vectors and convert to DiskANN binary format.

Supports three input formats:
  - bin      DiskANN format: [npts: uint32, dim: uint32, data...]
  - fvecs    per-vector [dim: int32, float32 * dim] (float32, e.g., DEEP1B)
  - bvecs    per-vector [dim: int32, uint8   * dim] (uint8,   e.g., SIFT1B)

Output is always DiskANN binary format with two-uint32 header [npts, dim].
File extension chosen by --dtype: float32 -> .fbin, uint8 -> .bin, int8 -> .i8bin.

Usage examples (these mirror data/download_and_extract.sh):

  # SIFT 300M from a downloaded SIFT1B bvecs file
  python extract_subset.py \
      --input  $DATASETS_ROOT/raw/sift1B/bigann_base.bvecs \
      --output $DATASETS_ROOT/SIFT1B/sift300M_base.bin \
      --count 300_000_000 --dtype uint8 --format bvecs \
      --query-input  $DATASETS_ROOT/raw/sift1B/bigann_query.bvecs \
      --query-output $DATASETS_ROOT/SIFT1B/sift300M_query.bin \
      --query-format bvecs

  # DEEP 300M from a (concatenated) deep1B fvecs stream
  python extract_subset.py \
      --input  $DATASETS_ROOT/raw/deep1B/base.fvecs \
      --output $DATASETS_ROOT/DEEP1B/deep300M_base.fbin \
      --count 300_000_000 --dtype float32 --format fvecs \
      --query-input  $DATASETS_ROOT/raw/DEEP1B/deep1B_queries.fvecs \
      --query-output $DATASETS_ROOT/DEEP1B/deep300M_query.fbin \
      --query-format fvecs

  # SPACEV 300M from a SPACEV1B bin file (already DiskANN format, int8)
  python extract_subset.py \
      --input  $DATASETS_ROOT/raw/spacev1B/spacev1B_base.bin \
      --output $DATASETS_ROOT/SPACEV1B/spacev300M_base.i8bin \
      --count 300_000_000 --dtype int8 --format bin \
      --query-input  $DATASETS_ROOT/raw/spacev1B/spacev1B_query.bin \
      --query-output $DATASETS_ROOT/SPACEV1B/spacev300M_query.i8bin \
      --query-format bin
"""

import argparse
import os
import struct
import time

import numpy as np


DTYPE_MAP = {
    "float32": (np.float32, 4),
    "uint8":   (np.uint8,   1),
    "int8":    (np.int8,    1),
}


def _read_per_vec_header(path: str, elem_size: int):
    """fvecs / bvecs share the same per-vector header [dim: int32, data]."""
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    vec_size = 4 + dim * elem_size
    file_size = os.path.getsize(path)
    if file_size % vec_size != 0:
        raise ValueError(
            f"{path}: size {file_size} not divisible by per-vec size {vec_size} "
            f"(dim={dim}, elem_size={elem_size}). Wrong --format?"
        )
    npts = file_size // vec_size
    return npts, dim, vec_size


def extract_from_bin(input_path: str, output_path: str, count: int, dtype: str):
    """Slice DiskANN bin format → DiskANN bin (no conversion)."""
    _, elem_size = DTYPE_MAP[dtype]
    with open(input_path, "rb") as f:
        npts, dim = struct.unpack("II", f.read(8))

    print(f"Source (bin): {input_path}")
    print(f"  npts={npts:,}, dim={dim}, dtype={dtype}")

    count = min(count, npts)
    point_size = dim * elem_size
    total_bytes = 8 + count * point_size

    print(f"Output: {output_path}")
    print(f"  count={count:,}, size={total_bytes / (1024**3):.1f} GB")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()
    chunk = 100_000
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        fout.write(struct.pack("II", count, dim))
        fin.seek(8)

        written = 0
        while written < count:
            batch = min(chunk, count - written)
            data = fin.read(batch * point_size)
            if len(data) < batch * point_size:
                raise RuntimeError(f"Unexpected EOF at vec {written:,}")
            fout.write(data)
            written += batch
            if written % 10_000_000 == 0 or written == count:
                elapsed = time.time() - t0
                print(
                    f"  {written:,}/{count:,} ({100*written/count:.0f}%) — "
                    f"{written/elapsed/1e6:.1f}M vec/s"
                )
    print(f"Done in {time.time()-t0:.1f}s — "
          f"{os.path.getsize(output_path)/(1024**3):.1f} GB")


def extract_from_per_vec(input_path: str, output_path: str, count: int, dtype: str):
    """Slice fvecs (float32) or bvecs (uint8) → DiskANN bin format."""
    _, elem_size = DTYPE_MAP[dtype]
    npts, dim, vec_size = _read_per_vec_header(input_path, elem_size)

    print(f"Source (per-vec hdr, elem_size={elem_size}): {input_path}")
    print(f"  npts={npts:,}, dim={dim}, vec_size={vec_size}")

    count = min(count, npts)
    out_point_size = dim * elem_size
    total_bytes = 8 + count * out_point_size

    print(f"Output (bin): {output_path}")
    print(f"  count={count:,}, dtype={dtype}, size={total_bytes / (1024**3):.1f} GB")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()
    chunk = 100_000
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        fout.write(struct.pack("II", count, dim))

        written = 0
        while written < count:
            batch = min(chunk, count - written)
            raw = fin.read(batch * vec_size)
            if len(raw) < batch * vec_size:
                raise RuntimeError(f"Unexpected EOF at vec {written:,}")

            arr = np.frombuffer(raw, dtype=np.uint8).reshape(batch, vec_size)
            # Strip 4-byte dim prefix; remaining bytes are dim*elem_size of data
            fout.write(arr[:, 4:].tobytes())

            written += batch
            if written % 10_000_000 == 0 or written == count:
                elapsed = time.time() - t0
                print(
                    f"  {written:,}/{count:,} ({100*written/count:.0f}%) — "
                    f"{written/elapsed/1e6:.1f}M vec/s"
                )
    print(f"Done in {time.time()-t0:.1f}s — "
          f"{os.path.getsize(output_path)/(1024**3):.1f} GB")


def convert_query_per_vec(input_path: str, output_path: str, dtype: str):
    """Whole-file conversion of fvecs/bvecs query → DiskANN bin (queries are small)."""
    _, elem_size = DTYPE_MAP[dtype]
    npts, dim, vec_size = _read_per_vec_header(input_path, elem_size)

    print(f"Converting query: {input_path} → {output_path}")
    print(f"  npts={npts:,}, dim={dim}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(npts, vec_size)
    payload = arr[:, 4:].tobytes()

    with open(output_path, "wb") as f:
        f.write(struct.pack("II", npts, dim))
        f.write(payload)
    print(f"  Done ({os.path.getsize(output_path)/(1024**2):.1f} MB)")


def copy_query_bin(input_path: str, output_path: str):
    """Bin queries are already DiskANN-native; just copy."""
    import shutil
    print(f"Copying query: {input_path} → {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"  Done ({os.path.getsize(output_path)/(1024**2):.1f} MB)")


def parse_count(s: str) -> int:
    s = s.strip().upper().replace("_", "")
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def main():
    p = argparse.ArgumentParser(description="Extract subset + convert to DiskANN bin")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", required=True,
                   help="Vectors to keep (e.g. 300_000_000 or 300M)")
    p.add_argument("--dtype", default="float32",
                   choices=list(DTYPE_MAP.keys()))
    p.add_argument("--format", default="bin", choices=["bin", "fvecs", "bvecs"],
                   help="Input format")
    p.add_argument("--query-input")
    p.add_argument("--query-output")
    p.add_argument("--query-format", choices=["bin", "fvecs", "bvecs"],
                   help="Defaults to --format")
    args = p.parse_args()

    count = parse_count(args.count)

    if args.format == "bin":
        extract_from_bin(args.input, args.output, count, args.dtype)
    else:  # fvecs or bvecs
        extract_from_per_vec(args.input, args.output, count, args.dtype)

    if args.query_input and args.query_output:
        qfmt = args.query_format or args.format
        if qfmt == "bin":
            copy_query_bin(args.query_input, args.query_output)
        else:
            convert_query_per_vec(args.query_input, args.query_output, args.dtype)


if __name__ == "__main__":
    main()
