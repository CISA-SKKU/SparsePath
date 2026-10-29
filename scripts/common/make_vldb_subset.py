#!/usr/bin/env python3
"""Extract a subset of vectors from binary files and convert to DiskANN fbin/i8bin format.

Supports:
  - DiskANN bin format: [npts: uint32, dim: uint32, data...]
  - fvecs format: per-vector [dim: int32, floats...] (e.g., deep1B base.fvecs)

Output is always DiskANN binary format (fbin for float32, i8bin for int8, etc.)

Usage:
    # SIFT 500M from sift1B (already bin format, uint8)
    python extract_subset.py \\
        --input /path/to/sift1B/sift1B_base.bin \\
        --output /path/to/local_ssd/sift500M/sift500M_base.bin \\
        --count 500000000 --dtype uint8 --format bin \\
        --query-input /path/to/sift1B/sift1B_query.bin \\
        --query-output /path/to/local_ssd/sift500M/sift500M_query.bin

    # Deep 500M from deep1B (fvecs format, float32)
    python extract_subset.py \\
        --input /path/to/deep1b/base.fvecs \\
        --output /path/to/local_ssd/deep500M/deep500M_base.fbin \\
        --count 500000000 --dtype float32 --format fvecs \\
        --query-input /path/to/deep1b/deep1B_queries.fvecs \\
        --query-output /path/to/local_ssd/deep500M/deep500M_query.fbin --query-format fvecs

    # SpaceV 500M from spacev1B (bin format, int8)
    python extract_subset.py \\
        --input /path/to/spacev1B/spacev1B_base.bin \\
        --output /path/to/local_ssd/spacev500M/spacev500M_base.i8bin \\
        --count 500000000 --dtype int8 --format bin \\
        --query-input /path/to/spacev1B/spacev1B_query.bin \\
        --query-output /path/to/local_ssd/spacev500M/spacev500M_query.i8bin
"""

import argparse
import os
import struct
import time
import numpy as np


DTYPE_MAP = {
    'float32': (np.float32, 4),
    'uint8': (np.uint8, 1),
    'int8': (np.int8, 1),
}


def read_fvecs_header(path):
    """Read dimension from fvecs file (first 4 bytes = dim as int32)."""
    with open(path, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
    elem_size = 4  # fvecs = float32
    vec_size = 4 + dim * elem_size  # dim_prefix + data
    file_size = os.path.getsize(path)
    npts = file_size // vec_size
    return npts, dim, vec_size


def extract_from_bin(input_path, output_path, count, dtype):
    """Extract from DiskANN bin format [npts, dim, data...]."""
    np_dtype, elem_size = DTYPE_MAP[dtype]

    with open(input_path, 'rb') as f:
        npts, dim = struct.unpack('II', f.read(8))

    print(f"Source (bin): {input_path}")
    print(f"  npts={npts:,}, dim={dim}, dtype={dtype}")

    count = min(count, npts)
    point_size = dim * elem_size
    total_bytes = 8 + count * point_size

    print(f"Output: {output_path}")
    print(f"  count={count:,}, size={total_bytes / (1024**3):.1f} GB")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    t0 = time.time()
    chunk = 100_000

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        fout.write(struct.pack('II', count, dim))
        fin.seek(8)

        written = 0
        while written < count:
            batch = min(chunk, count - written)
            data = fin.read(batch * point_size)
            if len(data) < batch * point_size:
                raise RuntimeError(f"Unexpected EOF at {written:,}")
            fout.write(data)
            written += batch
            if written % 10_000_000 == 0 or written == count:
                elapsed = time.time() - t0
                print(f"  {written:,}/{count:,} ({100*written/count:.0f}%) — "
                      f"{written/elapsed/1e6:.1f}M vec/s")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {os.path.getsize(output_path)/(1024**3):.1f} GB")


def extract_from_fvecs(input_path, output_path, count, dtype):
    """Extract from fvecs format and convert to DiskANN bin format."""
    np_dtype, elem_size = DTYPE_MAP[dtype]
    npts, dim, vec_size = read_fvecs_header(input_path)

    print(f"Source (fvecs): {input_path}")
    print(f"  npts={npts:,}, dim={dim}, vec_size={vec_size}")

    count = min(count, npts)
    out_point_size = dim * elem_size
    total_bytes = 8 + count * out_point_size

    print(f"Output (bin): {output_path}")
    print(f"  count={count:,}, dtype={dtype}, size={total_bytes / (1024**3):.1f} GB")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    t0 = time.time()
    chunk = 100_000

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        fout.write(struct.pack('II', count, dim))

        written = 0
        while written < count:
            batch = min(chunk, count - written)
            raw = fin.read(batch * vec_size)
            if len(raw) < batch * vec_size:
                raise RuntimeError(f"Unexpected EOF at {written:,}")

            # Parse fvecs: strip 4-byte dim prefix from each vector
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(batch, vec_size)
            # Skip first 4 bytes (dim prefix), take dim*4 bytes of float data
            data = arr[:, 4:].tobytes()
            fout.write(data)

            written += batch
            if written % 10_000_000 == 0 or written == count:
                elapsed = time.time() - t0
                print(f"  {written:,}/{count:,} ({100*written/count:.0f}%) — "
                      f"{written/elapsed/1e6:.1f}M vec/s")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {os.path.getsize(output_path)/(1024**3):.1f} GB")


def convert_fvecs_query(input_path, output_path, dtype):
    """Convert fvecs query file to DiskANN bin format (small file, load all)."""
    np_dtype, elem_size = DTYPE_MAP[dtype]
    npts, dim, vec_size = read_fvecs_header(input_path)

    print(f"Converting query: {input_path} → {output_path}")
    print(f"  npts={npts:,}, dim={dim}")

    with open(input_path, 'rb') as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(npts, vec_size)
    data = arr[:, 4:].copy()
    vectors = np.frombuffer(data.tobytes(), dtype=np.float32).reshape(npts, dim)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('II', npts, dim))
        f.write(vectors.tobytes())
    print(f"  Done ({os.path.getsize(output_path)/(1024**2):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Extract subset + convert to DiskANN bin")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--dtype', default='float32', choices=['float32', 'uint8', 'int8'])
    parser.add_argument('--format', default='bin', choices=['bin', 'fvecs'],
                        help="Input format: bin (DiskANN) or fvecs")
    parser.add_argument('--query-input', help="Source query file")
    parser.add_argument('--query-output', help="Output query file")
    parser.add_argument('--query-format', default=None, choices=['bin', 'fvecs'],
                        help="Query format (defaults to --format)")
    args = parser.parse_args()

    if args.format == 'fvecs':
        extract_from_fvecs(args.input, args.output, args.count, args.dtype)
    else:
        extract_from_bin(args.input, args.output, args.count, args.dtype)

    if args.query_input and args.query_output:
        qfmt = args.query_format or args.format
        if qfmt == 'fvecs':
            convert_fvecs_query(args.query_input, args.query_output, args.dtype)
        else:
            import shutil
            print(f"\nCopying query: {args.query_input} → {args.query_output}")
            os.makedirs(os.path.dirname(args.query_output) or '.', exist_ok=True)
            shutil.copy2(args.query_input, args.query_output)
            print(f"  Done ({os.path.getsize(args.query_output)/(1024**2):.1f} MB)")


if __name__ == '__main__':
    main()
