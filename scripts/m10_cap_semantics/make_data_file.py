#!/usr/bin/env python3
"""
make_data_file.py — data-less checkpoint → DiskANN 표준 .data 파일 생성기 (2026-08-07).

왜: load_graph_only는 base 풀에서 tag 순서로 벡터를 재구성하며 temp 버퍼를 쓴다
(ctor _data + temp = 2× 피크 — laion100M은 ~410GB라 모든 노드 OOM). .data를 미리
만들어 두면 표준 Index::load() 경로(copy_aligned_data_from_file, chunk 복사·이중버퍼
없음·순차 I/O)를 쓸 수 있어 피크 ≈ 정상상태(~205GB)로 떨어진다. 반복 실험용 인프라.

포맷: [u32 npts = max_points+1][u32 dim][T row-major, loc 순서]
- tags[loc]==0 (frozen 슬롯/삭제) → 0 벡터
- loc == max_points (frozen 슬롯): <graph>.frozen 사이드카에서 벡터 스플라이스

메모리: 출력 배열 전체를 RAM에 유지 (npts*dim*sizeof — laion100M ≈ 205GB).
base 풀은 [gmin..gmax]를 슬랩 단위 순차 read (active gid 윈도우가 연속이라 낭비 적음).
⚠️ 실행 노드 가용 RAM > 출력 크기 + 슬랩(2GB) 필요. laion100M은 worker-host(377G)에서.

사용례 (호스트/컨테이너 어디서든; 경로만 일치시키면 됨):
  python3 make_data_file.py \
    --graph /experiments/artifacts/laion100M/graph/checkpoints/lits_k4_iter150 \
    --base  /workspace/datasets_nfs/LAION400M/laion400M_base.fbin \
    --dtype float32 --dim 512 --max-points 100000000
출력: <graph>.data (기본; --out으로 변경)
"""
import argparse
import os
import sys

import numpy as np

_DT = {"float32": np.float32, "uint8": np.uint8, "int8": np.int8}


def load_tags(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.uint32, count=2)
        npts = int(hdr[0])
        tags = np.fromfile(f, dtype=np.uint32, count=npts)
    if tags.shape[0] != npts:
        sys.exit(f"short tags read: {tags.shape[0]} != {npts}")
    return tags


def load_frozen(path, dim, npdt):
    """<graph>.frozen: [u32 count=1][u32 dim][T vec] — bindings와 동일 포맷."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.uint32, count=2)
        if hdr.shape[0] != 2 or int(hdr[0]) != 1 or int(hdr[1]) != dim:
            print(f"  [warn] frozen header mismatch {hdr} — frozen splice skip", flush=True)
            return None
        vec = np.fromfile(f, dtype=npdt, count=dim)
    return vec if vec.shape[0] == dim else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="graph checkpoint path (tags/frozen는 여기서 유도)")
    ap.add_argument("--base", required=True, help="base pool fbin ([u32 npts][u32 dim][T...])")
    ap.add_argument("--dtype", required=True, choices=list(_DT))
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--max-points", type=int, required=True)
    ap.add_argument("--out", default=None, help="default: <graph>.data")
    ap.add_argument("--slab-rows", type=int, default=1_000_000)
    args = ap.parse_args()

    npdt = _DT[args.dtype]
    dim = args.dim
    out_path = args.out or (args.graph + ".data")
    npts_out = args.max_points + 1  # + frozen slot

    tags = load_tags(args.graph + ".tags")
    if tags.shape[0] != npts_out:
        print(f"  [warn] tags npts {tags.shape[0]} != max_points+1 {npts_out}", flush=True)

    with open(args.base, "rb") as f:
        bh = np.fromfile(f, dtype=np.uint32, count=2)
    base_npts, base_dim = int(bh[0]), int(bh[1])
    if base_dim != dim:
        sys.exit(f"base dim {base_dim} != --dim {dim}")

    locs = np.nonzero(tags[: args.max_points])[0].astype(np.int64)
    gids = tags[locs].astype(np.int64) - 1  # tag = gid+1
    if gids.max() >= base_npts:
        sys.exit(f"tag gid {gids.max()} >= base npts {base_npts}")
    gmin, gmax = int(gids.min()), int(gids.max())
    print(f"active={locs.shape[0]:,} gid window=[{gmin:,}..{gmax:,}] span={gmax-gmin+1:,}", flush=True)

    order = np.argsort(gids, kind="stable")
    gids_s, locs_s = gids[order], locs[order]

    out = np.zeros((npts_out, dim), dtype=npdt)
    row_sz = dim * np.dtype(npdt).itemsize
    hdr_sz = 8
    done = 0
    with open(args.base, "rb") as f:
        for slab_start in range(gmin, gmax + 1, args.slab_rows):
            slab_end = min(slab_start + args.slab_rows, gmax + 1)
            lo = np.searchsorted(gids_s, slab_start, side="left")
            hi = np.searchsorted(gids_s, slab_end, side="left")
            if lo == hi:
                continue  # 이 슬랩에 active gid 없음 — read도 skip
            f.seek(hdr_sz + slab_start * row_sz)
            slab = np.fromfile(f, dtype=npdt, count=(slab_end - slab_start) * dim)
            slab = slab.reshape(slab_end - slab_start, dim)
            out[locs_s[lo:hi]] = slab[gids_s[lo:hi] - slab_start]
            done += hi - lo
            if done % 10_000_000 < (hi - lo):
                print(f"  {done:,}/{locs.shape[0]:,} rows", flush=True)

    if done != locs.shape[0]:
        sys.exit(f"filled {done} != active {locs.shape[0]}")

    fv = load_frozen(args.graph + ".frozen", dim, npdt)
    if fv is not None:
        out[args.max_points] = fv
        print("  frozen vector spliced", flush=True)

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        np.array([npts_out, dim], dtype=np.uint32).tofile(f)
        out.tofile(f)
    os.rename(tmp, out_path)
    print(f"OK: {out_path} ({(hdr_sz + npts_out * row_sz) / 2**30:.1f} GiB)", flush=True)


if __name__ == "__main__":
    main()
