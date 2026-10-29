#!/usr/bin/env python
"""런 직전 colocation 사전 점검 (2026-08-02, 사용자 상시 규칙).

왜 필요한가 — 08-02 실측 사고:
  200M/8p colocation 디렉토리에 **인덱스만** 있고 실페이로드 `partition<N>.coloc` 이 없었다.
  `du` 로는 20-23G 가 잡혀서(= local_to_tag/tag_to_local 인덱스) "준비됨" 으로 오판했고,
  8 노드 런이 RDMA 연결까지 다 끝난 뒤 노드 7 에서야
  `FAILED: Local co-location payload for node 7 (partition 6) not found` 로 죽었다.
  ⇒ **디렉토리 존재나 크기로 판단하면 안 된다. 노드별 `partition<p>.coloc` 을 직접 봐야 한다.**

사용:
  check_colocation.py <config.yaml> [--quiet]
  종료코드 0 = 전 노드 OK, 1 = 누락 있음(런 금지), 2 = config 파싱 실패
"""
import argparse, os, re, subprocess, sys
from pathlib import Path

import yaml

EXP = Path(os.environ.get("DANNS_EXPERIMENT_ROOT", Path(__file__).resolve().parents[1]))
# 컨테이너 /workspace/colocation_local 의 호스트 실체 (docker inspect danns 로 확인)
HOST_COLOC_ROOT = os.environ.get("WORKER_LOCAL_SSD", "/workspace/colocation_local")


def load(p: Path):
    with open(p) as f:
        return yaml.safe_load(f)


def nodes_of(cfg, cfg_path: Path):
    nf = cfg.get("cluster", {}).get("nodes_file")
    if not nf:
        return []
    np_ = (cfg_path.parent / nf).resolve()
    nodes = load(np_).get("nodes", [])
    out = []
    for n in nodes:
        h = str(n.get("host", ""))
        # nodes.yaml 은 RDMA IP(100.0.0.X) 로 적혀 있다 → swiftX 로 되돌린다
        m = re.match(r"100\.\d+\.\d+\.(\d+)$", h)
        out.append((int(n["id"]), f"worker{m.group(1)}" if m else h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cp = Path(a.config).resolve()
    try:
        cfg = load(cp)
        exp_name = cfg["experiment_name"]
        npart = int(cfg["partition"]["num_partitions"])
        strat = cfg["partition"]["insert_strategy"]
        # 🔴 graph_variant 는 `experiment` 가 아니라 **`build`** 아래에 있다(실측).
        gvar = cfg["build"]["graph_variant"]
    except Exception as e:                                  # noqa: BLE001
        print(f"🔴 config 파싱 실패: {e}", file=sys.stderr)
        return 2

    variant = f"{npart}_partitions_{strat}"
    rel = f"{exp_name}/colocation/{variant}/{gvar}"
    nodes = nodes_of(cfg, cp)
    if not nodes:
        print("🔴 nodes_file 을 못 읽었다", file=sys.stderr)
        return 2
    if len(nodes) != npart:
        print(f"🔴 노드 수({len(nodes)}) != 파티션 수({npart})", file=sys.stderr)
        return 1

    if not a.quiet:
        print(f"  coloc 점검: {rel}  (노드 {len(nodes)})")

    bad = 0
    # 엔진 규약(resolve_node_partition_map)과 동일한 dense-rank: 노드 id 정렬
    # 순서가 파티션 인덱스다. nid-1 하드코딩은 노드셋이 1로 시작하지 않으면
    # (예: worker hosts 4p) 틀린다 — 2026-08-23 frozen 시딩 소실 사고의 동족 버그.
    rank = {nid: i for i, (nid, _) in enumerate(sorted(nodes))}
    for nid, host in nodes:
        p = rank[nid]
        d = f"{HOST_COLOC_ROOT}/{rel}"
        # 자기 파티션 페이로드 + 공용 인덱스가 **둘 다** 있어야 한다
        cmd = (f'f={d}/partition{p}.coloc; '
               f'if [ -s "$f" ]; then s=$(stat -c %s "$f"); else s=0; fi; '
               f'm=$([ -s {d}/colocation_meta.json ] && echo 1 || echo 0); '
               f'echo "$s $m"')
        r = subprocess.run(["ssh", "-n", "-T", "-o", "BatchMode=yes",
                            "-o", "ConnectTimeout=10", host, cmd],
                           capture_output=True, text=True, timeout=60).stdout.split()
        sz, meta = (int(r[0]), r[1]) if len(r) == 2 else (0, "0")
        ok = sz > 0 and meta == "1"
        if not ok:
            bad += 1
        if not a.quiet or not ok:
            why = "" if ok else ("  ← partition%d.coloc 없음/빈파일" % p if sz == 0
                                 else "  ← colocation_meta.json 없음")
            print(f"    {'✅' if ok else '🔴'} {host} (partition {p}) "
                  f"{sz/2**30:8.2f} GB{why}")

    if bad:
        print(f"🔴 colocation 미비 {bad}/{len(nodes)} 노드 — **런 금지**. "
              f"coloc 빌드부터 (distributed_colocation_build.py --variant {gvar} --execute)")
        return 1
    if not a.quiet:
        print("  ✅ 전 노드 colocation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
