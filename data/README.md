# Datasets

The artifact does **not** ship dataset bytes. This directory contains the
scripts to download the four datasets used by the paper from their original
public sources and to slice the **first 300M vectors** out of each one as a
DiskANN binary file ready for the experiments.

## Datasets

The paper labels the workloads `<ds>100M` (e.g. **deep100M** is the headline
dataset). The "300M" file label below refers to the *backing* slice on disk:
each experiment maintains a live 100M-vector graph and consumes a 200M-vector
insert pool over 200 update iterations (1M inserts + 1M deletes per iteration
= 200 % cumulative turnover). All 300M vectors come from the dataset's
**1B** public release; we just take the first 300M.

| Workload  | Backing file | Public source                         | dtype   | dim | File size    |
|-----------|--------------|---------------------------------------|---------|-----|--------------|
| sift100M  | sift300M.bin | BIGANN / TEXMEX (IRISA, FTP)          | uint8   | 128 | ~38 GB bin   |
| deep100M  | deep300M.fbin| DEEP1B (Yandex Disk)                  | float32 |  96 | ~115 GB fbin |
| spacev100M| spacev300M.i8bin | Hugging Face (BAAI/SPACEV) + MSTuring | int8    | 100 | ~30 GB i8bin |
| laion100M | laion300M.fbin   | LAION-400M CLIP ViT-B/32 (deploy.laion.ai) | float32 | 512 | ~570 GB fbin |

For the headline figure the artifact only needs **deep100M** (i.e.
`deep1B` slice). The other three are listed for completeness and are
exercised by the same scripts if you wish to extend the experiment.

## Workflow

Set two environment variables before running anything:

```bash
export RAW_DIR=/path/to/raw_downloads        # large; tolerates ~1 TB across all 4
export DATASETS_ROOT=/path/to/artifact_data  # final datasets live here
```

Then download + extract one dataset at a time:

```bash
bash data/download_and_extract.sh sift1B
bash data/download_and_extract.sh deep1B
bash data/download_and_extract.sh spacev1B
bash data/download_and_extract.sh laion
```

Each invocation is idempotent — partial downloads resume on retry, and
already-extracted output files are detected and skipped.

The orchestrator places outputs at:

```
$DATASETS_ROOT/SIFT1B/sift300M_{base,query}.bin
$DATASETS_ROOT/DEEP1B/deep300M_{base,query}.fbin
$DATASETS_ROOT/SPACEV1B/spacev300M_{base,query}.i8bin
$DATASETS_ROOT/LAION400M/laion300M_{base,query}.fbin
```

These paths match the `dataset.data_root` + `dataset.base_file` fields used by
the experiment YAMLs in `scripts/configs/vldb/<dataset>/`.

## Disk requirements

Approximate worst case (raw download + extracted subset, both kept):

| Dataset    | Raw download | Extracted subset |
|------------|--------------|------------------|
| sift1B     | ~92 GB (.gz) + ~250 GB (.bvecs) | ~38 GB |
| deep1B     | ~120 GB (12 shards) + 60 GB (merged base.fvecs) | ~115 GB |
| spacev1B   | streamed; subset only | ~30 GB |
| laion      | ~290 GB (npy) | ~570 GB |

If disk is tight, you can delete `$RAW_DIR/<dataset>/` after the subset is
materialized — the artifact runtime only reads from `$DATASETS_ROOT/vldb/...`.

## Per-dataset notes

### sift1B (BIGANN)
- Source FTP: `ftp://ftp.irisa.fr/local/texmex/corpus/`
- Files: `bigann_base.bvecs.gz`, `bigann_query.bvecs.gz`
- Format: bvecs (uint8 vectors, dim=128)
- Decompression is single-threaded; allow >30 min for the base file.
- If the FTP is unreachable, mirrors are listed on the BIGANN dataset page.

### deep1B (Yandex)
- Distributed as 37 raw byte shards (`base/base_NN`, ~10 GB each) at the
  Yandex Disk public link `https://yadi.sk/d/11eDCm7Dsn9GA`.
- The shards are NOT independent fvecs files — they are raw byte chunks of a
  concatenated fvecs stream. A vector can span across two shards. The
  downloader concatenates them with `cat` before extraction.
- We pull 12 shards (~120 GB) which is sufficient to extract 300M vectors.
- The Yandex API occasionally rate-limits; the downloader is idempotent — just
  re-run on failure.

### spacev1B (Hugging Face)
- Streamed via `datasets.load_dataset("BAAI/SPACEV", split="base", streaming=True)`.
- Requires `pip install datasets`.
- **Queries are NOT in the BAAI/SPACEV HF dataset.** They are a separate file
  shipped by Microsoft's MS-Turing SPACEV release. Place the file at
  `$RAW_DIR/spacev1B/spacev1B_query.bin` (DiskANN bin format) before running
  the extractor, or skip the query step and use a fallback query set.

### laion (LAION-400M)
- Source: `https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_<NNN>.npy`
- 410 numpy files, ~1M float16 vectors per file, dim=512.
- Distribution does NOT include a separate query set. The downloader
  splits off the last 10K vectors of the 300M to use as queries
  (override with `--query-count`).
- The extractor converts float16 → float32 (DiskANN's bin format requires
  fixed-width payloads).

## Extending to other subset sizes

`data/extract_subset.py` is generic. To extract a different subset (e.g. 100M
or 500M) once the 1B raw is on disk:

```bash
python data/extract_subset.py \
    --input  $RAW_DIR/sift1B/bigann_base.bvecs \
    --output $DATASETS_ROOT/vldb/sift500M/sift500M_base.bin \
    --count  500M \
    --dtype  uint8 --format bvecs
```
