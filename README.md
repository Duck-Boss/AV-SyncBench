# AV-SyncBench

Reproducible evaluation for **AV-SyncBench: Decoupled Benchmarking of Temporal
and Semantic Audio-Visual Synchronization** (Interspeech 2026).

AV-SyncBench asks a controlled pairwise question: does a model score the
original audio-video pair above a version in which only timing or source timbre
has been changed? This repository makes the data path, preprocessing, scoring,
aggregation, environment, and error accounting explicit.

- 3,269 in-the-wild videos and 38,390 evaluation samples
- 10 scenarios across voice, music, and environmental sound
- temporal challenges: global offset, local jitter, and global speed
- semantic challenges: voice and instrument timbre replacement
- no model training or fine-tuning

Paper: [Interspeech 2026](https://www.isca-archive.org/interspeech_2026/zhou26g_interspeech.html)
· [arXiv](https://arxiv.org/abs/2607.00726) · Dataset:
[ModelScope](https://modelscope.cn/datasets/coming245/AVSyncBench) · Project:
[website](https://fgt7t6g.github.io/AV-SyncBench/)

## What is included

- an end-to-end Global Offset tutorial, from dataset download to result files;
- two synthetic, redistributable end-to-end model demos;
- a complete Synchformer VGGSound classifier adapter using `p(delta=0)` and
  the author-side outer-window aggregation;
- a complete ImageBind-Huge adapter with the author-side paper transforms and
  scoring (`paper_legacy`, using a documented Decord decoder substitution) and
  an explicitly named official-style 0.64-second adaptation
  (`official_0p64`);
- verified official model links and checkpoint metadata;
- a pinned CUDA 12.6 / PyTorch 2.10 ImageBind environment, with Decord video
  decoding and no dependency on the retired torchvision video reader;
- manifest validation, strict pairwise metrics, stratified summaries, and a
  machine-readable run card for reproducibility.

The model source and weight table is in [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md).

## Evaluation in one line

For every original/negative pair:

```text
correct = score(original video, original audio)
          > score(original video, perturbed audio)
```

Ties are incorrect. Synchformer uses the probability of its zero-offset class;
ImageBind uses mean segment-wise cosine similarity. These are different score
functions under the same pairwise decision rule.

## Two model demos

Prerequisites: Linux, Git, FFmpeg, and the corresponding model environment
described in the tutorial. Each script clones the pinned upstream source when
missing, creates a synthetic pair, downloads official weights, validates the
manifest, and runs one evaluation:

```bash
python -m pip install -e .

# Demo 1: Synchformer VGGSound classifier
bash examples/run_synchformer_demo.sh

# Demo 2: ImageBind-Huge feature similarity
bash examples/run_imagebind_demo.sh
```

Set `DEVICE`, `SYNCHFORMER_ROOT`, `IMAGEBIND_ROOT`, or `CHECKPOINT_ROOT` to
override the defaults. Each demo writes a per-pair JSONL, a stratified
`*.summary.json`, and a sanitized `*.run.json`.

The synthetic pair only checks that the pipeline runs; it is not a benchmark
sample and has no expected scientific accuracy.

## Full reproduction

Follow [docs/REPRODUCE_GLOBAL_OFFSET.md](docs/REPRODUCE_GLOBAL_OFFSET.md). The
guide covers:

1. the 68.19 GB, seven-part ModelScope download;
2. pinned upstream repositories and model weights;
3. manifest validation and required Synchformer media canonicalization;
4. Synchformer and ImageBind commands;
5. expected paper numbers and the exact comparison protocol;
6. information to attach when reporting a discrepancy.

For format details see [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md); for common
replication failures see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md); for
the relationship between author-side and upstream preprocessing paths, see
[docs/PROTOCOL_PROVENANCE.md](docs/PROTOCOL_PROVENANCE.md).

## Paper reference results

Pairwise accuracy reported by the AV-SyncBench paper:

| Model | 50 ms | 100 ms | 200 ms | 300 ms | 500 ms | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Synchformer (VGGSound) | 0.510 | 0.541 | 0.582 | 0.622 | 0.662 | **0.583** |
| SparseSync (VGGSound-Sparse) | 0.518 | 0.514 | 0.561 | 0.602 | 0.648 | **0.569** |

These are AV-SyncBench pairwise accuracies, not the upstream models'
offset-classification accuracies. See the tutorial for the exact scoring and
aggregation protocol.

## Citation

Please cite AV-SyncBench and each evaluated upstream model:

```bibtex
@inproceedings{zhou26g_interspeech,
  title     = {{AV-SyncBench: Decoupled Benchmarking of Temporal and Semantic Audio-Visual Synchronization}},
  author    = {Tianhong Zhou and Mingyang Han and Boyu Li and Yuxuan Jiang and Jiaxin Ye and Dongxiao Wang and Haoxiang Shi and Kunpeng Wang and Jun Song and Cheng Yu and Bo Zheng},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {1137--1141},
  doi       = {10.21437/Interspeech.2026-2177},
  issn      = {2958-1796}
}
```

Project and release updates are posted on the
[AV-SyncBench website](https://fgt7t6g.github.io/AV-SyncBench/). Enable the
repository issue tracker before directing public replication reports there.

## Licenses and media rights

The adapters in this repository are released separately from upstream models
and weights. Synchformer code is MIT; ImageBind's upstream `LICENSE` is CC
BY-NC-SA 4.0. Weight licensing may differ from code licensing. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset media may retain
rights belonging to original creators. The current repository does not declare
a separate license for the new AV-SyncBench adapter code; maintainers should
add the approved project license before granting redistribution rights.
