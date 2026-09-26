# Model sources

AV-SyncBench does not redistribute third-party model weights. Download the
weights from the original authors and review their licenses before use.

## Implemented evaluation adapters

| Model | Upstream source | Version used by AV-SyncBench | Training / upstream test data | Official checkpoint | Upstream reference metric |
|---|---|---|---|---|---|
| Synchformer | [GitHub](https://github.com/v-iashin/Synchformer) · [project](https://www.robots.ox.ac.uk/~vgg/research/synchformer/) · [paper](https://arxiv.org/abs/2401.16423) | Commit [`b66668a`](https://github.com/v-iashin/Synchformer/commit/b66668a1521d7567cc760e5544b2b5b53179b687), model ID `24-01-02T10-00-53` | VGGSound / VGGSound-Sparse | [config](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-02T10-00-53/cfg-24-01-02T10-00-53.yaml) · [checkpoint](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-02T10-00-53/24-01-02T10-00-53.pt) | Acc@1 `43.8`; Acc@1±1 class `60.2` on VGGSound-Sparse |
| ImageBind-Huge | [GitHub](https://github.com/facebookresearch/ImageBind) · [project](https://facebookresearch.github.io/ImageBind/) · [paper](https://arxiv.org/abs/2305.05665) | Commit [`53680b0`](https://github.com/facebookresearch/ImageBind/commit/53680b02d7e37b19b124fa37bae4b6c98c38f5be) | ImageBind pretraining mixture | [checkpoint](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) | See the upstream model card; AV-SyncBench uses pairwise A/V cosine, not an upstream leaderboard metric |

Synchformer file verification:

| File | Bytes | MD5 |
|---|---:|---|
| `cfg-24-01-02T10-00-53.yaml` | 6,878 | `606ca81ca850a220009fde67233f89a4` |
| `24-01-02T10-00-53.pt` | 1,131,154,181 | `19592ed5e64e3560b93772c5d46e3334` |

The upstream ImageBind checkpoint is 4,803,584,173 bytes. The authors do not
publish a trustworthy SHA-256/MD5 for it, so `scripts/download_models.py`
verifies its byte size and records locally computed MD5 and SHA-256 values in
each run file. The adapter requires `torch.load(..., weights_only=True)` and
refuses an older PyTorch version that cannot provide that safer loading mode.
Still use only the fixed official URL above.

The ImageBind adapter exposes two named preprocessing modes. `paper_legacy`
ports the audited author-side AV-SyncBench transforms and scoring used for
paper-table comparison, while substituting Decord for the legacy
torchvision/PyAV video decoder. `official_0p64` applies an explicitly different
official-style 0.64-second adaptation for ablation. Never merge results from
the two modes without labeling them.

## Reference-only baseline

SparseSync is another offset-classification baseline reported in the paper. A
public adapter may be added later; its official source is listed here so users
do not have to infer the checkpoint from the paper.

| Model | Upstream source | Version / checkpoint | Official checkpoint | Upstream reference metric |
|---|---|---|---|---|
| SparseSync | [GitHub](https://github.com/v-iashin/SparseSync) · [project](https://v-iashin.github.io/SparseSync/) · [paper](https://arxiv.org/abs/2210.07055) | Commit [`a5bee8a`](https://github.com/v-iashin/SparseSync/commit/a5bee8a047c0ebeec66b18d4ddf4c5f0ef098a4f), model ID `22-07-28T15-49-45` (VGGSound-Sparse) | [config](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/22-07-28T15-49-45/cfg-22-07-28T15-49-45.yaml) · [checkpoint](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/22-07-28T15-49-45/22-07-28T15-49-45.pt) | 21-class Acc@1 `44.3` |

SparseSync checkpoint verification: 653,037,985 bytes; MD5
`a26f2079cb93eacd9eafdd1b78227824`.

## What the AV-SyncBench numbers mean

The metrics in the upstream model tables and the AV-SyncBench paper are not the
same metric:

- Upstream Synchformer/SparseSync tables report offset-class prediction
  accuracy.
- AV-SyncBench uses the probability assigned to the zero-offset class as a
  continuous synchronization score, then asks whether the original pair scores
  above its controlled negative.
- ImageBind produces embeddings. AV-SyncBench averages the diagonal audio/video
  cosine similarity over fixed, non-overlapping 0.64-second segments.

Do not compare the values in the last column above directly with AV-SyncBench
pairwise accuracy.

The Synchformer adapter fixes the benchmark-level outer-window protocol,
checkpoint identity, score definition, and aggregation so external runs can
be compared with all five paper strata.
