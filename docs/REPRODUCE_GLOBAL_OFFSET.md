# Reproduce the Global Offset evaluation

This guide starts from a prepared Linux machine and ends with per-sample
results, an aggregate summary, and a sanitized run card. Commands assume that
the current directory is the AV-SyncBench repository. Conda (or an equivalent
environment manager), Git, FFmpeg, and an NVIDIA CUDA stack must already be
available.

> **Protocol note.** The Synchformer path below implements the VGGSound
> `p(delta=0)` score and makes the author-side outer-window convention explicit.
> Section 9 gives the paper-reported values for direct comparison.

## 1. Hardware, storage, and this package

- Linux with Git and FFmpeg
- one NVIDIA GPU with a working CUDA PyTorch stack
- about 70 GB for the downloaded split archive
- additional space for extraction and outputs
- roughly 250 GB free if you retain the download, extracted data, and a full
  canonical Synchformer copy at the same time
- Python 3.8 for the pinned Synchformer environment
- Python 3.10 for the upstream ImageBind environment

Install the lightweight AV-SyncBench package in the Python environment used
for manifest checks:

```bash
python -m pip install -e .
```

## 2. Download AV-SyncBench

Install the official ModelScope client, then download and extract:

```bash
python -m pip install modelscope
bash scripts/download_dataset.sh \
  data/downloads/AVSyncBench \
  data/AVSyncBench
```

Source: [coming245/AVSyncBench on ModelScope](https://modelscope.cn/datasets/coming245/AVSyncBench).
The script pins dataset revision
`86c06579529a6e7b2cafb0dc386a50152a37fb98`, verifies the published SHA-256
and byte size of every part, refuses to extract into a non-empty directory,
then streams the seven parts into `tar`. The parts total 68,191,839,983 bytes
(68.19 GB / 63.51 GiB).

Discover the temporal manifest and, critically, use its parent as the dataset
root. The paths inside `metadata.jsonl` are relative to
`hf_temporal_challenge/`, not to the archive's outer directory.

```bash
temporal_manifest="$(find data/AVSyncBench -type f \
  -path '*/hf_temporal_challenge/metadata.jsonl' -print -quit)"
test -n "$temporal_manifest"
temporal_root="$(dirname "$temporal_manifest")"

python scripts/validate_manifest.py \
  --manifest "$temporal_manifest" \
  --dataset-root "$temporal_root" \
  --edit-type time_shifted \
  --expect-count 15000 \
  --require-offset 50 \
  --require-offset 100 \
  --require-offset 200 \
  --require-offset 300 \
  --require-offset 500 \
  --require-direction advance \
  --require-direction delay \
  --require-complete-offsets \
  --require-complete-directions \
  --exact-offset-set \
  --exact-direction-set \
  --expect-per-offset 3000 \
  --expect-per-direction 7500 \
  --expect-per-offset-direction 1500
```

The author-side construction creates 1,500 source videos × five magnitudes ×
two directions = 15,000 Global Offset pairs. The validator prints the actual
counts; do not start a full GPU run if the count, edit type, magnitude, or
direction coverage differs. A zero-record filter is an error.

## 3. Clone exact upstream model versions

### Synchformer

```bash
mkdir -p third_party
git clone https://github.com/v-iashin/Synchformer.git third_party/Synchformer
git -C third_party/Synchformer checkout b66668a1521d7567cc760e5544b2b5b53179b687
conda env create \
  --name avsync-synchformer \
  --file third_party/Synchformer/conda_env.yml
conda activate avsync-synchformer
python -m pip install -e .
python - <<'PY'
import av
print("PyAV", av.__version__)
assert av.__version__ == "9.0.0"
PY
```

The PyAV check is important. The upstream authors document large result changes
with newer PyAV releases. The adapter also enforces the core versions from this
official environment (`torch 2.0.0`, `torchvision 0.15.0`, `torchaudio 2.0.0`,
`av 9.0.0`, and NumPy `1.23.5`) before model loading.

Download the exact VGGSound model used in the AV-SyncBench paper:

```bash
python scripts/download_models.py \
  --model synchformer-vggsound \
  --output-dir checkpoints
```

The downloader and evaluator both verify the official file size and MD5 before
the pickle-based PyTorch checkpoint is loaded:

```text
config MD5:     606ca81ca850a220009fde67233f89a4
checkpoint MD5: 19592ed5e64e3560b93772c5d46e3334
```

Only load checkpoints obtained from the official links in
[MODEL_ZOO.md](MODEL_ZOO.md).

### ImageBind (optional second evaluator)

Use a separate environment because its upstream dependency set differs:

```bash
git clone https://github.com/facebookresearch/ImageBind.git third_party/ImageBind
git -C third_party/ImageBind checkout 53680b02d7e37b19b124fa37bae4b6c98c38f5be
conda create --name avsync-imagebind python=3.10 -y
conda activate avsync-imagebind
conda install -c conda-forge ffmpeg -y
python -m pip install -r requirements/imagebind-cu126.txt
python -m pip install -e .
python scripts/download_models.py \
  --model imagebind-huge \
  --output-dir checkpoints
```

Both modes decode video with Decord 0.6.0 and send positive and negative audio
through the FFmpeg executable as mono 16 kHz signed-16 PCM. PyAV and
`torchvision.io.read_video` are not used. The run card records the actual
versions. ImageBind is subject to its upstream non-commercial share-alike
terms.

The `cu126` file pins the official compatible trio PyTorch `2.10.0`,
torchvision `0.25.0`, and torchaudio `2.10.0`, plus NumPy `1.26.4` and the
small set of packages imported by the pinned ImageBind model. PyTorch `2.10.0`
is required because it fixes
[GHSA-63cw-57p8-fm3p](https://github.com/advisories/GHSA-63cw-57p8-fm3p),
which also affects `torch.load(..., weights_only=True)` in earlier releases.
The version trio and CUDA 12.6 wheel command come from the official
[PyTorch version matrix](https://pytorch.org/get-started/previous-versions/).
Do **not** additionally install `third_party/ImageBind/requirements.txt`: it
allows old affected PyTorch versions and installs the unused legacy media
stack. If CUDA 12.6 is unsuitable for the host, use another official PyTorch
2.10.0 wheel index and keep the same base versions; the adapter checks them
before loading the 4.8 GB checkpoint.

## 4. Exact pairwise score

For every record, both models apply the same strict decision:

```text
correct = score(original video, original audio)
          > score(original video, perturbed audio)
```

Ties are incorrect.

### Synchformer

The VGGSound model uses a 21-class grid:

```text
[-2.0, -1.8, ..., -0.2, 0.0, 0.2, ..., 1.8, 2.0] seconds
```

Zero offset is class index 10:

```text
positive = softmax(model(original pair))[10]
negative = softmax(model(perturbed pair))[10]
```

Do not pass the known perturbation into the model transform. The negative WAV
is already shifted; both members must use `offset_sec=0.0` or the negative will
be shifted twice.

One model input is five seconds. Internally, Synchformer forms fourteen
0.64-second segments at a 0.32-second stride and produces one 21-class output.
Those are not fourteen independent predictions.

The default outer-window policy named `repository_sliding` is:

1. round common media duration up to a multiple of five seconds, minimum five;
2. launch a five-second model window every 0.64 seconds;
3. repeat the last video frame and zero-pad audio at the tail;
4. calculate `p(delta=0)` for each window;
5. average probabilities, not logits.

This follows the author-side classifier aggregation convention. Alternative
`first`, `center`, and valid-only `sliding` policies are available for
discrepancy diagnosis and are always recorded.

### ImageBind

The default `paper_legacy` mode ports the audited author-side transforms and
scoring: both audio conditions are decoded by FFmpeg to mono 16 kHz signed-16
PCM; evaluation uses 0.64-second non-overlapping segments, first/last video
frames warped to 224×224, one spatial crop, custom log MelSpectrogram 128×204
without the official fbank normalization, and at most 18 uniformly selected
segments. The public runtime substitutes Decord for the legacy
torchvision/PyAV video decoder; the run card records that backend explicitly.
The positive-audio LRU cache changes only decoding cost, not scores.

The alternative `official_0p64` mode is a deliberately different,
official-style feature adaptation: fixed two-frame sampling, short-side 224,
three spatial crops, Kaldi fbank 128×204, and official mean/std normalization.
It retains the benchmark's common FFmpeg audio decode rather than claiming to
be the upstream two-second media loader. It is useful for ablation but must not
be presented as the paper-table pipeline.

Both modes re-normalize model embeddings and compute mean diagonal audio/video
cosine. Incomplete final segments are dropped.

## 5. Canonicalize media for Synchformer

Synchformer's official VGGSound path expects 25 FPS, short-side 256 video and
mono 16 kHz audio. Do not treat this as optional when running the strict
Synchformer adapter. Create a separate canonical copy; source files are never
modified:

```bash
conda activate avsync-synchformer
python scripts/prepare_media.py \
  --manifest "$temporal_manifest" \
  --dataset-root "$temporal_root" \
  --edit-type time_shifted \
  --output-root data/AVSyncBench-synchformer

synchformer_manifest="data/AVSyncBench-synchformer/manifest.canonical.jsonl"
synchformer_root="data/AVSyncBench-synchformer"
python scripts/validate_manifest.py \
  --manifest "$synchformer_manifest" \
  --dataset-root "$synchformer_root" \
  --edit-type time_shifted \
  --expect-count 15000 \
  --require-complete-offsets \
  --require-complete-directions \
  --require-offset 50 \
  --require-offset 100 \
  --require-offset 200 \
  --require-offset 300 \
  --require-offset 500 \
  --require-direction advance \
  --require-direction delay \
  --exact-offset-set \
  --exact-direction-set \
  --expect-per-offset 3000 \
  --expect-per-direction 7500 \
  --expect-per-offset-direction 1500
```

`prepare_media.py` deduplicates shared source videos and original audio, writes
safe relative paths, and deliberately emits only public manifest fields.

## 6. Run the synthetic smoke test

The generated five-second flash/beep pair contains no third-party media. Add
`--force` when regenerating existing demo files.

```bash
python scripts/make_demo.py
python scripts/validate_manifest.py \
  --manifest demo/data/global_offset_demo.jsonl \
  --dataset-root demo/data \
  --expect-count 1
```

Synchformer:

```bash
conda activate avsync-synchformer
python -m avsyncbench.evaluators.synchformer_global_offset \
  --manifest demo/data/global_offset_demo.jsonl \
  --dataset-root demo/data \
  --synchformer-root third_party/Synchformer \
  --config checkpoints/synchformer-vggsound/cfg-24-01-02T10-00-53.yaml \
  --checkpoint checkpoints/synchformer-vggsound/24-01-02T10-00-53.pt \
  --device cuda:0 \
  --window-policy repository_sliding \
  --output outputs/demo_synchformer.jsonl
```

ImageBind:

```bash
conda activate avsync-imagebind
python -m avsyncbench.evaluators.imagebind \
  --manifest demo/data/global_offset_demo.jsonl \
  --dataset-root demo/data \
  --imagebind-root third_party/ImageBind \
  --checkpoint checkpoints/imagebind-huge/imagebind_huge.pth \
  --device cuda:0 \
  --preprocess-mode paper_legacy \
  --output outputs/demo_imagebind.jsonl
```

Each command should create a JSONL file plus sibling `.summary.json` and
`.run.json` files. A synthetic score has no scientific target; this test only
proves that model loading, decoding, scoring, and serialization complete.

## 7. Run a ten-pair preflight

Before a full GPU run, test ten records. Synchformer uses the canonical
manifest; ImageBind paper reproduction uses the released media and raw
manifest.

```bash
conda activate avsync-synchformer
python -m avsyncbench.evaluators.synchformer_global_offset \
  --manifest data/AVSyncBench-synchformer/manifest.canonical.jsonl \
  --dataset-root data/AVSyncBench-synchformer \
  --edit-type time_shifted \
  --synchformer-root third_party/Synchformer \
  --config checkpoints/synchformer-vggsound/cfg-24-01-02T10-00-53.yaml \
  --checkpoint checkpoints/synchformer-vggsound/24-01-02T10-00-53.pt \
  --device cuda:0 \
  --limit 10 \
  --output outputs/synchformer_preflight10.jsonl
```

Do not use `--continue-on-error` for release results. If it is used for
diagnosis, the adapter writes all files but exits non-zero when any record
fails; the summary includes both valid-only accuracy and
`accuracy_all_records`, where failures count as incorrect.

## 8. Run the full benchmark

Synchformer protocol:

```bash
conda activate avsync-synchformer
synchformer_manifest="data/AVSyncBench-synchformer/manifest.canonical.jsonl"
synchformer_root="data/AVSyncBench-synchformer"

python -m avsyncbench.evaluators.synchformer_global_offset \
  --manifest "$synchformer_manifest" \
  --dataset-root "$synchformer_root" \
  --edit-type time_shifted \
  --synchformer-root third_party/Synchformer \
  --config checkpoints/synchformer-vggsound/cfg-24-01-02T10-00-53.yaml \
  --checkpoint checkpoints/synchformer-vggsound/24-01-02T10-00-53.pt \
  --device cuda:0 \
  --batch-size 1 \
  --window-policy repository_sliding \
  --window-stride 0.64 \
  --round-duration-to 5.0 \
  --output outputs/synchformer_vggsound_global_offset.jsonl
```

ImageBind paper-generation protocol:

```bash
conda activate avsync-imagebind
temporal_manifest="$(find data/AVSyncBench -type f \
  -path '*/hf_temporal_challenge/metadata.jsonl' -print -quit)"
test -n "$temporal_manifest"
temporal_root="$(dirname "$temporal_manifest")"

python -m avsyncbench.evaluators.imagebind \
  --manifest "$temporal_manifest" \
  --dataset-root "$temporal_root" \
  --edit-type time_shifted \
  --imagebind-root third_party/ImageBind \
  --checkpoint checkpoints/imagebind-huge/imagebind_huge.pth \
  --device cuda:0 \
  --batch-size 4 \
  --segment-seconds 0.64 \
  --preprocess-mode paper_legacy \
  --max-segments 18 \
  --output outputs/imagebind_paper_legacy_global_offset.jsonl
```

For the official-style ablation, change only
`--preprocess-mode official_0p64`; the run card preserves the distinction.

## 9. Inspect and compare outputs

For `outputs/synchformer_vggsound_global_offset.jsonl`, the adapter also writes:

- `outputs/synchformer_vggsound_global_offset.summary.json`
- `outputs/synchformer_vggsound_global_offset.run.json`

The summary contains overall and stratified counts, valid-only accuracy,
all-record accuracy, ties, and failures. The run card records sanitized file
names, model and manifest hashes, upstream and adapter commits, dependency and
decoder versions, preprocessing, score definition, filtering, window policy,
and aggregation. It does not expose the user's absolute directory tree.

Paper reference values for Synchformer VGGSound are:

| Offset | Pairwise accuracy |
|---:|---:|
| 50 ms | 0.510 |
| 100 ms | 0.541 |
| 200 ms | 0.582 |
| 300 ms | 0.622 |
| 500 ms | 0.662 |
| Overall | **0.583** |

A directly comparable run contains all 15,000 records, zero errors, both
directions, and all five magnitudes. If numbers differ, follow
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) and attach the sanitized run card,
summary, exact command, and a short JSONL excerpt to the issue or email.

## 10. Custom media

For custom files, run `scripts/prepare_media.py` exactly as in Section 5. It
creates a new 25 FPS / short-side-256 H.264 video, separate mono 16 kHz PCM WAVs
for both audio conditions, and a rewritten relative-path manifest. It never
modifies the input dataset.
