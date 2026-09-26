# Reproduction troubleshooting

## First compare the protocol card

When reporting a discrepancy, attach the generated `*.run.json`,
`*.summary.json`, and the exact command. At minimum compare:

1. checkpoint ID and local MD5;
2. upstream repository commit;
3. Python, PyTorch, torchvision, torchaudio, Decord, PyAV (Synchformer), and
   FFmpeg versions;
4. input FPS, audio sample rate, resize/crop, and duration;
5. offset magnitudes and directions;
6. crop/window starts and aggregation;
7. score definition and tie policy;
8. per-offset counts, errors, and accuracy.

Run cards contain sanitized basenames rather than absolute host paths. Include
your exact command separately after removing any private directories.

## No records or missing files after extraction

Set `--dataset-root` to the directory containing the discovered
`metadata.jsonl`, `original_video/`, and `edited_audio/` entries. For the
temporal archive this is the `hf_temporal_challenge/` directory, not the outer
extraction directory. The validator treats a zero-record filter as an error.

## Synchformer: the most common mismatches

### PyAV newer than 9.0.0

The upstream README explicitly warns that `av>=9.1.1` changes the evaluation
result. Upstream [Issue #11](https://github.com/v-iashin/Synchformer/issues/11)
reports a large Acc@1 drop after the version change. Reproduce in the exact
official environment with `av==9.0.0` before comparing numbers; the adapter
enforces that version.

### Wrong checkpoint

The AV-SyncBench paper uses the VGGSound model `24-01-02T10-00-53`, not the
AudioSet model `24-01-04T16-39-21`. Check the generated config/checkpoint MD5.

### Wrong metric

AV-SyncBench does **not** ask whether the predicted class equals the known
50/100/200/300/500 ms perturbation. It computes:

```text
score(pair) = softmax(offset_logits)[10]  # class 10 is delta = 0
correct = score(original) > score(negative)
```

The strict comparison means a tie is incorrect.

### Applying the offset twice

Released negatives are already shifted. Both positive and negative samples are
passed to the upstream transform with `offset_sec=0.0`. Passing the manifest
offset into Synchformer again creates a second shift and invalidates the pair.

### Random test-time crop or offset

The upstream transform becomes random if **both** `v_start_i_sec` and
`offset_sec` are absent. This adapter always provides both values. Compare the
recorded window starts when two runs differ.

### Treating 0.64-second segments as independent predictions

The VGGSound checkpoint takes one 5-second crop, forms fourteen 0.64-second
segments at 0.32-second stride internally, and emits one 21-class prediction.
It does not emit fourteen independent probabilities to average.

### Inconsistent positive/negative windows

The positive and negative members must use identical video frames and identical
window starts. This implementation derives one start list from their common
duration and records it in the JSONL output.

### Different outer-window aggregation

Synchformer itself defines one 5-second prediction, but not how to aggregate a
3--13 second benchmark clip. The release default mirrors the repository's
classifier convention: 5-second windows every 0.64 seconds, tail padding, then
mean probability. A first/center-only crop or mean-logit aggregation is a
different protocol. Compare `window_policy`, `window_starts`,
`round_duration_to`, `tail_padding`, and `window_aggregation` in `*.run.json`.

### Non-canonical video

The strict adapter rejects video that is not 25 FPS with short side 256. Run
`scripts/prepare_media.py` and evaluate its rewritten manifest; do not bypass
the check when comparing with the Synchformer protocol.


## ImageBind: the most common mismatches

- Use the pinned PyTorch `2.10.0` / torchvision `0.25.0` / torchaudio `2.10.0`
  / NumPy `1.26.4` / Decord `0.6.0` environment. Do not install the pinned
  ImageBind repository's unconstrained requirements on top of it. The adapter
  uses Decord directly and rejects a different core environment.
- Confirm `preprocess_mode` first. `paper_legacy` ports the audited author-side
  transforms and scoring, including FFmpeg mono/16 kHz signed-16 PCM audio
  decoding. Its public runtime uses Decord in place of the legacy
  torchvision/PyAV video decoder. `official_0p64` is an explicit, different
  official-style adaptation and is not expected to reproduce the same number.
- In `official_0p64`, the upstream video loader cannot simply be called with
  `clip_duration=0.64`: it is designed around 2-second clips and uses the clip
  duration as the number of sampled frames. Passing `0.64` directly is not a
  valid adaptation. This adapter selects a fixed two frames from each 0.64 s
  window, then applies the official 224-pixel transform and three spatial crops.
- Both modes keep a `128 x 204` audio feature. `paper_legacy` uses the
  author-side log-Mel implementation; `official_0p64` uses the upstream Kaldi
  fbank and mean/std. Both use one benchmark-level FFmpeg signed-16 PCM decode,
  with a bounded positive-audio cache. Do not mix these modes within one
  comparison.
- Normalize both output embeddings again before cosine scoring. ImageBind's
  audio postprocessor includes a fixed scale; a raw dot product is not cosine.
- Do not softmax similarities across a batch: that makes each score depend on
  which unrelated samples happen to share the batch.

## Checkpoint trust

PyTorch checkpoints are serialized objects. Use only the official download
links. Synchformer size and MD5 are enforced before loading. ImageBind upstream
publishes a byte size but no trustworthy digest; the downloader uses the fixed
official HTTPS URL and the run card records locally computed MD5 and SHA-256.
The ImageBind adapter additionally requires PyTorch 2.10.0, the first release
patched for the `weights_only=True` checkpoint vulnerability documented in
[GHSA-63cw-57p8-fm3p](https://github.com/advisories/GHSA-63cw-57p8-fm3p).
