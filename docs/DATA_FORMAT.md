# Manifest format

Evaluation input is JSON Lines: one original-versus-perturbed comparison per
line. All paths must be relative to `--dataset-root`; absolute paths and paths
that escape the root are rejected by default so a shared manifest cannot read
arbitrary host files.

```json
{
  "sample_id": "example-delay-050",
  "video_id": "example",
  "original_video": "video/example.mp4",
  "original_audio": "audio/original/example.wav",
  "edited_audio": "audio/global_offset/example_delay_050.wav",
  "task_type": "global_offset",
  "target_attribute": "50ms delay",
  "offset_ms": 50,
  "direction": "delay",
  "category": "single_instrument",
  "duration": 5.0
}
```

Required fields:

- `sample_id`: stable unique identifier.
- `original_video`: visual stream used for both members of the pair.
- Exactly one of `edited_audio` / `negative_audio` or `edited_video` /
  `negative_video`. The two evaluators in this release implement audio
  perturbations and therefore require `edited_audio` / `negative_audio`;
  video-negative records are accepted by the generic parser for other tasks
  but rejected by these evaluators.

Recommended fields:

- `original_audio`: a mono 16 kHz PCM WAV. If omitted, the audio track embedded
  in `original_video` is used.
- `task_type`, `offset_ms`, `direction`, `category`: used for stratified summary
  metrics.

Legacy aliases accepted by the parser are documented in
`avsyncbench/manifest.py`. New manifests should use the canonical names above.
The validator and canonicalization tool expose `--allow-absolute-paths` only
for explicit maintainer conversion of legacy manifests; model evaluators keep
absolute paths disabled.

## Result files

Each evaluator writes three files:

1. `*.jsonl`: one record per pair, including the two scores, strict pairwise
   decision, per-window/per-segment scores, ties, and errors.
2. `*.summary.json`: overall, per-offset, per-category, and per-direction counts
   and accuracy.
3. `*.run.json`: sanitized file names, model/adapter commits,
   checkpoint/config/manifest hashes, dependency and FFmpeg versions,
   preprocessing, filtering, aggregation, and tie policy.

Per-record error messages replace the supplied dataset and upstream checkout
roots with labels such as `<dataset_root>` before they are serialized. Review a
JSONL excerpt for any other application-specific metadata before sharing it.

`accuracy` uses valid records only. `accuracy_all_records` counts decode or
inference failures as incorrect. When `--continue-on-error` is used, outputs
are still written but the evaluator exits non-zero if any failure occurred;
release comparisons require zero failures.
