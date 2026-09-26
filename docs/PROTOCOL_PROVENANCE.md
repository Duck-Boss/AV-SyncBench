# Protocol provenance

This note maps each public adapter to the author-side and upstream behavior it
implements.

## Synchformer

The author-side codebase contains feature-level evaluation and
classifier-style aggregation in separate runners. The public Global Offset
entrypoint makes the selected classifier path explicit through two auditable
components:

1. official Synchformer VGGSound model loading, transforms, 21-class grid, and
   zero-class probability; and
2. the author-side classifier outer-window convention already used for the
   SparseSync baseline: five-second windows every 0.64 seconds, rounded/padded
   duration, and mean zero-class probability.

The public implementation records the outer-window starts and aggregation in
each run card so the protocol can be audited independently of the model output.

## ImageBind

The author-side result-generation runner and upstream ImageBind reference
preprocessing differ. Both are preserved as explicit modes:

- `paper_legacy`: the author-side 224×224 warp, custom log-Mel 128×204, no
  upstream audio mean/std, and at most 18 segments. The public runtime keeps
  the FFmpeg signed-16 audio path and replaces the legacy torchvision/PyAV
  video decoder with Decord;
- `official_0p64`: short-side resize, three crops, upstream Kaldi fbank and
  mean/std, all complete segments, and the same benchmark-level FFmpeg audio
  decode as `paper_legacy`.

The run card records the mode. `paper_legacy` is the paper-table path;
`official_0p64` is an ablation.

## Reproduction checks

For a directly comparable result:

- validate all 15,000 Global Offset pairs and the expected magnitude/direction
  coverage;
- require zero decode or inference failures;
- archive sanitized run and summary JSON files;
- compare every reported stratum, not only overall accuracy;
- record any correction to the protocol rather than tuning it silently.
