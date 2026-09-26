#!/usr/bin/env bash
set -euo pipefail

# Run inside the Synchformer environment described in the tutorial.
# This creates synthetic media, fetches the official VGGSound checkpoint, and
# writes predictions, a summary, and a reproducibility run card.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

synchformer_root="${SYNCHFORMER_ROOT:-third_party/Synchformer}"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints}"
device="${DEVICE:-cuda:0}"
expected_commit="b66668a1521d7567cc760e5544b2b5b53179b687"

if [[ ! -d "$synchformer_root/.git" ]]; then
  mkdir -p "$(dirname "$synchformer_root")"
  git clone https://github.com/v-iashin/Synchformer.git "$synchformer_root"
  git -C "$synchformer_root" checkout "$expected_commit"
fi

actual_commit="$(git -C "$synchformer_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "Synchformer must be checked out at $expected_commit" >&2
  exit 2
fi

if [[ ! -f demo/data/global_offset_demo.jsonl ]]; then
  python scripts/make_demo.py
fi
python scripts/validate_manifest.py \
  --manifest demo/data/global_offset_demo.jsonl \
  --dataset-root demo/data \
  --expect-count 1
python scripts/download_models.py \
  --model synchformer-vggsound \
  --output-dir "$checkpoint_root"

mkdir -p outputs
python -m avsyncbench.evaluators.synchformer_global_offset \
  --manifest demo/data/global_offset_demo.jsonl \
  --dataset-root demo/data \
  --synchformer-root "$synchformer_root" \
  --config "$checkpoint_root/synchformer-vggsound/cfg-24-01-02T10-00-53.yaml" \
  --checkpoint "$checkpoint_root/synchformer-vggsound/24-01-02T10-00-53.pt" \
  --device "$device" \
  --window-policy repository_sliding \
  --output outputs/demo_synchformer.jsonl
