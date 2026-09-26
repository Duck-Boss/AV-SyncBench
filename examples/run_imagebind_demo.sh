#!/usr/bin/env bash
set -euo pipefail

# Run inside the pinned ImageBind environment described in the tutorial.
# The official checkpoint is about 4.8 GB.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

imagebind_root="${IMAGEBIND_ROOT:-third_party/ImageBind}"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints}"
device="${DEVICE:-cuda:0}"
expected_commit="53680b02d7e37b19b124fa37bae4b6c98c38f5be"

if [[ ! -d "$imagebind_root/.git" ]]; then
  mkdir -p "$(dirname "$imagebind_root")"
  git clone https://github.com/facebookresearch/ImageBind.git "$imagebind_root"
  git -C "$imagebind_root" checkout "$expected_commit"
fi

actual_commit="$(git -C "$imagebind_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "ImageBind must be checked out at $expected_commit" >&2
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
  --model imagebind-huge \
  --output-dir "$checkpoint_root"

mkdir -p outputs
python -m avsyncbench.evaluators.imagebind \
  --manifest demo/data/global_offset_demo.jsonl \
  --dataset-root demo/data \
  --imagebind-root "$imagebind_root" \
  --checkpoint "$checkpoint_root/imagebind-huge/imagebind_huge.pth" \
  --device "$device" \
  --preprocess-mode paper_legacy \
  --output outputs/demo_imagebind.jsonl
