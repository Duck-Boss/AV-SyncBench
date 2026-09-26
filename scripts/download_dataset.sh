#!/usr/bin/env bash
set -euo pipefail

# Download the official ModelScope snapshot and stream-extract its seven split
# tar.gz parts. This avoids creating a second 68.19 GB concatenated archive.

RAW_DIR="${1:-data/downloads/AVSyncBench}"
OUTPUT_DIR="${2:-data/AVSyncBench}"
REVISION="86c06579529a6e7b2cafb0dc386a50152a37fb98"

if ! command -v modelscope >/dev/null 2>&1; then
  echo "ModelScope CLI is missing. Install it with: python -m pip install modelscope" >&2
  exit 2
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "sha256sum is required for dataset integrity verification." >&2
  exit 2
fi

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to extract into non-empty directory: $OUTPUT_DIR" >&2
  echo "Choose an empty output directory so existing files cannot be overwritten." >&2
  exit 2
fi

mkdir -p "$RAW_DIR" "$OUTPUT_DIR"
modelscope download \
  --dataset coming245/AVSyncBench \
  --revision "$REVISION" \
  --local_dir "$RAW_DIR"

shopt -s nullglob
parts=("$RAW_DIR"/AV_SyncBench.tar.gz.part_*)
if [[ ${#parts[@]} -ne 7 ]]; then
  echo "Expected 7 archive parts in $RAW_DIR, found ${#parts[@]}." >&2
  exit 3
fi

expected_sizes=(10737418240 10737418240 10737418240 10737418240 10737418240 10737418240 3767330543)
expected_sha256=(
  4e88e575b3c5e202cd2addbc1789bfed6a4cf036d85f0f70d5ca56be05b96d7d
  00e4b50c2b4b32d539d8c75a8e6f9308afa83983c3c0b221445935740354fdbd
  078c1d9a70beb6f9fa6d6b0c6afb21e3527d932c04f76cec2b726a388baf4d53
  7ef96fb3dce0b6f241e4d7aa19d873e518176ddf33e3b7a16bc5feb5a32eac01
  4dfbbd97087b025a0b16f46def669a953de566bbae275489dd5ef90330b4d712
  35ec67bc890589289f1644ecbba1a30337119db65fec99072e8bcc2a06196630
  3d71987d812814950fb940862fff1648d127f4cc15bbfe68c34e7ad4c6bcd2f1
)
for index in "${!parts[@]}"; do
  actual_size="$(stat -c %s "${parts[$index]}")"
  if [[ "$actual_size" != "${expected_sizes[$index]}" ]]; then
    echo "Size mismatch: ${parts[$index]} is $actual_size bytes; expected ${expected_sizes[$index]}." >&2
    exit 4
  fi
  actual_sha256="$(sha256sum "${parts[$index]}" | awk '{print $1}')"
  if [[ "$actual_sha256" != "${expected_sha256[$index]}" ]]; then
    echo "SHA-256 mismatch: ${parts[$index]}." >&2
    exit 4
  fi
done

cat "${parts[@]}" | tar --no-same-owner --no-same-permissions -xzf - -C "$OUTPUT_DIR"
echo "Dataset revision $REVISION extracted to $OUTPUT_DIR"

echo "Discovered manifests:"
find "$OUTPUT_DIR" -type f -name metadata.jsonl -print | sort
