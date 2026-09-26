#!/usr/bin/env python3
"""Create a five-second, dependency-light synthetic smoke-test pair.

The generated sample is not part of the benchmark and must not be used for a
scientific accuracy claim. It only verifies download, decoding, model loading,
scoring, and result serialization end to end.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List


def run(command: List[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo/data"))
    parser.add_argument("--offset-ms", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")
    if args.offset_ms <= 0 or args.offset_ms >= 5_000:
        raise SystemExit("--offset-ms must be between 1 and 4999")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "synthetic_sync.mp4"
    original_audio = output_dir / "synthetic_sync.wav"
    negative_audio = output_dir / f"synthetic_delay_{args.offset_ms}ms.wav"
    manifest_path = output_dir / "global_offset_demo.jsonl"
    outputs = [video_path, original_audio, negative_audio, manifest_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"outputs already exist; pass --force to replace them: {existing}")

    # A bright square and an 880 Hz pulse are active for 80 ms at each integer
    # second. The visual stream is encoded without audio; PCM WAV files avoid
    # codec priming delay in the 50--500 ms synchronization test.
    visual_filter = (
        r"color=c=0x101820:s=256x256:r=25:d=5,"
        r"drawbox=x=64:y=64:w=128:h=128:color=0x2ED6C9:t=fill:"
        r"enable=lt(mod(t\,1)\,0.08)"
    )
    audio_filter = (
        r"aevalsrc=if(lt(mod(t\,1)\,0.08)\,"
        r"0.25*sin(2*PI*880*t)\,0):s=16000:d=5"
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            visual_filter,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            audio_filter,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(original_audio),
        ]
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(original_audio),
            "-af",
            f"adelay={args.offset_ms}:all=1,apad,atrim=duration=5",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(negative_audio),
        ]
    )

    record = {
        "sample_id": f"synthetic-delay-{args.offset_ms}ms",
        "video_id": "synthetic-pulse",
        "original_video": video_path.name,
        "original_audio": original_audio.name,
        "edited_audio": negative_audio.name,
        "edit_type": "global_offset",
        "task_type": "global_offset",
        "target_attribute": f"{args.offset_ms}ms delay",
        "offset_ms": args.offset_ms,
        "direction": "delay",
        "category": "synthetic_smoke_test",
        "duration": 5.0,
    }
    manifest_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"created: {manifest_path}")


if __name__ == "__main__":
    main()
