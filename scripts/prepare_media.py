#!/usr/bin/env python3
"""Canonicalize a manifest to 25 FPS video and mono 16 kHz PCM audio.

The strict Synchformer adapter requires this step; it is also useful for custom
clips and for diagnosing decoder/version differences. It writes a new manifest
and never modifies the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from avsyncbench.manifest import (
    iter_manifest,
    validate_audio_negative_records,
    validate_unique_ids,
)


def stable_name(path: Path, dataset_root: Path) -> str:
    """Hash a relative name, or file content for an external legacy path."""

    resolved = path.resolve()
    try:
        relative_source = resolved.relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        hasher = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()[:20]
    return hashlib.sha256(relative_source.encode("utf-8")).hexdigest()[:20]


def safe_sample_name(sample_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", sample_id).strip(".-") or "sample"
    suffix = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}-{suffix}"


def run(command: List[str]) -> None:
    subprocess.run(command, check=True)


def make_video(ffmpeg: str, source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part.mp4")
    vf = (
        "fps=25,"
        "scale=iw*256/min(iw\\,ih):ih*256/min(iw\\,ih),"
        "crop=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            str(temporary),
        ]
    )
    temporary.replace(destination)


def make_audio(ffmpeg: str, source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part.wav")
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
    )
    temporary.replace(destination)


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--edit-type",
        action="append",
        dest="edit_types",
        help="canonicalize only this manifest edit_type; repeat for multiple values",
    )
    parser.add_argument(
        "--allow-absolute-paths",
        action="store_true",
        help="allow maintainer conversion of legacy manifests with absolute media paths",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")
    output_root = args.output_root.resolve()
    output_manifest = (
        args.output_manifest or output_root / "manifest.canonical.jsonl"
    ).resolve()
    if output_manifest == args.manifest.expanduser().resolve():
        raise SystemExit("--output-manifest must not overwrite the input manifest")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    video_cache: Dict[Path, Path] = {}
    original_audio_cache: Dict[Path, Path] = {}
    records = list(
        iter_manifest(
            args.manifest,
            args.dataset_root,
            limit=args.limit,
            edit_types=args.edit_types,
            allow_absolute_paths=args.allow_absolute_paths,
        )
    )
    if not records:
        raise SystemExit("no records matched the requested manifest/filter")
    validate_unique_ids(records)
    validate_audio_negative_records(records)
    output_records = []
    for record in records:
        if record.negative_audio is None:
            raise ValueError(
                f"{record.sample_id}: media canonicalization requires negative audio"
            )
        source_video = record.original_video.resolve()
        canonical_video = video_cache.setdefault(
            source_video,
            output_root
            / "media"
            / "video"
            / f"{stable_name(source_video, args.dataset_root)}.mp4",
        )
        source_original_audio = (record.original_audio or record.original_video).resolve()
        canonical_original_audio = original_audio_cache.setdefault(
            source_original_audio,
            output_root
            / "media"
            / "audio"
            / "original"
            / f"{stable_name(source_original_audio, args.dataset_root)}.wav",
        )
        canonical_negative_audio = (
            output_root
            / "media"
            / "audio"
            / "negative"
            / f"{safe_sample_name(record.sample_id)}.wav"
        )
        make_video(ffmpeg, source_video, canonical_video, args.force)
        make_audio(ffmpeg, source_original_audio, canonical_original_audio, args.force)
        make_audio(ffmpeg, record.negative_audio.resolve(), canonical_negative_audio, args.force)

        # Emit a public, canonical schema instead of copying arbitrary source
        # fields that may contain internal paths or private metadata.
        converted = {
            "sample_id": record.sample_id,
            "original_video": relative(canonical_video, output_root),
            "original_audio": relative(canonical_original_audio, output_root),
            "edited_audio": relative(canonical_negative_audio, output_root),
            "task_type": record.task,
            "edit_type": record.edit_type,
        }
        for key in ("video_id", "target_attribute", "duration"):
            value = record.raw.get(key)
            if value not in (None, ""):
                converted[key] = value
        for key, value in (
            ("offset_ms", record.offset_ms),
            ("direction", record.direction),
            ("category", record.category),
        ):
            if value is not None:
                converted[key] = value
        output_records.append(converted)

    with output_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(output_records):,} records to {output_manifest}")


if __name__ == "__main__":
    main()
