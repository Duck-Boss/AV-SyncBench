"""Manifest parsing shared by all AV-SyncBench evaluators.

The public benchmark manifest uses paths relative to the extracted dataset root.
This module deliberately accepts a small set of legacy aliases so that internal
manifests can be validated and converted without hard-coded server paths.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Union


@dataclass(frozen=True)
class ManifestRecord:
    """One original-versus-perturbed comparison."""

    sample_id: str
    original_video: Path
    negative_audio: Optional[Path]
    negative_video: Optional[Path]
    original_audio: Optional[Path]
    task: str
    edit_type: str
    offset_ms: Optional[int]
    direction: Optional[str]
    category: Optional[str]
    raw: Mapping[str, Any]

    @property
    def negative_media(self) -> Path:
        if self.negative_video is not None:
            return self.negative_video
        if self.negative_audio is not None:
            return self.negative_audio
        raise ValueError(f"{self.sample_id}: neither negative_audio nor negative_video is set")


def _first(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve(root: Path, value: Any, *, allow_absolute_paths: bool = False) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        if not allow_absolute_paths:
            raise ValueError(
                f"absolute manifest paths are disabled: {path}; use paths relative "
                "to --dataset-root"
            )
        return path.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"relative manifest path escapes dataset root: {path}")
    return resolved


def _parse_offset_ms(record: Mapping[str, Any]) -> Optional[int]:
    direct = _first(record, "offset_ms", "shift_ms", "magnitude_ms")
    if direct is not None:
        return abs(int(round(float(direct))))

    seconds = _first(record, "offset_sec", "shift_sec")
    if seconds is not None:
        return abs(int(round(float(seconds) * 1000)))

    candidates = [
        str(record.get(key) or "")
        for key in (
            "target_attribute",
            "attribute",
            "audio_caption",
            "edited_audio",
            "negative_audio",
            "edited_video",
            "negative_video",
        )
    ]
    for candidate in candidates:
        match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*ms\b", candidate, flags=re.I)
        if match:
            return abs(int(round(float(match.group(1)))))
    for candidate in candidates:
        match = re.search(
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:sec(?:ond)?s?|s)\b",
            candidate,
            flags=re.I,
        )
        if match:
            return abs(int(round(float(match.group(1)) * 1000)))
    return None


def _parse_direction(record: Mapping[str, Any]) -> Optional[str]:
    direct = _first(record, "direction", "shift_direction")
    if direct is not None:
        return str(direct).lower()
    text = " ".join(
        str(record.get(key) or "")
        for key in ("target_attribute", "audio_caption", "edited_audio", "negative_audio")
    ).lower()
    if re.search(r"(?:^|[^a-z])advance(?:d)?(?:$|[^a-z])", text):
        return "advance"
    if re.search(r"(?:^|[^a-z])delay(?:ed)?(?:$|[^a-z])", text):
        return "delay"
    return None


def parse_record(
    record: Mapping[str, Any],
    dataset_root: Path,
    line_number: int,
    *,
    allow_absolute_paths: bool = False,
) -> ManifestRecord:
    raw_sample_id = _first(record, "sample_id", "id", "uid")
    if raw_sample_id is None:
        raise ValueError(f"line {line_number}: sample_id (or id/uid alias) is required")
    sample_id = str(raw_sample_id)
    original_video = _resolve(
        dataset_root,
        _first(record, "original_video", "video", "video_path", "positive_video"),
        allow_absolute_paths=allow_absolute_paths,
    )
    if original_video is None:
        raise ValueError(f"line {line_number} ({sample_id}): original_video is required")

    negative_audio = _resolve(
        dataset_root,
        _first(record, "edited_audio", "negative_audio", "perturbed_audio", "audio_path"),
        allow_absolute_paths=allow_absolute_paths,
    )
    negative_video = _resolve(
        dataset_root,
        _first(record, "edited_video", "negative_video", "perturbed_video"),
        allow_absolute_paths=allow_absolute_paths,
    )
    original_audio = _resolve(
        dataset_root,
        _first(record, "original_audio", "positive_audio"),
        allow_absolute_paths=allow_absolute_paths,
    )
    if negative_audio is None and negative_video is None:
        raise ValueError(
            f"line {line_number} ({sample_id}): one of edited_audio/negative_audio "
            "or edited_video/negative_video is required"
        )
    if negative_audio is not None and negative_video is not None:
        raise ValueError(
            f"line {line_number} ({sample_id}): provide exactly one negative medium, "
            "not both audio and video"
        )

    task = str(_first(record, "task_type", "task") or "unknown")
    edit_type = str(_first(record, "edit_type") or task)
    direction = _parse_direction(record)
    category = _first(record, "category", "scenario")
    return ManifestRecord(
        sample_id=sample_id,
        original_video=original_video,
        negative_audio=negative_audio,
        negative_video=negative_video,
        original_audio=original_audio,
        task=task,
        edit_type=edit_type,
        offset_ms=_parse_offset_ms(record),
        direction=direction,
        category=None if category is None else str(category),
        raw=dict(record),
    )


def iter_manifest(
    manifest_path: Union[Path, str],
    dataset_root: Union[Path, str],
    *,
    limit: Optional[int] = None,
    require_files: bool = True,
    edit_types: Optional[Iterable[str]] = None,
    allow_absolute_paths: bool = False,
) -> Iterator[ManifestRecord]:
    """Yield validated records from a JSONL manifest."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    allowed_edit_types = None if edit_types is None else {value.lower() for value in edit_types}
    with manifest_path.open("r", encoding="utf-8") as handle:
        yielded = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number}: each JSONL item must be an object")
            parsed = parse_record(
                raw,
                dataset_root,
                line_number,
                allow_absolute_paths=allow_absolute_paths,
            )
            if allowed_edit_types is not None and parsed.edit_type.lower() not in allowed_edit_types:
                continue
            if require_files:
                paths = [parsed.original_video, parsed.negative_media]
                if parsed.original_audio is not None:
                    paths.append(parsed.original_audio)
                missing = [str(path) for path in paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"line {line_number} ({parsed.sample_id}) references missing files: {missing}"
                    )
            yield parsed
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def validate_unique_ids(records: Iterable[ManifestRecord]) -> None:
    seen: Dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        if record.sample_id in seen:
            raise ValueError(
                f"duplicate sample_id {record.sample_id!r} at records {seen[record.sample_id]} and {index}"
            )
        seen[record.sample_id] = index


def validate_audio_negative_records(records: Iterable[ManifestRecord]) -> None:
    """Require the audio-negative pair schema used by the public evaluators."""

    invalid = [record.sample_id for record in records if record.negative_audio is None]
    if invalid:
        preview = ", ".join(repr(value) for value in invalid[:5])
        suffix = " ..." if len(invalid) > 5 else ""
        raise ValueError(
            "this evaluator requires edited_audio/negative_audio records; "
            f"found {len(invalid)} video-negative record(s): {preview}{suffix}"
        )
