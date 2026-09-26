#!/usr/bin/env python3
"""Validate an AV-SyncBench JSONL manifest before starting a GPU job."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from avsyncbench.manifest import iter_manifest, validate_unique_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--edit-type", action="append", dest="edit_types")
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--require-offset", action="append", type=int, default=[])
    parser.add_argument("--require-direction", action="append", default=[])
    parser.add_argument("--require-complete-offsets", action="store_true")
    parser.add_argument("--require-complete-directions", action="store_true")
    parser.add_argument(
        "--exact-offset-set",
        action="store_true",
        help="reject offset values not listed by --require-offset",
    )
    parser.add_argument(
        "--exact-direction-set",
        action="store_true",
        help="reject directions not listed by --require-direction",
    )
    parser.add_argument(
        "--expect-per-offset",
        type=int,
        help="require this many records for every required offset",
    )
    parser.add_argument(
        "--expect-per-direction",
        type=int,
        help="require this many records for every required direction",
    )
    parser.add_argument(
        "--expect-per-offset-direction",
        type=int,
        help="require this many records for every required offset/direction pair",
    )
    parser.add_argument(
        "--allow-absolute-paths",
        action="store_true",
        help="allow legacy absolute media paths (disabled by default for safe sharing)",
    )
    args = parser.parse_args()

    records = list(
        iter_manifest(
            args.manifest,
            args.dataset_root,
            limit=args.limit,
            require_files=not args.skip_file_check,
            edit_types=args.edit_types,
            allow_absolute_paths=args.allow_absolute_paths,
        )
    )
    if not records:
        raise SystemExit("no records matched the requested manifest/filter")
    validate_unique_ids(records)
    tasks = Counter(record.task for record in records)
    edit_types = Counter(record.edit_type for record in records)
    offsets = Counter(record.offset_ms for record in records)
    directions = Counter(record.direction for record in records)
    offset_directions = Counter(
        (record.offset_ms, record.direction) for record in records
    )
    print(f"valid records: {len(records):,}")
    print(f"tasks: {dict(sorted(tasks.items()))}")
    print(f"edit_types: {dict(sorted(edit_types.items()))}")
    print(f"offset_ms: {dict(sorted(offsets.items(), key=lambda item: (item[0] is None, item[0])))}")
    print(f"directions: {dict(sorted(directions.items(), key=lambda item: str(item[0])))}")

    if args.expect_count is not None and len(records) != args.expect_count:
        raise SystemExit(f"expected {args.expect_count:,} records, found {len(records):,}")
    required_offsets = set(args.require_offset)
    required_directions = {value.lower() for value in args.require_direction}
    actual_offsets = set(offsets) - {None}
    actual_directions = set(directions) - {None}
    missing_offsets = sorted(required_offsets - actual_offsets)
    if missing_offsets:
        raise SystemExit(f"required offsets are missing: {missing_offsets}")
    missing_directions = sorted(
        required_directions - actual_directions
    )
    if missing_directions:
        raise SystemExit(f"required directions are missing: {missing_directions}")
    if args.require_complete_offsets and offsets.get(None, 0):
        raise SystemExit(f"{offsets[None]:,} records have no parseable offset")
    if args.require_complete_directions and directions.get(None, 0):
        raise SystemExit(f"{directions[None]:,} records have no parseable direction")
    if args.exact_offset_set:
        if not required_offsets:
            raise SystemExit("--exact-offset-set requires at least one --require-offset")
        unexpected_offsets = sorted(actual_offsets - required_offsets)
        if unexpected_offsets:
            raise SystemExit(f"unexpected offsets are present: {unexpected_offsets}")
    if args.exact_direction_set:
        if not required_directions:
            raise SystemExit(
                "--exact-direction-set requires at least one --require-direction"
            )
        unexpected_directions = sorted(actual_directions - required_directions)
        if unexpected_directions:
            raise SystemExit(
                f"unexpected directions are present: {unexpected_directions}"
            )
    if args.expect_per_offset is not None:
        if args.expect_per_offset <= 0 or not required_offsets:
            raise SystemExit(
                "--expect-per-offset must be positive and requires --require-offset"
            )
        wrong = {
            value: offsets[value]
            for value in sorted(required_offsets)
            if offsets[value] != args.expect_per_offset
        }
        if wrong:
            raise SystemExit(
                f"per-offset counts differ from {args.expect_per_offset:,}: {wrong}"
            )
    if args.expect_per_direction is not None:
        if args.expect_per_direction <= 0 or not required_directions:
            raise SystemExit(
                "--expect-per-direction must be positive and requires --require-direction"
            )
        wrong = {
            value: directions[value]
            for value in sorted(required_directions)
            if directions[value] != args.expect_per_direction
        }
        if wrong:
            raise SystemExit(
                f"per-direction counts differ from {args.expect_per_direction:,}: {wrong}"
            )
    if args.expect_per_offset_direction is not None:
        if (
            args.expect_per_offset_direction <= 0
            or not required_offsets
            or not required_directions
        ):
            raise SystemExit(
                "--expect-per-offset-direction must be positive and requires "
                "--require-offset and --require-direction"
            )
        wrong = {
            f"{offset_ms}:{direction}": offset_directions[(offset_ms, direction)]
            for offset_ms in sorted(required_offsets)
            for direction in sorted(required_directions)
            if offset_directions[(offset_ms, direction)]
            != args.expect_per_offset_direction
        }
        if wrong:
            raise SystemExit(
                "per-offset/direction counts differ from "
                f"{args.expect_per_offset_direction:,}: {wrong}"
            )


if __name__ == "__main__":
    main()
