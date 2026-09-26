#!/usr/bin/env python3
"""Download official evaluation checkpoints with available verification."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Artifact:
    model: str
    filename: str
    url: str
    size: int
    md5: Optional[str] = None


ARTIFACTS = (
    Artifact(
        model="synchformer-vggsound",
        filename="cfg-24-01-02T10-00-53.yaml",
        url=(
            "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/"
            "sync/sync_models/24-01-02T10-00-53/cfg-24-01-02T10-00-53.yaml"
        ),
        size=6_878,
        md5="606ca81ca850a220009fde67233f89a4",
    ),
    Artifact(
        model="synchformer-vggsound",
        filename="24-01-02T10-00-53.pt",
        url=(
            "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/"
            "sync/sync_models/24-01-02T10-00-53/24-01-02T10-00-53.pt"
        ),
        size=1_131_154_181,
        md5="19592ed5e64e3560b93772c5d46e3334",
    ),
    Artifact(
        model="imagebind-huge",
        filename="imagebind_huge.pth",
        url="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
        size=4_803_584_173,
        # Upstream does not publish a trustworthy SHA/MD5 for this multipart object.
        md5=None,
    ),
)


class Progress:
    def __init__(self, label: str, initial: int, total: int) -> None:
        self.label = label
        self.downloaded = initial
        self.total = total
        self.last_percent = -1

    def update(self, count: int) -> None:
        self.downloaded += count
        percent = int(self.downloaded * 100 / self.total) if self.total else 0
        if percent != self.last_percent:
            print(
                f"\r{self.label}: {self.downloaded:,}/{self.total:,} bytes ({percent:3d}%)",
                end="",
                flush=True,
            )
            self.last_percent = percent

    def finish(self) -> None:
        print()


def digest(path: Path, algorithm: str = "md5") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def is_valid(path: Path, artifact: Artifact) -> bool:
    if not path.is_file() or path.stat().st_size != artifact.size:
        return False
    return artifact.md5 is None or digest(path) == artifact.md5


def download(artifact: Artifact, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if artifact.md5 is None:
        print(
            f"warning: upstream publishes no digest for {artifact.filename}; "
            "verifying official URL and byte size only",
            file=sys.stderr,
        )
    if destination.exists() and not force:
        if is_valid(destination, artifact):
            print(f"verified: {destination}")
            return destination
        raise RuntimeError(
            f"existing file failed verification: {destination}; pass --force to replace it"
        )

    partial = destination.with_name(destination.name + ".part")
    if force and partial.exists():
        partial.unlink()
    start = partial.stat().st_size if partial.exists() else 0
    if start == artifact.size and start > 0:
        if is_valid(partial, artifact):
            os.replace(partial, destination)
            print(f"verified completed partial: {destination}")
            return destination
        raise RuntimeError(
            f"completed partial file failed verification: {partial}; "
            "pass --force to replace it"
        )
    if start > artifact.size:
        partial.unlink()
        start = 0

    headers = {"User-Agent": "AV-SyncBench/0.1"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(artifact.url, headers=headers)
    with urllib.request.urlopen(request) as response:  # nosec B310: fixed HTTPS URLs above
        resumed = start > 0 and getattr(response, "status", None) == 206
        if start and not resumed:
            start = 0
        mode = "ab" if resumed else "wb"
        progress = Progress(artifact.filename, start, artifact.size)
        with partial.open(mode) as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                progress.update(len(block))
        progress.finish()

    if partial.stat().st_size != artifact.size:
        raise RuntimeError(
            f"size mismatch for {artifact.filename}: got {partial.stat().st_size:,}, "
            f"expected {artifact.size:,}; partial file kept for resume"
        )
    if artifact.md5 is not None:
        actual_md5 = digest(partial)
        if actual_md5 != artifact.md5:
            raise RuntimeError(
                f"MD5 mismatch for {artifact.filename}: got {actual_md5}, "
                f"expected {artifact.md5}; partial file kept for inspection"
            )
    os.replace(partial, destination)
    print(f"ready: {destination}")
    return destination


def selected_artifacts(model: str) -> Iterable[Artifact]:
    for artifact in ARTIFACTS:
        if model == "all" or artifact.model == model:
            yield artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("synchformer-vggsound", "imagebind-huge", "all"),
        default="synchformer-vggsound",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for artifact in selected_artifacts(args.model):
        model_dir = args.output_dir / artifact.model
        download(artifact, model_dir / artifact.filename, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
