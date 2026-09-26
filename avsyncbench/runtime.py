"""Capture the environment needed to diagnose reproduction differences."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union


def file_digests(
    path: Union[Path, str], algorithms: Sequence[str] = ("md5", "sha256")
) -> Dict[str, str]:
    """Hash a file once with every requested algorithm.

    MD5 is retained only because the upstream Synchformer release publishes an
    MD5 identifier. SHA-256 is also recorded for stronger local provenance.
    """

    hashers = {name: hashlib.new(name) for name in algorithms}
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            for hasher in hashers.values():
                hasher.update(block)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def file_md5(path: Union[Path, str]) -> str:
    return file_digests(path, ("md5",))["md5"]


def verify_artifact(
    path: Union[Path, str],
    *,
    expected_size: int,
    expected_md5: Optional[str] = None,
) -> Mapping[str, str]:
    """Verify an official model artifact before unsafe checkpoint loading."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path.name}: expected {expected_size:,} bytes, got {actual_size:,}"
        )
    digests = file_digests(path)
    if expected_md5 is not None and digests["md5"].lower() != expected_md5.lower():
        raise ValueError(
            f"{path.name}: MD5 does not match the official release identifier"
        )
    return digests


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_package_versions(expected: Mapping[str, str]) -> None:
    """Require exact public release versions, allowing wheel local suffixes."""

    mismatches = []
    for name, wanted in expected.items():
        actual = package_version(name)
        comparable = None if actual is None else actual.split("+", 1)[0]
        if comparable != wanted:
            mismatches.append(f"{name}={actual!r} (expected {wanted})")
    if mismatches:
        raise RuntimeError(
            "incompatible package versions: "
            + ", ".join(mismatches)
            + "; install the pinned release environment"
        )


def command_output(command: Iterable[str]) -> Optional[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_revision(path: Union[Path, str]) -> Optional[str]:
    return command_output(["git", "-C", str(path), "rev-parse", "HEAD"])


def require_upstream_checkout(
    root: Union[Path, str],
    *,
    expected_commit: str,
    required_files: Sequence[str],
) -> Path:
    """Require the exact upstream Git checkout named by the release protocol."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"upstream checkout is not a directory: {root}")
    missing = [relative for relative in required_files if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"upstream checkout {root} is missing required files: {missing}"
        )
    actual_commit = git_revision(root)
    if actual_commit is None:
        raise ValueError(f"upstream checkout is not a readable Git repository: {root}")
    if actual_commit.lower() != expected_commit.lower():
        raise ValueError(
            f"upstream checkout commit {actual_commit} does not match pinned "
            f"commit {expected_commit}"
        )
    status = command_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"]
    )
    if status is None:
        raise ValueError(f"could not inspect upstream checkout status: {root}")
    if status:
        raise ValueError(
            "upstream checkout contains modified or untracked files; use a clean "
            f"checkout at {expected_commit}"
        )
    return root


def _portable_name(path: Union[Path, str]) -> str:
    """Return a shareable path label without leaking a user's directory tree."""

    return Path(path).name


def sanitized_exception(
    exception: BaseException,
    private_paths: Optional[Mapping[str, Union[Path, str]]] = None,
) -> str:
    """Format an error while replacing caller-provided absolute path roots.

    Per-record errors are useful for diagnosis and are commonly shared with a
    replication report. Dataset and checkout roots are therefore replaced by
    stable labels before serialization.
    """

    message = f"{type(exception).__name__}: {exception}"
    replacements = []
    for label, value in (private_paths or {}).items():
        resolved = Path(value).expanduser().resolve()
        replacements.append((str(resolved), f"<{label}>"))
        replacements.append((resolved.as_posix(), f"<{label}>"))
    ordered = sorted(
        set(replacements),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for source, replacement in ordered:
        message = message.replace(source, replacement)
    return message


def require_distinct_paths(paths: Mapping[str, Union[Path, str]]) -> None:
    """Reject accidental input/output aliases before any expensive work."""

    seen: Dict[Path, str] = {}
    for label, value in paths.items():
        resolved = Path(value).expanduser().resolve()
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(
                f"path collision: {label} and {previous} both resolve to {resolved}"
            )
        seen[resolved] = label


def require_paths_not_in(
    paths: Mapping[str, Union[Path, str]],
    protected_paths: Iterable[Union[Path, str]],
) -> None:
    """Reject outputs that would overwrite any protected input file."""

    protected = {Path(value).expanduser().resolve() for value in protected_paths}
    for label, value in paths.items():
        resolved = Path(value).expanduser().resolve()
        if resolved in protected:
            raise ValueError(
                f"path collision: {label} would overwrite protected input "
                f"{resolved.name}"
            )


def collect_runtime(
    *,
    model_name: str,
    upstream_root: Union[Path, str],
    checkpoint: Union[Path, str],
    config: Optional[Union[Path, str]] = None,
    checkpoint_digests: Optional[Mapping[str, str]] = None,
    config_digests: Optional[Mapping[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "av",
        "numpy",
        "omegaconf",
        "timm",
        "transformers",
        "pytorchvideo",
        "decord",
    ]
    checkpoint = Path(checkpoint)
    checkpoint_hashes = dict(checkpoint_digests or file_digests(checkpoint))
    adapter_root = Path(__file__).resolve().parents[1]
    output: Dict[str, Any] = {
        "model": model_name,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
        "ffmpeg": command_output(["ffmpeg", "-version"]),
        "adapter_commit": git_revision(adapter_root),
        "upstream_root": _portable_name(upstream_root),
        "upstream_commit": git_revision(upstream_root),
        "checkpoint": checkpoint.name,
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_md5": checkpoint_hashes["md5"],
        "checkpoint_sha256": checkpoint_hashes["sha256"],
    }
    if config is not None:
        config = Path(config)
        config_hashes = dict(config_digests or file_digests(config))
        output.update(
            {
                "config": config.name,
                "config_size": config.stat().st_size,
                "config_md5": config_hashes["md5"],
                "config_sha256": config_hashes["sha256"],
            }
        )
    if extra:
        output.update(extra)
    return output


def write_run_metadata(path: Union[Path, str], metadata: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
