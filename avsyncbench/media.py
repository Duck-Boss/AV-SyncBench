"""Media decoding helpers used by the public model adapters."""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


@dataclass(frozen=True)
class VideoFrames:
    frames: torch.Tensor  # T, C, H, W; uint8 in [0, 255]
    fps: float

    @property
    def duration(self) -> float:
        return self.frames.shape[0] / self.fps


@dataclass(frozen=True)
class AudioWaveform:
    waveform: torch.Tensor  # mono, T
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.waveform.numel() / self.sample_rate


def _mono(audio: torch.Tensor) -> torch.Tensor:
    # Keep torch optional at module import time so the pure manifest/window
    # helpers remain usable in lightweight environments.
    import torch

    if audio.ndim == 1:
        return audio.float()
    if audio.ndim != 2:
        raise ValueError(f"expected 1D/2D audio, got shape {tuple(audio.shape)}")
    if audio.shape[0] <= 8:
        return audio.float().mean(dim=0)
    if audio.shape[1] <= 8:
        return audio.float().mean(dim=1)
    raise ValueError(f"cannot infer channel dimension for audio shape {tuple(audio.shape)}")


def load_video(path: Union[Path, str]) -> VideoFrames:
    import torchvision

    path = Path(path)
    if not hasattr(torchvision.io, "read_video"):
        raise RuntimeError(
            "this adapter requires torchvision.io.read_video; install the pinned "
            "model environment from docs/REPRODUCE_GLOBAL_OFFSET.md"
        )
    frames, _, info = torchvision.io.read_video(
        str(path), pts_unit="sec", output_format="TCHW"
    )
    fps = float(info.get("video_fps") or 0)
    if fps <= 0 or frames.numel() == 0:
        raise ValueError(f"failed to decode a valid video stream from {path}")
    return VideoFrames(frames=frames, fps=fps)


def load_video_decord(path: Union[Path, str]) -> VideoFrames:
    """Decode a video to CPU RGB frames with Decord.

    ImageBind's public reference loader uses Decord. Keeping this loader
    separate also avoids relying on the video-decoding APIs removed from newer
    torchvision releases.
    """

    import torch

    try:
        import decord
    except ImportError as exc:
        raise RuntimeError(
            "decord is required for ImageBind video decoding; install the pinned "
            "ImageBind environment from docs/REPRODUCE_GLOBAL_OFFSET.md"
        ) from exc

    path = Path(path)
    try:
        reader = decord.VideoReader(
            str(path),
            ctx=decord.cpu(0),
            num_threads=1,
        )
        frame_count = len(reader)
        fps = float(reader.get_avg_fps())
        if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("decoded stream has no frames or a non-positive FPS")
        batch = reader.get_batch(list(range(frame_count)))
        frames = batch if torch.is_tensor(batch) else torch.from_numpy(batch.asnumpy())
    except Exception as exc:
        raise ValueError(f"failed to decode a valid video stream from {path}") from exc

    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] != frame_count:
        raise ValueError(
            f"unexpected Decord frame shape {tuple(frames.shape)} from {path}"
        )
    frames = frames.permute(0, 3, 1, 2).contiguous()
    return VideoFrames(frames=frames, fps=fps)


def load_audio(path: Union[Path, str], target_rate: int = 16_000) -> AudioWaveform:
    import torchaudio
    import torchvision

    path = Path(path)
    if path.suffix.lower() in VIDEO_SUFFIXES:
        if not hasattr(torchvision.io, "read_video"):
            raise RuntimeError(
                "video audio decoding requires torchvision.io.read_video; install "
                "the pinned release environment"
            )
        _, audio, info = torchvision.io.read_video(
            str(path), pts_unit="sec", output_format="TCHW"
        )
        sample_rate = int(info.get("audio_fps") or 0)
        if sample_rate <= 0 or audio.numel() == 0:
            raise ValueError(f"failed to decode a valid audio stream from {path}")
        waveform = _mono(audio)
    else:
        audio, sample_rate = torchaudio.load(str(path))
        waveform = _mono(audio)

    if sample_rate <= 0 or waveform.numel() == 0:
        raise ValueError(f"failed to decode a valid audio stream from {path}")
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
        sample_rate = target_rate
    return AudioWaveform(waveform=waveform.contiguous(), sample_rate=sample_rate)


def load_audio_ffmpeg(
    path: Union[Path, str],
    target_rate: int = 16_000,
    ffmpeg: Optional[str] = None,
) -> AudioWaveform:
    """Decode through FFmpeg's mono 16 kHz signed-16 PCM path."""

    import torch

    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    executable = ffmpeg or shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required for paper_legacy audio decoding")
    path = Path(path)
    completed = subprocess.run(
        [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(target_rate),
            "-acodec",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode audio from {path}: {detail[-1000:]}")
    if not completed.stdout:
        raise ValueError(f"ffmpeg decoded an empty audio stream from {path}")
    waveform = (
        torch.frombuffer(bytearray(completed.stdout), dtype=torch.int16)
        .to(torch.float32)
        .div_(32768.0)
    )
    return AudioWaveform(waveform=waveform, sample_rate=target_rate)


def non_overlapping_starts(duration: float, segment_seconds: float) -> List[float]:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    count = int(math.floor((duration + 1e-9) / segment_seconds))
    return [index * segment_seconds for index in range(count)]


def fixed_window_starts(
    duration: float,
    window_seconds: float,
    policy: str,
    stride_seconds: Optional[float] = None,
) -> List[float]:
    """Create deterministic starts for a fixed-length classifier.

    This is an AV-SyncBench adapter policy, not an upstream Synchformer policy.
    """

    if duration + 1e-9 < window_seconds:
        raise ValueError(
            f"common duration {duration:.3f}s is shorter than required window "
            f"{window_seconds:.3f}s"
        )
    last = max(0.0, duration - window_seconds)
    if policy == "first":
        return [0.0]
    if policy == "center":
        return [last / 2.0]
    if policy != "sliding":
        raise ValueError(f"unknown window policy: {policy}")
    stride = stride_seconds or window_seconds
    if stride <= 0:
        raise ValueError("stride_seconds must be positive")
    starts: List[float] = []
    current = 0.0
    while current <= last + 1e-9:
        starts.append(current)
        current += stride
    if not starts or abs(starts[-1] - last) > 1e-6:
        starts.append(last)
    return starts


def padded_sliding_starts(
    duration: float,
    *,
    window_seconds: float = 5.0,
    stride_seconds: float = 0.64,
    round_duration_to: float = 5.0,
) -> Tuple[float, List[float]]:
    """Return the repository classifier's padded sliding-window schedule."""

    if duration < 0:
        raise ValueError("duration must be non-negative")
    if window_seconds <= 0 or stride_seconds <= 0 or round_duration_to <= 0:
        raise ValueError("window, stride, and rounding values must be positive")
    target_duration = max(
        math.ceil(duration / round_duration_to) * round_duration_to,
        window_seconds,
    )
    starts = [
        round(index * stride_seconds, 8)
        for index in range(int(math.ceil(target_duration / stride_seconds)))
        if index * stride_seconds < target_duration
    ]
    return target_duration, starts
