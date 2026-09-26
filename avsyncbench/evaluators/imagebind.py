"""Evaluate ImageBind with AV-SyncBench's 0.64-second pairwise protocol.

The default `paper_legacy` preprocessing tracks the audited author-side result
runner. The explicit `official_0p64` alternative uses upstream-style video
transforms and audio features. Both reuse the official ImageBind model and
vendor neither upstream code nor weights. Video decoding uses Decord in both
modes.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F

from avsyncbench.manifest import (
    ManifestRecord,
    iter_manifest,
    validate_audio_negative_records,
    validate_unique_ids,
)
from avsyncbench.media import (
    AudioWaveform,
    VideoFrames,
    load_audio_ffmpeg,
    load_video_decord,
    non_overlapping_starts,
)
from avsyncbench.metrics import PairwiseResult, summarize_results, write_jsonl, write_summary
from avsyncbench.runtime import (
    collect_runtime,
    file_digests,
    require_distinct_paths,
    require_package_versions,
    require_paths_not_in,
    require_upstream_checkout,
    sanitized_exception,
    verify_artifact,
    write_run_metadata,
)


IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
AUDIO_MEAN = -4.268
AUDIO_STD = 9.138
PINNED_IMAGEBIND_COMMIT = "53680b02d7e37b19b124fa37bae4b6c98c38f5be"
OFFICIAL_CHECKPOINT_SIZE = 4_803_584_173


def _uniform_temporal_subsample(clip: torch.Tensor, samples: int = 2) -> torch.Tensor:
    """Select evenly spaced frames from a C,T,H,W tensor."""

    if clip.ndim != 4 or clip.shape[1] <= 0 or samples <= 0:
        raise ValueError("expected a non-empty C,T,H,W clip and a positive sample count")
    indices = torch.linspace(0, clip.shape[1] - 1, samples).long()
    return torch.index_select(clip, 1, indices)


def _short_side_scale(clip: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Resize a C,T,H,W tensor while preserving its aspect ratio."""

    if clip.ndim != 4 or size <= 0:
        raise ValueError("expected a C,T,H,W clip and a positive target size")
    _, _, height, width = clip.shape
    if height <= 0 or width <= 0:
        raise ValueError("video frames must have positive spatial dimensions")
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        new_width = size
    else:
        new_height = size
        new_width = int(math.floor((float(width) / height) * size))
    return F.interpolate(
        clip,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )


def _three_spatial_crops(clip: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Return upstream-style leading, center, and trailing square crops."""

    if clip.ndim != 4 or size <= 0:
        raise ValueError("expected a C,T,H,W clip and a positive crop size")
    _, _, height, width = clip.shape
    if height < size or width < size:
        raise ValueError(
            f"cannot take a {size} x {size} crop from {height} x {width} frames"
        )
    if height > width:
        offsets = (0, int(math.ceil((height - size) / 2)), height - size)
        crops = [clip[:, :, offset : offset + size, :] for offset in offsets]
    else:
        offsets = (0, int(math.ceil((width - size) / 2)), width - size)
        crops = [clip[:, :, :, offset : offset + size] for offset in offsets]
    return torch.stack(crops, dim=0)


class ImageBindScorer:
    def __init__(
        self,
        upstream_root: Path,
        checkpoint: Path,
        device: str = "cuda:0",
        batch_size: int = 4,
        segment_seconds: float = 0.64,
        sample_rate: int = 16_000,
        preprocess_mode: str = "paper_legacy",
        max_segments: int = 18,
        positive_audio_cache_size: int = 8,
    ) -> None:
        if (
            batch_size <= 0
            or segment_seconds <= 0
            or sample_rate <= 0
            or max_segments <= 0
            or positive_audio_cache_size < 0
        ):
            raise ValueError(
                "batch_size, segment_seconds, sample_rate, and max_segments must be "
                "positive; positive_audio_cache_size must be non-negative"
            )
        if preprocess_mode not in {"paper_legacy", "official_0p64"}:
            raise ValueError(f"unknown preprocess_mode: {preprocess_mode}")
        require_package_versions(
            {
                "torch": "2.10.0",
                "torchvision": "0.25.0",
                "torchaudio": "2.10.0",
                "numpy": "1.26.4",
                "decord": "0.6.0",
                "timm": "1.0.30",
            }
        )
        upstream_root = require_upstream_checkout(
            upstream_root,
            expected_commit=PINNED_IMAGEBIND_COMMIT,
            required_files=(
                "LICENSE",
                "imagebind/models/imagebind_model.py",
            ),
        )
        checkpoint = checkpoint.expanduser().resolve()
        # Upstream publishes the byte size but no cryptographic digest. The
        # downloader uses the official HTTPS URL; only load that trusted file.
        self.checkpoint_digests = verify_artifact(
            checkpoint,
            expected_size=OFFICIAL_CHECKPOINT_SIZE,
        )
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))

        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType

        self.ModalityType = ModalityType
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.segment_seconds = segment_seconds
        self.sample_rate = sample_rate
        self.preprocess_mode = preprocess_mode
        self.max_segments = max_segments
        self.positive_audio_cache_size = positive_audio_cache_size
        self._positive_audio_cache: "OrderedDict[Path, AudioWaveform]" = OrderedDict()
        import torchaudio

        self.torchaudio = torchaudio
        if preprocess_mode == "paper_legacy":
            self.legacy_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=128,
                f_min=0.0,
                f_max=sample_rate / 2,
                power=2.0,
            )

        self.model = imagebind_model.imagebind_huge(pretrained=False)
        try:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "this adapter requires a PyTorch version whose torch.load supports "
                "weights_only=True; upgrade PyTorch instead of using unsafe pickle loading"
            ) from exc
        self.model.load_state_dict(state_dict, strict=True)
        del state_dict
        self.model.eval().to(self.device)

    def _load_pair_audio(self, path: Path, *, cache_positive: bool) -> AudioWaveform:
        key = path.expanduser().resolve()
        if cache_positive and self.positive_audio_cache_size:
            cached = self._positive_audio_cache.get(key)
            if cached is not None:
                self._positive_audio_cache.move_to_end(key)
                return cached
        audio = load_audio_ffmpeg(key, target_rate=self.sample_rate)
        if cache_positive and self.positive_audio_cache_size:
            self._positive_audio_cache[key] = audio
            self._positive_audio_cache.move_to_end(key)
            while len(self._positive_audio_cache) > self.positive_audio_cache_size:
                self._positive_audio_cache.popitem(last=False)
        return audio

    @staticmethod
    def _official_frame_bounds(
        start: float,
        segment_seconds: float,
        fps: float,
    ) -> tuple:
        """Match Decord's inclusive-start/exclusive-end clip boundaries.

        The epsilon removes positive binary rounding residue from exact frame
        boundaries such as ``7 * 0.64 * 25`` without changing real fractional
        boundaries.
        """

        return (
            int(math.ceil(start * fps - 1e-9)),
            int(math.ceil((start + segment_seconds) * fps - 1e-9)),
        )

    def _video_feature(self, video: VideoFrames, start: float) -> torch.Tensor:
        if self.preprocess_mode == "paper_legacy":
            frame_count = video.frames.shape[0]
            first = max(0, min(frame_count - 1, int(round(start * video.fps))))
            final = max(
                0,
                min(
                    frame_count - 1,
                    int(round((start + self.segment_seconds - 1e-3) * video.fps)),
                ),
            )
            clip = video.frames[[first, final]].float() / 255.0  # T=2, C, H, W
            clip = F.interpolate(
                clip,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            mean = torch.as_tensor(IMAGE_MEAN).view(1, 3, 1, 1)
            std = torch.as_tensor(IMAGE_STD).view(1, 3, 1, 1)
            clip = ((clip - mean) / std).permute(1, 0, 2, 3)
            return clip.unsqueeze(0)  # one crop, C, T=2, H, W

        first, last = self._official_frame_bounds(
            start,
            self.segment_seconds,
            video.fps,
        )
        clip = video.frames[first:last]
        if clip.shape[0] < 2:
            raise ValueError(f"video segment at {start:.3f}s has fewer than two frames")
        clip = clip.permute(1, 0, 2, 3).float()
        clip = _uniform_temporal_subsample(clip, 2) / 255.0
        clip = _short_side_scale(clip, 224)
        mean = torch.as_tensor(IMAGE_MEAN).view(3, 1, 1, 1)
        std = torch.as_tensor(IMAGE_STD).view(3, 1, 1, 1)
        clip = (clip - mean) / std
        return _three_spatial_crops(clip, 224)  # 3, C, T=2, H, W

    def _audio_feature(self, audio: AudioWaveform, start: float) -> torch.Tensor:
        first = int(round(start * audio.sample_rate))
        last = int(round((start + self.segment_seconds) * audio.sample_rate))
        waveform = audio.waveform[first:last].clone().unsqueeze(0)
        expected = int(round(self.segment_seconds * audio.sample_rate))
        if waveform.shape[-1] != expected:
            raise ValueError(
                f"audio segment at {start:.3f}s has {waveform.shape[-1]} samples; expected {expected}"
            )
        if self.preprocess_mode == "paper_legacy":
            mel = torch.log(self.legacy_mel(waveform) + 1e-6)
            difference = 204 - mel.shape[-1]
            if difference > 0:
                mel = F.pad(mel, (0, difference))
            elif difference < 0:
                mel = mel[:, :, :204]
        else:
            centered = waveform - waveform.mean()
            mel = self.torchaudio.compliance.kaldi.fbank(
                centered,
                htk_compat=True,
                sample_frequency=audio.sample_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_length=25,
                frame_shift=10,
            ).transpose(0, 1)
            difference = 204 - mel.shape[-1]
            if difference > 0:
                mel = F.pad(mel, (0, difference))
            elif difference < 0:
                mel = mel[:, :204]
            mel = mel.unsqueeze(0)
            mel = (mel - AUDIO_MEAN) / AUDIO_STD
        return mel.unsqueeze(0)  # S=1, C=1, mel=128, time=204

    @torch.inference_mode()
    def _encode_video(self, video: VideoFrames, starts: Sequence[float]) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for index in range(0, len(starts), self.batch_size):
            batch_starts = starts[index : index + self.batch_size]
            batch = torch.stack([self._video_feature(video, start) for start in batch_starts])
            embeddings = self.model(
                {self.ModalityType.VISION: batch.to(self.device, non_blocking=True)}
            )[self.ModalityType.VISION]
            outputs.append(F.normalize(embeddings.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def _encode_audio(self, audio: AudioWaveform, starts: Sequence[float]) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for index in range(0, len(starts), self.batch_size):
            batch_starts = starts[index : index + self.batch_size]
            batch = torch.stack([self._audio_feature(audio, start) for start in batch_starts])
            embeddings = self.model(
                {self.ModalityType.AUDIO: batch.to(self.device, non_blocking=True)}
            )[self.ModalityType.AUDIO]
            outputs.append(F.normalize(embeddings.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    def score_record(self, record: ManifestRecord) -> PairwiseResult:
        if record.negative_audio is None:
            raise ValueError(
                f"{record.sample_id}: ImageBind pairwise evaluation requires negative audio"
            )
        video = load_video_decord(record.original_video)
        positive_audio = self._load_pair_audio(
            record.original_audio or record.original_video,
            cache_positive=True,
        )
        negative_audio = self._load_pair_audio(record.negative_audio, cache_positive=False)
        common_duration = min(video.duration, positive_audio.duration, negative_audio.duration)
        starts = non_overlapping_starts(common_duration, self.segment_seconds)
        if self.preprocess_mode == "paper_legacy" and len(starts) > self.max_segments:
            indices = (
                torch.linspace(0, len(starts) - 1, steps=self.max_segments)
                .round()
                .long()
                .tolist()
            )
            starts = [starts[index] for index in indices]
        if not starts:
            raise ValueError(
                f"{record.sample_id}: common duration {common_duration:.3f}s is shorter "
                f"than one {self.segment_seconds:.3f}s segment"
            )

        visual = self._encode_video(video, starts)
        positive = self._encode_audio(positive_audio, starts)
        negative = self._encode_audio(negative_audio, starts)
        positive_segments = (visual * positive).sum(dim=-1)
        negative_segments = (visual * negative).sum(dim=-1)
        positive_score = float(positive_segments.mean())
        negative_score = float(negative_segments.mean())
        if not math.isfinite(positive_score) or not math.isfinite(negative_score):
            raise FloatingPointError(
                f"{record.sample_id}: model produced a non-finite pairwise score"
            )
        return PairwiseResult(
            sample_id=record.sample_id,
            task=record.task,
            positive_score=positive_score,
            negative_score=negative_score,
            edit_type=record.edit_type,
            offset_ms=record.offset_ms,
            direction=record.direction,
            category=record.category,
            metadata={
                "segment_seconds": self.segment_seconds,
                "segment_count": len(starts),
                "preprocess_mode": self.preprocess_mode,
                "max_segments": (
                    self.max_segments if self.preprocess_mode == "paper_legacy" else None
                ),
                "segment_starts": starts,
                "positive_segment_scores": positive_segments.tolist(),
                "negative_segment_scores": negative_segments.tolist(),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--imagebind-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=0.64)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--preprocess-mode",
        choices=("paper_legacy", "official_0p64"),
        default="paper_legacy",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=18,
        help="paper_legacy only; official_0p64 always uses all complete segments",
    )
    parser.add_argument("--positive-audio-cache-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--edit-type",
        action="append",
        dest="edit_types",
        help="evaluate only this manifest edit_type; repeat for multiple values",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    metadata_path = args.run_metadata or args.output.with_suffix(".run.json")
    require_distinct_paths(
        {
            "manifest": args.manifest,
            "checkpoint": args.checkpoint,
            "output": args.output,
            "summary": summary_path,
            "run_metadata": metadata_path,
        }
    )
    records = list(
        iter_manifest(
            args.manifest,
            args.dataset_root,
            limit=args.limit,
            edit_types=args.edit_types,
        )
    )
    if not records:
        raise SystemExit(
            "no records matched the requested manifest/filter; check --dataset-root and --edit-type"
        )
    validate_unique_ids(records)
    validate_audio_negative_records(records)
    require_paths_not_in(
        {
            "output": args.output,
            "summary": summary_path,
            "run_metadata": metadata_path,
        },
        (
            path
            for record in records
            for path in (
                record.original_video,
                record.original_audio,
                record.negative_media,
            )
            if path is not None
        ),
    )
    scorer = ImageBindScorer(
        upstream_root=args.imagebind_root,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        preprocess_mode=args.preprocess_mode,
        max_segments=args.max_segments,
        positive_audio_cache_size=args.positive_audio_cache_size,
    )
    results: List[PairwiseResult] = []
    for index, record in enumerate(records, start=1):
        try:
            result = scorer.score_record(record)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            result = PairwiseResult(
                sample_id=record.sample_id,
                task=record.task,
                positive_score=None,
                negative_score=None,
                edit_type=record.edit_type,
                offset_ms=record.offset_ms,
                direction=record.direction,
                category=record.category,
                error=sanitized_exception(
                    exc,
                    {
                        "dataset_root": args.dataset_root,
                        "imagebind_root": args.imagebind_root,
                    },
                ),
            )
        results.append(result)
        print(
            f"[{index}] {record.sample_id}: "
            + ("ERROR " + result.error if result.error else f"correct={result.correct}")
        )

    summary = summarize_results(results)
    write_jsonl(args.output, results)
    write_summary(summary_path, summary)
    write_run_metadata(
        metadata_path,
        collect_runtime(
            model_name="ImageBind-Huge",
            upstream_root=args.imagebind_root,
            checkpoint=args.checkpoint,
            checkpoint_digests=scorer.checkpoint_digests,
            extra={
                "pinned_upstream_commit": PINNED_IMAGEBIND_COMMIT,
                "manifest": args.manifest.name,
                "manifest_size": args.manifest.stat().st_size,
                "manifest_sha256": file_digests(args.manifest, ("sha256",))["sha256"],
                "dataset_root": args.dataset_root.name,
                "edit_type_filter": args.edit_types,
                "limit": args.limit,
                "continue_on_error": args.continue_on_error,
                "device": args.device,
                "batch_size": args.batch_size,
                "segment_seconds": args.segment_seconds,
                "segment_stride_seconds": args.segment_seconds,
                "sample_rate": args.sample_rate,
                "preprocess_mode": args.preprocess_mode,
                "max_segments": (
                    args.max_segments if args.preprocess_mode == "paper_legacy" else None
                ),
                "positive_audio_cache_size": args.positive_audio_cache_size,
                "video_decoder": "Decord 0.6.0 (CPU, one decode thread)",
                "audio_decoder": "FFmpeg mono 16 kHz signed-16 PCM",
                "video_frames_per_segment": 2,
                "video_resize": (
                    "warp 224 x 224"
                    if args.preprocess_mode == "paper_legacy"
                    else "short side 224"
                ),
                "video_spatial_crops": (
                    1 if args.preprocess_mode == "paper_legacy" else 3
                ),
                "video_normalization_mean": IMAGE_MEAN,
                "video_normalization_std": IMAGE_STD,
                "audio_features": (
                    "MelSpectrogram(n_fft=400, hop=160), log, 128 x 204"
                    if args.preprocess_mode == "paper_legacy"
                    else "Kaldi fbank 128 x 204"
                ),
                "audio_normalization_mean": (
                    None if args.preprocess_mode == "paper_legacy" else AUDIO_MEAN
                ),
                "audio_normalization_std": (
                    None if args.preprocess_mode == "paper_legacy" else AUDIO_STD
                ),
                "remainder_policy": "drop incomplete final segment",
                "score": "mean diagonal cosine over non-overlapping segments",
                "tie_policy": "strict positive > negative; ties incorrect",
            },
        ),
    )
    print(json.dumps(summary, indent=2))
    if summary["overall"]["errors"]:
        raise SystemExit(
            "evaluation completed with errors; outputs were written, but this run "
            "must not be used for a paper comparison"
        )


if __name__ == "__main__":
    main()
