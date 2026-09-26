"""Evaluate the Synchformer VGGSound checkpoint on AV-SyncBench Global Offset.

The synchronization score is softmax(logits)[zero-offset class]. Each manifest
record is correct only when the original pair receives a strictly higher score
than its pre-generated offset negative. The manifest offset is never applied a
second time inside the upstream transform.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from avsyncbench.manifest import (
    ManifestRecord,
    iter_manifest,
    validate_audio_negative_records,
    validate_unique_ids,
)
from avsyncbench.media import (
    AudioWaveform,
    VideoFrames,
    fixed_window_starts,
    load_audio,
    load_video,
    padded_sliding_starts,
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


PINNED_SYNCHFORMER_COMMIT = "b66668a1521d7567cc760e5544b2b5b53179b687"
OFFICIAL_CONFIG_SIZE = 6_878
OFFICIAL_CONFIG_MD5 = "606ca81ca850a220009fde67233f89a4"
OFFICIAL_CHECKPOINT_SIZE = 1_131_154_181
OFFICIAL_CHECKPOINT_MD5 = "19592ed5e64e3560b93772c5d46e3334"


class SynchformerScorer:
    def __init__(
        self,
        upstream_root: Path,
        config: Path,
        checkpoint: Path,
        device: str = "cuda:0",
        batch_size: int = 1,
        use_amp: Optional[bool] = None,
        window_policy: str = "repository_sliding",
        window_stride: float = 0.64,
        round_duration_to: float = 5.0,
        max_input_seconds: float = 10.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if window_stride <= 0 or round_duration_to <= 0 or max_input_seconds <= 0:
            raise ValueError(
                "window_stride, round_duration_to, and max_input_seconds must be positive"
            )
        require_package_versions(
            {
                "torch": "2.0.0",
                "torchvision": "0.15.0",
                "torchaudio": "2.0.0",
                "av": "9.0.0",
                "numpy": "1.23.5",
            }
        )
        upstream_root = require_upstream_checkout(
            upstream_root,
            expected_commit=PINNED_SYNCHFORMER_COMMIT,
            required_files=(
                "example.py",
                "dataset/transforms.py",
                "scripts/train_utils.py",
            ),
        )
        config = config.expanduser().resolve()
        checkpoint = checkpoint.expanduser().resolve()
        self.config_digests = verify_artifact(
            config,
            expected_size=OFFICIAL_CONFIG_SIZE,
            expected_md5=OFFICIAL_CONFIG_MD5,
        )
        self.checkpoint_digests = verify_artifact(
            checkpoint,
            expected_size=OFFICIAL_CHECKPOINT_SIZE,
            expected_md5=OFFICIAL_CHECKPOINT_MD5,
        )
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))

        from dataset.transforms import make_class_grid
        from omegaconf import OmegaConf
        from scripts.train_utils import get_model, get_transforms, prepare_inputs

        self.prepare_inputs = prepare_inputs
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.window_policy = window_policy
        self.window_stride = window_stride
        self.round_duration_to = round_duration_to
        self.max_input_seconds = max_input_seconds

        cfg = OmegaConf.load(config)
        # The Stage-II checkpoint already contains both feature extractors.
        cfg.model.params.afeat_extractor.params.ckpt_path = None
        cfg.model.params.vfeat_extractor.params.ckpt_path = None
        cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
            ".modules.feature_selector.", ".sync_model."
        )
        self.cfg = cfg
        self.transform = get_transforms(cfg, ["test"])["test"]
        _, self.model = get_model(cfg, self.device)
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state["model"], strict=True)
        del state
        self.model.eval()

        max_offset = float(cfg.data.max_off_sec)
        class_count = int(cfg.model.params.transformer.params.off_head_cfg.params.out_features)
        grid = torch.as_tensor(make_class_grid(-max_offset, max_offset, class_count)).float()
        self.grid = grid
        self.zero_index = int(torch.argmin(torch.abs(grid)).item())
        if abs(float(grid[self.zero_index])) > 1e-8:
            raise ValueError(f"offset class grid does not contain zero: {grid.tolist()}")

        cfg_amp = bool(cfg.training.get("use_half_precision", False))
        self.use_amp = cfg_amp if use_amp is None else use_amp
        if self.use_amp and self.device.type != "cuda":
            raise ValueError("AMP is only supported for a CUDA device in this adapter")

    @staticmethod
    def _validate_video(video: VideoFrames, path: Path) -> None:
        if abs(video.fps - 25.0) > 1e-3:
            raise ValueError(f"{path}: expected 25 FPS, got {video.fps}")
        short_side = min(video.frames.shape[-2:])
        if short_side != 256:
            raise ValueError(
                f"{path}: expected preprocessed short side 256, got {short_side}; "
                "run scripts/prepare_media.py first"
            )

    def _build_item(
        self,
        video: VideoFrames,
        audio: AudioWaveform,
        start: float,
        label: str,
    ) -> dict:
        target_video_frames = int(round(5.0 * video.fps))
        target_audio_frames = int(round(5.0 * audio.sample_rate))
        video_start = int(round(start * video.fps))
        audio_start = int(round(start * audio.sample_rate))
        video_clip = video.frames[video_start : video_start + target_video_frames]
        audio_clip = audio.waveform[audio_start : audio_start + target_audio_frames]
        if video_clip.shape[0] == 0:
            video_clip = video.frames[-1:].clone()
        if video_clip.shape[0] < target_video_frames:
            padding = video_clip[-1:].repeat(
                target_video_frames - video_clip.shape[0], 1, 1, 1
            )
            video_clip = torch.cat([video_clip, padding], dim=0)
        if audio_clip.shape[0] < target_audio_frames:
            audio_clip = torch.nn.functional.pad(
                audio_clip, (0, target_audio_frames - audio_clip.shape[0])
            )
        # Both values must be present. Omitting both makes the upstream test
        # transform randomly sample an offset and a crop.
        return {
            "video": video_clip,
            "audio": audio_clip,
            "meta": {
                "video": {"fps": [video.fps]},
                "audio": {"framerate": [audio.sample_rate]},
            },
            "path": label,
            "split": "test",
            "targets": {"v_start_i_sec": 0.0, "offset_sec": 0.0},
        }

    @torch.inference_mode()
    def _score_windows(
        self,
        video: VideoFrames,
        audio: AudioWaveform,
        starts: Sequence[float],
        label: str,
    ) -> List[float]:
        scores: List[float] = []
        for index in range(0, len(starts), self.batch_size):
            batch_starts = starts[index : index + self.batch_size]
            items = [
                self.transform(self._build_item(video, audio, start, label))
                for start in batch_starts
            ]
            batch = torch.utils.data.default_collate(items)
            aud, vid, _ = self.prepare_inputs(batch, self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                _, logits = self.model(vid, aud)
            p_zero = torch.softmax(logits.float(), dim=-1)[:, self.zero_index]
            scores.extend(float(value) for value in p_zero.cpu())
        return scores

    def score_record(self, record: ManifestRecord) -> PairwiseResult:
        if record.negative_audio is None:
            raise ValueError(
                f"{record.sample_id}: Synchformer pairwise evaluation requires negative audio"
            )
        positive_audio = load_audio(record.original_audio or record.original_video, 16_000)
        negative_audio = load_audio(record.negative_audio, 16_000)
        video = load_video(record.original_video)
        self._validate_video(video, record.original_video)
        common_duration = min(video.duration, positive_audio.duration, negative_audio.duration)
        window_seconds = 5.0
        if self.window_policy == "repository_sliding":
            # Mirrors the repository's classifier protocol: start a 5 s model
            # window every 0.64 s and pad the tail (video=last frame,
            # audio=zeros), then average p(delta=0) across windows.
            target_duration, starts = padded_sliding_starts(
                common_duration,
                window_seconds=window_seconds,
                stride_seconds=self.window_stride,
                round_duration_to=self.round_duration_to,
            )
        else:
            target_duration = min(common_duration, self.max_input_seconds)
            starts = fixed_window_starts(
                target_duration,
                window_seconds,
                self.window_policy,
                self.window_stride,
            )
        positive_windows = self._score_windows(
            video, positive_audio, starts, f"{record.sample_id}:positive"
        )
        negative_windows = self._score_windows(
            video, negative_audio, starts, f"{record.sample_id}:negative"
        )
        positive_score = sum(positive_windows) / len(positive_windows)
        negative_score = sum(negative_windows) / len(negative_windows)
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
                "window_seconds": window_seconds,
                "window_policy": self.window_policy,
                "window_stride": self.window_stride,
                "round_duration_to": self.round_duration_to,
                "common_duration": common_duration,
                "target_duration": target_duration,
                "tail_padding": self.window_policy == "repository_sliding",
                "aggregation": "mean_probability",
                "window_starts": starts,
                "positive_p_zero": positive_windows,
                "negative_p_zero": negative_windows,
                "zero_class_index": self.zero_index,
                "offset_class_grid_seconds": self.grid.tolist(),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--synchformer-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    parser.add_argument(
        "--window-policy",
        choices=("repository_sliding", "first", "center", "sliding"),
        default="repository_sliding",
    )
    parser.add_argument("--window-stride", type=float, default=0.64)
    parser.add_argument("--round-duration-to", type=float, default=5.0)
    parser.add_argument("--max-input-seconds", type=float, default=10.0)
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
            "config": args.config,
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
    scorer = SynchformerScorer(
        upstream_root=args.synchformer_root,
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        use_amp=args.amp,
        window_policy=args.window_policy,
        window_stride=args.window_stride,
        round_duration_to=args.round_duration_to,
        max_input_seconds=args.max_input_seconds,
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
                        "synchformer_root": args.synchformer_root,
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
            model_name="Synchformer VGGSound 24-01-02T10-00-53",
            upstream_root=args.synchformer_root,
            checkpoint=args.checkpoint,
            config=args.config,
            checkpoint_digests=scorer.checkpoint_digests,
            config_digests=scorer.config_digests,
            extra={
                "pinned_upstream_commit": PINNED_SYNCHFORMER_COMMIT,
                "manifest": args.manifest.name,
                "manifest_size": args.manifest.stat().st_size,
                "manifest_sha256": file_digests(args.manifest, ("sha256",))["sha256"],
                "dataset_root": args.dataset_root.name,
                "edit_type_filter": args.edit_types,
                "limit": args.limit,
                "continue_on_error": args.continue_on_error,
                "device": args.device,
                "batch_size": args.batch_size,
                "amp": scorer.use_amp,
                "video_fps": 25,
                "audio_sample_rate": 16_000,
                "window_policy": args.window_policy,
                "window_stride": args.window_stride,
                "round_duration_to": args.round_duration_to,
                "max_input_seconds": args.max_input_seconds,
                "tail_padding": args.window_policy == "repository_sliding",
                "window_aggregation": "mean_probability",
                "offset_class_grid_seconds": scorer.grid.tolist(),
                "zero_class_index": scorer.zero_index,
                "score": "softmax(offset_logits)[zero_class_index]",
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
