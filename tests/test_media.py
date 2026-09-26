import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # lightweight manifest-only environments
    torch = None

from avsyncbench.media import (
    VideoFrames,
    _mono,
    fixed_window_starts,
    load_audio,
    load_audio_ffmpeg,
    load_video_decord,
    padded_sliding_starts,
)


class WindowTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_empty_wave_file_is_rejected(self) -> None:
        fake_torchaudio = SimpleNamespace(
            load=lambda path: (torch.empty((1, 0)), 16_000),
            functional=SimpleNamespace(),
        )
        fake_torchvision = SimpleNamespace(io=SimpleNamespace())
        with patch.dict(
            sys.modules,
            {"torchaudio": fake_torchaudio, "torchvision": fake_torchvision},
        ):
            with self.assertRaisesRegex(ValueError, "valid audio stream"):
                load_audio("empty.wav")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_ffmpeg_signed_16_pcm_is_normalized(self) -> None:
        completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": struct.pack("<hhh", -32768, 0, 32767),
                "stderr": b"",
            },
        )()
        with patch("avsyncbench.media.shutil.which", return_value="ffmpeg"), patch(
            "avsyncbench.media.subprocess.run", return_value=completed
        ):
            audio = load_audio_ffmpeg("example.mp4")
        self.assertEqual(audio.sample_rate, 16_000)
        self.assertTrue(
            torch.allclose(
                audio.waveform,
                torch.tensor([-1.0, 0.0, 32767 / 32768]),
            )
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_mono_accepts_channel_first_and_channel_last(self) -> None:
        channel_first = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        expected = channel_first.mean(dim=0)
        self.assertTrue(torch.equal(_mono(channel_first), expected))
        self.assertTrue(torch.equal(_mono(channel_first.transpose(0, 1)), expected))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_imagebind_legacy_video_feature_shape(self) -> None:
        from avsyncbench.evaluators.imagebind import ImageBindScorer

        scorer = ImageBindScorer.__new__(ImageBindScorer)
        scorer.preprocess_mode = "paper_legacy"
        scorer.segment_seconds = 0.64
        video = VideoFrames(
            frames=torch.zeros((20, 3, 256, 320), dtype=torch.uint8),
            fps=25.0,
        )
        feature = scorer._video_feature(video, 0.0)
        self.assertEqual(tuple(feature.shape), (1, 3, 2, 224, 224))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_imagebind_official_frame_bounds_use_ceil(self) -> None:
        from avsyncbench.evaluators.imagebind import ImageBindScorer

        self.assertEqual(
            ImageBindScorer._official_frame_bounds(0.64, 0.64, 30.0),
            (20, 39),
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_imagebind_official_frame_bounds_ignore_float_residue(self) -> None:
        from avsyncbench.evaluators.imagebind import ImageBindScorer

        self.assertEqual(
            ImageBindScorer._official_frame_bounds(5 * 0.64, 0.64, 25.0),
            (80, 96),
        )
        self.assertEqual(
            ImageBindScorer._official_frame_bounds(7 * 0.64, 0.64, 25.0),
            (112, 128),
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_decord_video_loader_returns_tchw_rgb(self) -> None:
        array = torch.zeros((3, 8, 12, 3), dtype=torch.uint8)

        class FakeReader:
            def __init__(self, path, ctx, num_threads):
                self.path = path
                self.ctx = ctx
                self.num_threads = num_threads

            def __len__(self):
                return 3

            def get_avg_fps(self):
                return 25.0

            def get_batch(self, indices):
                self.indices = indices
                return array

        fake_decord = SimpleNamespace(
            VideoReader=FakeReader,
            cpu=lambda index: ("cpu", index),
        )
        with patch.dict(sys.modules, {"decord": fake_decord}):
            video = load_video_decord("example.mp4")
        self.assertEqual(tuple(video.frames.shape), (3, 3, 8, 12))
        self.assertEqual(video.frames.dtype, torch.uint8)
        self.assertEqual(video.fps, 25.0)

    def test_repository_sliding_rounds_and_pads(self) -> None:
        target, starts = padded_sliding_starts(6.2)
        self.assertEqual(target, 10.0)
        self.assertEqual(starts[0], 0.0)
        self.assertEqual(starts[-1], 9.6)
        self.assertEqual(len(starts), 16)

    def test_valid_sliding_includes_last_window(self) -> None:
        starts = fixed_window_starts(8.0, 5.0, "sliding", 5.0)
        self.assertEqual(starts, [0.0, 3.0])


if __name__ == "__main__":
    unittest.main()
