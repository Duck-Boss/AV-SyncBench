import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from avsyncbench.manifest import iter_manifest, validate_audio_negative_records
from scripts.prepare_media import stable_name
from scripts.validate_manifest import main as validate_manifest_main


class ManifestTests(unittest.TestCase):
    def test_stable_name_does_not_depend_on_dataset_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            relative = Path("video") / "same.mp4"
            (first_root / relative).parent.mkdir()
            (second_root / relative).parent.mkdir()
            (first_root / relative).write_bytes(b"first")
            (second_root / relative).write_bytes(b"second")
            self.assertEqual(
                stable_name(first_root / relative, first_root),
                stable_name(second_root / relative, second_root),
            )

    def test_sample_id_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v.mp4").touch()
            (root / "n.wav").touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "original_video": "v.mp4",
                        "edited_audio": "n.wav",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample_id"):
                list(iter_manifest(manifest, root))

    def test_manifest_resolves_paths_and_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "videos").mkdir()
            (root / "audio").mkdir()
            (root / "videos" / "a.mp4").touch()
            (root / "audio" / "a_shift.wav").touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "a+050",
                        "original_video": "videos/a.mp4",
                        "edited_audio": "audio/a_shift.wav",
                        "task_type": "global_offset",
                        "edit_type": "time_shifted",
                        "target_attribute": "50ms advance",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            record = next(iter_manifest(manifest, root))
            self.assertEqual(record.sample_id, "a+050")
            self.assertEqual(record.original_video, root / "videos" / "a.mp4")
            self.assertEqual(record.offset_ms, 50)
            self.assertEqual(record.edit_type, "time_shifted")
            self.assertEqual(record.direction, "advance")

    def test_offset_parser_ignores_date_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v.mp4").touch()
            (root / "clip__delay_300ms.wav").touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "dated",
                        "original_video": "v.mp4",
                        "edited_audio": "clip__delay_300ms.wav",
                        "target_attribute": "Original Partition: 20260203_sampled_0_time_shifted",
                        "edit_type": "time_shifted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            record = next(iter_manifest(manifest, root))
            self.assertEqual(record.offset_ms, 300)
            self.assertEqual(record.direction, "delay")

    def test_absolute_paths_are_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "v.mp4"
            audio = root / "n.wav"
            video.touch()
            audio.touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "absolute",
                        "original_video": str(video.resolve()),
                        "edited_audio": str(audio.resolve()),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "absolute manifest paths are disabled"):
                list(iter_manifest(manifest, root))

    def test_two_negative_media_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("v.mp4", "n.wav", "n.mp4"):
                (root / name).touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "ambiguous",
                        "original_video": "v.mp4",
                        "edited_audio": "n.wav",
                        "edited_video": "n.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one negative medium"):
                list(iter_manifest(manifest, root))

    def test_audio_evaluator_rejects_video_negative_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v.mp4").touch()
            (root / "negative.mp4").touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "visual-negative",
                        "original_video": "v.mp4",
                        "edited_video": "negative.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = list(iter_manifest(manifest, root))
            with self.assertRaisesRegex(ValueError, "requires edited_audio"):
                validate_audio_negative_records(records)

    def test_limit_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "limit must be a positive integer"):
                list(iter_manifest(manifest, root, limit=0))

    def test_validator_rejects_offset_outside_exact_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "original_video": "v.mp4",
                            "edited_audio": "n.wav",
                            "offset_ms": offset,
                            "direction": "delay",
                        }
                    )
                    for sample_id, offset in (("expected", 500), ("unexpected", 999))
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "validate_manifest.py",
                "--manifest",
                str(manifest),
                "--dataset-root",
                str(root),
                "--skip-file-check",
                "--require-offset",
                "500",
                "--exact-offset-set",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "unexpected offsets are present"):
                    validate_manifest_main()

    def test_validator_rejects_unbalanced_offset_direction_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            cells = (
                (50, "advance"),
                (50, "advance"),
                (100, "delay"),
                (100, "delay"),
            )
            manifest.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "sample_id": f"sample-{index}",
                            "original_video": "v.mp4",
                            "edited_audio": "n.wav",
                            "offset_ms": offset,
                            "direction": direction,
                        }
                    )
                    for index, (offset, direction) in enumerate(cells)
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "validate_manifest.py",
                "--manifest",
                str(manifest),
                "--dataset-root",
                str(root),
                "--skip-file-check",
                "--require-offset",
                "50",
                "--require-offset",
                "100",
                "--require-direction",
                "advance",
                "--require-direction",
                "delay",
                "--expect-per-offset",
                "2",
                "--expect-per-direction",
                "2",
                "--expect-per-offset-direction",
                "1",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    SystemExit, "per-offset/direction counts differ"
                ):
                    validate_manifest_main()


if __name__ == "__main__":
    unittest.main()
