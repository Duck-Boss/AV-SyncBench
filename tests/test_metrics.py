import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avsyncbench.metrics import PairwiseResult, summarize_results, write_jsonl
from avsyncbench.runtime import (
    require_distinct_paths,
    require_package_versions,
    require_paths_not_in,
    require_upstream_checkout,
    sanitized_exception,
)


class MetricTests(unittest.TestCase):
    def test_nonfinite_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.jsonl"
            result = PairwiseResult("nan", "global_offset", math.nan, 0.0)
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                write_jsonl(output, [result])

    def test_package_version_check_accepts_local_wheel_suffix(self) -> None:
        with patch("avsyncbench.runtime.package_version", return_value="2.0.1+cu118"):
            require_package_versions({"torch": "2.0.1"})

    def test_package_version_check_rejects_mismatch(self) -> None:
        with patch("avsyncbench.runtime.package_version", return_value="2.5.1"):
            with self.assertRaisesRegex(RuntimeError, "incompatible package versions"):
                require_package_versions({"torch": "2.0.1"})

    def test_path_collisions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "path collision"):
            require_distinct_paths({"input": Path("same"), "output": Path("same")})

    def test_output_cannot_overwrite_protected_media(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected input"):
            require_paths_not_in(
                {"output": Path("media.mp4")},
                [Path("other.wav"), Path("media.mp4")],
            )

    def test_upstream_checkout_must_be_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "required.py").touch()
            with patch(
                "avsyncbench.runtime.git_revision", return_value="a" * 40
            ), patch(
                "avsyncbench.runtime.command_output", return_value=" M required.py"
            ):
                with self.assertRaisesRegex(ValueError, "modified or untracked"):
                    require_upstream_checkout(
                        root,
                        expected_commit="a" * 40,
                        required_files=("required.py",),
                    )

    def test_sanitized_exception_redacts_dataset_root(self) -> None:
        root = Path("private") / "dataset"
        absolute = root.resolve()
        error = FileNotFoundError(absolute / "audio" / "sample.wav")
        rendered = sanitized_exception(error, {"dataset_root": absolute})
        self.assertIn("<dataset_root>", rendered)
        self.assertNotIn(str(absolute), rendered)

    def test_strict_pairwise_metric_counts_tie_as_incorrect(self) -> None:
        results = [
            PairwiseResult("a", "global_offset", 0.9, 0.3, offset_ms=50),
            PairwiseResult("b", "global_offset", 0.4, 0.4, offset_ms=50),
            PairwiseResult("c", "global_offset", None, None, offset_ms=100, error="decode"),
        ]
        summary = summarize_results(results)
        self.assertEqual(
            summary["overall"],
            {
                "total": 3,
                "valid": 2,
                "errors": 1,
                "correct": 1,
                "ties": 1,
                "accuracy": 0.5,
                "accuracy_all_records": 1 / 3,
                "error_rate": 1 / 3,
            },
        )
        self.assertEqual(summary["by_offset_ms"]["50"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
