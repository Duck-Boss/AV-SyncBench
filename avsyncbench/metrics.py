"""Pairwise AV-SyncBench metrics and deterministic result serialization."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union


@dataclass(frozen=True)
class PairwiseResult:
    sample_id: str
    task: str
    positive_score: Optional[float]
    negative_score: Optional[float]
    edit_type: Optional[str] = None
    offset_ms: Optional[int] = None
    direction: Optional[str] = None
    category: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.error is None
            and self.positive_score is not None
            and self.negative_score is not None
            and math.isfinite(self.positive_score)
            and math.isfinite(self.negative_score)
        )

    @property
    def is_tie(self) -> bool:
        return self.is_valid and self.positive_score == self.negative_score

    @property
    def correct(self) -> bool:
        # AV-SyncBench uses a strict comparison; ties are counted as incorrect.
        return self.is_valid and self.positive_score > self.negative_score

    def to_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output["valid"] = self.is_valid
        output["tie"] = self.is_tie
        output["correct"] = self.correct
        return output


def _group_summary(results: List[PairwiseResult]) -> Dict[str, Any]:
    valid = [result for result in results if result.is_valid]
    correct = sum(result.correct for result in valid)
    ties = sum(result.is_tie for result in valid)
    return {
        "total": len(results),
        "valid": len(valid),
        "errors": len(results) - len(valid),
        "correct": correct,
        "ties": ties,
        # `accuracy` matches the valid-only research metric. The all-record
        # variant prevents a permissive continue-on-error run from looking
        # better simply because failures disappeared from the denominator.
        "accuracy": None if not valid else correct / len(valid),
        "accuracy_all_records": None if not results else correct / len(results),
        "error_rate": None if not results else (len(results) - len(valid)) / len(results),
    }


def summarize_results(results: Iterable[PairwiseResult]) -> Dict[str, Any]:
    results = list(results)
    by_offset: Dict[str, List[PairwiseResult]] = defaultdict(list)
    by_edit_type: Dict[str, List[PairwiseResult]] = defaultdict(list)
    by_category: Dict[str, List[PairwiseResult]] = defaultdict(list)
    by_direction: Dict[str, List[PairwiseResult]] = defaultdict(list)
    for result in results:
        if result.offset_ms is not None:
            by_offset[str(result.offset_ms)].append(result)
        if result.edit_type:
            by_edit_type[result.edit_type].append(result)
        if result.category:
            by_category[result.category].append(result)
        if result.direction:
            by_direction[result.direction].append(result)
    return {
        "overall": _group_summary(results),
        "by_offset_ms": {
            key: _group_summary(group)
            for key, group in sorted(by_offset.items(), key=lambda item: int(item[0]))
        },
        "by_edit_type": {key: _group_summary(group) for key, group in sorted(by_edit_type.items())},
        "by_category": {key: _group_summary(group) for key, group in sorted(by_category.items())},
        "by_direction": {key: _group_summary(group) for key, group in sorted(by_direction.items())},
    }


def write_jsonl(path: Union[Path, str], results: Iterable[PairwiseResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def write_summary(path: Union[Path, str], summary: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
