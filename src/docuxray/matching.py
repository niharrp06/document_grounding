from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher

from .geometry import merge_boxes
from .normalization import normalize_text
from .types import BoundingBox, OCRWord


def find_value_span(
    words: list[OCRWord],
    value: str,
    *,
    min_score: float = 0.84,
    max_extra_words: int = 2,
) -> tuple[list[int], float]:
    target = normalize_text(value)
    if not target:
        return [], 0.0

    expected_words = max(1, len(value.split()))
    best_indices: list[int] = []
    best_score = 0.0
    min_window = max(1, expected_words - 1)
    max_window = min(len(words), expected_words + max_extra_words)

    for start in range(len(words)):
        for size in range(min_window, max_window + 1):
            end = start + size
            if end > len(words):
                break
            candidate = normalize_text(" ".join(word.text for word in words[start:end]))
            score = SequenceMatcher(None, candidate, target).ratio()
            if score > best_score:
                best_indices = list(range(start, end))
                best_score = score
    return (best_indices, best_score) if best_score >= min_score else ([], best_score)


def box_for_indices(words: list[OCRWord], indices: Iterable[int]) -> BoundingBox | None:
    return merge_boxes(words[index].box for index in indices)


def bio_labels_for_fields(words: list[OCRWord], fields: dict[str, str]) -> list[str]:
    labels = ["O"] * len(words)
    for field, value in fields.items():
        indices, _ = find_value_span(words, value)
        if not indices:
            continue
        labels[indices[0]] = f"B-{field}"
        for index in indices[1:]:
            labels[index] = f"I-{field}"
    return labels
