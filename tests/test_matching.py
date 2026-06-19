from docuxray.matching import bio_labels_for_fields, box_for_indices, find_value_span
from docuxray.types import BoundingBox, OCRWord


def word(text: str, x1: int, x2: int) -> OCRWord:
    return OCRWord(text=text, box=BoundingBox(x1, 2, x2, 14), confidence=0.99)


def test_find_value_span_handles_punctuation_and_case() -> None:
    words = [word("Invoice", 0, 42), word("INV-001", 50, 95), word("Total", 100, 130)]
    indices, score = find_value_span(words, "inv001")
    assert indices == [1]
    assert score == 1.0


def test_find_value_span_merges_multiple_words() -> None:
    words = [word("Name", 0, 30), word("Ada", 40, 62), word("Lovelace", 68, 120)]
    indices, _ = find_value_span(words, "Ada Lovelace")
    assert indices == [1, 2]
    assert box_for_indices(words, indices) == BoundingBox(40, 2, 120, 14)


def test_bio_labels_for_fields_marks_matching_tokens() -> None:
    words = [word("INV-001", 0, 48), word("Ada", 60, 82), word("Lovelace", 88, 140)]
    labels = bio_labels_for_fields(
        words, {"invoice_number": "INV-001", "customer_name": "Ada Lovelace"}
    )
    assert labels == ["B-invoice_number", "B-customer_name", "I-customer_name"]
