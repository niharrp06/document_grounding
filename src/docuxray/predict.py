from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .geometry import merge_boxes, normalize_box
from .matching import box_for_indices, find_value_span
from .ocr import PaddleOCRExtractor
from .types import FieldBox, OCRWord


def predict_fields(
    image_path: str | Path,
    requested_fields: dict[str, str],
    *,
    model_dir: str | Path | None = None,
    lang: str = "en",
) -> list[FieldBox]:
    words = PaddleOCRExtractor(lang=lang).extract(image_path)
    model_spans = _predict_model_spans(image_path, words, model_dir) if model_dir else {}
    results = []
    for field, value in requested_fields.items():
        indices, match_score = find_value_span(words, value)
        if indices:
            results.append(
                FieldBox(field, value, box_for_indices(words, indices), match_score, "ocr_text_match")
            )
            continue
        model_prediction = model_spans.get(field)
        if model_prediction:
            box, confidence = model_prediction
            results.append(FieldBox(field, value, box, confidence, "layoutlmv3"))
            continue
        results.append(FieldBox(field, value, None, 0.0, "not_found"))
    return results


def _predict_model_spans(
    image_path: str | Path,
    words: list[OCRWord],
    model_dir: str | Path | None,
) -> dict[str, tuple[Any, float]]:
    if not words:
        return {}
    try:
        import torch
        from PIL import Image
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    except ImportError as exc:
        raise RuntimeError(
            "Model inference dependencies are not installed. Install them with: pip install -e '.[ml]'"
        ) from exc

    processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    model.eval()
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        boxes = [normalize_box(word.box, *rgb_image.size) for word in words]
        encoding = processor(
            rgb_image,
            [word.text for word in words],
            boxes=boxes,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    with torch.no_grad():
        probabilities = model(**encoding).logits.softmax(dim=-1)[0]

    by_word: dict[int, tuple[str, float]] = {}
    for token_index, word_index in enumerate(encoding.word_ids(batch_index=0)):
        if word_index is None:
            continue
        confidence, label_id = probabilities[token_index].max(dim=-1)
        label = model.config.id2label[int(label_id)]
        score = float(confidence)
        if word_index not in by_word or score > by_word[word_index][1]:
            by_word[word_index] = (label, score)

    spans: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for word_index, (label, score) in by_word.items():
        if label != "O" and "-" in label:
            _, field = label.split("-", maxsplit=1)
            spans[field].append((word_index, score))
    return {
        field: (
            merge_boxes(words[index].box for index, _ in tokens),
            sum(score for _, score in tokens) / len(tokens),
        )
        for field, tokens in spans.items()
    }
