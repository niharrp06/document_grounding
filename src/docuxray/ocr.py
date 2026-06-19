from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import polygon_to_box
from .io import ocr_words_from_dicts
from .types import OCRWord


class JsonOCRExtractor:
    def __init__(self, ocr_path: str | Path) -> None:
        self.ocr_path = Path(ocr_path)

    def extract(self, image_path: str | Path | None = None) -> list[OCRWord]:
        import json

        data = json.loads(self.ocr_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("words", [])
        if not isinstance(data, list):
            raise ValueError("OCR JSON must be a list of words or an object with a words list")
        return ocr_words_from_dicts(data)


class PaddleOCRExtractor:
    def __init__(self, *, lang: str = "en", use_angle_cls: bool = True) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed or is unavailable for this Python version. "
                "On Python 3.14, run grounding with --ocr-json, or use a Python version "
                "supported by PaddlePaddle for live OCR."
            ) from exc
        self._engine = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls)
        self._use_angle_cls = use_angle_cls

    def extract(self, image_path: str | Path) -> list[OCRWord]:
        result = self._engine.ocr(str(image_path), cls=self._use_angle_cls)
        words: list[OCRWord] = []
        for line in _iter_lines(result):
            points, text_and_confidence = line
            text, confidence = text_and_confidence
            if text.strip():
                words.append(
                    OCRWord(
                        text=text,
                        box=polygon_to_box(points),
                        confidence=float(confidence),
                    )
                )
        return words


def _iter_lines(result: Any):
    if not result:
        return
    for page in result:
        if not page:
            continue
        for line in page:
            if _is_ocr_line(line):
                yield line


def _is_ocr_line(line: Any) -> bool:
    return (
        isinstance(line, (list, tuple))
        and len(line) == 2
        and isinstance(line[1], (list, tuple))
        and len(line[1]) == 2
    )
