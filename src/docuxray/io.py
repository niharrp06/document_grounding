from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import BoundingBox, OCRWord
from .normalization import normalize_field_requests


def read_fields(path: str | Path) -> dict[str, str]:
    data = _read_json(path)
    if not isinstance(data, dict) or not data:
        raise ValueError("Fields file must be a non-empty JSON object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
        raise ValueError("Every field name and field value must be a string")
    return data


def read_field_requests(path: str | Path):
    return normalize_field_requests(_read_json(path))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ocr_words_from_dicts(items: list[dict[str, Any]]) -> list[OCRWord]:
    return [
        OCRWord(
            text=item["text"],
            box=BoundingBox(*item["box"]),
            confidence=float(item.get("confidence", 1.0)),
        )
        for item in items
    ]


def _read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
