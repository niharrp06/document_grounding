from __future__ import annotations

import json
from pathlib import Path

from .io import read_jsonl
from .matching import bio_labels_for_fields
from .ocr import PaddleOCRExtractor


def prepare_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    lang: str = "en",
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is not installed. Install dependencies with: pip install -e .") from exc

    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path)
    extractor = PaddleOCRExtractor(lang=lang)
    prepared = []

    for record in read_jsonl(manifest_path):
        image_path = _resolve_image_path(manifest_path, record["image"])
        fields = record["fields"]
        words = extractor.extract(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        prepared.append(
            {
                "image": str(image_path),
                "width": width,
                "height": height,
                "fields": fields,
                "words": [word.to_dict() for word in words],
                "labels": bio_labels_for_fields(words, fields),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in prepared:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_image_path(manifest_path: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
