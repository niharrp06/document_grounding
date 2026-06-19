from __future__ import annotations

import json
from pathlib import Path

from .graph import build_document_graph
from .grounding import GroundingEngine
from .io import read_field_requests
from .ocr import JsonOCRExtractor, PaddleOCRExtractor
from .tables import attach_tables, load_tables
from .types import DocumentPage, GroundedField


def run_grounding_pipeline(
    image_path: str | Path,
    fields_path: str | Path,
    *,
    tables_path: str | Path | None = None,
    ocr_json_path: str | Path | None = None,
    ocr_backend: str = "paddle",
    output_graph_path: str | Path | None = None,
    lang: str = "en",
) -> list[GroundedField]:
    image_path = Path(image_path)
    requests = read_field_requests(fields_path)
    words = _extract_words(image_path, ocr_json_path=ocr_json_path, ocr_backend=ocr_backend, lang=lang)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read image dimensions. Install with: pip install -e .") from exc
    with Image.open(image_path) as image:
        page = DocumentPage(page=0, width=image.width, height=image.height, image_path=str(image_path))
    graph = build_document_graph([page], words)
    tables, rows, cells = load_tables(tables_path)
    attach_tables(graph, tables, rows, cells)
    if output_graph_path:
        Path(output_graph_path).write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
    return GroundingEngine().ground(graph, requests)


def _extract_words(
    image_path: Path,
    *,
    ocr_json_path: str | Path | None,
    ocr_backend: str,
    lang: str,
):
    if ocr_json_path:
        return JsonOCRExtractor(ocr_json_path).extract(image_path)
    if ocr_backend == "none":
        return []
    if ocr_backend == "paddle":
        return PaddleOCRExtractor(lang=lang).extract(image_path)
    raise ValueError("ocr_backend must be one of: paddle, none")
