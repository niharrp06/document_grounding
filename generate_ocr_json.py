from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest import result


def generate_ocr_json(
    image_path: str | Path,
    output_path: str | Path | None = None,
    *,
    lang: str = "en",
    use_angle_cls: bool = True,
    min_confidence: float = 0.0,
) -> Path:
    """Run PaddleOCR and write DocuXray-compatible OCR JSON.

    PaddleOCR returns detected text regions with quadrilateral boxes. DocuXray accepts
    them as OCR word/line entries in this format:

    {
      "words": [
        {
          "id": "w0",
          "text": "Invoice No: 2921",
          "box": [x1, y1, x2, y2],
          "confidence": 0.98,
          "page": 0
        }
      ]
    }
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed in this Python environment. Install it in a Paddle-compatible "
            "environment, for example: pip install paddleocr paddlepaddle"
        ) from exc

    image_path = Path(image_path)
    if output_path is None:
        output_path = Path("outputs") / "ocr" / f"{image_path.stem}.ocr.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls)
    result = engine.predict(str(image_path))

    print("OCR completed")
    print(type(result))

    words: list[dict[str, Any]] = []
    for line in iter_paddle_lines(result):
        points, text_and_confidence = line
        text, confidence = text_and_confidence
        text = str(text).strip()
        confidence = float(confidence)
        if not text or confidence < min_confidence:
            continue
        words.append(
            {
                "id": f"w{len(words)}",
                "text": text,
                "box": polygon_to_box(points),
                "confidence": confidence,
                "page": 0,
            }
        )

    output_path.write_text(json.dumps({"words": words}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def iter_paddle_lines(result: Any):
    if not result:
        return
    for page in result:
        if not page:
            continue
        for line in page:
            if is_paddle_line(line):
                yield line


def is_paddle_line(line: Any) -> bool:
    return (
        isinstance(line, (list, tuple))
        and len(line) == 2
        and isinstance(line[1], (list, tuple))
        and len(line[1]) == 2
    )


def polygon_to_box(points: Any) -> list[int]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DocuXray OCR JSON from an image using PaddleOCR.")
    parser.add_argument("--image", required=True, help="Path to the input document image")
    parser.add_argument("--output", help="Optional output JSON path. Defaults to outputs/ocr/<image>.ocr.json")
    parser.add_argument("--lang", default="en", help="PaddleOCR language code, default: en")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence from 0.0 to 1.0")
    parser.add_argument("--no-angle-cls", action="store_true", help="Disable PaddleOCR angle classifier")
    args = parser.parse_args()

    try:
        output_path = generate_ocr_json(
            args.image,
            args.output,
            lang=args.lang,
            use_angle_cls=not args.no_angle_cls,
            min_confidence=args.min_confidence,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"OCR JSON written to: {output_path}")


if __name__ == "__main__":
    main()
